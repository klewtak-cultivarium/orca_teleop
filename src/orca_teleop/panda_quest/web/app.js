const BUTTON = document.getElementById("start-button");
const STATUS = document.getElementById("status");
const CANVAS = document.getElementById("xr-canvas");

let session = null;
let refSpace = null;
let gl = null;
let xrLayer = null;
let pc = null;
let telemetryChannel = null;
let controlChannel = null;
let lastTelemetrySentAtMs = -Infinity;
let lastTelemetryReadyState = null;
let telemetryPacketCount = 0;

const TARGET_TELEMETRY_HZ = 30;
const MAX_TELEMETRY_BUFFERED_AMOUNT = 128 * 1024;
const DEFAULT_ICE_SERVERS = [{ urls: ["stun:stun.l.google.com:19302"] }];

function setStatus(message) {
  STATUS.textContent = message;
}

function matrixToArray(matrix) {
  return Array.from(matrix);
}

function reportClientDebug(event, message, extra = null, level = "info") {
  console.log(`[QuestClient/${level}] ${event}: ${message}`, extra ?? "");
  void fetch("/debug/client-log", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ level, event, message, extra, timestamp_ms: Date.now() }),
    keepalive: true,
  }).catch(() => {});
}

async function sendSessionConfig() {
  await fetch("/session-config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      framebuffer_width: xrLayer.framebufferWidth,
      framebuffer_height: xrLayer.framebufferHeight,
    }),
  });
}

function collectControllerTelemetry(frame, inputSource, referenceSpace) {
  if (!inputSource.gripSpace) {
    return null;
  }
  const gripPose = frame.getPose(inputSource.gripSpace, referenceSpace);
  if (!gripPose) {
    return null;
  }
  return {
    grip: matrixToArray(gripPose.transform.matrix),
    axes: inputSource.gamepad ? Array.from(inputSource.gamepad.axes) : [],
    buttons: inputSource.gamepad ? inputSource.gamepad.buttons.map((button) => button.value) : [],
  };
}

async function negotiatePeerConnection() {
  pc = new RTCPeerConnection({ iceServers: DEFAULT_ICE_SERVERS });
  telemetryChannel = pc.createDataChannel("telemetry", { ordered: false, maxRetransmits: 0 });
  controlChannel = pc.createDataChannel("control");
  telemetryChannel.bufferedAmountLowThreshold = Math.floor(MAX_TELEMETRY_BUFFERED_AMOUNT / 2);

  pc.onconnectionstatechange = () => {
    setStatus(`WebRTC connection: ${pc.connectionState}`);
    reportClientDebug("pc-connection-state", `RTCPeerConnection state changed to ${pc.connectionState}.`);
  };

  telemetryChannel.onopen = () => {
    setStatus("Telemetry channel open. Hold the right controller trigger to drive the arm.");
    reportClientDebug("telemetry-channel-open", "Telemetry data channel opened.");
  };
  telemetryChannel.onclose = () => {
    reportClientDebug("telemetry-channel-close", "Telemetry data channel closed.");
  };
  controlChannel.onopen = () => {
    reportClientDebug("control-channel-open", "Control channel opened.");
  };

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  const response = await fetch("/offer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sdp: offer.sdp, type: offer.type }),
  });
  if (!response.ok) {
    throw new Error(`Offer failed with ${response.status}`);
  }

  const answer = await response.json();
  await pc.setRemoteDescription(answer);
}

function sendTelemetry(frame, pose, referenceSpace, nowMs) {
  if (!telemetryChannel || telemetryChannel.readyState !== "open") {
    const readyState = telemetryChannel ? telemetryChannel.readyState : "missing";
    if (readyState !== lastTelemetryReadyState) {
      lastTelemetryReadyState = readyState;
      reportClientDebug("telemetry-not-ready", "Telemetry channel is not open.", { ready_state: readyState });
    }
    return;
  }

  if (nowMs - lastTelemetrySentAtMs < 1000.0 / TARGET_TELEMETRY_HZ) {
    return;
  }
  if (telemetryChannel.bufferedAmount > MAX_TELEMETRY_BUFFERED_AMOUNT) {
    return;
  }

  const payload = {
    type: "telemetry",
    timestamp_ms: nowMs,
    head: matrixToArray(pose.transform.matrix),
    controllers: {},
    hands: {},
  };

  for (const inputSource of session.inputSources) {
    const controllerTelemetry = collectControllerTelemetry(frame, inputSource, referenceSpace);
    if (controllerTelemetry) {
      payload.controllers[inputSource.handedness] = controllerTelemetry;
    }
  }

  telemetryChannel.send(JSON.stringify(payload));
  lastTelemetrySentAtMs = nowMs;
  telemetryPacketCount += 1;
  if (telemetryPacketCount <= 3) {
    reportClientDebug("telemetry-sent", `Sent telemetry packet ${telemetryPacketCount}.`, {
      input_source_count: session?.inputSources?.length ?? 0,
      controller_sides: Object.keys(payload.controllers),
    });
  }
}

function onXrFrame(nowMs, frame) {
  session.requestAnimationFrame(onXrFrame);
  const pose = frame.getViewerPose(refSpace);
  if (!pose) {
    return;
  }
  sendTelemetry(frame, pose, refSpace, nowMs);

  gl.bindFramebuffer(gl.FRAMEBUFFER, xrLayer.framebuffer);
  gl.clearColor(0.0, 0.0, 0.0, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT);
}

async function startImmersiveSession() {
  if (!window.isSecureContext) {
    throw new Error("Open this page over HTTPS in Quest Browser. An ngrok HTTPS URL is the quickest path.");
  }
  if (!navigator.xr) {
    throw new Error("WebXR is unavailable here. Use Quest Browser on the headset.");
  }
  const supported = await navigator.xr.isSessionSupported("immersive-vr");
  if (!supported) {
    throw new Error("This browser does not support immersive-vr.");
  }

  setStatus("Requesting VR session...");
  session = await navigator.xr.requestSession("immersive-vr", {
    optionalFeatures: ["local-floor"],
  });
  gl = CANVAS.getContext("webgl", { xrCompatible: true });
  xrLayer = new XRWebGLLayer(session, gl);
  session.updateRenderState({ baseLayer: xrLayer });
  reportClientDebug("xr-session-started", "Started VR controller telemetry session.");
  await sendSessionConfig();

  try {
    refSpace = await session.requestReferenceSpace("local");
  } catch (_error) {
    refSpace = await session.requestReferenceSpace("local-floor");
  }

  session.addEventListener("end", () => {
    setStatus("XR session ended.");
    session = null;
  });

  await negotiatePeerConnection();
  setStatus("XR session started. Waiting for controller telemetry...");
  session.requestAnimationFrame(onXrFrame);
}

BUTTON.addEventListener("click", async () => {
  BUTTON.disabled = true;
  try {
    setStatus("Connecting to host...");
    await startImmersiveSession();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus(message || "Failed to start teleop.");
    reportClientDebug("start-failed", message || "Failed to start teleop.", null, "error");
    BUTTON.disabled = false;
  }
});
