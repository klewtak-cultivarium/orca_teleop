const BUTTON = document.getElementById("start-button");
const STATUS = document.getElementById("status");
const CANVAS = document.getElementById("xr-canvas");

let session = null;
let refSpace = null;
let gl = null;
let xrLayer = null;
let xrSessionMode = null;
let socket = null;
let reconnectTimer = null;
let lastTelemetrySentAtMs = -Infinity;
let lastSocketReadyState = null;
let telemetryPacketCount = 0;

const TARGET_TELEMETRY_HZ = 30;
const MAX_TELEMETRY_BUFFERED_AMOUNT = 128 * 1024;
const RECONNECT_DELAY_MS = 750;
const WEBXR_HAND_JOINTS = [
  "wrist",
  "thumb-metacarpal",
  "thumb-phalanx-proximal",
  "thumb-phalanx-distal",
  "thumb-tip",
  "index-finger-metacarpal",
  "index-finger-phalanx-proximal",
  "index-finger-phalanx-intermediate",
  "index-finger-phalanx-distal",
  "index-finger-tip",
  "middle-finger-metacarpal",
  "middle-finger-phalanx-proximal",
  "middle-finger-phalanx-intermediate",
  "middle-finger-phalanx-distal",
  "middle-finger-tip",
  "ring-finger-metacarpal",
  "ring-finger-phalanx-proximal",
  "ring-finger-phalanx-intermediate",
  "ring-finger-phalanx-distal",
  "ring-finger-tip",
  "pinky-finger-metacarpal",
  "pinky-finger-phalanx-proximal",
  "pinky-finger-phalanx-intermediate",
  "pinky-finger-phalanx-distal",
  "pinky-finger-tip",
];

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

function collectHandTelemetry(frame, inputSource, referenceSpace) {
  if (!inputSource.hand || !inputSource.handedness) {
    return null;
  }

  const wristJoint = inputSource.hand.get("wrist");
  const wristPose = wristJoint ? frame.getJointPose(wristJoint, referenceSpace) : null;
  if (!wristPose) {
    return null;
  }

  const landmarks = [];
  for (const jointName of WEBXR_HAND_JOINTS) {
    const jointSpace = inputSource.hand.get(jointName);
    const jointPose = jointSpace ? frame.getJointPose(jointSpace, referenceSpace) : null;
    if (!jointPose) {
      return null;
    }
    landmarks.push([
      jointPose.transform.position.x,
      jointPose.transform.position.y,
      jointPose.transform.position.z,
    ]);
  }

  return {
    wrist: matrixToArray(wristPose.transform.matrix),
    landmarks,
  };
}

function connectTelemetrySocket() {
  if (reconnectTimer !== null) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
  socket = new WebSocket(wsUrl);

  socket.onopen = () => {
    setStatus("Telemetry WebSocket connected. Quest poses are streaming to the host.");
    reportClientDebug("telemetry-socket-open", `WebSocket connected to ${wsUrl}.`);
  };
  socket.onclose = (event) => {
    reportClientDebug(
      "telemetry-socket-close",
      `WebSocket closed (code=${event.code}, reason=${event.reason || "n/a"}). Reconnecting.`,
    );
    setStatus("Telemetry WebSocket closed; reconnecting...");
    socket = null;
    if (reconnectTimer === null) {
      reconnectTimer = setTimeout(() => {
        reconnectTimer = null;
        connectTelemetrySocket();
      }, RECONNECT_DELAY_MS);
    }
  };
  socket.onerror = (event) => {
    reportClientDebug(
      "telemetry-socket-error",
      event && event.message ? event.message : "WebSocket error",
      null,
      "warning",
    );
  };
}

function sendTelemetry(frame, pose, referenceSpace, nowMs) {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    const readyState = socket ? socket.readyState : "missing";
    if (readyState !== lastSocketReadyState) {
      lastSocketReadyState = readyState;
      reportClientDebug("telemetry-socket-not-ready", "WebSocket is not open.", {
        ready_state: readyState,
      });
    }
    return;
  }
  lastSocketReadyState = socket.readyState;

  if (nowMs - lastTelemetrySentAtMs < 1000.0 / TARGET_TELEMETRY_HZ) {
    return;
  }
  if (socket.bufferedAmount > MAX_TELEMETRY_BUFFERED_AMOUNT) {
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

    const handTelemetry = collectHandTelemetry(frame, inputSource, referenceSpace);
    if (handTelemetry) {
      payload.hands[inputSource.handedness] = handTelemetry;
    }
  }

  socket.send(JSON.stringify(payload));
  lastTelemetrySentAtMs = nowMs;
  telemetryPacketCount += 1;
  if (telemetryPacketCount <= 3) {
    reportClientDebug("telemetry-sent", `Sent telemetry packet ${telemetryPacketCount}.`, {
      input_source_count: session?.inputSources?.length ?? 0,
      controller_sides: Object.keys(payload.controllers),
      hand_sides: Object.keys(payload.hands),
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
  if (xrSessionMode === "immersive-ar") {
    gl.clearColor(0.0, 0.0, 0.0, 0.0);
  } else {
    gl.clearColor(0.02, 0.04, 0.10, 1.0);
  }
  gl.clear(gl.COLOR_BUFFER_BIT);
}

async function selectXrSessionMode() {
  if (await navigator.xr.isSessionSupported("immersive-ar")) {
    return "immersive-ar";
  }
  if (await navigator.xr.isSessionSupported("immersive-vr")) {
    return "immersive-vr";
  }
  throw new Error("This browser supports neither immersive-ar nor immersive-vr.");
}

async function startImmersiveSession() {
  if (!window.isSecureContext) {
    throw new Error(
      "Open this page over HTTPS (or http://localhost via adb reverse) in Quest Browser.",
    );
  }
  if (!navigator.xr) {
    throw new Error("WebXR is unavailable here. Use Quest Browser on the headset.");
  }

  xrSessionMode = await selectXrSessionMode();
  setStatus(`Requesting ${xrSessionMode} session...`);
  session = await navigator.xr.requestSession(xrSessionMode, {
    optionalFeatures: ["local-floor", "hand-tracking"],
  });
  gl = CANVAS.getContext("webgl", {
    alpha: xrSessionMode === "immersive-ar",
    xrCompatible: true,
  });
  xrLayer = new XRWebGLLayer(session, gl, {
    alpha: xrSessionMode === "immersive-ar",
  });
  session.updateRenderState({ baseLayer: xrLayer });
  reportClientDebug(
    "xr-session-started",
    `Started ${xrSessionMode} controller/hand telemetry session.`,
  );
  await sendSessionConfig();

  try {
    refSpace = await session.requestReferenceSpace("local");
  } catch (_error) {
    refSpace = await session.requestReferenceSpace("local-floor");
  }

  session.addEventListener("end", () => {
    setStatus("XR session ended.");
    session = null;
    xrSessionMode = null;
    if (socket) {
      socket.close();
      socket = null;
    }
    if (reconnectTimer !== null) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
  });

  connectTelemetrySocket();
  setStatus("XR session started. Waiting for Quest telemetry...");
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
