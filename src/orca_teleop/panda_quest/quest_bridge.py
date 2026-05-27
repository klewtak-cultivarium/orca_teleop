from __future__ import annotations

import asyncio
import copy
import json
import ssl
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from orca_teleop.panda_quest.transforms import xr_matrix_to_mujoco_matrix

HAND_SIDES = ("left", "right")
EVENT_NAMES = ("sync", "done", "reset", "recenter")


class QuestTelemetryState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._head_matrix: np.ndarray | None = None
        self._controller_matrices: dict[str, np.ndarray | None] = {
            side: None for side in HAND_SIDES
        }
        self._controller_axes: dict[str, list[float]] = {side: [] for side in HAND_SIDES}
        self._controller_buttons: dict[str, list[float]] = {side: [] for side in HAND_SIDES}
        self._hand_wrist_matrices: dict[str, np.ndarray | None] = {
            side: None for side in HAND_SIDES
        }
        self._hand_landmarks: dict[str, np.ndarray | None] = {side: None for side in HAND_SIDES}
        self._events: dict[str, bool] = {name: False for name in EVENT_NAMES}
        self._last_update_monotonic = 0.0
        self._last_update_wall = 0.0
        self._last_client_wall_ms: float | None = None

    def update(self, payload: Mapping[str, Any]) -> None:
        with self._lock:
            head = payload.get("head")
            if head is not None:
                self._head_matrix = xr_matrix_to_mujoco_matrix(head)

            controllers = payload.get("controllers", {})
            for side in HAND_SIDES:
                controller_payload = controllers.get(side)
                if controller_payload is None:
                    self._controller_matrices[side] = None
                    self._controller_axes[side] = []
                    self._controller_buttons[side] = []
                    continue

                grip = controller_payload.get("grip")
                self._controller_matrices[side] = (
                    xr_matrix_to_mujoco_matrix(grip) if grip is not None else None
                )
                self._controller_axes[side] = list(controller_payload.get("axes", []))
                self._controller_buttons[side] = list(controller_payload.get("buttons", []))

            hands = payload.get("hands", {})
            for side in HAND_SIDES:
                hand_payload = hands.get(side)
                if hand_payload is None:
                    self._hand_wrist_matrices[side] = None
                    self._hand_landmarks[side] = None
                    continue

                wrist = hand_payload.get("wrist")
                self._hand_wrist_matrices[side] = (
                    xr_matrix_to_mujoco_matrix(wrist) if wrist is not None else None
                )
                landmarks = hand_payload.get("landmarks")
                if landmarks is None:
                    self._hand_landmarks[side] = None
                else:
                    landmarks_array = np.asarray(landmarks, dtype=np.float64)
                    self._hand_landmarks[side] = (
                        landmarks_array if landmarks_array.shape == (25, 3) else None
                    )

            for event_name in EVENT_NAMES:
                if payload.get("events", {}).get(event_name):
                    self._events[event_name] = True

            client_wall_ms = payload.get("client_wall_ms")
            self._last_client_wall_ms = (
                float(client_wall_ms) if isinstance(client_wall_ms, (int, float)) else None
            )
            self._last_update_wall = time.time()
            self._last_update_monotonic = time.monotonic()

    def push_event(self, event_name: str) -> None:
        if event_name not in EVENT_NAMES:
            raise KeyError(f"Unknown Quest control event: {event_name}")
        with self._lock:
            self._events[event_name] = True

    def get_head_matrix(self) -> np.ndarray | None:
        with self._lock:
            return None if self._head_matrix is None else self._head_matrix.copy()

    def get_controller_matrix(self, side: str) -> np.ndarray | None:
        with self._lock:
            matrix = self._controller_matrices[side]
            return None if matrix is None else matrix.copy()

    def get_controller_axes(self, side: str) -> list[float]:
        with self._lock:
            return list(self._controller_axes[side])

    def get_controller_buttons(self, side: str) -> list[float]:
        with self._lock:
            return list(self._controller_buttons[side])

    def get_hand_wrist_matrix(self, side: str) -> np.ndarray | None:
        with self._lock:
            matrix = self._hand_wrist_matrices[side]
            return None if matrix is None else matrix.copy()

    def get_hand_landmarks(self, side: str) -> np.ndarray | None:
        with self._lock:
            landmarks = self._hand_landmarks[side]
            return None if landmarks is None else landmarks.copy()

    def pop_event(self, event_name: str) -> bool:
        with self._lock:
            value = self._events[event_name]
            self._events[event_name] = False
            return value

    @property
    def last_update_monotonic(self) -> float:
        with self._lock:
            return self._last_update_monotonic

    @property
    def last_update_wall(self) -> float:
        with self._lock:
            return self._last_update_wall

    @property
    def last_client_wall_ms(self) -> float | None:
        with self._lock:
            return self._last_client_wall_ms


class QuestTelemetryBridge:
    """WebXR signaling + WebSocket telemetry bridge for Quest input.

    Telemetry runs over a persistent WebSocket on the same aiohttp server, so
    the full pipe (page load + telemetry) goes through one TCP connection.
    That makes USB tunneling via ``adb reverse tcp:8765 tcp:8765`` work end to
    end and removes the WebRTC ICE/consent failure modes from the equation.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.ssl_context = ssl_context
        self.state = QuestTelemetryState()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._runner: Any | None = None
        self._connected = threading.Event()
        self._started = threading.Event()
        self._last_telemetry_payload: dict[str, Any] | None = None
        self._telemetry_packet_count = 0
        self._websockets: set[Any] = set()

    @property
    def url(self) -> str:
        scheme = "https" if self.ssl_context is not None else "http"
        return f"{scheme}://{self.host}:{self.port}"

    def wait_until_connected(self, timeout: float | None = None) -> bool:
        return self._connected.wait(timeout)

    def get_last_telemetry_payload(self) -> dict[str, Any] | None:
        return (
            None
            if self._last_telemetry_payload is None
            else copy.deepcopy(self._last_telemetry_payload)
        )

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="quest-telemetry")
        self._thread.start()
        if not self._started.wait(timeout=10.0):
            raise RuntimeError("Quest telemetry bridge did not start within 10 seconds.")

    def stop(self) -> None:
        if self._loop is None:
            return
        future = asyncio.run_coroutine_threadsafe(self._async_shutdown(), self._loop)
        future.result(timeout=10.0)
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._async_start())
        self._started.set()
        loop.run_forever()
        loop.close()

    async def _async_start(self) -> None:
        from aiohttp import web

        web_dir = Path(__file__).resolve().parent / "web"
        self._index_path = web_dir / "index.html"
        self._app_js_path = web_dir / "app.js"
        self._style_css_path = web_dir / "style.css"

        app = web.Application()
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/config.json", self._handle_config)
        app.router.add_get("/app.js", self._handle_app_js)
        app.router.add_get("/style.css", self._handle_style_css)
        app.router.add_get("/ws", self._handle_ws)
        app.router.add_post("/session-config", self._handle_session_config)
        app.router.add_post("/debug/client-log", self._handle_debug_client_log)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port, ssl_context=self.ssl_context)
        await site.start()

    async def _async_shutdown(self) -> None:
        for ws in list(self._websockets):
            await ws.close()
        self._websockets.clear()
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def _handle_index(self, request: Any) -> Any:
        from aiohttp import web

        response = web.FileResponse(self._index_path)
        response.headers["Cache-Control"] = "no-store"
        return response

    async def _handle_config(self, request: Any) -> Any:
        from aiohttp import web

        return web.json_response(
            {"telemetry_only": True, "quest_input_mode": "controller"},
            headers={"Cache-Control": "no-store"},
        )

    async def _handle_app_js(self, request: Any) -> Any:
        from aiohttp import web

        response = web.FileResponse(self._app_js_path)
        response.headers["Cache-Control"] = "no-store"
        return response

    async def _handle_style_css(self, request: Any) -> Any:
        from aiohttp import web

        response = web.FileResponse(self._style_css_path)
        response.headers["Cache-Control"] = "no-store"
        return response

    async def _handle_session_config(self, request: Any) -> Any:
        from aiohttp import web

        await request.json()
        return web.json_response({"ok": True})

    async def _handle_debug_client_log(self, request: Any) -> Any:
        from aiohttp import web

        payload = await request.json()
        level = str(payload.get("level", "info")).upper()
        event = str(payload.get("event", "unknown"))
        message = str(payload.get("message", ""))
        print(f"[QuestClient/{level}] {event}: {message}", flush=True)
        return web.json_response({"ok": True})

    async def _handle_ws(self, request: Any) -> Any:
        from aiohttp import WSMsgType, web

        ws = web.WebSocketResponse(heartbeat=10.0, max_msg_size=4 * 1024 * 1024)
        await ws.prepare(request)
        self._websockets.add(ws)
        peer = request.remote
        print(f"Quest telemetry WebSocket opened from {peer}.", flush=True)

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        payload = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue
                    self._dispatch_payload(payload, transport=f"ws:{peer}")
                elif msg.type == WSMsgType.ERROR:
                    print(
                        f"Quest telemetry WebSocket error: {ws.exception()!r}",
                        flush=True,
                    )
                    break
        finally:
            self._websockets.discard(ws)
            print(f"Quest telemetry WebSocket closed from {peer}.", flush=True)
            self._connected.clear()

        return ws

    def _dispatch_payload(self, payload: Mapping[str, Any], *, transport: str) -> None:
        payload_type = payload.get("type")
        if payload_type == "telemetry":
            self._ingest_telemetry(payload, transport=transport)
            return
        if payload_type == "control_event":
            event_name = payload.get("event")
            if isinstance(event_name, str):
                try:
                    self.state.push_event(event_name)
                except KeyError:
                    print(
                        f"Ignoring unknown Quest control event {event_name!r}.",
                        flush=True,
                    )

    def _ingest_telemetry(self, payload: Mapping[str, Any], *, transport: str) -> None:
        self.state.update(payload)
        self._last_telemetry_payload = copy.deepcopy(dict(payload))
        self._connected.set()
        self._telemetry_packet_count += 1
        if self._telemetry_packet_count <= 3:
            controllers = payload.get("controllers", {})
            hands = payload.get("hands", {})
            print(
                "Telemetry packet "
                f"{self._telemetry_packet_count} via {transport}: "
                f"controllers={list(controllers)} hands={list(hands)}",
                flush=True,
            )
