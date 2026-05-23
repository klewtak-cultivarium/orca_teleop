"""Manus MetaGloves Pro gRPC publisher.

Subscribes to the C++ ``SharpaManusClient`` over ZMQ, converts the 25-joint
Manus skeleton to the 21-joint MANO layout expected by the retargeter, and
streams ``HandFrame`` messages to the ``IngressServer`` via gRPC.

Usage::

    # Stream right-hand glove data to local pipeline
    python -m orca_teleop.ingress.manus.publisher

    # Stream to a remote robot
    python -m orca_teleop.ingress.manus.publisher --server 192.168.1.42:50051

    # Left hand
    python -m orca_teleop.ingress.manus.publisher --hand left
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time

import grpc
import numpy as np

# Generated protobuf module for the Manus ZMQ wire format.
_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "source")
sys.path.insert(0, _SOURCE_DIR)

import manus_stream_pb2  # noqa: E402

from orca_teleop.ingress import hand_stream_pb2, hand_stream_pb2_grpc  # noqa: E402
from orca_teleop.ingress.manus.conversion import MANUS_TO_MANO, manus_zmq_to_mano_keypoints  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_ZMQ_ADDRESS = "tcp://127.0.0.1:2044"


class ManusPublisher:
    """Receives Manus glove data over ZMQ and streams it to IngressServer via gRPC."""

    def __init__(
        self,
        server_address: str = "localhost:50051",
        handedness: str = "right",
        zmq_address: str = DEFAULT_ZMQ_ADDRESS,
    ) -> None:
        self._server_address = server_address
        self._handedness = handedness.lower()
        self._zmq_address = zmq_address

        self._lock = threading.Lock()
        self._latest_keypoints: np.ndarray | None = None
        self._fresh = False
        self._stop = False

    def _zmq_receiver(self) -> None:
        """Background thread: subscribe to ZMQ, parse protobuf, extract keypoints."""
        import zmq

        ctx = zmq.Context()
        sub = ctx.socket(zmq.SUB)
        sub.setsockopt(zmq.CONFLATE, 1)
        sub.setsockopt_string(zmq.SUBSCRIBE, "")
        sub.connect(self._zmq_address)
        logger.info("ZMQ subscriber connected to %s", self._zmq_address)

        try:
            while not self._stop:
                try:
                    raw = sub.recv(zmq.NOBLOCK)
                except zmq.Again:
                    time.sleep(0.001)
                    continue

                msg = manus_stream_pb2.GloveState()
                msg.ParseFromString(raw)

                if self._handedness == "left":
                    poses = msg.left_mocap_pose
                else:
                    poses = msg.right_mocap_pose

                if len(poses) == 0:
                    continue

                # Extract (25, 3) positions from protobuf Pose messages
                all_positions = np.array(
                    [[p.position.x, p.position.y, p.position.z] for p in poses],
                    dtype=np.float32,
                )

                if all_positions.shape[0] < 25:
                    logger.debug(
                        "Expected 25 joints, got %d — skipping frame.", all_positions.shape[0]
                    )
                    continue

                # Select 21 MANO joints and convert Z-up → Y-up
                keypoints = manus_zmq_to_mano_keypoints(all_positions[MANUS_TO_MANO])

                with self._lock:
                    self._latest_keypoints = keypoints
                    self._fresh = True
        finally:
            sub.close()
            ctx.term()

    def _frame_generator(self):
        """Yield HandFrame protos as fast as new data arrives."""
        while not self._stop:
            with self._lock:
                if not self._fresh:
                    kp = None
                else:
                    kp = self._latest_keypoints.copy()
                    self._fresh = False

            if kp is None:
                time.sleep(0.001)
                continue

            yield hand_stream_pb2.HandFrame(
                keypoints=kp.ravel().tolist(),
                handedness=self._handedness,
                timestamp_ns=time.time_ns(),
            )

    def run(self) -> None:
        """Start ZMQ receiver thread and gRPC stream, block until interrupted."""
        logger.info(
            "ManusPublisher starting (server=%s, hand=%s, zmq=%s)",
            self._server_address,
            self._handedness,
            self._zmq_address,
        )

        zmq_thread = threading.Thread(target=self._zmq_receiver, name="zmq-receiver", daemon=True)
        zmq_thread.start()

        channel = grpc.insecure_channel(self._server_address)
        stub = hand_stream_pb2_grpc.HandStreamStub(channel)
        stream_future = stub.StreamHandFrames.future(self._frame_generator())

        try:
            stream_future.result()
        except KeyboardInterrupt:
            pass
        except grpc.RpcError as e:
            logger.error("gRPC stream ended: %s", e)
        finally:
            self._stop = True
            stream_future.cancel()
            channel.close()
            zmq_thread.join(timeout=2.0)
            logger.info("ManusPublisher shut down.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream Manus glove data to the orca_teleop server via gRPC.",
    )
    parser.add_argument(
        "--server",
        default="localhost:50051",
        help="gRPC server address (default: localhost:50051)",
    )
    parser.add_argument(
        "--hand",
        default="right",
        choices=["left", "right"],
        help="Which hand to stream (default: right)",
    )
    parser.add_argument(
        "--zmq-address",
        default=DEFAULT_ZMQ_ADDRESS,
        help=f"ZMQ address of C++ client (default: {DEFAULT_ZMQ_ADDRESS})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    publisher = ManusPublisher(
        server_address=args.server,
        handedness=args.hand,
        zmq_address=args.zmq_address,
    )
    publisher.run()


if __name__ == "__main__":
    main()
