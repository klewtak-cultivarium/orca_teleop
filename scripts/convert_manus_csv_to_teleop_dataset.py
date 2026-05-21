"""Convert a raw Manus CSV export into teleop-ready hand landmarks.

The teleop ingress accepts 21 MediaPipe/MANO-style landmarks as 63 row-major
floats in ``HandFrame.keypoints``. Manus exports many more columns; this script
keeps only the position columns that map to that 21-landmark surface and writes
a Hugging Face ``Dataset`` with:

    frame, timestamp_ms, timestamp_s, timestamp_ns, handedness, keypoints

The conversion is intentionally the same one used by the Manus replay publisher
so a live Manus SDK publisher can share this contract later.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_REPO_ID = "fracapuano/manus-mano-poses"

MANO_LANDMARK_NAMES: tuple[str, ...] = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)

# Manus has one additional non-thumb CMC position per finger. The teleop
# pipeline consumes the MediaPipe/MANO-style 21-point surface, so those extra
# CMCs are intentionally skipped.
MANUS_POSITION_JOINTS_FOR_MANO: tuple[str, ...] = (
    "Hand",
    "Thumb_CMC",
    "Thumb_MCP",
    "Thumb_DIP",
    "Thumb_TIP",
    "Index_MCP",
    "Index_PIP",
    "Index_DIP",
    "Index_TIP",
    "Middle_MCP",
    "Middle_PIP",
    "Middle_DIP",
    "Middle_TIP",
    "Ring_MCP",
    "Ring_PIP",
    "Ring_DIP",
    "Ring_TIP",
    "Pinky_MCP",
    "Pinky_PIP",
    "Pinky_DIP",
    "Pinky_TIP",
)


def manus_unity_positions_to_mano_keypoints(positions_cm: np.ndarray) -> np.ndarray:
    """Convert selected Manus Unity-style positions to teleop hand landmarks.

    Args:
        positions_cm: ``(21, 3)`` or ``(N, 21, 3)`` array in centimeters,
            ordered as ``MANUS_POSITION_JOINTS_FOR_MANO``.

    Returns:
        ``(21, 3)`` or ``(N, 21, 3)`` float32 array in meters, wrist-relative,
        ordered as ``MANO_LANDMARK_NAMES``. The coordinates are ready to be
        flattened row-major into the teleop gRPC ``HandFrame.keypoints`` field.

    Coordinate contract:
        keypoints_m = [-X_cm, Z_cm, -Y_cm] / 100
        after subtracting ``Hand_Position`` from every selected joint.
    """
    positions = np.asarray(positions_cm, dtype=np.float32)
    single_frame = positions.ndim == 2
    if single_frame:
        positions = positions[None, ...]

    if positions.ndim != 3 or positions.shape[1:] != (21, 3):
        raise ValueError(
            f"positions_cm must have shape (21, 3) or (N, 21, 3); got {positions.shape}"
        )

    relative_cm = positions - positions[:, [0], :]
    keypoints = np.empty_like(relative_cm, dtype=np.float32)
    keypoints[..., 0] = -relative_cm[..., 0] / 100.0
    keypoints[..., 1] = relative_cm[..., 2] / 100.0
    keypoints[..., 2] = -relative_cm[..., 1] / 100.0
    return keypoints[0] if single_frame else keypoints


def load_manus_csv_as_teleop_arrays(csv_path: str | Path) -> dict[str, np.ndarray]:
    """Return teleop-ready arrays from a raw Manus glove CSV.

    The returned ``keypoints`` array has shape ``(N, 21, 3)`` and is already in
    the exact format expected by ``TargetPose.joint_positions`` and by the gRPC
    ``HandFrame.keypoints`` field after ``keypoints.ravel().tolist()``.
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install the manus extra: `uv sync --extra manus`.") from exc

    df = pd.read_csv(csv_path)

    positions = []
    missing_columns = []
    for joint_name in MANUS_POSITION_JOINTS_FOR_MANO:
        cols = [f"{joint_name}_Position_{axis}" for axis in "XYZ"]
        missing_columns.extend([col for col in cols if col not in df.columns])
        if not any(col not in df.columns for col in cols):
            positions.append(df[cols].to_numpy(dtype=np.float32))

    if missing_columns:
        raise ValueError(f"Manus CSV is missing expected position columns: {missing_columns}")

    positions_cm = np.stack(positions, axis=1)
    keypoints = manus_unity_positions_to_mano_keypoints(positions_cm)

    if "Frame" in df.columns:
        frame = df["Frame"].to_numpy(dtype=np.int32)
    else:
        frame = np.arange(len(df), dtype=np.int32)

    if "Elapsed_Time_In_Milliseconds" not in df.columns:
        raise ValueError("Manus CSV must contain Elapsed_Time_In_Milliseconds")

    timestamp_ms = df["Elapsed_Time_In_Milliseconds"].to_numpy(dtype=np.float32)
    timestamp_s = (timestamp_ms / 1000.0).astype(np.float32)
    timestamp_ns = np.rint(timestamp_ms.astype(np.float64) * 1_000_000).astype(np.int64)

    return {
        "frame": frame,
        "timestamp_ms": timestamp_ms,
        "timestamp_s": timestamp_s,
        "timestamp_ns": timestamp_ns,
        "keypoints": keypoints.astype(np.float32),
    }


def build_teleop_dataset(csv_path: str | Path, handedness: str):
    """Build a Hugging Face Dataset from a Manus CSV."""
    try:
        from datasets import Array2D, Dataset, Features, Value
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install the manus extra: `uv sync --extra manus`.") from exc

    arrays = load_manus_csv_as_teleop_arrays(csv_path)
    n_frames = int(arrays["keypoints"].shape[0])

    features = Features(
        {
            "frame": Value("int32"),
            "timestamp_ms": Value("float32"),
            "timestamp_s": Value("float32"),
            "timestamp_ns": Value("int64"),
            "handedness": Value("string"),
            "keypoints": Array2D(shape=(21, 3), dtype="float32"),
        }
    )
    return Dataset.from_dict(
        {
            "frame": arrays["frame"].tolist(),
            "timestamp_ms": arrays["timestamp_ms"].tolist(),
            "timestamp_s": arrays["timestamp_s"].tolist(),
            "timestamp_ns": arrays["timestamp_ns"].tolist(),
            "handedness": [handedness] * n_frames,
            "keypoints": arrays["keypoints"].tolist(),
        },
        features=features,
    )


def write_metadata(csv_path: str | Path, dataset, output_path: Path, handedness: str) -> None:
    keypoints = np.asarray(dataset["keypoints"], dtype=np.float32)
    timestamp_s = np.asarray(dataset["timestamp_s"], dtype=np.float32)
    metadata = {
        "source_csv": str(csv_path),
        "num_frames": int(len(dataset)),
        "duration_s": float(timestamp_s[-1] - timestamp_s[0]) if len(timestamp_s) else 0.0,
        "fps_estimate": float((len(timestamp_s) - 1) / (timestamp_s[-1] - timestamp_s[0]))
        if len(timestamp_s) > 1 and timestamp_s[-1] > timestamp_s[0]
        else None,
        "handedness": handedness,
        "landmark_names": list(MANO_LANDMARK_NAMES),
        "manus_source_joints": list(MANUS_POSITION_JOINTS_FOR_MANO),
        "dropped_position_joints": ["Index_CMC", "Middle_CMC", "Ring_CMC", "Pinky_CMC"],
        "dropped_channel_groups": [
            "rotations",
            "linear_velocities",
            "linear_accelerations",
            "pinch_features",
            "joint_flex_spread_angles",
            "angular_velocities",
            "angular_accelerations",
        ],
        "coordinate_transform": {
            "input": "Manus Unity-style position columns in centimeters",
            "output": "MediaPipe/MANO-style 21 landmarks, wrist-relative meters",
            "formula": "keypoints_m = [-X_cm, Z_cm, -Y_cm] / 100 after subtracting Hand_Position",
        },
        "keypoints_min": float(np.min(keypoints)) if len(keypoints) else None,
        "keypoints_max": float(np.max(keypoints)) if len(keypoints) else None,
    }
    output_path.write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a raw Manus CSV export into teleop-ready 21-landmark rows.",
    )
    parser.add_argument("csv_path", help="Raw Manus CSV export")
    parser.add_argument(
        "--handedness",
        default="right",
        choices=["left", "right"],
        help="Handedness to store in the output dataset",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/manus_teleop_dataset",
        help="Directory for save_to_disk output and conversion_metadata.json",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the converted dataset to Hugging Face",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face dataset repo used with --push-to-hub",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split name used with --push-to-hub",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_teleop_dataset(csv_path, args.handedness)
    dataset.save_to_disk(str(output_dir))
    write_metadata(csv_path, dataset, output_dir / "conversion_metadata.json", args.handedness)

    print(f"wrote dataset with {len(dataset)} frames to {output_dir}")
    print("keypoints shape per row: (21, 3); gRPC payload: row['keypoints'].ravel().tolist()")

    if args.push_to_hub:
        url = dataset.push_to_hub(
            args.repo_id,
            split=args.split,
            commit_message="Convert Manus CSV to teleop landmarks",
        )
        print(f"pushed to {url}")


if __name__ == "__main__":
    main()
