"""Tests for orca_teleop.smoothing."""

import numpy as np
import pytest
from conftest import CANONICAL_LANDMARK_SHAPE, plausible_hand_keypoints

from orca_teleop.smoothing import (
    LandmarkSmoother,
    OneEuroFilter,
)


def test_one_euro_first_sample_seeds():
    filt = OneEuroFilter()
    assert filt(42.0, 1.0 / 30.0) == 42.0


def test_one_euro_reduces_stationary_noise():
    rng = np.random.default_rng(0)
    filt = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    dt = 1.0 / 30.0
    filt(1.0, dt)

    raw, out = [], []
    for _ in range(300):
        noisy = 1.0 + rng.normal(0.0, 0.1)
        raw.append(noisy)
        out.append(filt(noisy, dt))
    # Compare steady-state spread after the transient warm-up.
    assert np.std(out[50:]) < np.std(raw[50:]) * 0.5


def test_one_euro_rejects_non_positive_dt():
    filt = OneEuroFilter()
    with pytest.raises(ValueError, match="dt"):
        filt(1.0, 0.0)


def test_one_euro_rejects_non_positive_cutoff():
    with pytest.raises(ValueError, match="min_cutoff"):
        OneEuroFilter(min_cutoff=0.0)
    with pytest.raises(ValueError, match="d_cutoff"):
        OneEuroFilter(d_cutoff=-1.0)


def test_one_euro_reset_clears_state():
    filt = OneEuroFilter()
    filt(1.0, 1.0 / 30.0)
    filt(5.0, 2.0 / 30.0)
    filt.reset()
    assert filt(7.0, 3.0 / 30.0) == 7.0


def test_landmark_smoother_preserves_shape_and_dtype():
    smoother = LandmarkSmoother()
    out = smoother(plausible_hand_keypoints(), timestamp_ns=1_000_000)
    assert out.shape == CANONICAL_LANDMARK_SHAPE
    assert out.dtype == np.float32


def test_landmark_smoother_first_frame_is_identity():
    smoother = LandmarkSmoother()
    pts = plausible_hand_keypoints()
    out = smoother(pts, timestamp_ns=1)
    assert np.array_equal(out, pts)


def test_landmark_smoother_reduces_stationary_noise():
    smoother = LandmarkSmoother()
    rng = np.random.default_rng(3)
    base = plausible_hand_keypoints()
    smoother(base, timestamp_ns=0)

    raw, filtered = [], []
    t = 1_000_000
    step = 33_000_000  # ~30 fps
    for _ in range(200):
        noisy = base + rng.normal(0.0, 0.01, size=CANONICAL_LANDMARK_SHAPE).astype(np.float32)
        raw.append(noisy)
        filtered.append(smoother(noisy, timestamp_ns=t))
        t += step
    # Jitter = per-channel temporal spread; smoothing must reduce it.
    raw_jitter = np.std(np.stack(raw)[10:], axis=0).mean()
    out_jitter = np.std(np.stack(filtered)[10:], axis=0).mean()
    assert out_jitter < raw_jitter * 0.5


def test_landmark_smoother_non_monotonic_timestamp_passes_through():
    smoother = LandmarkSmoother()
    smoother(plausible_hand_keypoints(), timestamp_ns=5_000_000)
    doubled = plausible_hand_keypoints() * 2.0
    out = smoother(doubled, timestamp_ns=5_000_000)
    assert np.array_equal(out, doubled)


def test_landmark_smoother_rejects_wrong_shape():
    smoother = LandmarkSmoother()
    with pytest.raises(ValueError, match="shape"):
        smoother(np.zeros((20, 3), dtype=np.float32), timestamp_ns=0)


def test_landmark_smoother_reset_reseeds():
    smoother = LandmarkSmoother()
    smoother(plausible_hand_keypoints(), timestamp_ns=1)
    smoother(plausible_hand_keypoints(), timestamp_ns=2)
    smoother.reset()
    pts = plausible_hand_keypoints()
    out = smoother(pts, timestamp_ns=3)
    assert np.array_equal(out, pts)
