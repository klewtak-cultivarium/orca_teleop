"""One-Euro signal smoothing for noisy landmark streams.

Implements the classic 1€ filter (Casiez et al., CHI 2012) as a per-keypoint
stateful smoother.  The 1€ filter is the de-facto choice for teleop because it
is adaptive: it applies heavy low-pass filtering when the signal is slow (low
jitter) and light filtering when the signal is fast (low lag).

Reference: https://gery.casiez.net/1euro/
"""

from dataclasses import dataclass

import numpy as np

DEFAULT_MIN_CUTOFF = 1.0
DEFAULT_BETA = 0.007
DEFAULT_D_CUTOFF = 1.0


def _smoothing_factor(dt: float, cutoff: float) -> float:
    """Compute the alpha smoothing factor for a first-order low-pass filter."""
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


@dataclass
class _OneEuroAxisState:
    """Internal state for a single scalar channel of the filter."""

    x_prev: float = 0.0
    dx_prev: float = 0.0
    initialised: bool = False


class OneEuroFilter:
    """Stateful 1€ filter operating on a scalar time-series.

    The filter is parameterised by ``min_cutoff`` (the cutoff frequency when the
    signal is stationary — lower means smoother but laggier), ``beta`` (the
    speed coefficient — higher means less lag but more jitter when moving) and
    ``d_cutoff`` (the cutoff for the derivative estimate; rarely tuned).
    """

    def __init__(
        self,
        min_cutoff: float = DEFAULT_MIN_CUTOFF,
        beta: float = DEFAULT_BETA,
        d_cutoff: float = DEFAULT_D_CUTOFF,
    ) -> None:
        if min_cutoff <= 0.0:
            raise ValueError(f"min_cutoff must be > 0, got {min_cutoff}")
        if d_cutoff <= 0.0:
            raise ValueError(f"d_cutoff must be > 0, got {d_cutoff}")
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._state = _OneEuroAxisState()

    def reset(self) -> None:
        """Forget all history; the next call to :meth:`__call__` re-seeds."""
        self._state = _OneEuroAxisState()

    def __call__(self, x: float, dt: float) -> float:
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt}")
        s = self._state
        if not s.initialised:
            s.x_prev = x
            s.dx_prev = 0.0
            s.initialised = True
            return x

        # Derivative low-pass.
        a_d = _smoothing_factor(dt, self.d_cutoff)
        dx = (x - s.x_prev) / dt
        dx_hat = a_d * dx + (1.0 - a_d) * s.dx_prev

        # Value low-pass with speed-dependent cutoff.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _smoothing_factor(dt, cutoff)
        x_hat = a * x + (1.0 - a) * s.x_prev

        s.x_prev = x_hat
        s.dx_prev = dx_hat
        return x_hat


class LandmarkSmoother:
    """Apply an independent 1€ filter to every component of a landmark array.

    Wraps ``N`` scalar :class:`OneEuroFilter` instances (one per element of the
    flattened ``(21, 3)`` landmark tensor, i.e. 63 channels).  Per-channel state
    is essential because each coordinate has a different scale and motion
    profile (e.g. fingertip z vs. wrist x).
    """

    def __init__(
        self,
        n_points: int = 21,
        n_dims: int = 3,
        min_cutoff: float = DEFAULT_MIN_CUTOFF,
        beta: float = DEFAULT_BETA,
        d_cutoff: float = DEFAULT_D_CUTOFF,
    ) -> None:
        self.n_points = n_points
        self.n_dims = n_dims
        self._filters = [
            OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
            for _ in range(n_points * n_dims)
        ]
        self._last_ts_ns: int | None = None

    def reset(self) -> None:
        """Clear all channel state and the last-seen timestamp."""
        for f in self._filters:
            f.reset()
        self._last_ts_ns = None

    def __call__(self, points: np.ndarray, timestamp_ns: int) -> np.ndarray:
        """Smooth a ``(n_points, n_dims)`` landmark array at the given timestamp.

        Args:
            points: ``(n_points, n_dims)`` array of landmark coordinates.
            timestamp_ns: Monotonic capture timestamp in nanoseconds.

        Returns:
            Smoothed array of the same shape as ``points`` (float32).
        """
        if points.shape != (self.n_points, self.n_dims):
            raise ValueError(
                f"expected points shape ({self.n_points}, {self.n_dims}), got {points.shape}"
            )
        if self._last_ts_ns is None:
            # First frame: seed the filters and return the input unchanged.
            self._last_ts_ns = timestamp_ns
            for i, v in enumerate(points.reshape(-1)):
                self._filters[i]._state.x_prev = float(v)
                self._filters[i]._state.initialised = True
            return points.astype(np.float32, copy=True)

        dt = (timestamp_ns - self._last_ts_ns) * 1e-9
        self._last_ts_ns = timestamp_ns
        if dt <= 0.0:
            # Non-monotonic timestamp: skip smoothing, return input.
            return points.astype(np.float32, copy=True)

        flat = points.reshape(-1).astype(np.float64)
        out = np.empty_like(flat)
        for i, v in enumerate(flat):
            out[i] = self._filters[i](float(v), dt)
        return out.reshape(self.n_points, self.n_dims).astype(np.float32)


__all__ = [
    "DEFAULT_BETA",
    "DEFAULT_MIN_CUTOFF",
    "LandmarkSmoother",
    "OneEuroFilter",
]
