"""1-Euro Filter for 2D point smoothing.

Reference: Casiez et al., "1 Euro Filter: A Simple Speed-based Low-pass Filter
for Noisy Input in Interactive Systems", CHI 2012.

The filter adapts its cutoff frequency based on signal speed:
- Stationary signals get heavy smoothing (low cutoff)
- Fast-moving signals get minimal lag (high cutoff)
"""

from __future__ import annotations

import math

import numpy as np


def _smoothing_factor(te: float, cutoff: float) -> float:
    """Compute exponential smoothing factor alpha from time step and cutoff freq."""
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / te)


class LowPassFilter:
    """Simple first-order low-pass filter for a scalar or array value."""

    __slots__ = ("_hatx", "_initialized")

    def __init__(self) -> None:
        self._hatx: np.ndarray | None = None
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def value(self) -> np.ndarray:
        assert self._hatx is not None
        return self._hatx

    def filter(self, x: np.ndarray, alpha: float) -> np.ndarray:
        if not self._initialized:
            self._hatx = x.copy()
            self._initialized = True
        else:
            self._hatx = alpha * x + (1.0 - alpha) * self._hatx  # type: ignore[operator]
        return self._hatx  # type: ignore[return-value]

    def reset(self, x: np.ndarray) -> None:
        self._hatx = x.copy()
        self._initialized = True


class OneEuroFilter:
    """1-Euro Filter for N-dimensional signals (typically 2D points).

    Parameters
    ----------
    min_cutoff : float
        Minimum cutoff frequency (Hz). Lower = more smoothing when stationary.
    beta : float
        Speed coefficient. Higher = less lag during fast motion.
    dcutoff : float
        Cutoff frequency for the derivative filter (Hz). Usually 1.0.
    fps : float
        Expected frame rate. Used to compute time step when no timestamp given.
    """

    __slots__ = ("min_cutoff", "beta", "dcutoff", "fps", "_x_filter", "_dx_filter", "_last_time")

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        dcutoff: float = 1.0,
        fps: float = 60.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.fps = fps
        self._x_filter = LowPassFilter()
        self._dx_filter = LowPassFilter()
        self._last_time: float | None = None

    @property
    def initialized(self) -> bool:
        return self._x_filter.initialized

    def reset(self, x: np.ndarray, t: float | None = None) -> np.ndarray:
        """Hard-reset filter state to x (e.g. after a teleport)."""
        self._x_filter.reset(x)
        self._dx_filter.reset(np.zeros_like(x))
        self._last_time = t
        return x.copy()

    def __call__(self, x: np.ndarray, t: float | None = None) -> np.ndarray:
        """Filter a new measurement.

        Parameters
        ----------
        x : np.ndarray
            New measurement (flat array, e.g. shape (8,) for 4 corners x 2D).
        t : float | None
            Timestamp in seconds. If None, assumes constant fps.

        Returns
        -------
        np.ndarray
            Filtered value, same shape as x.
        """
        x = np.asarray(x, dtype=np.float64)

        if not self._x_filter.initialized:
            # First sample: initialize both filters, no smoothing
            self._dx_filter.filter(np.zeros_like(x), 1.0)
            self._x_filter.filter(x, 1.0)
            self._last_time = t
            return x.copy()

        # Compute time step
        if t is not None and self._last_time is not None:
            te = t - self._last_time
            if te <= 0:
                te = 1.0 / self.fps
        else:
            te = 1.0 / self.fps
        self._last_time = t

        # Filter the derivative (speed)
        dx = (x - self._x_filter.value) / te
        alpha_d = _smoothing_factor(te, self.dcutoff)
        dx_hat = self._dx_filter.filter(dx, alpha_d)

        # Compute adaptive cutoff based on speed
        speed = np.abs(dx_hat)
        cutoff = self.min_cutoff + self.beta * speed

        # Per-element alpha from adaptive cutoff
        alpha = np.array([_smoothing_factor(te, float(c)) for c in cutoff.flat]).reshape(
            cutoff.shape
        )

        # Apply element-wise alpha (use mean for the low-pass call)
        # For per-element smoothing, we do it manually
        prev = self._x_filter.value
        filtered = alpha * x + (1.0 - alpha) * prev
        self._x_filter._hatx = filtered  # noqa: SLF001
        self._x_filter._initialized = True  # noqa: SLF001

        return filtered.copy()


class OneEuroFilter2D:
    """Convenience wrapper: filters an array of 2D points (Nx2 or Nx1x2).

    Internally flattens to 1D, applies OneEuroFilter, reshapes back.
    """

    __slots__ = ("_filter", "_shape")

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        dcutoff: float = 1.0,
        fps: float = 60.0,
    ) -> None:
        self._filter = OneEuroFilter(
            min_cutoff=min_cutoff, beta=beta, dcutoff=dcutoff, fps=fps
        )
        self._shape: tuple[int, ...] | None = None

    @property
    def initialized(self) -> bool:
        return self._filter.initialized

    def reset(self, points: np.ndarray, t: float | None = None) -> np.ndarray:
        """Hard-reset to given points."""
        self._shape = points.shape
        flat = points.reshape(-1).astype(np.float64)
        result = self._filter.reset(flat, t)
        return result.reshape(self._shape).astype(np.float32)

    def __call__(self, points: np.ndarray, t: float | None = None) -> np.ndarray:
        """Filter new point measurements."""
        self._shape = points.shape
        flat = points.reshape(-1).astype(np.float64)
        result = self._filter(flat, t)
        return result.reshape(self._shape).astype(np.float32)
