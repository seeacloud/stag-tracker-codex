"""CCV-style image preprocessing pipeline for marker detection.

Inspired by Community Core Vision (CCV) and reacTIVision: aggressive preprocessing
that converts the raw image into a clean binary image where marker structure is
extremely sharp, regardless of original blur/contrast.

Pipeline:
    Raw → BG Subtract → Smooth → Highpass → Amplify → Adaptive Threshold → Binary

The binary output makes STag's edge detection trivial — the "blur problem" largely
disappears because adaptive thresholding restores hard black/white boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CCVConfig:
    bg_subtract: bool = True
    bg_speed: float = 0.0  # 0 = static bg (manual capture), >0 = dynamic running average
    smooth_ksize: int = 5  # Gaussian blur kernel size
    highpass_blur: int = 29  # large blur for highpass; subtract this from input
    highpass_noise: int = 2  # suppress small variations after highpass
    amplify: float = 4.0  # contrast amplification factor
    threshold_tile: int = 21  # adaptive threshold tile size (must be odd)
    threshold_c: int = 5  # adaptive threshold constant subtracted from mean
    invert: bool = False  # invert binary output (white markers on black bg vs black on white)


class CCVPreprocessor:
    """CCV-inspired pipeline producing a binary image optimized for marker detection."""

    def __init__(self, config: CCVConfig | None = None) -> None:
        self.config = config or CCVConfig()
        self._background: np.ndarray | None = None

    def capture_background(self, gray: np.ndarray) -> None:
        """Snapshot the current frame as static background reference."""
        self._background = gray.astype(np.float32).copy()

    def has_background(self) -> bool:
        return self._background is not None

    def reset_background(self) -> None:
        self._background = None

    def process(self, gray: np.ndarray, return_stages: bool = False):
        """Run the full CCV pipeline on a grayscale image.

        Returns the final binary image, or (binary, stages_dict) if return_stages=True.
        """
        cfg = self.config
        stages = {} if return_stages else None
        if return_stages:
            stages["raw"] = gray.copy()

        work = gray.astype(np.float32)

        # Step 1: Background subtraction
        if cfg.bg_subtract and self._background is not None:
            if cfg.bg_speed > 0:
                # Running average update
                self._background = (
                    (1.0 - cfg.bg_speed) * self._background
                    + cfg.bg_speed * work
                )
            diff = np.abs(work - self._background)
            work = diff
        if return_stages:
            stages["bg_sub"] = work.clip(0, 255).astype(np.uint8)

        # Step 2: Smooth (reduce sensor noise)
        if cfg.smooth_ksize > 1:
            k = cfg.smooth_ksize | 1  # ensure odd
            work_uint = work.clip(0, 255).astype(np.uint8)
            smoothed = cv2.GaussianBlur(work_uint, (k, k), 0)
            work = smoothed.astype(np.float32)
        if return_stages:
            stages["smooth"] = work.clip(0, 255).astype(np.uint8)

        # Step 3: Highpass (subtract heavy blur from input → enhances edges/structure)
        if cfg.highpass_blur > 1:
            k = cfg.highpass_blur | 1
            work_uint = work.clip(0, 255).astype(np.uint8)
            heavy_blur = cv2.GaussianBlur(work_uint, (k, k), 0).astype(np.float32)
            highpass = work - heavy_blur + 128.0  # center around 128
            # Suppress small variations (noise)
            if cfg.highpass_noise > 0:
                deviation = highpass - 128.0
                mask = np.abs(deviation) < cfg.highpass_noise
                deviation[mask] = 0
                highpass = 128.0 + deviation
            work = highpass
        if return_stages:
            stages["highpass"] = work.clip(0, 255).astype(np.uint8)

        # Step 4: Amplify contrast
        if cfg.amplify != 1.0:
            centered = work - 128.0
            work = 128.0 + centered * cfg.amplify
        if return_stages:
            stages["amplify"] = work.clip(0, 255).astype(np.uint8)

        work_uint = work.clip(0, 255).astype(np.uint8)

        # Step 5: Adaptive threshold → binary
        tile = max(3, cfg.threshold_tile | 1)
        thresh_type = cv2.THRESH_BINARY_INV if cfg.invert else cv2.THRESH_BINARY
        binary = cv2.adaptiveThreshold(
            work_uint,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            thresh_type,
            tile,
            cfg.threshold_c,
        )
        if return_stages:
            stages["binary"] = binary.copy()

        if return_stages:
            return binary, stages
        return binary
