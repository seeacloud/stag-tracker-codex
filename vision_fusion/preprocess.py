from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(slots=True)
class EnhanceConfig:
    clahe: bool = False
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    sharpen: bool = False
    sharpen_amount: float = 1.0
    sharpen_radius: float = 1.2
    sharpen_threshold: int = 0


def apply_clahe(image: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """Apply CLAHE for low-contrast / low-light recovery.

    Works on grayscale or BGR. STag accepts both, but we keep the input shape so
    downstream code is unaffected.
    """
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    if image.ndim == 2:
        return clahe.apply(image)
    if image.ndim == 3 and image.shape[2] == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return image


def apply_unsharp_mask(
    image: np.ndarray,
    amount: float = 1.0,
    radius: float = 1.2,
    threshold: int = 0,
) -> np.ndarray:
    """Sharpen with unsharp mask: out = image + amount * (image - blurred).

    Cheap defense against motion / focus blur — usually rescues markers whose
    edges are still locatable but whose internal cells have softened past the
    decoder's tolerance. ``threshold`` skips low-amplitude differences to avoid
    amplifying sensor noise.
    """
    if amount <= 0:
        return image
    blurred = cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=max(radius, 0.1))
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0.0)
    if threshold > 0:
        diff = cv2.absdiff(image, blurred)
        if diff.ndim == 3:
            diff = diff.max(axis=2, keepdims=True)
            diff = np.broadcast_to(diff, image.shape)
        mask = diff > threshold
        result = image.copy()
        np.copyto(result, sharpened, where=mask)
        return result
    return sharpened


def enhance_for_detection(image: np.ndarray, config: EnhanceConfig) -> np.ndarray:
    """Run all enabled enhancements in order. Returns a new image."""
    output = image
    if config.clahe:
        output = apply_clahe(output, clip=config.clahe_clip, grid=config.clahe_grid)
    if config.sharpen:
        output = apply_unsharp_mask(
            output,
            amount=config.sharpen_amount,
            radius=config.sharpen_radius,
            threshold=config.sharpen_threshold,
        )
    return output
