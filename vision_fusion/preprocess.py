from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(slots=True)
class EnhanceConfig:
    gamma: float = 1.0
    deconv_radius: int = 0
    deconv_snr: float = 0.002
    clahe: bool = False
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    sharpen: bool = False
    sharpen_amount: float = 1.0
    sharpen_radius: float = 1.2
    sharpen_threshold: int = 0

    def to_dict(self) -> dict:
        return {
            "gamma": self.gamma,
            "deconv_radius": self.deconv_radius,
            "deconv_snr": self.deconv_snr,
            "clahe": self.clahe,
            "clahe_clip": self.clahe_clip,
            "clahe_grid": self.clahe_grid,
            "sharpen": self.sharpen,
            "sharpen_amount": self.sharpen_amount,
            "sharpen_radius": self.sharpen_radius,
            "sharpen_threshold": self.sharpen_threshold,
        }


_gamma_lut_cache: dict[float, np.ndarray] = {}


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Brighten/darken via gamma curve. gamma < 1 brightens, > 1 darkens.

    Uses a precomputed LUT for speed (~0.3ms on 720p crop).
    """
    if abs(gamma - 1.0) < 1e-3:
        return image
    lut = _gamma_lut_cache.get(gamma)
    if lut is None:
        lut = np.array(
            [((i / 255.0) ** gamma) * 255 for i in range(256)],
            dtype=np.uint8,
        )
        _gamma_lut_cache[gamma] = lut
    return cv2.LUT(image, lut)


def _disc_psf(radius: int) -> np.ndarray:
    """Create a disc-shaped point spread function for defocus blur."""
    size = 2 * radius + 1
    psf = np.zeros((size, size), dtype=np.float32)
    cv2.circle(psf, (radius, radius), radius, 1.0, -1)
    psf /= psf.sum()
    return psf


def apply_wiener_deconv(image: np.ndarray, radius: int = 3, snr: float = 0.002) -> np.ndarray:
    """Wiener deconvolution for disc-shaped defocus blur.

    Models out-of-focus blur as a uniform disc PSF and inverts it in the
    frequency domain. Much more effective than unsharp mask for defocus.
    snr is the noise-to-signal power ratio (regularization). Lower = more
    aggressive sharpening but more ringing. 0.001-0.01 is typical.
    Cost: ~2-4ms per 720p crop (acceptable in async workers).
    """
    if radius <= 0:
        return image
    is_color = image.ndim == 3
    if is_color:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    h, w = gray.shape
    psf = _disc_psf(radius)
    psf_padded = np.zeros((h, w), dtype=np.float32)
    kh, kw = psf.shape
    psf_padded[:kh, :kw] = psf
    psf_padded = np.roll(psf_padded, -kh // 2, axis=0)
    psf_padded = np.roll(psf_padded, -kw // 2, axis=1)

    img_f = np.float32(gray) / 255.0
    IMG = cv2.dft(img_f, flags=cv2.DFT_COMPLEX_OUTPUT)
    PSF = cv2.dft(psf_padded, flags=cv2.DFT_COMPLEX_OUTPUT)

    # Wiener filter: H* / (|H|^2 + NSR)
    psf_re = PSF[:, :, 0]
    psf_im = PSF[:, :, 1]
    psf_sq = psf_re * psf_re + psf_im * psf_im + snr

    out_re = (IMG[:, :, 0] * psf_re + IMG[:, :, 1] * psf_im) / psf_sq
    out_im = (IMG[:, :, 1] * psf_re - IMG[:, :, 0] * psf_im) / psf_sq

    OUT = np.zeros_like(IMG)
    OUT[:, :, 0] = out_re
    OUT[:, :, 1] = out_im

    result = cv2.idft(OUT, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    result = np.clip(result * 255, 0, 255).astype(np.uint8)

    if is_color:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = result
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return result


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
    if config.gamma != 1.0:
        output = apply_gamma(output, config.gamma)
    if config.deconv_radius > 0:
        output = apply_wiener_deconv(output, config.deconv_radius, config.deconv_snr)
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
