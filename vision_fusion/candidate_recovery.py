"""Recover marker IDs from rejected STag candidates via warp + deblur + re-detect.

When STag locates a quad but can't decode it (rejected candidate), this module:
1. CNN classifier (if model loaded) — batch GPU inference, <1ms
2. Perspective-warps the candidate to a canonical square
3. Applies aggressive deblurring (multiple strategies)
4. Re-runs STag detection on the cleaned patch
5. Falls back to template matching against previously-seen markers
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import cv2
import numpy as np

from .preprocess import (
    apply_clahe,
    apply_gamma,
    apply_unsharp_mask,
    apply_wiener_deconv,
)

if TYPE_CHECKING:
    from .cnn_classifier import MarkerClassifier


CANONICAL_SIZE = 128
WARP_PADDING = 24


class CandidateRecovery:
    """Attempt to decode rejected STag candidates via warp + deblur."""

    def __init__(self, library_hd: int = 17, ncc_threshold: float = 0.6,
                 classifier: Optional["MarkerClassifier"] = None) -> None:
        self._library_hd = library_hd
        self._ncc_threshold = ncc_threshold
        self._classifier = classifier
        self._templates: dict[int, np.ndarray] = {}
        self._dst_pts = np.array([
            [WARP_PADDING, WARP_PADDING],
            [WARP_PADDING + CANONICAL_SIZE, WARP_PADDING],
            [WARP_PADDING + CANONICAL_SIZE, WARP_PADDING + CANONICAL_SIZE],
            [WARP_PADDING, WARP_PADDING + CANONICAL_SIZE],
        ], dtype=np.float32)
        self._total_size = CANONICAL_SIZE + 2 * WARP_PADDING

    def learn_template(self, marker_id: int, frame: np.ndarray, corners: np.ndarray) -> None:
        """Save a canonical template from a successfully detected marker."""
        src_pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        M = cv2.getPerspectiveTransform(src_pts, self._dst_pts)
        warped = cv2.warpPerspective(frame, M, (self._total_size, self._total_size))
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if warped.ndim == 3 else warped
        self._templates[marker_id] = gray

    def try_recover(
        self,
        frame: np.ndarray,
        rejected_quads: list[np.ndarray],
    ) -> list[tuple[int, np.ndarray, float]]:
        """Try to decode rejected candidates.

        Returns list of (marker_id, corners, confidence) for recovered markers.
        """
        if not rejected_quads:
            return []

        results: list[tuple[int, np.ndarray, float]] = []
        remaining_quads: list[tuple[int, np.ndarray]] = list(enumerate(rejected_quads))

        # Strategy 0: CNN classifier (batch GPU inference, <1ms for all candidates)
        if self._classifier is not None:
            patches = []
            valid_indices = []
            for idx, quad in remaining_quads:
                src_pts = np.asarray(quad, dtype=np.float32).reshape(4, 2)
                if src_pts.shape != (4, 2):
                    continue
                M = cv2.getPerspectiveTransform(src_pts, self._dst_pts)
                warped = cv2.warpPerspective(frame, M, (self._total_size, self._total_size))
                if warped.ndim == 3:
                    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                patches.append(warped)
                valid_indices.append(idx)

            if patches:
                predictions = self._classifier.classify(patches)
                recovered_indices = set()
                for (marker_id, conf), idx in zip(predictions, valid_indices):
                    if marker_id is not None:
                        src_pts = np.asarray(rejected_quads[idx], dtype=np.float32).reshape(4, 2)
                        results.append((marker_id, src_pts, conf))
                        recovered_indices.add(idx)
                remaining_quads = [(i, q) for i, q in remaining_quads if i not in recovered_indices]

        # Strategy 1 & 2: STag re-detect + template matching (for remaining candidates)
        for idx, quad in remaining_quads:
            src_pts = np.asarray(quad, dtype=np.float32).reshape(4, 2)
            if src_pts.shape != (4, 2):
                continue

            M = cv2.getPerspectiveTransform(src_pts, self._dst_pts)
            warped = cv2.warpPerspective(frame, M, (self._total_size, self._total_size))

            marker_id = self._try_redetect(warped)
            if marker_id is not None:
                results.append((marker_id, src_pts, 1.0))
                continue

            if self._templates:
                match = self._try_template_match(warped)
                if match is not None:
                    results.append((match[0], src_pts, match[1]))

        return results

    def _try_redetect(self, warped: np.ndarray) -> Optional[int]:
        """Apply multiple deblur strategies and re-run STag."""
        import stag

        strategies = [
            self._enhance_sharpen_heavy,
            self._enhance_deconv_clahe,
            self._enhance_deconv_sharpen,
            self._enhance_gamma_clahe_heavy,
        ]

        for enhance_fn in strategies:
            enhanced = enhance_fn(warped)
            try:
                corners, ids, _ = stag.detectMarkers(enhanced, libraryHD=self._library_hd)
            except TypeError:
                corners, ids, _ = stag.detectMarkers(enhanced, self._library_hd)
            if ids is not None and len(ids) > 0:
                return int(ids[0][0]) if hasattr(ids[0], '__len__') else int(ids[0])
        return None

    def _try_template_match(self, warped: np.ndarray) -> Optional[tuple[int, float]]:
        """NCC template matching against learned templates."""
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if warped.ndim == 3 else warped
        inner = gray[WARP_PADDING:WARP_PADDING + CANONICAL_SIZE,
                     WARP_PADDING:WARP_PADDING + CANONICAL_SIZE]
        norm_candidate = cv2.normalize(inner, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        best_id: Optional[int] = None
        best_score = self._ncc_threshold

        for marker_id, template in self._templates.items():
            tmpl_inner = template[WARP_PADDING:WARP_PADDING + CANONICAL_SIZE,
                                  WARP_PADDING:WARP_PADDING + CANONICAL_SIZE]
            norm_tmpl = cv2.normalize(tmpl_inner, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Try all 4 rotations (marker orientation unknown)
            for rot in range(4):
                rotated = np.rot90(norm_candidate, rot)
                result = cv2.matchTemplate(
                    rotated, norm_tmpl, cv2.TM_CCOEFF_NORMED
                )
                score = float(result.max())
                if score > best_score:
                    best_score = score
                    best_id = marker_id

        if best_id is not None:
            return (best_id, best_score)
        return None

    @staticmethod
    def _enhance_sharpen_heavy(img: np.ndarray) -> np.ndarray:
        out = apply_gamma(img, 0.6)
        out = apply_clahe(out, clip=4.5)
        return apply_unsharp_mask(out, amount=5.0, radius=3.0)

    @staticmethod
    def _enhance_deconv_clahe(img: np.ndarray) -> np.ndarray:
        out = apply_gamma(img, 0.6)
        out = apply_wiener_deconv(out, radius=3, snr=0.005)
        return apply_clahe(out, clip=4.0)

    @staticmethod
    def _enhance_deconv_sharpen(img: np.ndarray) -> np.ndarray:
        out = apply_gamma(img, 0.6)
        out = apply_wiener_deconv(out, radius=4, snr=0.003)
        out = apply_clahe(out, clip=3.5)
        return apply_unsharp_mask(out, amount=3.0, radius=2.0)

    @staticmethod
    def _enhance_gamma_clahe_heavy(img: np.ndarray) -> np.ndarray:
        out = apply_gamma(img, 0.5)
        out = apply_clahe(out, clip=6.0)
        return apply_unsharp_mask(out, amount=4.0, radius=2.5)
