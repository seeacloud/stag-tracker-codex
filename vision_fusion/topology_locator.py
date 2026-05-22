"""Topology-based STag marker locator: finds 'square containing circle' patterns.

Inspired by reacTIVision's topological fiducial tracking. Works on binary images
(typically from CCV pipeline) by analyzing contour hierarchy:

1. Find all contours with full hierarchy info
2. Filter quadrilateral parent contours (4-corner polygons of reasonable size)
3. For each quad, check if it contains a circular child contour (the marker's
   inner pattern circle)
4. Return quads that match the 'square outside, circle inside' topology

This detects STag markers regardless of training data — works on ANY HD17 marker
because the outer-square + inner-circle topology is universal to all STag markers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .models import BBox


@dataclass
class TopologyConfig:
    min_area: int = 800       # minimum quad area in pixels (filters tiny noise)
    max_area: int = 80000     # maximum quad area
    min_circularity: float = 0.55  # 4*pi*A/P^2; 1.0=perfect circle, 0=line
    poly_epsilon: float = 0.04     # contour approximation tolerance (fraction of perimeter)
    aspect_min: float = 0.5        # min w/h aspect ratio (filters thin shapes)
    aspect_max: float = 2.0        # max w/h aspect ratio
    inner_min_area_ratio: float = 0.05  # inner circle must be at least 5% of quad area
    inner_max_area_ratio: float = 0.95  # inner circle must be less than 95% of quad area


class TopologyLocator:
    """Find STag markers by their 'square + inner circle' topology in binary images."""

    def __init__(self, config: TopologyConfig | None = None) -> None:
        self.config = config or TopologyConfig()

    def locate(
        self,
        binary: np.ndarray,
        existing_bboxes: Optional[list[BBox]] = None,
    ) -> list[tuple[np.ndarray, BBox]]:
        """Find marker-like regions in a binary image.

        Returns list of (corners_4x2, bbox) for each match.
        """
        cfg = self.config

        # Find all contours WITH hierarchy info
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        if hierarchy is None or len(contours) == 0:
            return []

        hierarchy = hierarchy[0]  # shape (N, 4): [next, prev, first_child, parent]

        results: list[tuple[np.ndarray, BBox]] = []
        h_img, w_img = binary.shape[:2]

        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < cfg.min_area or area > cfg.max_area:
                continue

            # Approximate to polygon and check for quadrilateral
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            approx = cv2.approxPolyDP(contour, cfg.poly_epsilon * perimeter, True)
            if len(approx) != 4:
                continue
            if not cv2.isContourConvex(approx):
                continue

            # Bounding box + aspect ratio filter
            x, y, w, h = cv2.boundingRect(approx)
            if h == 0:
                continue
            aspect = w / h
            if aspect < cfg.aspect_min or aspect > cfg.aspect_max:
                continue

            # Check children for circular structure
            child_idx = hierarchy[idx][2]
            has_circle = False
            while child_idx != -1:
                child_contour = contours[child_idx]
                child_area = cv2.contourArea(child_contour)
                area_ratio = child_area / max(area, 1.0)
                if (cfg.inner_min_area_ratio <= area_ratio <= cfg.inner_max_area_ratio
                        and child_area > 50):
                    child_perim = cv2.arcLength(child_contour, True)
                    if child_perim > 0:
                        circularity = 4.0 * np.pi * child_area / (child_perim * child_perim)
                        if circularity >= cfg.min_circularity:
                            has_circle = True
                            break
                # Walk to next sibling
                child_idx = hierarchy[child_idx][0]

            if not has_circle:
                continue

            corners = approx.reshape(4, 2).astype(np.float32)
            bbox = (int(x), int(y), int(w), int(h))

            if existing_bboxes and self._overlaps_any(bbox, existing_bboxes):
                continue

            results.append((corners, bbox))

        return results

    @staticmethod
    def _overlaps_any(bbox: BBox, existing: list[BBox]) -> bool:
        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        for ex, ey, ew, eh in existing:
            if ex <= cx <= ex + ew and ey <= cy <= ey + eh:
                return True
        return False
