"""Bit Marker 识别器:2×2 象限相对灰度二值化(无需CNN,抗糊+免疫光照不均)。

BitRecognizerTri 与 DigitClassifierTri **同接口**(read_batch/read_debug/recognize +
min_cell_conf/orient 属性),直接塞进 digit_detect_tri 的完整管线
(YOLO/定向/tracker/TUIO/screen-map 全复用)。用 `python -m vision_fusion.digit_detect_tri --bit` 启动。

识别:warp方图→定向→切4格→每格2×2象限→max-gap二值化→匹配13码→mod13校验→id。
之前的数字/符号识别保留不动。
"""
from __future__ import annotations
import json
from pathlib import Path
import cv2, numpy as np

from .bit_marker import CODE_BITS, codes_to_id, N_CODES
from .symbol_marker import cell_boxes
from .digit_detect_tri import (slice_by_boxes, symbol_orient, _ensure_gray,
                               _SYM_ROT_TO_TL)

_BITS_TO_CODE = {b: i for i, b in enumerate(CODE_BITS)}


def quad_means(cell: np.ndarray) -> list[float]:
    g = _ensure_gray(cell); H, W = g.shape; h, w = H // 2, W // 2
    return [float(g[0:h, 0:w].mean()), float(g[0:h, w:].mean()),
            float(g[h:, 0:w].mean()), float(g[h:, w:].mean())]


def binarize_cell(cell: np.ndarray, min_gap: float = 0.0):
    """max-gap 二值化。返回 (码号 or -1, bits, gap_ratio)。"""
    vals = quad_means(cell)
    order = sorted(range(4), key=lambda i: vals[i])
    sv = [vals[i] for i in order]
    gaps = [sv[i + 1] - sv[i] for i in range(3)]
    cut = int(np.argmax(gaps))
    span = sv[3] - sv[0] + 1e-6
    gap_ratio = gaps[cut] / span
    bits = [0, 0, 0, 0]
    for k in range(cut + 1):
        bits[order[k]] = 1
    code = _BITS_TO_CODE.get(tuple(bits), -1)
    if gap_ratio < min_gap:
        return -1, tuple(bits), gap_ratio
    return code, tuple(bits), gap_ratio


class BitRecognizerTri:
    """与 DigitClassifierTri 同接口的 bit 识别器。min_cell_conf 复用为象限 min_gap 门槛。"""

    def __init__(self, boxes=None, min_cell_conf: float = 0.15, max_orient: int = 4):
        if boxes is None:
            bs = json.loads(Path("bit_marker_settings.json").read_text(encoding="utf-8")) \
                if Path("bit_marker_settings.json").exists() else {}
            lp = {k: bs[k] for k in ("line_ratio", "padding_ratio") if k in bs}
            boxes = cell_boxes(**lp)
        self.boxes = boxes
        self.min_cell_conf = min_cell_conf      # = min_gap
        self.max_orient = max_orient
        self.orient = symbol_orient
        self.rot_map = _SYM_ROT_TO_TL

    def _decode_oriented(self, o):
        """已定向方图 → (id, 最低gap, codes, bits_list, gaps)。"""
        cells = slice_by_boxes(o, self.boxes)
        codes, gaps, bits_l = [], [], []
        for cell in cells:
            c, b, gr = binarize_cell(cell, 0.0)     # 先不卡门槛,拿原始
            codes.append(c); gaps.append(gr); bits_l.append(b)
        mid = codes_to_id(codes)
        return mid, (min(gaps) if gaps else 0.0), codes, bits_l, gaps

    def read_debug(self, square: np.ndarray):
        order, dark = self.orient(square)
        info = {"ranking": order, "corner": order[0], "orient_ok": True,
                "orient_conf": (dark[order[0]] - dark[order[1]]) / (dark[order[0]] + 1e-6),
                "top": "", "bot": "", "top_conf": 0.0, "bot_conf": 0.0}
        first = None
        for corner in order[:self.max_orient]:
            rot = self.rot_map[corner]
            o = square if rot is None else cv2.rotate(square, rot)
            mid, mgap, codes, bits_l, gaps = self._decode_oriented(o)
            if first is None:
                first = (corner, mid, mgap_str(codes), mgap_val(gaps), o)
            if mid >= 0 and min(gaps) >= self.min_cell_conf:
                info.update(corner=corner, top=str(codes[:2]), bot=str(codes[2:]),
                            top_conf=min(gaps), bot_conf=min(gaps), min_conf=min(gaps), _oriented=o)
                return mid, min(gaps), info
        c, mid, chars, conf, o = first
        info.update(corner=c, top=chars, bot="", top_conf=conf, bot_conf=conf,
                    min_conf=conf, _oriented=o)
        return -1, 0.0, info

    def recognize(self, square: np.ndarray, min_conf: float = 0.0):
        mid, conf, _ = self.read_debug(square)
        return (mid, conf) if mid >= 0 else (-1, 0.0)

    def read_batch(self, squares: list):
        return [self.read_debug(sq) for sq in squares]


def mgap_str(codes):
    return "-".join(str(c) for c in codes)


def mgap_val(gaps):
    return min(gaps) if gaps else 0.0
