"""Tri-Bit Marker 识别器:4×4 格,左上 2×2 大三角定向 + 12 格数据(无校验)。

与 DigitClassifierTri 同接口(read_batch/read_debug/recognize),塞进 digit_detect_tri
管线复用 YOLO/tracker/TUIO 等。识别管线:
  warp → 4 角 2×2 块对角对比 → 旋正到大三角在 TL → 大三角区均值=基准 → 12 数据格二值化 → id。
"""
from __future__ import annotations
import json
from pathlib import Path
import cv2, numpy as np

from .tri_bit_marker import (decode_bits, encode_id, MAX_ID, N_DATA,
                              DATA_CELL_IDX, id_to_bits)
from .digit_detect_tri import _ensure_gray


_ROT = {0: None, 1: cv2.ROTATE_90_COUNTERCLOCKWISE,
        2: cv2.ROTATE_180, 3: cv2.ROTATE_90_CLOCKWISE}


def _load_settings():
    p = Path("tri_bit_marker_settings.json")
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _grid_geometry(square, border_ratio, gap_ratio, fill_ratio):
    """16 格的填充区坐标 (4×4, 行扫描)。"""
    h, w = square.shape[:2]
    s = min(h, w)
    B = int(s * border_ratio)
    inner = s - 2 * B
    cell = inner / 4.0
    g = int(cell * gap_ratio / 2)
    boxes = []
    for idx in range(16):
        row, col = idx // 4, idx % 4
        cx0 = int(B + col * cell) + g
        cy0 = int(B + row * cell) + g
        cx1 = int(B + (col + 1) * cell) - g
        cy1 = int(B + (row + 1) * cell) - g
        cw = cx1 - cx0; ch = cy1 - cy0
        px = int(cw * (1 - fill_ratio) / 2)
        py = int(ch * (1 - fill_ratio) / 2)
        boxes.append((cx0 + px, cy0 + py, cx1 - px, cy1 - py))
    return boxes


def _quad_corner_block(square, border_ratio, gap_ratio, fill_ratio, corner_idx):
    """返回 4 角 2×2 块的(x0,y0,x1,y1)。corner_idx: 0=TL 1=TR 2=BR 3=BL。"""
    h, w = square.shape[:2]
    s = min(h, w)
    B = int(s * border_ratio)
    inner = s - 2 * B
    half = inner // 2
    if corner_idx == 0:    # TL
        return (B, B, B + half, B + half)
    if corner_idx == 1:    # TR
        return (B + half, B, B + 2 * half, B + half)
    if corner_idx == 2:    # BR
        return (B + half, B + half, B + 2 * half, B + 2 * half)
    return (B, B + half, B + half, B + 2 * half)   # BL


def _make_tri_template(corner_idx, size):
    """生成 corner_idx 朝向的"理想三角"模板(黑=0,白=255)。
    TL→黑右下;TR→黑左下;BR→黑左上;BL→黑右上。"""
    t = np.full((size, size), 255, np.uint8)
    h = size
    if corner_idx == 0:
        pts = [[h-1, 0], [h-1, h-1], [0, h-1]]
    elif corner_idx == 1:
        pts = [[0, 0], [0, h-1], [h-1, h-1]]
    elif corner_idx == 2:
        pts = [[0, 0], [h-1, 0], [0, h-1]]
    else:
        pts = [[h-1, 0], [h-1, h-1], [0, 0]]
    cv2.fillPoly(t, [np.array(pts, np.int32)], 0)
    return t


def _template_match_score(roi, corner_idx):
    """ROI 与"理想 corner_idx 朝向三角"模板的归一化互相关。
    越大=越像该角是大三角格(三角顶点贴合度)。比对角灰度差稳得多——
    纯色数据格组合无法模拟"严格 50/50 对角分布"的模板。"""
    if roi.size == 0 or roi.shape[0] < 4 or roi.shape[1] < 4:
        return 0.0
    T = cv2.resize(_make_tri_template(corner_idx, roi.shape[0]),
                   (roi.shape[1], roi.shape[0]))
    r = roi.astype(np.float32)
    t = T.astype(np.float32)
    r = (r - r.mean()) / (r.std() + 1e-6)
    t = (t - t.mean()) / (t.std() + 1e-6)
    return float((r * t).mean())


def _diagonal_score(roi, corner_idx):
    """旧:对角灰度分(整块对角两半的灰度差)。保留作辅助。"""
    roi = roi.astype(np.float32)
    h, w = roi.shape
    if h < 4 or w < 4:
        return 0.0
    yy, xx = np.mgrid[0:h, 0:w]
    if corner_idx == 0:
        mask = (xx / max(w - 1, 1)) + (yy / max(h - 1, 1)) > 1.0
    elif corner_idx == 1:
        mask = (xx / max(w - 1, 1)) < (yy / max(h - 1, 1))
    elif corner_idx == 2:
        mask = (xx / max(w - 1, 1)) + (yy / max(h - 1, 1)) < 1.0
    else:
        mask = (xx / max(w - 1, 1)) > (1.0 - yy / max(h - 1, 1))
    dark = roi[mask].mean() if mask.sum() else 128
    light = roi[~mask].mean() if (~mask).sum() else 128
    return float(light - dark)


def _decode_oriented(gray, boxes, big_tri_region, min_contrast):
    """已旋正(大三角在左上): 大三角区均值=基准,12 数据格(中心90%采样)二值化。
    数据格采中心 90% 避边角渗墨;基准用大三角整块。
    返回 (id, base_centrality)。base_centrality 衡量基准位于数据格 [min,max] 的居中度。"""
    # 基准:大三角全块均值
    bx0, by0, bx1, by1 = big_tri_region
    base = float(gray[by0:by1, bx0:bx1].mean())
    # 数据格中心 90% 采样(各边各缩 5%)
    data_means = []
    for ci in DATA_CELL_IDX:
        x0, y0, x1, y1 = boxes[ci]
        cw = x1 - x0; ch = y1 - y0
        px = int(cw * 0.05); py = int(ch * 0.05)
        sx0, sy0, sx1, sy1 = x0 + px, y0 + py, x1 - px, y1 - py
        if sx1 > sx0 and sy1 > sy0:
            data_means.append(float(gray[sy0:sy1, sx0:sx1].mean()))
        else:
            data_means.append(float(gray[y0:y1, x0:x1].mean()))
    bits12 = [1 if dm < base else 0 for dm in data_means]
    lo, hi = min(data_means), max(data_means)
    span = hi - lo + 1e-6
    mid_pt = (lo + hi) / 2.0
    half = span / 2.0
    base_centrality = max(0.0, 1.0 - abs(base - mid_pt) / max(half, 1e-6))
    if base_centrality < min_contrast:
        return -1, base_centrality, bits12
    return decode_bits(bits12), base_centrality, bits12


class TriBitRecognizer:
    """与 DigitClassifierTri 同接口的 tri-bit 识别器。
    min_cell_conf 复用为基准居中度门槛(0=不卡)。"""

    def __init__(self, settings_path=None, min_cell_conf=0.0, max_orient=4):
        s = _load_settings() if settings_path is None \
            else json.loads(Path(settings_path).read_text(encoding="utf-8"))
        self.border_ratio = s.get("border_ratio", 0.08)
        self.gap_ratio = s.get("gap_ratio", 0.0)
        self.fill_ratio = s.get("fill_ratio", 1.0)
        self.min_cell_conf = min_cell_conf
        self.max_orient = max_orient
        self.orient = self._orient
        self.rot_map = _ROT

    def _orient(self, square):
        """模板匹配定向:每个 2×2 角块与"理想三角"模板做归一化互相关。
        真三角的对角分布严格 50/50,纯色数据格组合无法模拟。模板匹配比单一灰度差/边缘检测稳得多。"""
        g = _ensure_gray(square)
        scores = []
        for ci in range(4):
            x0, y0, x1, y1 = _quad_corner_block(square, self.border_ratio,
                                                  self.gap_ratio, self.fill_ratio, ci)
            scores.append(_template_match_score(g[y0:y1, x0:x1], ci))
        order = sorted(range(4), key=lambda i: scores[i], reverse=True)
        return order, scores

    def _big_tri_region(self, square):
        """旋正后(大三角在 TL),返回大三角覆盖的 2×2 块区域。"""
        return _quad_corner_block(square, self.border_ratio,
                                    self.gap_ratio, self.fill_ratio, 0)

    def _read_one(self, square):
        boxes = _grid_geometry(square, self.border_ratio, self.gap_ratio, self.fill_ratio)
        region = self._big_tri_region(square)
        g = _ensure_gray(square)
        return _decode_oriented(g, boxes, region, self.min_cell_conf)

    def read_debug(self, square):
        """4 朝向都试,优先按定向分(对角分)选朝向,定向分相近时按基准居中度选。"""
        order, score = self.orient(square)
        info = {"ranking": order, "corner": order[0], "orient_ok": True,
                "orient_conf": (score[order[0]] - score[order[1]]) / (abs(score[order[0]]) + 1e-6),
                "top": "", "bot": "", "top_conf": 0.0, "bot_conf": 0.0}
        cands = {}
        for corner in range(4):
            rot = self.rot_map[corner]
            o = square if rot is None else cv2.rotate(square, rot)
            mid, conf, bits12 = self._read_one(o)
            cands[corner] = (mid, bits12, conf, o, score[corner])
        # 按定向分降序遍历:第一个能解出 id 且 conf>=门槛 的朝向胜出
        for corner in order:
            mid, bits12, conf, o, sc = cands[corner]
            if mid >= 0 and conf >= self.min_cell_conf:
                info.update(corner=corner, top=f"{mid}", bot=str(bits12[:6]),
                            top_conf=conf, bot_conf=conf,
                            min_conf=conf, _oriented=o)
                return mid, conf, info
        # 全失败:返回定向首选朝向的(原始)结果
        corner = order[0]
        mid, bits12, conf, o, _ = cands[corner]
        info.update(corner=corner, top=str(bits12[:6]), bot=str(bits12[6:]),
                    top_conf=conf, bot_conf=conf,
                    min_conf=conf, _oriented=o)
        return -1, 0.0, info

    def recognize(self, square, min_conf=0.0):
        mid, conf, _ = self.read_debug(square)
        return (mid, conf) if mid >= 0 else (-1, 0.0)

    def read_batch(self, squares):
        return [self.read_debug(sq) for sq in squares]
