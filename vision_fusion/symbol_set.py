"""符号集唯一真相源:v8 的 11 个高区分度几何符号,参数化绘制(统一线宽 + 圆角线头)。

逻辑标签沿用 "0123456789X",故 checksum_char / decode_id / slice_cells / DigitCNN
全部复用——符号只是这些逻辑标签的"高区分度物理外观"。
"""
from __future__ import annotations

import cv2
import numpy as np

SYMBOLS = "0123456789X"
# 逻辑标签 -> 画法 key(v8 锁定的 11 符号)
_KIND = {"0": "dring", "1": "L", "2": "hbar", "3": "bslash", "4": "Tl",
         "5": "corner", "6": "Y", "7": "J", "8": "Tr", "9": "fslash", "X": "box"}


def draw_symbol(char: str, size: int = 64, stroke: int = 10, round_cap: bool = False) -> np.ndarray:
    """画单个符号(白底 255、墨 0)。

    size: 画布边长;stroke: 统一线宽(所有符号同一值);round_cap: 圆角线头。
    坐标以 64px 为参考,按 size 比例缩放。
    """
    g = np.full((size, size), 255, np.uint8)
    s = size / 64.0
    t = max(2, int(round(stroke)))
    aa = cv2.LINE_AA if round_cap else cv2.LINE_8     # 直角:LINE_8 不抗锯齿,端点方

    def P(x, y):
        return (int(round(x * s)), int(round(y * s)))

    def line(a, b):
        cv2.line(g, P(*a), P(*b), 0, t, lineType=aa)
        if round_cap:                                  # 仅圆角模式补端点圆
            r = max(1, t // 2)
            cv2.circle(g, P(*a), r, 0, -1, lineType=aa)
            cv2.circle(g, P(*b), r, 0, -1, lineType=aa)

    k = _KIND[char]
    # 所有符号坐标限定在 [16,48] 安全框内(留 ≥16px 边距),
    # 即使粗线宽+圆头也不触及 cell 边缘 → 切格不会把符号切残。
    if k == "dring":
        cv2.circle(g, P(32, 32), int(round(16 * s)), 0, -1, lineType=aa)
    elif k == "box":
        cv2.rectangle(g, P(18, 18), P(46, 46), 0, t, lineType=aa)
    elif k == "vbar":
        line((32, 16), (32, 48))
    elif k == "hbar":
        line((16, 32), (48, 32))
    elif k == "bslash":
        line((18, 18), (46, 46))
    elif k == "fslash":
        line((46, 18), (18, 46))
    elif k == "L":
        line((24, 16), (24, 46)); line((24, 46), (46, 46))
    elif k == "J":
        line((40, 16), (40, 46)); line((40, 46), (18, 46))
    elif k == "Tl":
        line((20, 16), (20, 48)); line((20, 32), (48, 32))
    elif k == "Tr":
        line((44, 16), (44, 48)); line((44, 32), (16, 32))
    elif k == "corner":
        line((16, 18), (48, 18)); line((44, 18), (44, 44))
    elif k == "Y":
        line((32, 48), (32, 33)); line((32, 33), (16, 16)); line((32, 33), (48, 16))
    return g
