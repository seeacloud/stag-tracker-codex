"""符号集唯一真相源:v8 的 11 个高区分度几何符号,参数化绘制(统一线宽 + 圆角线头)。

逻辑标签沿用 "0123456789X",故 checksum_char / decode_id / slice_cells / DigitCNN
全部复用——符号只是这些逻辑标签的"高区分度物理外观"。
"""
from __future__ import annotations

import cv2
import numpy as np

SYMBOLS = "0123456789X"
# 逻辑标签 -> 画法 key(v8 锁定的 11 符号)
_KIND = {"0": "dring", "1": "L", "2": "vbar", "3": "bslash", "4": "Tl",
         "5": "corner", "6": "Y", "7": "J", "8": "Tr", "9": "fslash", "X": "box"}


def draw_symbol(char: str, size: int = 64, stroke: int = 10, round_cap: bool = True) -> np.ndarray:
    """画单个符号(白底 255、墨 0)。

    size: 画布边长;stroke: 统一线宽(所有符号同一值);round_cap: 圆角线头。
    坐标以 64px 为参考,按 size 比例缩放。
    """
    g = np.full((size, size), 255, np.uint8)
    s = size / 64.0
    t = max(2, int(round(stroke)))
    aa = cv2.LINE_AA

    def P(x, y):
        return (int(round(x * s)), int(round(y * s)))

    def line(a, b):
        cv2.line(g, P(*a), P(*b), 0, t, lineType=aa)
        if round_cap:
            r = max(1, t // 2)
            cv2.circle(g, P(*a), r, 0, -1, lineType=aa)
            cv2.circle(g, P(*b), r, 0, -1, lineType=aa)

    k = _KIND[char]
    if k == "dring":
        cv2.circle(g, P(32, 32), int(round(22 * s)), 0, -1, lineType=aa)
    elif k == "box":
        cv2.rectangle(g, P(12, 12), P(52, 52), 0, t, lineType=aa)
    elif k == "vbar":
        line((32, 6), (32, 58))
    elif k == "bslash":
        line((12, 12), (52, 52))
    elif k == "fslash":
        line((52, 12), (12, 52))
    elif k == "L":
        line((23, 8), (23, 50)); line((23, 50), (50, 50))
    elif k == "J":
        line((41, 8), (41, 50)); line((41, 50), (14, 50))
    elif k == "Tl":
        line((17, 8), (17, 56)); line((17, 32), (54, 32))
    elif k == "Tr":
        line((47, 8), (47, 56)); line((47, 32), (10, 32))
    elif k == "corner":
        line((12, 15), (48, 15)); line((44, 15), (44, 44))
    elif k == "Y":
        line((32, 56), (32, 33)); line((32, 33), (12, 10)); line((32, 33), (52, 10))
    return g
