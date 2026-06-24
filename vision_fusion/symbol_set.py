"""符号集唯一真相源:11 个数字图标(0-9X)直接用用户设计的 PNG 贴图。

逻辑标签沿用 "0123456789X",故 checksum_char / decode_id / slice_cells / DigitCNN
全部复用——符号只是这些逻辑标签的"高区分度物理外观"。

图标来自仓库根目录 0-9X/<char>.png(200x200 RGBA),render 时合成到白底、缩放贴入格子。
朝向由 marker 外框定位,符号本身不自带定向。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

SYMBOLS = "0123456789X"
_ICON_DIR = Path(__file__).resolve().parent.parent / "0-9X"


@lru_cache(maxsize=16)
def _load_icon(char: str) -> np.ndarray:
    """加载 PNG 图标,合成到白底,返回灰度(白底255墨0)。"""
    path = _ICON_DIR / f"{char}.png"
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(f"符号图标缺失: {path}")
    im = im.astype(np.float32)
    if im.ndim == 3 and im.shape[2] == 4:                  # RGBA → 合成白底
        a = im[:, :, 3:4] / 255.0
        g = (im[:, :, :3] * a + 255 * (1 - a)).mean(2)
    elif im.ndim == 3:
        g = im[:, :, :3].mean(2)
    else:
        g = im
    return g.astype(np.uint8)


def draw_symbol(char: str, size: int = 64, stroke: int = 0, round_cap: bool = False) -> np.ndarray:
    """返回符号灰度图(白底255墨0,size×size)。

    贴图模式:直接缩放用户设计的 PNG。stroke/round_cap 仅为兼容旧签名,已忽略。
    """
    icon = _load_icon(char)
    return cv2.resize(icon, (size, size), interpolation=cv2.INTER_AREA)
