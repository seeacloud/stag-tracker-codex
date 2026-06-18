"""离线评测画面增强对识别的影响(合成淡+梯度 ground-truth)。

用法: python -m vision_fusion.eval_enhance              # 基线
评测器给定一个 square 预处理函数,输出 正确/误读/漏读。误读必须保持 0。
"""
from __future__ import annotations
import glob
import os
import random

import cv2
import numpy as np

from .digit_detect_tri import DigitClassifierTri
from .digit_detect import warp_square
from .nn_synth_ui import overlay_marker, load_settings, CANVAS_W, CANVAS_H


def faint_samples(n_per=4, seed=9):
    """用 overlay_marker(透视+混合+缩放)+ 线性梯度造"淡"样本,返回 (true_id, square)。"""
    s = load_settings()
    params = dict(brightness=s.get("brightness", 100.0), contrast=s.get("contrast", 0.5),
                  blur_sigma=s.get("blur_sigma", 2.5), scatter_sigma=s.get("scatter_sigma", 5.0),
                  opacity=s.get("opacity", 0.8), levels_black=0.0, levels_white=255.0,
                  levels_gamma=1.0)
    mk = [(int(os.path.basename(f).split("_")[1]), cv2.imread(f, cv2.IMREAD_GRAYSCALE))
          for f in sorted(glob.glob("digit_markers_tri/digit_*.png"))]
    rng = random.Random(seed)
    out = []
    for mid, g in mk:
        for _ in range(n_per):
            scene = np.full((CANVAS_H, CANVAS_W), rng.randint(125, 150), np.uint8)
            yy, xx = np.mgrid[0:CANVAS_H, 0:CANVAS_W]
            a = rng.uniform(0, 6.28)
            r = (np.cos(a) * xx + np.sin(a) * yy)
            r = (r - r.min()) / (r.max() - r.min())
            scene = np.clip(scene + (r - 0.5) * 70, 0, 255).astype(np.uint8)
            scene, cor, _ = overlay_marker(scene, g, CANVAS_W // 2, CANVAS_H // 2,
                                           rng.uniform(-180, 180), rng.uniform(0.2, 0.45), **params)
            sq = warp_square(scene, cor, 200)
            if sq.size:
                out.append((mid, sq))
    return out


def evaluate(pre=None, samples=None):
    """pre: 可选 square→square 预处理。返回 (正确, 误读, 漏读, 总)。"""
    rec = DigitClassifierTri(min_cell_conf=0.9, max_orient=1)
    samples = samples or faint_samples()
    c = w = m = 0
    for mid, sq in samples:
        s2 = pre(sq) if pre else sq
        got, _ = rec.recognize(s2, min_conf=0.0)
        if got == mid:
            c += 1
        elif got >= 0:
            w += 1
        else:
            m += 1
    return c, w, m, c + w + m


def main() -> int:
    samples = faint_samples()
    c, w, m, n = evaluate(None, samples)
    print(f"基线(无额外增强): 正确 {c}/{n} ({100*c/n:.0f}%)  误读 {w}  漏 {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
