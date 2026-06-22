"""符号区分度评测器:量化候选字体/符号集里"最易混对"的可分性。

对每个候选(字体或符号集),渲染 11 个符号(0-9+X)→ 同一套真实退化(模糊+IR压对比+噪声)
多次采样 → 算每对符号在退化后的平均 L2 距离 → 报告**最小对距离**(越大越好,代表
最易混的那对也拉得开)。用数据选字体/设计符号,而非凭感觉。

用法: python -m vision_fusion.eval_glyph_separability
"""
from __future__ import annotations
import itertools
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

CHARS = "0123456789X"
SZ = 64
FONTS = {
    "Gothic(现用)": "C:/Windows/Fonts/Gothic.ttf",
    "Consolas-Bold": "C:/Windows/Fonts/consolab.ttf",
    "CascadiaMono": "C:/Windows/Fonts/CascadiaMono.ttf",
    "Roboto-Black": "C:/Windows/Fonts/Roboto-Black.ttf",
    "ArialBold": "C:/Windows/Fonts/arialbd.ttf",
}


def render_glyph(ch: str, font_path: str, sz: int = SZ) -> np.ndarray:
    img = Image.new("L", (sz, sz), 255)
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(font_path, int(sz * 0.72))
    bb = f.getbbox(ch)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    d.text(((sz - tw) / 2 - bb[0], (sz - th) / 2 - bb[1]), ch, fill=0, font=f)
    return np.array(img)


def degrade(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sigma = rng.uniform(1.5, 3.5)
    b = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    lo, hi = rng.integers(110, 135), rng.integers(160, 185)
    b = (lo + b.astype(np.float32) / 255 * (hi - lo))
    b = b + rng.normal(0, 8, b.shape)
    return np.clip(b, 0, 255).astype(np.uint8)


def mean_degraded(ch_img: np.ndarray, n: int, rng) -> np.ndarray:
    """退化样本的平均特征向量(归一化后展平)——代表该符号在退化下的"长相中心"。"""
    acc = np.zeros((SZ, SZ), np.float32)
    for _ in range(n):
        d = degrade(ch_img, rng).astype(np.float32)
        d = (d - d.mean()) / (d.std() + 1e-6)      # 去亮度/对比,只看形状
        acc += d
    v = acc / n
    return v.ravel()


def separability(glyphs: dict, n=40, seed=0):
    """glyphs: {符号: 64x64图}。返回 (最小对距离, 最易混对, 全部对距离表)。"""
    rng = np.random.default_rng(seed)
    feats = {c: mean_degraded(g, n, rng) for c, g in glyphs.items()}
    pairs = {}
    for a, b in itertools.combinations(glyphs, 2):
        pairs[(a, b)] = float(np.linalg.norm(feats[a] - feats[b]))
    worst = min(pairs, key=pairs.get)
    return pairs[worst], worst, pairs


def main() -> int:
    print(f"{'字体':16} 最小对距离  最易混对  (越大越好)")
    results = {}
    for name, path in FONTS.items():
        try:
            glyphs = {c: render_glyph(c, path) for c in CHARS}
        except Exception as e:
            print(f"{name:16} 跳过({e})"); continue
        mind, worst, pairs = separability(glyphs)
        results[name] = (mind, worst, pairs)
        top3 = sorted(pairs.items(), key=lambda x: x[1])[:3]
        print(f"{name:16} {mind:8.1f}   {worst[0]}/{worst[1]}   "
              f"最难3对={[(f'{a}/{b}', round(d, 1)) for (a, b), d in top3]}")
    if results:
        best = max(results, key=lambda k: results[k][0])
        print(f"\n最高区分度字体: {best} (最小对距离 {results[best][0]:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
