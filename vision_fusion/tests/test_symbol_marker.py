import itertools

import numpy as np

from vision_fusion.symbol_set import SYMBOLS, draw_symbol


def test_symbols_11_classes():
    assert SYMBOLS == "0123456789X"


def test_draw_symbol_shape_and_ink():
    for ch in SYMBOLS:
        g = draw_symbol(ch, size=64, stroke=10)
        assert g.shape == (64, 64) and g.dtype == np.uint8
        assert int(g.min()) < 60      # 有墨
        assert int(g.max()) > 200     # 有白底


def test_draw_symbol_stroke_scales_ink():
    thin = int((draw_symbol("4", 64, 6) < 128).sum())     # ⊢ 竖+横
    thick = int((draw_symbol("4", 64, 16) < 128).sum())
    assert thick > thin               # 线宽变大→墨更多


def test_symbol_separability_beats_digits():
    from vision_fusion.eval_glyph_separability import degrade
    rng = np.random.default_rng(0)

    def feat(g):
        a = np.zeros((64, 64), np.float32)
        for _ in range(30):
            d = degrade(g, rng).astype(np.float32)
            d = (d - d.mean()) / (d.std() + 1e-6)
            a += d
        return (a / 30).ravel()

    F = {c: feat(draw_symbol(c, 64, 10)) for c in SYMBOLS}
    mind = min(float(np.linalg.norm(F[a] - F[b]))
               for a, b in itertools.combinations(SYMBOLS, 2))
    assert mind > 30.0, f"区分度 {mind:.1f} 不及预期(数字基线 23.5)"
