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


def test_render_marker_symbol():
    from vision_fusion.symbol_marker import render_marker_symbol, marker_chars
    assert marker_chars(83) == "0833"        # 复用 checksum_char
    g = render_marker_symbol(83, pixels=300)
    assert g.shape == (300, 300) and g.dtype == np.uint8
    bottom = g[int(300 * 0.93):, 20:280].mean()
    top = g[:int(300 * 0.04), 20:280].mean()
    assert bottom < 80, f"底边应是黑条 mean={bottom:.0f}"
    assert top < 80, f"顶边是普通黑框 mean={top:.0f}"   # 顶也是边框(黑),但更细


def test_edge_ranking_finds_bottom():
    import cv2
    from vision_fusion.symbol_marker import render_marker_symbol
    from vision_fusion.digit_detect_tri import edge_ranking
    g = render_marker_symbol(283, pixels=200)            # 底边黑条
    order, _ = edge_ranking(g)
    assert order[0] == 2, f"正放最暗边应为底(2),实际 {order[0]}"   # 0上1右2下3左
    assert edge_ranking(cv2.rotate(g, cv2.ROTATE_180))[0][0] == 0   # 旋180→黑条到顶


def test_edge_ranking_survives_gradient():
    import cv2
    from vision_fusion.symbol_marker import render_marker_symbol
    from vision_fusion.digit_detect_tri import edge_ranking
    g = render_marker_symbol(283, pixels=200).astype(np.float32)
    h, w = g.shape
    yy, xx = np.mgrid[0:h, 0:w]
    ramp = ((xx + yy) / (h + w)).astype(np.float32)
    g2 = np.clip(g + (ramp - 0.5) * 80, 0, 255).astype(np.uint8)   # 叠强梯度
    assert edge_ranking(g2)[0][0] == 2, "去平面后梯度不该带偏底边判断"


import os, pytest


@pytest.mark.skipif(not os.path.exists("models/symbol_cnn.pt"), reason="symbol_cnn 未训练")
def test_symbol_recognize_roundtrip():
    from vision_fusion.symbol_marker import render_marker_symbol
    from vision_fusion.digit_detect_tri import DigitClassifierTri, edge_ranking
    from vision_fusion.symbol_marker import SYM_CENTER_Y
    rec = DigitClassifierTri(model_path="models/symbol_cnn.pt",
                             orient=edge_ranking, mask_tri=False, mask_bottom=True, center_y=SYM_CENTER_Y)
    for mid in (83, 7, 20, 99, 283):       # 7→007X 含校验 X
        got, _ = rec.recognize(render_marker_symbol(mid, pixels=200), min_conf=0.0)
        assert got == mid, f"{mid} → {got}"
