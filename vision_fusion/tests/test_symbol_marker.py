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


def test_draw_symbol_loads_png():
    # 贴图模式:每个符号来自 0-9X/<char>.png,有墨且非空白
    for ch in SYMBOLS:
        g = draw_symbol(ch, size=64)
        assert (g < 128).sum() > 50       # 有实质墨量


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


import os, pytest


def test_render_marker_symbol():
    from vision_fusion.symbol_marker import render_marker_symbol, marker_chars
    assert marker_chars(83) == "0833"
    g = render_marker_symbol(83, pixels=300)
    assert g.shape == (300, 300) and g.dtype == np.uint8


def test_symbol_orient_all_rotations():
    import cv2
    from vision_fusion.symbol_marker import render_marker_symbol
    from vision_fusion.digit_detect_tri import symbol_orient, orient_by_symbol
    g = render_marker_symbol(248, pixels=200)            # 封闭格在 TL
    assert symbol_orient(g)[0][0] == 0
    for rot in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
        # 任意旋转后,orient_by_symbol 旋正应使封闭角回 TL
        assert symbol_orient(orient_by_symbol(cv2.rotate(g, rot)))[0][0] == 0


@pytest.mark.skipif(not os.path.exists("models/symbol_cnn.pt"), reason="symbol_cnn 未训练")
def test_symbol_recognize_roundtrip():
    import json
    from vision_fusion.symbol_marker import render_marker_symbol, cell_boxes
    from vision_fusion.digit_detect_tri import DigitClassifierTri, symbol_orient
    s = json.loads(open("symbol_marker_settings.json", encoding="utf-8").read())
    lp = {k: s[k] for k in ("line_ratio", "padding_ratio") if k in s}
    rp = {k: s[k] for k in ("line_ratio", "padding_ratio", "sym_fill", "stroke_ratio", "round_cap") if k in s}
    rec = DigitClassifierTri(model_path="models/symbol_cnn.pt",
                             orient=symbol_orient, boxes=cell_boxes(**lp), min_cell_conf=0.0)
    for mid in (248, 296, 83, 7, 283, 995):       # 7→007X 含校验 X
        got, _ = rec.recognize(render_marker_symbol(mid, pixels=200, **rp), min_conf=0.0)
        assert got == mid, f"{mid} → {got}"
