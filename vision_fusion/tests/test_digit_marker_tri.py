import numpy as np
import pytest

from vision_fusion.digit_marker_tri import checksum_char, generate_marker_tri


def test_checksum_known_values():
    # 083: 1*0 + 2*8 + 3*3 = 25, 25 % 11 = 3
    assert checksum_char(83) == "3"
    # 000: 0 → "0"
    assert checksum_char(0) == "0"


def test_checksum_emits_X_when_value_is_10():
    # 022: 1*0 + 2*2 + 3*2 = 10 → "X"
    assert checksum_char(22) == "X"
    # 999: 9 + 18 + 27 = 54, 54 % 11 = 10 → "X"
    assert checksum_char(999) == "X"


def test_checksum_detects_transposition():
    # 换位必被抓到：123 与 213 校验不同
    assert checksum_char(123) != checksum_char(213)


def test_checksum_all_ids_single_legal_char():
    legal = set("0123456789X")
    for mid in range(1000):
        c = checksum_char(mid)
        assert len(c) == 1 and c in legal


def test_generate_shape_and_dtype():
    arr = generate_marker_tri(83, pixels=600)
    assert arr.shape == (600, 600)
    assert arr.dtype == np.uint8


def test_generate_chamfer_only_top_left():
    p = 600
    arr = generate_marker_tri(83, pixels=p, border_ratio=0.07, chamfer_ratio=0.18)
    b = int(p * 0.07)
    q = b // 2  # 落在边框带内、且在切角三角内(2q < chamfer)
    assert arr[q, q] == 255            # 左上被切掉 → 白
    assert arr[q, p - 1 - q] == 0      # 右上黑框完好 → 黑
    assert arr[p - 1 - q, q] == 0      # 左下黑框完好 → 黑


def test_generate_digits_have_ink_near_cells():
    p = 600
    arr = generate_marker_tri(83, pixels=p)
    b = int(p * 0.07); pad = int(p * 0.06)
    c = (b + pad + p - 1 - b - pad) / 2.0
    col = p * 0.34; row = p * 0.34; w = int(p * 0.28) // 2
    centers = [(c - col / 2, c - row / 2), (c + col / 2, c - row / 2),
               (c - col / 2, c + row / 2), (c + col / 2, c + row / 2)]
    for gx, gy in centers:
        win = arr[int(gy - w):int(gy + w), int(gx - w):int(gx + w)]
        assert win.min() < 128  # 该格窗口内存在墨色像素


def test_stroke_ratio_adds_ink():
    base = generate_marker_tri(888, pixels=400, stroke_ratio=0.0)
    bold = generate_marker_tri(888, pixels=400, stroke_ratio=0.08)
    assert int((bold < 128).sum()) >= int((base < 128).sum())


import argparse

from vision_fusion.digit_marker_tri import _ids_from_args, main


def test_ids_from_args_single():
    ns = argparse.Namespace(id=83, range=None)
    assert _ids_from_args(ns) == [83]


def test_ids_from_args_inclusive_range():
    ns = argparse.Namespace(id=None, range=[5, 25])
    ids = _ids_from_args(ns)
    assert ids == list(range(5, 26))
    assert len(ids) == 21  # 闭区间


def test_ids_from_args_default_is_0_99():
    ns = argparse.Namespace(id=None, range=None)
    assert _ids_from_args(ns) == list(range(0, 100))


def test_main_batch_exports_range(tmp_path, monkeypatch):
    out = tmp_path / "markers_5_25"
    monkeypatch.setattr(
        "sys.argv",
        ["digit_marker_tri", "--range", "5", "25", "--output", str(out),
         "--pixels", "120"],
    )
    rc = main()
    assert rc == 0
    pngs = sorted(out.glob("*.png"))
    assert len(pngs) == 21
    # 文件名带校验位：5 → 005 + checksum
    assert (out / f"digit_005_{checksum_char(5)}.png").exists()
