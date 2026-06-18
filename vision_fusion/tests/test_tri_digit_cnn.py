import cv2
import numpy as np
import pytest

from vision_fusion.digit_marker_tri import generate_marker_tri
from vision_fusion.digit_detect_tri import CHARS, cell_centers, slice_cells


def test_chars_is_11_classes():
    assert CHARS == "0123456789X"


def test_cell_centers_order_and_values():
    c = cell_centers(col_gap=0.3765, row_gap=0.4176)
    assert len(c) == 4
    assert c[0] == pytest.approx((0.5 - 0.3765 / 2, 0.5 - 0.4176 / 2))  # TL d1
    assert c[1] == pytest.approx((0.5 + 0.3765 / 2, 0.5 - 0.4176 / 2))  # TR d2
    assert c[2] == pytest.approx((0.5 - 0.3765 / 2, 0.5 + 0.4176 / 2))  # BL d3
    assert c[3] == pytest.approx((0.5 + 0.3765 / 2, 0.5 + 0.4176 / 2))  # BR check


def test_slice_cells_shapes():
    g = generate_marker_tri(83, pixels=200)
    cells = slice_cells(g)
    assert len(cells) == 4
    for cell in cells:
        assert cell.shape == (64, 64)


def test_slice_cells_capture_ink():
    # 每个格子都应含黑色字形像素(白底 255, 字 0)
    g = generate_marker_tri(283, pixels=200)   # 2,8,3 + 校验, 四格都有字
    for cell in slice_cells(g):
        assert int(cell.min()) < 100


def test_tl_cell_not_triangle_contaminated():
    # 定向黑三角应被抹掉:TL 格(d1)左上角区域应为白(无三角墨迹)
    g = generate_marker_tri(283, pixels=300)   # d1='2'
    tl = slice_cells(g)[0]
    H, W = tl.shape
    corner = tl[:int(0.08 * H), :int(0.08 * W)]   # 极左上角(只该有三角的区域)
    assert int(corner.min()) > 150, "TL 左上角有黑像素(三角未抹掉)"


def test_marker_chars_maps_cells():
    from vision_fusion.nn_synth_tri_digit import marker_chars
    assert marker_chars(83) == "0833"   # 083 + 校验3
    assert marker_chars(7) == "007X"    # 007 + 校验X(=10)


import os
from pathlib import Path

_HAS_MODEL = os.path.exists("models/tri_digit_cnn.pt")
_MDIR = Path("digit_markers_tri")


def _real_markers(ids):
    out = []
    for mid in ids:
        hits = sorted(_MDIR.glob(f"digit_{mid:03d}_*.png"))
        if hits:
            out.append((mid, str(hits[0])))
    return out


@pytest.mark.skipif(not _HAS_MODEL, reason="tri_digit_cnn.pt 未训练")
def test_classifier_reads_real_markers():
    """在用户导出的真实 marker(GUI 字体/参数)上端到端读 ID,含校验 X(007X)。"""
    from vision_fusion.digit_detect_tri import DigitClassifierTri
    samples = _real_markers([7, 20, 83, 99])      # 7→007X
    if not samples:
        pytest.skip("digit_markers_tri 里无导出 marker")
    rec = DigitClassifierTri()
    for mid, p in samples:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        got, _ = rec.recognize(img, min_conf=0.0)
        assert got == mid, f"id {mid} read as {got}"


@pytest.mark.skipif(not _HAS_MODEL, reason="tri_digit_cnn.pt 未训练")
def test_classifier_reads_all_rotations():
    from vision_fusion.digit_detect_tri import DigitClassifierTri
    samples = _real_markers([83])
    if not samples:
        pytest.skip("digit_markers_tri 里无 83")
    mid, p = samples[0]
    g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    rec = DigitClassifierTri()
    for rot in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180,
                cv2.ROTATE_90_COUNTERCLOCKWISE):
        img = g if rot is None else cv2.rotate(g, rot)
        got, _ = rec.recognize(img, min_conf=0.0)
        assert got == mid, f"rot {rot} read as {got}"
