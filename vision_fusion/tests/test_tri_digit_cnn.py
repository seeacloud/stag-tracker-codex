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


class _FakeRec:
    """假识别器,计 read_debug 调用次数(不需 torch)。"""
    def __init__(self):
        self.calls = 0
    def read_debug(self, square):
        self.calls += 1
        return 83, 0.9, {"corner": 0, "ranking": [0, 1, 2, 3]}


def test_decode_cache_skips_cnn_after_lock():
    from vision_fusion.digit_detect_tri import decode_markers_cached
    from vision_fusion.digit_detect import MarkerTracker
    tr = MarkerTracker(); rec = _FakeRec()
    sq = np.zeros((200, 200), np.uint8)
    pts = np.array([[100, 100], [140, 100], [140, 140], [100, 140]], np.float32)
    items = None
    for _ in range(6):
        items, n_cnn = decode_markers_cached([(pts, sq)], rec, tr)
    # 锁定(投票权重过阈)后应停止再调 CNN
    assert rec.calls <= 2, f"locked 后仍调用 CNN {rec.calls} 次"
    assert items[0]["id"] == 83            # 显示投票后的稳定 id
    assert "corner" in items[0]            # 方向(箭头)信息总在


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


@pytest.mark.skipif(not _HAS_MODEL, reason="tri_digit_cnn.pt 未训练")
def test_read_batch_matches_read_debug():
    """批处理识别必须与逐个识别结果一致(提速不改判)。"""
    from vision_fusion.digit_detect_tri import DigitClassifierTri
    samples = _real_markers([0, 7, 20, 83])
    if not samples:
        pytest.skip("digit_markers_tri 里无导出 marker")
    rec = DigitClassifierTri()
    sqs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for _, p in samples]
    batch = rec.read_batch(sqs)
    for (mid, p), (bid, _, _) in zip(samples, batch):
        did, _, _ = rec.read_debug(cv2.imread(p, cv2.IMREAD_GRAYSCALE))
        assert bid == did == mid, f"{mid}: batch={bid} debug={did}"


def test_corner_ranking_survives_illumination_gradient():
    """方向判断要抗光照梯度(去平面)——一侧整片偏暗不该把方向带偏。"""
    from vision_fusion.digit_detect_tri import corner_ranking
    cases = [(None, 0), (cv2.ROTATE_90_CLOCKWISE, 1),
             (cv2.ROTATE_180, 2), (cv2.ROTATE_90_COUNTERCLOCKWISE, 3)]
    g = generate_marker_tri(283, pixels=200)
    h, w = g.shape
    yy, xx = np.mgrid[0:h, 0:w]
    ramp = ((xx + yy) / (h + w)).astype(np.float32)        # 左上→右下线性梯度
    faint = (120 + g.astype(np.float32) / 255 * 45)        # 压成窄灰阶(很淡)
    for rot, exp in cases:
        im = faint if rot is None else cv2.rotate(faint, rot)
        r = ramp if rot is None else cv2.rotate(ramp, rot)
        im2 = np.clip(im + (r - 0.5) * 80, 0, 255).astype(np.uint8)   # 叠强梯度
        assert corner_ranking(im2)[0][0] == exp, f"rot {rot} 方向判错"
