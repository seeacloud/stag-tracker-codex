import cv2
import numpy as np
import pytest

from vision_fusion.digit_marker_tri import generate_marker_tri
from vision_fusion.digit_detect_tri import find_triangle_corner, orient_by_triangle

# cv2 旋转 → 三角从 TL 移到的角(0=TL 1=TR 2=BR 3=BL)
CASES = [
    (None, 0),
    (cv2.ROTATE_90_CLOCKWISE, 1),
    (cv2.ROTATE_180, 2),
    (cv2.ROTATE_90_COUNTERCLOCKWISE, 3),
]


def _degrade(img, sigma=3.0, lo=90, hi=170):
    """模拟实拍：高斯模糊 + 压低对比到窄灰阶。"""
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    return (lo + (blur.astype(np.float32) / 255.0) * (hi - lo)).astype(np.uint8)


def test_find_triangle_corner_all_rotations():
    gen = generate_marker_tri(83, pixels=200)
    for rot, expected in CASES:
        img = gen if rot is None else cv2.rotate(gen, rot)
        corner, conf = find_triangle_corner(img)
        assert corner == expected
        assert conf > 0


def test_orient_by_triangle_returns_to_top_left():
    gen = generate_marker_tri(83, pixels=200)
    for rot, _ in CASES:
        img = gen if rot is None else cv2.rotate(gen, rot)
        oriented, ok = orient_by_triangle(img)
        assert ok
        assert find_triangle_corner(oriented)[0] == 0


def test_find_triangle_corner_survives_blur():
    gen = generate_marker_tri(83, pixels=200)
    for rot, expected in CASES:
        img = gen if rot is None else cv2.rotate(gen, rot)
        assert find_triangle_corner(_degrade(img))[0] == expected


from vision_fusion.digit_detect_tri import decode_id


def test_decode_id_valid():
    assert decode_id("0204") == 20    # 020 校验 4
    assert decode_id("0833") == 83    # 083 校验 3
    assert decode_id("007X") == 7     # 007 校验 X(=10)


def test_decode_id_strips_noise():
    assert decode_id("0 2 0 4") == 20      # 空格被过滤
    assert decode_id("[0204]") == 20       # 括号等杂字符被过滤(只留 0-9/X)
    assert decode_id("0833\n") == 83


def test_decode_id_rejects_bad_checksum():
    assert decode_id("0205") == -1


def test_decode_id_rejects_short_or_x_in_data():
    assert decode_id("007") == -1     # 不足 4 位
    assert decode_id("0X34") == -1    # 前 3 位出现 X（数据位不该有 X）


try:
    import rapidocr_onnxruntime  # noqa: F401
    _HAS_OCR = True
except Exception:
    _HAS_OCR = False


@pytest.mark.skipif(not _HAS_OCR, reason="rapidocr-onnxruntime not installed")
def test_recognize_real_markers():
    """在用户导出的真实 marker(digit_markers_tri)上端到端读 ID。

    用部署用的实际 marker（用户参数、大字），不是默认小字合成图。
    """
    from pathlib import Path
    from vision_fusion.digit_detect_tri import DigitRecognizerTri
    mdir = Path("digit_markers_tri")
    samples = []
    for mid in (83, 7, 20, 99):          # 含 X(007X) 与普通
        hits = sorted(mdir.glob(f"digit_{mid:03d}_*.png"))
        if hits:
            samples.append((mid, hits[0]))
    if not samples:
        pytest.skip("digit_markers_tri 里无导出 marker")
    rec = DigitRecognizerTri()
    for mid, p in samples:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        got, _ = rec.recognize(img, min_conf=0.0)
        assert got == mid, f"id {mid} read as {got}"


def test_module_api_exists():
    import vision_fusion.digit_detect_tri as m
    for name in ["find_triangle_corner", "orient_by_triangle", "decode_id",
                 "DigitRecognizerTri", "main"]:
        assert hasattr(m, name)




