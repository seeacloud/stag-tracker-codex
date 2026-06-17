import numpy as np
import pytest

from vision_fusion.digit_marker_tri import checksum_char


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
