"""Unit-style assertions for vision_fusion.stag_only.parse_passes.

Run directly: `python docs/test_parse_passes.py`. Asserts; non-zero exit on
failure. Covers backward-compat 3-field syntax and new optional 4th sharpen
field.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make package importable when run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision_fusion.preprocess import EnhanceConfig
from vision_fusion.stag_only import parse_passes


def make_base(sharpen: bool, amount: float = 1.7) -> EnhanceConfig:
    return EnhanceConfig(
        clahe=True,
        clahe_clip=2.0,
        clahe_grid=8,
        sharpen=sharpen,
        sharpen_amount=amount,
        sharpen_radius=1.2,
        sharpen_threshold=0,
    )


def test_empty_returns_no_passes() -> None:
    assert parse_passes("", make_base(False)) == []
    assert parse_passes("   ", make_base(False)) == []


def test_three_field_inherits_global_sharpen() -> None:
    """Backward-compat: 3 fields → sharpen comes from base_enhance."""
    base_off = make_base(sharpen=False)
    p = parse_passes("3.5:0.75,1.0,1.5:140", base_off)
    assert len(p) == 1
    assert p[0].enhance.clahe is True
    assert p[0].enhance.clahe_clip == 3.5
    assert p[0].enhance.sharpen is False, "sharpen should inherit base (off)"
    assert p[0].scales == (0.75, 1.0, 1.5)
    assert p[0].roi_min_short_side == 140

    base_on = make_base(sharpen=True, amount=1.7)
    p2 = parse_passes("3.5:1.0:0", base_on)
    assert p2[0].enhance.sharpen is True, "sharpen should inherit base (on)"
    assert p2[0].enhance.sharpen_amount == 1.7


def test_clahe_off_token() -> None:
    p = parse_passes("off:1.0,1.5:200", make_base(False))
    assert p[0].enhance.clahe is False
    assert p[0].scales == (1.0, 1.5)


def test_four_field_sharpen_off_overrides_global_on() -> None:
    """Critical: even when global sharpen is on, a pass can opt out."""
    base_on = make_base(sharpen=True, amount=1.7)
    p = parse_passes("3.5:1.0:0:off", base_on)
    assert p[0].enhance.sharpen is False, "explicit 'off' must override global on"


def test_four_field_sharpen_on_with_global_off() -> None:
    """A pass can opt-in to sharpen even when global is off, inheriting amount."""
    base_off = make_base(sharpen=False, amount=1.7)
    p = parse_passes("4.5:1.0,2.0:100:on", base_off)
    assert p[0].enhance.sharpen is True
    assert p[0].enhance.sharpen_amount == 1.7  # inherits global amount


def test_four_field_sharpen_numeric_overrides_amount() -> None:
    """Numeric sharpen token = sharpen on with that amount."""
    base = make_base(sharpen=False, amount=1.0)
    p = parse_passes("3.5:1.0:0:1.5", base)
    assert p[0].enhance.sharpen is True
    assert p[0].enhance.sharpen_amount == 1.5


def test_multiple_passes_independent_sharpen() -> None:
    """The bug we are fixing: per-pass sharpen must not leak across passes."""
    base_on = make_base(sharpen=True, amount=1.7)
    spec = "3.5:0.75,1.0,1.5:140:off;off:1.0,1.5:200:off;4.5:0.6,1.0,2.0:100:on"
    p = parse_passes(spec, base_on)
    assert len(p) == 3
    assert p[0].enhance.sharpen is False, "pass 0 explicit off"
    assert p[1].enhance.sharpen is False, "pass 1 explicit off"
    assert p[2].enhance.sharpen is True, "pass 2 explicit on"
    assert p[2].enhance.sharpen_amount == 1.7  # inherits global amount


def test_invalid_field_count_raises() -> None:
    try:
        parse_passes("3.5:1.0", make_base(False))
    except SystemExit:
        pass
    else:
        raise AssertionError("2-field spec must be rejected")

    try:
        parse_passes("3.5:1.0:0:on:extra", make_base(False))
    except SystemExit:
        pass
    else:
        raise AssertionError("5-field spec must be rejected")


def test_invalid_sharpen_token_raises() -> None:
    try:
        parse_passes("3.5:1.0:0:maybe", make_base(False))
    except SystemExit:
        pass
    else:
        raise AssertionError("non-numeric/non-keyword sharpen token must be rejected")


def main() -> int:
    tests = [
        test_empty_returns_no_passes,
        test_three_field_inherits_global_sharpen,
        test_clahe_off_token,
        test_four_field_sharpen_off_overrides_global_on,
        test_four_field_sharpen_on_with_global_off,
        test_four_field_sharpen_numeric_overrides_amount,
        test_multiple_passes_independent_sharpen,
        test_invalid_field_count_raises,
        test_invalid_sharpen_token_raises,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except SystemExit as e:
            failed += 1
            print(f"FAIL {t.__name__}: SystemExit({e.code}) — parse_passes rejected input the test expected to be valid")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
