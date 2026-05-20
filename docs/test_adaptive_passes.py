"""Tests for adaptive multi-pass: skip later passes when baseline already covers expected ids.

Run: `python docs/test_adaptive_passes.py`. Asserts; non-zero exit on failure.

Strategy: monkey-patch StagDetector._run_pass so each PassConfig deterministically
returns a chosen (observations, candidates) pair. Verify that:
  1) without expected_ids, all passes run (baseline behaviour preserved)
  2) with expected_ids subset of baseline pass result, later passes are skipped
  3) with expected_ids NOT covered by baseline, all passes still run
  4) result of skipped path equals just the baseline pass observations
  5) per-id frame counts of adaptive >= per-id of baseline (monotonicity)
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision_fusion.models import StagObservation
from vision_fusion.preprocess import EnhanceConfig
from vision_fusion.stag_detector import PassConfig, StagDetector


def _obs(marker_id: int, x: float = 0.0, y: float = 0.0) -> StagObservation:
    corners = np.asarray(
        [[x, y], [x + 10, y], [x + 10, y + 10], [x, y + 10]],
        dtype=np.float32,
    )
    return StagObservation(
        marker_id=marker_id,
        corners=corners,
        bbox=(int(x), int(y), 10, 10),
        pose=None,
    )


def _make_detector(passes: list[PassConfig], expected_ids=None, workers: int = 1) -> StagDetector:
    """Build a StagDetector but skip importing the real stag binding."""
    d = StagDetector.__new__(StagDetector)
    d._stag = MagicMock()
    d.library_hd = 17
    d.marker_size = None
    d.calibration = None
    d.roi_padding = 12
    d.passes = list(passes)
    d.last_candidates = []
    d.expected_ids = frozenset(expected_ids) if expected_ids is not None else None
    d.last_skipped_passes = 0
    workers = max(1, int(workers))
    d._pass_workers = min(workers, len(d.passes))
    d._executor = None  # serial path keeps tests deterministic
    return d


def _frame() -> np.ndarray:
    return np.zeros((100, 100, 3), dtype=np.uint8)


def _patch_passes(detector: StagDetector, returns: list[list[StagObservation]]) -> list[int]:
    """Replace _run_pass so that pass i returns (returns[i], []). Records call order."""
    call_order: list[int] = []
    pass_to_idx = {id(p): i for i, p in enumerate(detector.passes)}

    def fake(frame, rois, pass_cfg):
        idx = pass_to_idx[id(pass_cfg)]
        call_order.append(idx)
        return list(returns[idx]), []

    detector._run_pass = fake  # type: ignore[method-assign]
    return call_order


P_BASE = PassConfig(enhance=EnhanceConfig(), scales=(1.0,), roi_min_short_side=0)
P_AGG = PassConfig(
    enhance=EnhanceConfig(clahe_clip=4.5),
    scales=(0.6, 1.0, 2.0),
    roi_min_short_side=100,
)


def test_no_expected_ids_runs_all_passes() -> None:
    d = _make_detector([P_BASE, P_AGG], expected_ids=None)
    order = _patch_passes(d, [[_obs(0), _obs(1)], [_obs(2)]])
    obs = d.detect(_frame())
    assert order == [0, 1], f"expected both passes to run, got {order}"
    assert {o.marker_id for o in obs} == {0, 1, 2}


def test_expected_covered_by_baseline_skips_aggressive() -> None:
    d = _make_detector([P_BASE, P_AGG], expected_ids={0, 1})
    order = _patch_passes(d, [[_obs(0), _obs(1)], [_obs(2)]])
    obs = d.detect(_frame())
    assert order == [0], f"expected aggressive pass to be skipped, got {order}"
    assert {o.marker_id for o in obs} == {0, 1}


def test_expected_partial_runs_all_passes() -> None:
    d = _make_detector([P_BASE, P_AGG], expected_ids={0, 1, 2})
    order = _patch_passes(d, [[_obs(0), _obs(1)], [_obs(2)]])
    obs = d.detect(_frame())
    assert order == [0, 1], f"expected baseline missing id 2, both passes should run, got {order}"
    assert {o.marker_id for o in obs} == {0, 1, 2}


def test_skip_path_yields_only_baseline_observations() -> None:
    d = _make_detector([P_BASE, P_AGG], expected_ids={0})
    _patch_passes(d, [[_obs(0)], [_obs(99)]])
    obs = d.detect(_frame())
    assert {o.marker_id for o in obs} == {0}, "aggressive pass MUST NOT contribute when skipped"


def test_monotonicity_adaptive_supseteq_baseline() -> None:
    """Per the design, adaptive must yield >= 1-pass on every input."""
    # baseline alone yields {0}
    d_base = _make_detector([P_BASE], expected_ids=None)
    _patch_passes(d_base, [[_obs(0)]])
    base_obs = {o.marker_id for o in d_base.detect(_frame())}

    # adaptive 2-pass with expected_ids={0,1,2} — baseline misses 1,2 → fall through to agg
    d_adp = _make_detector([P_BASE, P_AGG], expected_ids={0, 1, 2})
    _patch_passes(d_adp, [[_obs(0)], [_obs(1), _obs(2)]])
    adp_obs = {o.marker_id for o in d_adp.detect(_frame())}

    assert base_obs.issubset(adp_obs), f"adaptive must include baseline: base={base_obs} adp={adp_obs}"


def test_skip_counter_records_skips() -> None:
    d = _make_detector([P_BASE, P_AGG], expected_ids={0})
    _patch_passes(d, [[_obs(0)], [_obs(99)]])
    assert d.last_skipped_passes == 0
    d.detect(_frame())
    assert d.last_skipped_passes == 1, f"expected 1 skipped pass, got {d.last_skipped_passes}"

    # Now miss expected → no skip
    d2 = _make_detector([P_BASE, P_AGG], expected_ids={0, 99})
    _patch_passes(d2, [[_obs(0)], [_obs(99)]])
    d2.detect(_frame())
    assert d2.last_skipped_passes == 0, "no skip when baseline incomplete"


def test_single_pass_with_expected_no_op() -> None:
    """expected_ids on a 1-pass detector doesn't crash and doesn't skip the only pass."""
    d = _make_detector([P_BASE], expected_ids={0, 1, 2})
    order = _patch_passes(d, [[_obs(0)]])
    obs = d.detect(_frame())
    assert order == [0]
    assert {o.marker_id for o in obs} == {0}
    assert d.last_skipped_passes == 0


def main() -> int:
    tests = [
        test_no_expected_ids_runs_all_passes,
        test_expected_covered_by_baseline_skips_aggressive,
        test_expected_partial_runs_all_passes,
        test_skip_path_yields_only_baseline_observations,
        test_monotonicity_adaptive_supseteq_baseline,
        test_skip_counter_records_skips,
        test_single_pass_with_expected_no_op,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
