"""Sweep 6 2-pass configs vs dim_id2.mp4 replay, summarize per-id + FPS + monotonicity."""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "docs" / "test-screenshots" / "dim_id2.mp4"
OUT_DIR = ROOT / "docs" / "test-screenshots"

BASELINE = "3.5:0.75,1.0,1.5:140:off"

# (label, aggressive_pass_spec)
CONFIGS = [
    ("C1_baseline", "4.5:0.6,1.0,2.0:100:on"),
    ("C2_no_low",   "4.5:1.0,2.0:100:on"),
    ("C3_super",    "4.5:1.0,2.5:80:on"),
    ("C4_clip55",   "5.5:0.6,1.0,2.0:100:on"),
    ("C5_amt15",    "4.5:0.6,1.0,2.0:100:1.5"),
    ("C6_no_clahe", "off:1.0,2.0:100:on"),
]


def run(label: str, agg: str) -> Path:
    out = OUT_DIR / f"sweep_{label}.jsonl"
    spec = f"{BASELINE};{agg}"
    cmd = [
        sys.executable, "-m", "vision_fusion.stag_only",
        "--source", str(VIDEO),
        "--no-memory", "--no-mirror",
        "--max-frames", "600",
        "--detect-passes", spec,
        "--pass-workers", "2",
        "--enhance-sharpen", "--sharpen-amount", "1.0", "--sharpen-radius", "1.2",
        "--log-jsonl", str(out),
        "--log-every", "0",
    ]
    print(f"[run] {label}  spec={spec}", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)
    return out


def stats(path: Path) -> dict:
    rows = [json.loads(l) for l in open(path, "r", encoding="utf-8")]
    ids = Counter()
    obs = 0
    fps = []
    for r in rows:
        for i in r["observed_ids"]:
            ids[i] += 1
            obs += 1
        fps.append(r["fps"])
    fps_sorted = sorted(fps)
    return {
        "frames": len(rows),
        "obs": obs,
        "ids": ids,
        "fps_med": fps_sorted[len(fps_sorted) // 2] if fps_sorted else 0.0,
        "rows": rows,
    }


def main() -> int:
    baseline_path = OUT_DIR / "replay_1pass.jsonl"
    if not baseline_path.exists():
        raise SystemExit(f"missing {baseline_path}; rerun the 1-pass replay first")
    base = stats(baseline_path)

    results = {"_baseline_1pass": base}
    for label, agg in CONFIGS:
        out = run(label, agg)
        results[label] = stats(out)

    # Print table
    all_ids = sorted({k for r in results.values() for k in r["ids"]})
    print()
    print(f"{'config':<18} {'obs/f':>6} {'fpsM':>6} " + " ".join(f"id{i:<4}" for i in all_ids) + "  mono")
    for label, r in results.items():
        per = " ".join(f"{r['ids'].get(i,0):>5}" for i in all_ids)
        if label == "_baseline_1pass":
            mono = "—"
        else:
            v = sum(
                1 for a, b in zip(base["rows"], r["rows"])
                if not set(a["observed_ids"]).issubset(set(b["observed_ids"]))
            )
            mono = f"{600 - v}/600"
        print(f"{label:<18} {r['obs']/600:>6.2f} {r['fps_med']:>6.2f} {per}  {mono}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
