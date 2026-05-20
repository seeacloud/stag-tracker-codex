"""Sweep 720p-tuned 2-pass configs vs dim_id2_720p.mp4."""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "docs" / "test-screenshots" / "dim_id2_720p.mp4"
OUT_DIR = ROOT / "docs" / "test-screenshots"

# (label, baseline_pass, aggressive_pass)
CONFIGS = [
    ("H1_no_15x",     "3.5:0.75,1.0:200:off",       "4.5:0.6,1.0,2.0:100:on"),
    ("H2_plain",      "3.5:1.0:0:off",              "4.5:0.6,1.0,2.0:100:on"),
    ("H3_more_super", "3.5:0.75,1.0,1.5:140:off",   "4.5:1.0,2.0,3.0:150:on"),
    ("H4_combo",      "3.5:0.75,1.0:200:off",       "4.5:1.0,2.0,3.0:150:on"),
    ("H5_aggr",       "3.5:0.75,1.0,1.5:140:off",   "5.5:0.6,1.0,2.0,3.0:120:on"),
]


def run(label: str, base: str, agg: str) -> Path:
    out = OUT_DIR / f"sweep720_{label}.jsonl"
    spec = f"{base};{agg}"
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
    base_path = OUT_DIR / "replay_720p_1pass.jsonl"
    c1_path = OUT_DIR / "replay_720p_C1.jsonl"
    base = stats(base_path)
    c1 = stats(c1_path)
    results = {"_1pass_720p": base, "C1_720p": c1}
    for label, b, a in CONFIGS:
        out = run(label, b, a)
        results[label] = stats(out)

    all_ids = sorted({k for r in results.values() for k in r["ids"]})
    print()
    print(f"{'config':<18} {'obs/f':>6} {'fpsM':>6} " + " ".join(f"id{i:<4}" for i in all_ids) + "  mono(vs 1pass)")
    for label, r in results.items():
        per = " ".join(f"{r['ids'].get(i,0):>5}" for i in all_ids)
        if label == "_1pass_720p":
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
