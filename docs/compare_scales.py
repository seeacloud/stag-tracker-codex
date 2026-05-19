"""Diff two stag_only --log-jsonl files and report detection metrics."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def metrics(records: Iterable[dict]) -> dict:
    records = list(records)
    n = len(records) or 1
    total_obs = sum(len(r["observed_ids"]) for r in records)
    detected_frames = sum(1 for r in records if r["observed_ids"])
    unique_ids: set[int] = set()
    for r in records:
        unique_ids.update(r["observed_ids"])
    fps_values = [r.get("fps", 0.0) for r in records if r.get("fps", 0.0) > 0]
    avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0.0
    return {
        "frames": n,
        "obs_total": total_obs,
        "obs_per_frame": total_obs / n,
        "detect_rate": detected_frames / n,
        "unique_ids": sorted(unique_ids),
        "avg_fps": avg_fps,
    }


def main() -> int:
    pairs = [
        ("CONTROL  scales=1.0          ", "docs/test-screenshots/scales_off.jsonl"),
        ("MULTI    scales=0.75,1.0,1.5 ", "docs/test-screenshots/scales_on.jsonl"),
        ("SUPER OFF scales=1.0,mem on  ", "docs/test-screenshots/super_off.jsonl"),
        ("SUPER ON  min-short=240      ", "docs/test-screenshots/super_on.jsonl"),
    ]
    rows: list[tuple[str, dict]] = []
    for label, path in pairs:
        p = Path(path)
        if not p.exists():
            continue
        records = load(p)
        rows.append((label, metrics(records)))

    header = (
        f"{'run':<32}"
        + f"{'frames':>8}"
        + f"{'obs/frm':>10}"
        + f"{'detect%':>10}"
        + f"{'#ids':>6}"
        + f"{'fps':>8}"
        + "  ids"
    )
    print(header)
    print("-" * len(header))
    for label, m in rows:
        print(
            f"{label:<32}"
            f"{m['frames']:>8}"
            f"{m['obs_per_frame']:>10.3f}"
            f"{100*m['detect_rate']:>9.1f}%"
            f"{len(m['unique_ids']):>6}"
            f"{m['avg_fps']:>8.1f}"
            f"  {m['unique_ids']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
