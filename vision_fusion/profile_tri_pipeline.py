"""离线逐步计时,定位 digit_detect_tri 实时管线的瓶颈(不依赖摄像头)。

在含多 marker 的帧上,把 main() 每帧做的事原样跑一遍并分步计时:
  gray → clahe → yolo(定位) → 每 marker(warp_square + read_debug:定向+裁切+OCR+校验)
额外统计 OCR 调用次数与总耗时(read_debug 最坏每 marker 8 次 OCR)。

Usage:
    python -m vision_fusion.profile_tri_pipeline --frames 5 --iters 10
    python -m vision_fusion.profile_tri_pipeline --source datasets/tri_det/images/train --frames 5
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def _t() -> float:
    return time.perf_counter()


def main() -> int:
    ap = argparse.ArgumentParser(description="Profile tri decode pipeline offline.")
    ap.add_argument("--source", default="datasets/tri_det/images/train",
                    help="含 marker 的帧目录或单张图。")
    ap.add_argument("--model", default="models/tri_marker_obb.pt")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--frames", type=int, default=5, help="取多少张帧。")
    ap.add_argument("--iters", type=int, default=10, help="每帧重复跑多少次取均值。")
    ap.add_argument("--recognizer", choices=["cnn", "ocr"], default="cnn",
                    help="cnn=DigitClassifierTri(默认); ocr=RapidOCR(对照)。")
    ap.add_argument("--archive", default=None, help="把结果文本存到该目录。")
    args = ap.parse_args()

    from ultralytics import YOLO
    from .digit_detect import warp_square
    from .digit_detect_tri import DigitRecognizerTri, DigitClassifierTri

    src = Path(args.source)
    if src.is_dir():
        paths = sorted(src.glob("*.jpg"))[: args.frames]
    else:
        paths = [src]
    if not paths:
        print(f"ERROR: no frames under {src}")
        return 1

    yolo = YOLO(args.model)
    rec = DigitClassifierTri() if args.recognizer == "cnn" else DigitRecognizerTri()
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))

    # OCR 计数/计时:仅 ocr 识别器有 _ocr(cnn 不计,decode 总时已含 CNN)。
    ocr_stats = {"calls": 0, "secs": 0.0}
    if hasattr(rec, "_ocr"):
        _orig_ocr = rec._ocr

        def _timed_ocr(crop):
            t0 = _t()
            out = _orig_ocr(crop)
            ocr_stats["secs"] += _t() - t0
            ocr_stats["calls"] += 1
            return out

        rec._ocr = _timed_ocr  # type: ignore

    # 预热(首次推理含 CUDA/onnx 初始化,不计入)。
    warm = cv2.imread(str(paths[0]))
    g = cv2.cvtColor(warm, cv2.COLOR_BGR2GRAY)
    e = clahe.apply(g)
    r0 = yolo(cv2.cvtColor(e, cv2.COLOR_GRAY2BGR), verbose=False, conf=args.conf)
    dev = "?"
    try:
        dev = str(next(yolo.model.parameters()).device)
    except Exception:
        pass
    if r0 and r0[0].obb is not None and len(r0[0].obb):
        sq = warp_square(e, r0[0].obb.xyxyxyxy[0].cpu().numpy().reshape(4, 2), size=200)
        if sq.size:
            rec.read_debug(sq)
    ocr_stats["calls"] = 0
    ocr_stats["secs"] = 0.0

    lines = [f"YOLO device: {dev}   model: {args.model}",
             f"frames: {len(paths)}  iters/frame: {args.iters}", ""]
    agg = {"gray": 0.0, "clahe": 0.0, "yolo": 0.0, "warp": 0.0, "decode": 0.0,
           "ocr": 0.0, "frame": 0.0, "markers": 0, "ocr_calls": 0, "runs": 0}

    for p in paths:
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        for _ in range(args.iters):
            ocr_calls0, ocr_secs0 = ocr_stats["calls"], ocr_stats["secs"]
            f0 = _t()
            t = _t(); gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); t_gray = _t() - t
            t = _t(); enhanced = clahe.apply(gray); t_clahe = _t() - t
            t = _t()
            results = yolo(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR),
                           verbose=False, conf=args.conf)
            t_yolo = _t() - t
            t_warp = 0.0
            t_decode = 0.0
            n_markers = 0
            for rr in results:
                if rr.obb is None:
                    continue
                for i in range(len(rr.obb)):
                    pts = rr.obb.xyxyxyxy[i].cpu().numpy().reshape(4, 2)
                    t = _t(); square = warp_square(enhanced, pts, size=200); t_warp += _t() - t
                    if square.size == 0:
                        continue
                    n_markers += 1
                    t = _t(); rec.read_debug(square); t_decode += _t() - t
            t_frame = _t() - f0
            agg["gray"] += t_gray; agg["clahe"] += t_clahe; agg["yolo"] += t_yolo
            agg["warp"] += t_warp; agg["decode"] += t_decode; agg["frame"] += t_frame
            agg["ocr"] += ocr_stats["secs"] - ocr_secs0
            agg["markers"] += n_markers
            agg["ocr_calls"] += ocr_stats["calls"] - ocr_calls0
            agg["runs"] += 1

    n = max(1, agg["runs"])
    ms = lambda k: 1000.0 * agg[k] / n
    avg_markers = agg["markers"] / n
    avg_ocr = agg["ocr_calls"] / n
    lines += [
        f"avg markers/frame : {avg_markers:.1f}",
        f"avg OCR calls/frame: {avg_ocr:.1f}  (read_debug 最坏 = 8 × markers)",
        "",
        f"  read(cap)       :   (摄像头,离线不测;实机约 30ms@30fps)",
        f"  cvtColor gray   : {ms('gray'):7.2f} ms",
        f"  CLAHE           : {ms('clahe'):7.2f} ms",
        f"  YOLO 定位       : {ms('yolo'):7.2f} ms   ({dev})",
        f"  warp_square 合计: {ms('warp'):7.2f} ms",
        f"  decode 合计     : {ms('decode'):7.2f} ms   <-- 含全部 OCR",
        f"    └ OCR 纯耗时  : {ms('ocr'):7.2f} ms  (CPU, {avg_ocr:.0f} 次/帧)",
        f"  ----------------",
        f"  每帧合计(不含cap): {ms('frame'):7.2f} ms  →  {1000.0/max(1e-6, ms('frame')):.1f} FPS 上限",
        "",
        f"OCR 占每帧计算量: {100.0*agg['ocr']/max(1e-6, agg['frame']):.0f}%",
    ]
    report = "\n".join(lines)
    print(report)

    if args.archive:
        ad = Path(args.archive); ad.mkdir(parents=True, exist_ok=True)
        (ad / "profile.txt").write_text(report, encoding="utf-8")
        print(f"\n[archived] {ad / 'profile.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
