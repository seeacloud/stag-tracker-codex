"""数字 Marker 解码器（digit_marker_tri 配套）：YOLO-OBB 定位 + 内角暗度三角定向
+ RapidOCR 读 4 位 + 加权 mod11+X 校验。

Usage:
    python -m vision_fusion.digit_detect_tri --source 0
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from .digit_marker_tri import checksum_char

# 旋转映射：把三角所在角旋到左上。0=TL 1=TR 2=BR 3=BL。
_ROT_TO_TL = {
    0: None,
    1: cv2.ROTATE_90_COUNTERCLOCKWISE,
    2: cv2.ROTATE_180,
    3: cv2.ROTATE_90_CLOCKWISE,
}


def find_triangle_corner(square: np.ndarray, k_ratio: float = 0.26) -> tuple[int, float]:
    """4 个角各取贴角方块(含黑边框角)，相对最暗者=黑三角所在角。

    边框 4 角相等→当常数基线；三角只加在其所在角→该角相对最暗。patch 贴角(offset=0)
    且够大罩住整块三角，远离中央数字，排除数字笔画干扰(干净图实测三角角暗度≈其它 2.2×)。
    返回 (corner, conf)，conf=(最暗-次暗)/最暗 仅参考；不设阈值，朝向对错最终由 decode_id
    加权 mod11 校验兜底。
    """
    gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY) if square.ndim == 3 else square
    s = gray.shape[0]
    k = max(4, int(s * k_ratio))
    patches = [
        gray[0:k, 0:k],            # TL
        gray[0:k, s - k:s],        # TR
        gray[s - k:s, s - k:s],    # BR
        gray[s - k:s, 0:k],        # BL
    ]
    dark = [float((255.0 - p.astype(np.float32)).mean()) for p in patches]
    order = sorted(range(4), key=lambda i: dark[i], reverse=True)
    top, second = dark[order[0]], dark[order[1]]
    conf = (top - second) / (top + 1e-6)
    return order[0], conf


def orient_by_triangle(square: np.ndarray) -> tuple[np.ndarray, bool]:
    """把(相对)最暗角的黑三角旋到左上。

    IR 画面整体偏灰、无绝对黑白，所以只做 4 角的**相对**比较，相对最暗者=三角所在，
    **不设绝对/置信度阈值**（之前的 margin 门槛在灰图上会误杀正确朝向，真机实测
    8 个里 5 个被这个门槛扔掉）。朝向是否正确最终由 decode_id 的加权 mod11 校验兜底
    ——错朝向读出的串过不了校验返回 -1，不会输出错 ID。返回的 bool 恒为 True（保留
    签名兼容）。
    """
    corner, _ = find_triangle_corner(square)
    rot = _ROT_TO_TL[corner]
    return (square if rot is None else cv2.rotate(square, rot)), True


def decode_id(chars: str) -> int:
    """OCR 字符串 → marker_id。保留 0-9 与 X，取前 4 位，加权 mod11 校验。

    前 3 位须为数字、第 4 位为校验(0-9 或 X)；校验不过或非法返回 -1。
    """
    keep = "".join(c for c in chars if c in "0123456789X")
    if len(keep) < 4:
        return -1
    s = keep[:4]
    if not s[:3].isdigit():        # 数据位出现 X 等 → 非法
        return -1
    mid = int(s[:3])
    return mid if checksum_char(mid) == s[3] else -1


class DigitRecognizerTri:
    """RapidOCR(PP-OCR) 读 2×2 数字带 + 加权 mod11+X 校验。"""

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR  # 懒加载，缺依赖不影响纯逻辑
        self.reader = RapidOCR()

    def _ocr(self, crop: np.ndarray) -> tuple[str, float]:
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        # 位置已知，DBNet 检测在低对比噪声带上会失败 → rec-only。
        result, _ = self.reader(crop, use_det=False, use_cls=False, use_rec=True)
        if not result:
            return "", 0.0
        text = "".join(r[0] for r in result)
        conf = float(np.mean([r[1] for r in result]))
        return text, conf

    def read_debug(self, square: np.ndarray) -> tuple[int, float, dict]:
        """识别并返回中间量（定向/两行 OCR 原文）供调试。"""
        corner, oconf = find_triangle_corner(square)
        oriented, ok = orient_by_triangle(square)
        info = {"orient_ok": ok, "orient_conf": oconf, "corner": corner,
                "top": "", "bot": "", "top_conf": 0.0, "bot_conf": 0.0}
        if not ok:
            return -1, 0.0, info
        h, w = oriented.shape[:2]
        x0, x1 = int(w * 0.12), int(w * 0.88)
        top = cv2.resize(oriented[int(h * 0.12):int(h * 0.50), x0:x1], None,
                         fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        bot = cv2.resize(oriented[int(h * 0.50):int(h * 0.88), x0:x1], None,
                         fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        t_txt, t_c = self._ocr(top)
        b_txt, b_c = self._ocr(bot)
        info.update(top=t_txt, bot=b_txt, top_conf=t_c, bot_conf=b_c,
                    _oriented=oriented, _top=top, _bot=bot)
        if not (t_txt and b_txt):
            return -1, 0.0, info
        conf = (t_c + b_c) / 2.0
        mid = decode_id(t_txt + b_txt)
        return (mid if mid >= 0 else -1), conf, info

    def recognize(self, square: np.ndarray, min_conf: float = 0.5) -> tuple[int, float]:
        mid, conf, _ = self.read_debug(square)
        if mid < 0 or conf < min_conf:
            return -1, 0.0
        return mid, conf


def main() -> int:
    from ultralytics import YOLO
    from .digit_detect import (order_corners, warp_square, draw_corners,
                               marker_up_vector, id_anchor)

    parser = argparse.ArgumentParser(description="Digit marker (tri) decoder.")
    parser.add_argument("--source", default="0")
    parser.add_argument("--model", default="models/tri_marker_obb.pt")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--mirror", action="store_true", default=False)
    parser.add_argument("--camera-exposure", type=int, default=None)
    parser.add_argument("--debug-dir", default=None,
                        help="存每个检出 marker 的定向/裁切图 + decode_log.txt(OCR 原文)供调试。")
    args = parser.parse_args()

    dbg = None
    dbg_log = None
    dbg_n = 0
    if args.debug_dir:
        from pathlib import Path
        dbg = Path(args.debug_dir)
        dbg.mkdir(parents=True, exist_ok=True)
        dbg_log = open(dbg / "decode_log.txt", "w", encoding="utf-8")
        print(f"[debug] dumping per-marker decode info → {dbg}")

    yolo = YOLO(args.model)
    recognizer = DigitRecognizerTri()
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    win = "Digit Marker TRI"

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    exposure = args.camera_exposure if args.camera_exposure is not None else -6
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("Digit Marker TRI decoder. Esc=quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.mirror:
            frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced = clahe.apply(gray)
        results = yolo(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR), verbose=False, conf=args.conf)

        disp = frame
        n_ok = 0
        for r in results:
            if r.obb is None:
                continue
            for i in range(len(r.obb)):
                pts = r.obb.xyxyxyxy[i].cpu().numpy().reshape(4, 2)
                square = warp_square(enhanced, pts, size=200)
                if square.size == 0:
                    continue
                if dbg is not None:
                    marker_id, _conf, info = recognizer.read_debug(square)
                    corner = info["corner"]
                    if dbg_n < 80:
                        cv2.imwrite(str(dbg / f"m{dbg_n:03d}_orient.png"), info.get("_oriented", square))
                        if "_top" in info:
                            cv2.imwrite(str(dbg / f"m{dbg_n:03d}_top.png"), info["_top"])
                            cv2.imwrite(str(dbg / f"m{dbg_n:03d}_bot.png"), info["_bot"])
                        dbg_log.write(
                            f"m{dbg_n:03d} orient_ok={info['orient_ok']} conf={info['orient_conf']:.2f} "
                            f"top='{info['top']}'({info['top_conf']:.2f}) "
                            f"bot='{info['bot']}'({info['bot_conf']:.2f}) id={marker_id}\n")
                        dbg_log.flush()
                        dbg_n += 1
                else:
                    marker_id, _conf = recognizer.recognize(square)
                    corner, _ = find_triangle_corner(square)
                center = pts.mean(axis=0)
                if marker_id >= 0:
                    n_ok += 1
                    draw_corners(disp, pts, color=(0, 220, 0))
                    anchor = id_anchor(pts, corner)
                    cv2.putText(disp, f"{marker_id:03d}", (int(anchor[0]) - 14, int(anchor[1]) + 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 1, cv2.LINE_AA)
                    mc, up = marker_up_vector(pts, corner)
                    ordered = order_corners(pts)
                    arrow_len = 0.6 * np.hypot(*(ordered[0] - mc))
                    tip = (int(mc[0] + up[0] * arrow_len), int(mc[1] + up[1] * arrow_len))
                    cv2.arrowedLine(disp, (int(mc[0]), int(mc[1])), tip,
                                    (0, 0, 255), 1, cv2.LINE_AA, tipLength=0.3)
                else:
                    draw_corners(disp, pts, color=(0, 180, 180))
                    cv2.putText(disp, "?", (int(center[0]) - 5, int(center[1]) + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 180), 1, cv2.LINE_AA)

        cv2.putText(disp, f"IDs:{n_ok}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.imshow(win, disp)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    if dbg_log is not None:
        dbg_log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
