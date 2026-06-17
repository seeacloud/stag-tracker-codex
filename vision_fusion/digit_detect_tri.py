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


def find_triangle_corner(square: np.ndarray, b_ratio: float = 0.07,
                         k_ratio: float = 0.12) -> tuple[int, float]:
    """4 个内角缝隙各取贴角小块，积分暗度最大者=黑三角所在角。

    返回 (corner, conf)。conf=(最暗-次暗)/(最暗) 作相对 margin。
    积分暗度天然低通，模糊不改变哪个角更黑（形状糊掉也无妨）。
    """
    gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY) if square.ndim == 3 else square
    s = gray.shape[0]
    b = int(s * b_ratio)
    k = max(4, int(s * k_ratio))
    patches = [
        gray[b:b + k, b:b + k],                  # TL
        gray[b:b + k, s - b - k:s - b],          # TR
        gray[s - b - k:s - b, s - b - k:s - b],  # BR
        gray[s - b - k:s - b, b:b + k],          # BL
    ]
    dark = [float((255.0 - p.astype(np.float32)).mean()) for p in patches]
    order = sorted(range(4), key=lambda i: dark[i], reverse=True)
    top, second = dark[order[0]], dark[order[1]]
    conf = (top - second) / (top + 1e-6)
    return order[0], conf


def orient_by_triangle(square: np.ndarray,
                       min_conf: float = 0.15) -> tuple[np.ndarray, bool]:
    """把黑三角旋到左上。margin 不足 → 返回原图 + False（定向存疑）。"""
    corner, conf = find_triangle_corner(square)
    if conf < min_conf:
        return square, False
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

    def recognize(self, square: np.ndarray, min_conf: float = 0.5) -> tuple[int, float]:
        oriented, ok = orient_by_triangle(square)
        if not ok:
            return -1, 0.0
        h, w = oriented.shape[:2]
        # 上下对半分，各含一行两个数字（全宽两列）。大字 marker 两行几乎占满高度，
        # 固定窄带会把两行搅在一起；对半切对各种字号/行距都稳。三角在上半左上角，
        # 占比小，RapidOCR rec 仍聚焦数字（实测 "08" 0.96 conf）。
        x0, x1 = int(w * 0.12), int(w * 0.88)
        top = oriented[int(h * 0.12):int(h * 0.50), x0:x1]
        bot = oriented[int(h * 0.50):int(h * 0.88), x0:x1]
        top = cv2.resize(top, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        bot = cv2.resize(bot, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        t_txt, t_c = self._ocr(top)
        b_txt, b_c = self._ocr(bot)
        if not (t_txt and b_txt):
            return -1, 0.0
        conf = (t_c + b_c) / 2.0
        if conf < min_conf:
            return -1, 0.0
        mid = decode_id(t_txt + b_txt)
        return (mid, conf) if mid >= 0 else (-1, 0.0)


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
    args = parser.parse_args()

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
