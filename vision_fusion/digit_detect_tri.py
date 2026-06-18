"""数字 Marker 解码器（digit_marker_tri 配套）：YOLO-OBB 定位 + 内角暗度三角定向
+ RapidOCR 读 4 位 + 加权 mod11+X 校验。

Usage:
    python -m vision_fusion.digit_detect_tri --source 0
"""
from __future__ import annotations

import argparse
import time

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


_TRI_L = 0.30  # 角三角取样区的腿长占比(罩住 marker 的黑三角，排除中央数字)


CHARS = "0123456789X"            # 11 类：0-9 与校验位 X(=10)

# 切格几何锁定部署 marker 的 GUI 参数(digit_marker_tri_settings.json)。
# 若重调 marker 列距/行距，这两个值要同步更新并重训分类器。
TRI_COL_GAP = 0.3765
TRI_ROW_GAP = 0.4176
CELL_HALF = 0.21                 # 归一化裁剪半边长(罩住单字、不蹭邻格)
CELL_OUT = 64                    # 输出格子尺寸(与 DigitCNN 输入一致)


def cell_centers(col_gap: float = TRI_COL_GAP, row_gap: float = TRI_ROW_GAP):
    """4 个数字格中心(归一化)，顺序 [TL d1, TR d2, BL d3, BR check]，
    与 generate_marker_tri 的 glyph 顺序一致。"""
    return [(0.5 - col_gap / 2, 0.5 - row_gap / 2),
            (0.5 + col_gap / 2, 0.5 - row_gap / 2),
            (0.5 - col_gap / 2, 0.5 + row_gap / 2),
            (0.5 + col_gap / 2, 0.5 + row_gap / 2)]


def slice_cells(square: np.ndarray, half: float = CELL_HALF, out: int = CELL_OUT):
    """把拉正后的 marker 切成 4 个数字格(灰度 out×out)。训练与推理共用，
    保证几何一致。square 可为灰度或 BGR。"""
    gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY) if square.ndim == 3 else square
    s = gray.shape[0]
    h = int(half * s)
    cells = []
    for ncx, ncy in cell_centers():
        cx, cy = int(ncx * s), int(ncy * s)
        x0, x1 = max(0, cx - h), min(s, cx + h)
        y0, y1 = max(0, cy - h), min(s, cy + h)
        crop = gray[y0:y1, x0:x1]
        cells.append(cv2.resize(crop, (out, out), interpolation=cv2.INTER_AREA))
    return cells


def _corner_triangle_darkness(gray: np.ndarray) -> list[float]:
    """4 个角各取一块直角三角区(贴角、斜边朝中心，正好罩黑三角该在的楔形)，
    返回各自的平均暗度 (255-灰度)。三角区比方块更聚焦三角、少蹭数字/背景。"""
    s = gray.shape[0]
    L = max(6, int(s * _TRI_L))
    tris = [
        [(0, 0), (L, 0), (0, L)],              # TL
        [(s, 0), (s - L, 0), (s, L)],          # TR
        [(s, s), (s - L, s), (s, s - L)],      # BR
        [(0, s), (L, s), (0, s - L)],          # BL
    ]
    inv = 255.0 - gray.astype(np.float32)
    out = []
    for t in tris:
        m = np.zeros((s, s), np.uint8)
        cv2.fillConvexPoly(m, np.array(t, np.int32), 1)
        out.append(float(inv[m == 1].mean()))
    return out


def corner_ranking(square: np.ndarray) -> tuple[list[int], list[float]]:
    """按三角区暗度从大到小给 4 个角排序，返回 (排序后的角下标, 各角暗度)。"""
    gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY) if square.ndim == 3 else square
    dark = _corner_triangle_darkness(gray)
    order = sorted(range(4), key=lambda i: dark[i], reverse=True)
    return order, dark


def find_triangle_corner(square: np.ndarray) -> tuple[int, float]:
    """相对最暗(三角区)的角=黑三角所在。返回 (corner, conf)。

    用三角形取样而非方块：贴角直角三角区正好罩三角该在的楔形，排除中央数字笔画与
    背景，真实 IR 上区分度更高。conf=(最暗-次暗)/最暗 仅参考；最终朝向由 recognize
    的"暗度排序+逐朝向校验"确定，单角选错也会被加权 mod11 校验纠回。
    """
    order, dark = corner_ranking(square)
    conf = (dark[order[0]] - dark[order[1]]) / (dark[order[0]] + 1e-6)
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

    def _read_oriented(self, square: np.ndarray, corner: int):
        """按指定角旋正 → 上下对半裁 → 各行 OCR。返回中间量。"""
        rot = _ROT_TO_TL[corner]
        o = square if rot is None else cv2.rotate(square, rot)
        h, w = o.shape[:2]
        x0, x1 = int(w * 0.12), int(w * 0.88)
        top = cv2.resize(o[int(h * 0.12):int(h * 0.50), x0:x1], None, fx=3, fy=3,
                         interpolation=cv2.INTER_CUBIC)
        bot = cv2.resize(o[int(h * 0.50):int(h * 0.88), x0:x1], None, fx=3, fy=3,
                         interpolation=cv2.INTER_CUBIC)
        tt, tc = self._ocr(top)
        bt, bc = self._ocr(bot)
        return tt, tc, bt, bc, o, top, bot

    def read_debug(self, square: np.ndarray) -> tuple[int, float, dict]:
        """暗度排序 + 逐朝向试 + 第一个过加权 mod11 校验的即采纳。

        三角暗度给朝向先验(最可能的先试)，校验位确认；单角排错也会被校验纠回
        (落到排序里下一个)，且按序取首个有效解，避免旋转碰撞误接受。
        """
        order, dark = corner_ranking(square)
        info = {"ranking": order, "corner": order[0], "orient_ok": True,
                "orient_conf": (dark[order[0]] - dark[order[1]]) / (dark[order[0]] + 1e-6),
                "top": "", "bot": "", "top_conf": 0.0, "bot_conf": 0.0}
        first = None
        for corner in order:
            tt, tc, bt, bc, o, top, bot = self._read_oriented(square, corner)
            if first is None:
                first = (corner, tt, tc, bt, bc, o, top, bot)
            mid = decode_id(tt + bt) if (tt and bt) else -1
            if mid >= 0:
                info.update(corner=corner, top=tt, bot=bt, top_conf=tc, bot_conf=bc,
                            _oriented=o, _top=top, _bot=bot)
                return mid, (tc + bc) / 2.0, info
        c, tt, tc, bt, bc, o, top, bot = first
        info.update(corner=c, top=tt, bot=bt, top_conf=tc, bot_conf=bc,
                    _oriented=o, _top=top, _bot=bot)
        return -1, 0.0, info

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

    fps_ema = None          # 指数滑动平均,读数稳一点
    while True:
        f0 = time.perf_counter()
        t = time.perf_counter()
        ok, frame = cap.read()
        t_read = time.perf_counter() - t
        if not ok:
            break
        if args.mirror:
            frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced = clahe.apply(gray)
        t = time.perf_counter()
        results = yolo(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR), verbose=False, conf=args.conf)
        t_yolo = time.perf_counter() - t

        disp = frame
        n_ok = 0
        n_mk = 0
        t = time.perf_counter()
        for r in results:
            if r.obb is None:
                continue
            for i in range(len(r.obb)):
                pts = r.obb.xyxyxyxy[i].cpu().numpy().reshape(4, 2)
                square = warp_square(enhanced, pts, size=200)
                if square.size == 0:
                    continue
                n_mk += 1
                marker_id, _conf, info = recognizer.read_debug(square)
                corner = info["corner"]            # 校验通过时采纳的朝向角
                if dbg is not None and dbg_n < 80:
                    cv2.imwrite(str(dbg / f"m{dbg_n:03d}_orient.png"), info.get("_oriented", square))
                    if "_top" in info:
                        cv2.imwrite(str(dbg / f"m{dbg_n:03d}_top.png"), info["_top"])
                        cv2.imwrite(str(dbg / f"m{dbg_n:03d}_bot.png"), info["_bot"])
                    dbg_log.write(
                        f"m{dbg_n:03d} ranking={info['ranking']} chosen={corner} "
                        f"top='{info['top']}' bot='{info['bot']}' id={marker_id}\n")
                    dbg_log.flush()
                    dbg_n += 1
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
        t_dec = time.perf_counter() - t

        t_frame = time.perf_counter() - f0
        inst_fps = 1.0 / t_frame if t_frame > 0 else 0.0
        fps_ema = inst_fps if fps_ema is None else 0.9 * fps_ema + 0.1 * inst_fps
        # 左上角:FPS + 每步 ms(read/yolo/decode),decode 一般是大头。
        cv2.putText(disp, f"FPS:{fps_ema:4.1f}  IDs:{n_ok}/{n_mk}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(disp,
                    f"read {t_read*1000:.0f} | yolo {t_yolo*1000:.0f} | decode {t_dec*1000:.0f} ms",
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
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
