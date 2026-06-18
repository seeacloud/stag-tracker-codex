"""数字 Marker 实时检测器 v3（极简版）。

Pipeline: 摄像头 → CLAHE → YOLO-OBB找框 → 透视矫正 → 圆盘模板定向 → 上下两行分读 → RapidOCR → 校验位验证

Usage:
    python -m vision_fusion.digit_detect --source 0
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO
from rapidocr_onnxruntime import RapidOCR

from .digit_marker import checksum


def order_corners(pts: np.ndarray) -> np.ndarray:
    """把 4 个角点按 [左上, 右上, 右下, 左下] 排序（不依赖 OBB 给出的起始角）。"""
    c = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    order = np.argsort(angles)  # 逆时针
    pts = pts[order]
    # 让起点为最靠左上的角
    start = np.argmin(pts.sum(axis=1))
    pts = np.roll(pts, -start, axis=0)
    return pts.astype(np.float32)


def warp_square(img: np.ndarray, pts: np.ndarray, size: int = 200) -> np.ndarray:
    """用 4 角点把斜放的 marker 透视矫正成正方形（消除旋转/透视形变）。"""
    src = order_corners(pts)
    dst = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (size, size))


def _dot_template(radius: int) -> np.ndarray:
    """实心黑圆盘模板（白底），用于匹配 marker 定向圆点。"""
    t = np.full((2 * radius, 2 * radius), 255, np.uint8)
    cv2.circle(t, (radius, radius), radius, 0, -1)
    return t


def find_dot_corner(square: np.ndarray) -> int:
    """返回定向圆点所在的角：0=左上 1=右上 2=右下 3=左下。

    在 4 个角区域做圆盘模板的归一化相关匹配（TM_CCOEFF_NORMED），相关性最高的角
    即圆点所在。相关系数只看"圆盘形状"模式，免疫退化亮度偏移——圆点被模糊/散射
    糊成灰团后仍能匹配（实测 98.8%）。SQDIFF 看绝对差，圆点变灰就失配（91.8%）；
    纯比角落暗度更差（~50-70%），因为暗度丢了形状信息，分不清圆点黑/数字黑/边框黑。
    """
    s = square.shape[0]
    r = max(8, int(s * 0.085))
    tmpl = _dot_template(r)
    q = int(s * 0.42)  # 每个角取 42% 区域，确保覆盖圆点可能的位置
    regions = [
        square[:q, :q],          # 左上
        square[:q, s - q:],      # 右上
        square[s - q:, s - q:],  # 右下
        square[s - q:, :q],      # 左下
    ]
    scores = [cv2.matchTemplate(reg, tmpl, cv2.TM_CCOEFF_NORMED).max() for reg in regions]
    return int(np.argmax(scores))


def orient_by_dot(square: np.ndarray) -> np.ndarray:
    """把定向圆点旋到左上角。"""
    rotations = {
        0: None,
        1: cv2.ROTATE_90_COUNTERCLOCKWISE,  # 右上 -> 左上
        2: cv2.ROTATE_180,                  # 右下 -> 左上
        3: cv2.ROTATE_90_CLOCKWISE,         # 左下 -> 左上
    }
    rot = rotations[find_dot_corner(square)]
    return square if rot is None else cv2.rotate(square, rot)


class DigitRecognizer:
    """RapidOCR(PP-OCR) 读数字带 + 校验位验证。"""

    def __init__(self):
        self.reader = RapidOCR()

    def _ocr_digits(self, crop: np.ndarray) -> tuple[str, float]:
        """对裁剪区域纯识别(rec-only)，返回 (数字串, 平均置信度)。读不出返回 ('', 0)。"""
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        # 位置已知，DBNet 检测在低对比度噪声带上会失败，rec-only 才能读出。
        result, _ = self.reader(crop, use_det=False, use_cls=False, use_rec=True)
        if not result:
            return '', 0.0
        text = ''.join(r[0] for r in result)
        conf = float(np.mean([r[1] for r in result]))
        digits = ''.join(c for c in text if c.isdigit())
        return digits, conf

    def _validate(self, digits: str, conf: float, min_conf: float) -> tuple[int, float]:
        """4 位数字串 + 校验位验证 → (marker_id, conf)。失败返回 (-1, 0)。"""
        if len(digits) < 4 or conf < min_conf:
            return -1, 0.0
        d = [int(c) for c in digits[:4]]
        if d[3] != (d[0] + d[1] + d[2]) % 10:
            return -1, 0.0
        return d[0] * 100 + d[1] * 10 + d[2], conf

    def recognize(self, square: np.ndarray, min_conf: float = 0.5) -> tuple[int, float]:
        """黑点定向 → 上下两行分别裁 → RapidOCR 各读一行 → 拼 4 位 + 校验。

        2×2 版式：上行=d0 d1，下行=d2 d3。每行只有 2 个大数字，比单行 4 个挤一起
        好认得多，字形误读率大降（实测退化下 93% 正确、0 误判，旧单行版 ~78%）。
        数字区在右下（避开左上定向圆点）。
        """
        oriented = orient_by_dot(square)
        h, w = oriented.shape[:2]
        x0, x1 = int(w * 0.30), int(w * 0.95)  # 右侧，避开左上圆点
        top = oriented[int(h * 0.28):int(h * 0.55), x0:x1]
        bot = oriented[int(h * 0.55):int(h * 0.85), x0:x1]
        top = cv2.resize(top, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        bot = cv2.resize(bot, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        d_top, c_top = self._ocr_digits(top)
        d_bot, c_bot = self._ocr_digits(bot)
        digits = (d_top + d_bot)[:4]
        conf = (c_top + c_bot) / 2.0 if (d_top and d_bot) else 0.0
        return self._validate(digits, conf, min_conf)



class MarkerTracker:
    """平滑追踪 + ID 投票。"""

    def __init__(self, alpha=0.2, hold_frames=15):
        self.tracks: dict[int, dict] = {}
        self.alpha = alpha
        self.hold_frames = hold_frames
        self.next_id = 0

    def update(self, detections: list[tuple[int, float, np.ndarray]], decay: float = 1.0):
        """detections: list of (marker_id, conf, center_xy)。

        marker_id<0（本帧 OCR 失败）也参与位置追踪——刷新 track 位置与存活，
        但不投票，由历史成功帧的投票维持稳定 ID。这样本帧读不出的框也能查到历史 ID。

        投票按 OCR 置信度加权（不是 +1 等权），这样高信心的正确读数能碾压
        低信心的误读，解决弱校验位下的 95→94、91→84 类误判。

        decay<1：每帧先把所有 track 的累积票乘 decay 再加新票——让历史票随时间衰减，
        marker 被换/移走时旧 id 会褪去、当前帧实际读到的新 id 几帧内接管(避免位置记忆赖着旧 id)。
        """
        if decay < 1.0:
            for t in self.tracks.values():
                t["votes"] = {k: v * decay for k, v in t["votes"].items()}
        matched = set()
        for marker_id, conf, center in detections:
            best_tid = None
            best_dist = 80
            for tid, t in self.tracks.items():
                if tid in matched: continue
                d = np.hypot(*(np.array(t["center"]) - center))
                if d < best_dist:
                    best_dist = d
                    best_tid = tid
            if best_tid is not None:
                t = self.tracks[best_tid]
                t["center"] = [t["center"][0]*(1-self.alpha) + center[0]*self.alpha,
                               t["center"][1]*(1-self.alpha) + center[1]*self.alpha]
                t["miss"] = 0
                t["age"] += 1
                if marker_id >= 0:  # 只有成功识别才投票
                    t["conf"] = t["conf"]*0.7 + conf*0.3
                    # 按置信度加权投票（conf 越高贡献越大，误读通常低信心）
                    t["votes"][marker_id] = t["votes"].get(marker_id, 0) + conf
                    t["id"] = max(t["votes"], key=t["votes"].get)
                matched.add(best_tid)
            elif marker_id >= 0:  # 只为成功识别新建 track，避免噪声框无限建轨
                self.tracks[self.next_id] = {
                    "center": list(center), "id": marker_id,
                    "conf": conf, "age": 1, "miss": 0,
                    "votes": {marker_id: conf}  # 初始投票也用 conf 加权
                }
                self.next_id += 1
        for tid in list(self.tracks.keys()):
            if tid not in matched:
                self.tracks[tid]["miss"] += 1
                if self.tracks[tid]["miss"] > self.hold_frames:
                    del self.tracks[tid]

    def get_stable(self, min_age=3, margin=1.5):
        """返回稳定锁定的 tracks。

        margin: 投票第一名的累积权重必须 >= 第二名 * margin 才确认——
        防止 95 vs 94 这种强校验碰撞时两边票数接近就草率输出。
        """
        result = []
        for t in self.tracks.values():
            if t["age"] < min_age or t["id"] < 0:
                continue
            votes = t["votes"]
            top = max(votes.values())
            runner_up = sorted(votes.values(), reverse=True)[1] if len(votes) > 1 else 0
            if top >= runner_up * margin:
                result.append((t["id"], t["conf"], t["center"]))
        return result

    def query_at(self, center, max_dist=50, min_weight=1.5, margin=1.5):
        """按位置查该处 track 的稳定投票 ID；查不到返回 -1。

        min_weight: 第一名累积权重最低门槛。
        margin: 第一名必须 >= 第二名 * margin 才输出(防 95/94 碰撞)。
        """
        best_id, best_dist = -1, max_dist
        for t in self.tracks.values():
            if t["id"] < 0:
                continue
            votes = t["votes"]
            top_w = votes.get(t["id"], 0)
            if top_w < min_weight:
                continue
            runner_up = max((v for k, v in votes.items() if k != t["id"]), default=0)
            if top_w < runner_up * margin:
                continue
            d = np.hypot(*(np.array(t["center"]) - center))
            if d < best_dist:
                best_dist = d
                best_id = t["id"]
        return best_id


def marker_up_vector(pts: np.ndarray, dot_corner: int) -> tuple[np.ndarray, np.ndarray]:
    """返回 (marker中心, 指向marker顶边中点的单位向量)。

    pts 是 OBB 四角；dot_corner 是圆点所在角（find_dot_corner 的返回，
    对应 order_corners 输出 [TL,TR,BR,BL] 的下标）。圆点=物理左上角，
    顶边 = 物理左上→物理右上 = ordered[dot] → ordered[(dot+1)%4]。
    """
    ordered = order_corners(pts)
    center = ordered.mean(axis=0)
    tl = ordered[dot_corner]
    tr = ordered[(dot_corner + 1) % 4]
    top_mid = (tl + tr) / 2.0
    up = top_mid - center
    n = np.hypot(*up)
    return center, (up / n if n > 1e-6 else np.array([0.0, -1.0]))


def id_anchor(pts: np.ndarray, dot_corner: int, u=0.78, v=0.21) -> np.ndarray:
    """ID 文字锚点：圆点关于 y 轴(垂直中线)的镜像位置——物理右上、留边距。

    圆点在 marker 内约归一化 (0.21, 0.21)（左上、带内边距）。镜像到右上即
    (0.78, 0.21)。用 marker 自身坐标系（物理左上→右上 = u 轴，左上→左下 = v 轴）
    把归一化坐标映射回图像，这样不管 marker 转到哪都跟着走、且始终留出边距。
    """
    ordered = order_corners(pts)
    tl = ordered[dot_corner]                  # 物理左上
    tr = ordered[(dot_corner + 1) % 4]        # 物理右上
    bl = ordered[(dot_corner + 3) % 4]        # 物理左下
    u_axis = tr - tl                          # 顶边方向
    v_axis = bl - tl                          # 左边方向
    return tl + u * u_axis + v * v_axis


def draw_corners(frame, pts, color=(0, 220, 0), length=14):
    pts = pts.astype(int)
    for i in range(4):
        p1 = pts[i]
        for p_next in [pts[(i+1)%4], pts[(i-1)%4]]:
            dx, dy = p_next[0]-p1[0], p_next[1]-p1[1]
            dist = max(1, int(np.hypot(dx, dy)))
            ux, uy = dx*length//dist, dy*length//dist
            cv2.line(frame, tuple(p1), (p1[0]+ux, p1[1]+uy), color, 1, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description="Digit marker detector v3.")
    parser.add_argument("--source", default="0")
    parser.add_argument("--model", default="models/digi_marker_obb.pt")
    parser.add_argument("--mirror", action="store_true", default=False)
    parser.add_argument("--camera-exposure", type=int, default=None)
    parser.add_argument("--conf", type=float, default=0.3)
    args = parser.parse_args()

    yolo = YOLO(args.model)
    recognizer = DigitRecognizer()
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    WIN = "Digit Marker v3"

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    exposure = args.camera_exposure if args.camera_exposure is not None else -6
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

    # 窗口 + 控制滑块（显示模式切换 / CLAHE 参数实时调节）
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("view 0raw1gray2clahe", WIN, 0, 2, lambda v: None)
    cv2.createTrackbar("clahe_clip x10", WIN, 50, 200, lambda v: None)  # 5.0 = 50/10
    cv2.createTrackbar("clahe_tile", WIN, 8, 32, lambda v: None)

    prev_time = time.perf_counter()
    fps = 0.0

    print("Digit Marker v3. Esc=quit, +/-=exposure. Trackbars: 切换显示底图 / 调 CLAHE")

    while True:
        ok, frame = cap.read()
        if not ok: break

        now = time.perf_counter()
        fps = fps * 0.9 + 0.1 / max(now - prev_time, 1e-6)
        prev_time = now

        if args.mirror:
            frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # CLAHE 参数从 trackbar 实时读取（clip 滑块是 x10 整数，tile 不能为 0）
        clip = max(1, cv2.getTrackbarPos("clahe_clip x10", WIN)) / 10.0
        tile = max(1, cv2.getTrackbarPos("clahe_tile", WIN))
        clahe.setClipLimit(clip)
        clahe.setTilesGridSize((tile, tile))
        enhanced = clahe.apply(gray)

        # YOLO 找 marker
        results = yolo(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR), verbose=False, conf=args.conf)

        draw_list = []  # (pts, marker_id, center, dot_corner) 延后绘制，等选好显示底图再画
        for r in results:
            if r.obb is None: continue
            for i in range(len(r.obb)):
                pts = r.obb.xyxyxyxy[i].cpu().numpy().reshape(4, 2)
                # 透视矫正：用 4 角点把斜放/旋转的 marker 拉正成正方形
                square = warp_square(enhanced, pts, size=200)
                if square.size == 0: continue

                # OCR（内部用黑点定向）
                marker_id, conf = recognizer.recognize(square)
                center = pts.mean(axis=0)
                dot_corner = find_dot_corner(square)  # 圆点所在角，用于画方向箭头

                draw_list.append((pts, marker_id, center, dot_corner))

        # 选显示底图：0=原图 1=灰度 2=CLAHE（识别始终跑在 CLAHE 上，这里只换显示）
        view = cv2.getTrackbarPos("view 0raw1gray2clahe", WIN)
        if view == 1:
            disp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif view == 2:
            disp = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        else:
            disp = frame

        # 在选定底图上画识别框 + ID + 方向箭头（纯单帧，无多帧投票——marker 频繁移动场景）
        n_ok = 0
        for pts, marker_id, center, dot_corner in draw_list:
            ordered = order_corners(pts)
            mc, up = marker_up_vector(pts, dot_corner)
            # ID 锚点 = 圆点关于 y 轴的镜像（物理右上、留内边距），呼应左上圆点
            anchor = id_anchor(pts, dot_corner)
            if marker_id >= 0:
                n_ok += 1
                color = (0, 220, 0)
                draw_corners(disp, pts, color=color)
                txt = str(marker_id)
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                org = (int(anchor[0] - tw / 2), int(anchor[1] + th / 2))  # 锚点居中
                cv2.putText(disp, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
                # 方向箭头：从中心沿 up 方向指向顶边外侧
                arrow_len = 0.6 * np.hypot(*(ordered[0] - mc))
                tip = (int(mc[0] + up[0] * arrow_len), int(mc[1] + up[1] * arrow_len))
                cv2.arrowedLine(disp, (int(mc[0]), int(mc[1])), tip,
                                (0, 0, 255), 1, cv2.LINE_AA, tipLength=0.3)
            else:
                draw_corners(disp, pts, color=(0, 180, 180))
                cv2.putText(disp, "?", (int(center[0])-5, int(center[1])+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 180), 1, cv2.LINE_AA)

        # HUD
        view_name = ["RAW", "GRAY", "CLAHE"][view]
        cv2.putText(disp, f"FPS:{fps:.0f} | Expo:{exposure} | View:{view_name} | clip:{clip:.1f} tile:{tile} | IDs:{n_ok}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow(WIN, disp)

        key = cv2.waitKey(1) & 0xFF
        if key == 27: break
        elif key == ord('+') or key == ord('='):
            exposure = min(-2, exposure + 1)
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        elif key == ord('-') or key == ord('_'):
            exposure = max(-10, exposure - 1)
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
