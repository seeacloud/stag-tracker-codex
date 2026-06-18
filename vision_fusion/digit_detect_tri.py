"""数字 Marker 解码器（digit_marker_tri 配套）：YOLO-OBB 定位 + 内角暗度三角定向
+ RapidOCR 读 4 位 + 加权 mod11+X 校验。

Usage:
    python -m vision_fusion.digit_detect_tri --source 0
"""
from __future__ import annotations

import argparse
import threading
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


def _ensure_gray(img: np.ndarray) -> np.ndarray:
    """转单通道灰度，兼容 (H,W)/(H,W,1)/(H,W,3)/(H,W,4)。
    ultralytics 会 patch cv2.imread，灰度读出可能是 (H,W,1)，单纯判 ndim==3 会误调 cvtColor。"""
    if img.ndim == 2:
        return img
    if img.shape[2] == 1:
        return img[:, :, 0]
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


CHARS = "0123456789X"            # 11 类：0-9 与校验位 X(=10)

# 切格几何锁定部署 marker 的 GUI 参数(digit_marker_tri_settings.json)。
# 若重调 marker 列距/行距，这两个值要同步更新并重训分类器。
TRI_COL_GAP = 0.3765
TRI_ROW_GAP = 0.4176
CELL_HALF = 0.20                 # 归一化裁剪半边长(罩住单字、不蹭邻格)
CELL_OUT = 64                    # 输出格子尺寸(与 DigitCNN 输入一致)
# 定向黑三角的几何(同 digit_marker_tri_settings.json)——切格前抹白,
# 否则 TL 格会把三角当墨迹。大字体下裁剪窗必然触到三角,只能抹掉而非靠缩窗避开。
TRI_BORDER_RATIO = 0.046
TRI_CHAMFER_RATIO = 0.187


def _mask_triangle(gray: np.ndarray) -> np.ndarray:
    """把左上内角的定向黑三角抹成白色(返回副本)。三角定向已在上游完成,
    此处只为净化 TL 数字格。楔形 [(0,0),(T,0),(0,T)],T 略大于 border+chamfer。"""
    out = gray.copy()
    s = out.shape[0]
    T = int((TRI_BORDER_RATIO + TRI_CHAMFER_RATIO) * 1.12 * s)
    cv2.fillConvexPoly(out, np.array([(0, 0), (T, 0), (0, T)], np.int32), 255)
    return out


def cell_centers(col_gap: float = TRI_COL_GAP, row_gap: float = TRI_ROW_GAP):
    """4 个数字格中心(归一化)，顺序 [TL d1, TR d2, BL d3, BR check]，
    与 generate_marker_tri 的 glyph 顺序一致。"""
    return [(0.5 - col_gap / 2, 0.5 - row_gap / 2),
            (0.5 + col_gap / 2, 0.5 - row_gap / 2),
            (0.5 - col_gap / 2, 0.5 + row_gap / 2),
            (0.5 + col_gap / 2, 0.5 + row_gap / 2)]


def slice_cells(square: np.ndarray, half: float = CELL_HALF, out: int = CELL_OUT):
    """把拉正后的 marker 切成 4 个数字格(灰度 out×out)。训练与推理共用，
    保证几何一致。先抹掉定向黑三角再切。square 可为灰度或 BGR。"""
    gray = _ensure_gray(square)
    gray = _mask_triangle(gray)
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


_GRID_CACHE: dict = {}


def _xy_grid(h: int, w: int):
    g = _GRID_CACHE.get((h, w))
    if g is None:
        yy, xx = np.mgrid[0:h, 0:w]
        g = (xx.astype(np.float32), yy.astype(np.float32))
        _GRID_CACHE[(h, w)] = g
    return g


def _corner_triangle_darkness(gray: np.ndarray) -> list[float]:
    """4 个角各取一块直角三角区(贴角、斜边朝中心，正好罩黑三角该在的楔形)，
    返回各自"比局部光照平面暗多少"。**先拟合并减去线性光照平面**(抗 IR 画面的
    明暗梯度——否则整片偏暗的角会被误当三角);三角是局部暗楔,梯度是大坡,平面只吃坡。
    在 64×64 下采样上算(方向是粗决策、梯度平滑,精度无损,省每帧 plane 拟合开销)。"""
    if gray.shape[0] != 64:
        gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    g = gray.astype(np.float32)
    h, w = g.shape
    s = h
    xx, yy = _xy_grid(h, w)
    # 3×3 正规方程拟合 z=a·x+b·y+c(O(N) 求和,比 lstsq 快),残差=去梯度后的图
    x = xx.ravel(); y = yy.ravel(); z = g.ravel(); n = z.size
    Sx = x.sum(); Sy = y.sum(); Sxx = x @ x; Syy = y @ y; Sxy = x @ y
    Sz = z.sum(); Sxz = x @ z; Syz = y @ z
    M = np.array([[Sxx, Sxy, Sx], [Sxy, Syy, Sy], [Sx, Sy, n]], np.float64)
    rhs = np.array([Sxz, Syz, Sz], np.float64)
    try:
        a, b, c = np.linalg.solve(M, rhs)
        resid = g - (a * xx + b * yy + c)
    except np.linalg.LinAlgError:
        resid = g - g.mean()
    L = max(6, int(s * _TRI_L))
    tris = [
        [(0, 0), (L, 0), (0, L)],              # TL
        [(s, 0), (s - L, 0), (s, L)],          # TR
        [(s, s), (s - L, s), (s, s - L)],      # BR
        [(0, s), (L, s), (0, s - L)],          # BL
    ]
    out = []
    for t in tris:
        m = np.zeros((s, s), np.uint8)
        cv2.fillConvexPoly(m, np.array(t, np.int32), 1)
        out.append(float(-resid[m == 1].mean()))   # 残差越暗(越负)→取负后越大→越像三角
    return out


def corner_ranking(square: np.ndarray) -> tuple[list[int], list[float]]:
    """按三角区暗度从大到小给 4 个角排序，返回 (排序后的角下标, 各角暗度)。"""
    gray = _ensure_gray(square)
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


class DigitClassifierTri:
    """轻量 CNN(0-9+X)读 2×2 数字 + 加权 mod11 校验，替代 RapidOCR。

    接口同 DigitRecognizerTri：read_debug/recognize。4 格一次 batch 推理；
    按角暗度排序逐朝向试，首个过 decode_id 校验的采纳。全 4 朝向也才 16 次微推理。
    """

    def __init__(self, model_path: str = "models/tri_digit_cnn.pt",
                 min_cell_conf: float = 0.9, max_orient: int = 1):
        import torch
        from .nn_train_digit import DigitCNN
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = DigitCNN(num_classes=len(CHARS)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        # 接受一个读数需:① 过加权 mod11 校验 ② 4 格里**最低**置信度 ≥ min_cell_conf
        # (淡 marker 误读常有某格 0.4~0.6 在瞎猜,挡掉避免凑巧撞合法编号的假阳性)。
        self.min_cell_conf = min_cell_conf
        # 只试暗度排序前 max_orient 个朝向(去梯度后暗度可靠),杜绝"末位朝向凑合法编号"。
        self.max_orient = max_orient

    def _classify(self, cells: list) -> tuple[str, float, float]:
        """4 格 → (4 字符串, 平均置信度, 最低单格置信度)。一次 batch 推理。"""
        batch = np.stack([c.astype(np.float32) / 255.0 for c in cells])[:, None]
        x = self.torch.from_numpy(batch).to(self.device)
        with self.torch.no_grad():
            prob = self.torch.softmax(self.model(x), dim=1)
            conf, idx = prob.max(dim=1)
        chars = "".join(CHARS[i] for i in idx.tolist())
        cc = conf.tolist()
        return chars, float(sum(cc) / 4.0), float(min(cc))

    def _classify_many(self, groups: list) -> list:
        """groups: list[4 格]。所有组的格子拼成一个大 batch 跑**一次** CNN,
        返回每组 (chars, mean_conf, min_conf)。消除多 marker×多朝向逐次调用的 GPU 同步开销。"""
        if not groups:
            return []
        flat = np.stack([c for g in groups for c in g]).astype(np.float32) / 255.0
        x = self.torch.from_numpy(flat)[:, None].to(self.device)
        with self.torch.no_grad():
            prob = self.torch.softmax(self.model(x), dim=1)
            conf, idx = prob.max(dim=1)
        idx = idx.tolist(); conf = conf.tolist()
        out = []
        for k in range(len(groups)):
            ii = idx[k * 4:(k + 1) * 4]; cc = conf[k * 4:(k + 1) * 4]
            out.append(("".join(CHARS[j] for j in ii), sum(cc) / 4.0, min(cc)))
        return out

    def read_batch(self, squares: list) -> list:
        """对多个 square 一次性解码:所有 square×前 max_orient 朝向的格子拼成单次 CNN 前向,
        挑首个**过校验且 4 格最低置信度达标**的。返回 list[(id, conf, info)],与 read_debug 同格式。"""
        groups = []          # 每个 = 一组 4 格
        owner = []           # (square_idx, corner)
        rankings = []
        for si, sq in enumerate(squares):
            order, dark = corner_ranking(sq)
            rankings.append((order, dark))
            for corner in order[:self.max_orient]:     # 只试暗度前 N 个朝向
                rot = _ROT_TO_TL[corner]
                o = sq if rot is None else cv2.rotate(sq, rot)
                groups.append(slice_cells(o))
                owner.append((si, corner))
        preds = self._classify_many(groups)        # 一次前向
        per_sq = {}
        for (si, corner), (chars, conf, mn) in zip(owner, preds):
            per_sq.setdefault(si, []).append((corner, chars, conf, mn))
        results = []
        for si in range(len(squares)):
            order, dark = rankings[si]
            info = {"ranking": order, "corner": order[0], "orient_ok": True,
                    "orient_conf": (dark[order[0]] - dark[order[1]]) / (dark[order[0]] + 1e-6),
                    "top": "", "bot": "", "top_conf": 0.0, "bot_conf": 0.0}
            cands = per_sq[si]                      # 已按 order 顺序
            first = cands[0]
            chosen = None
            for corner, chars, conf, mn in cands:
                if decode_id(chars) >= 0 and mn >= self.min_cell_conf:   # 校验 + 置信度门槛
                    chosen = (decode_id(chars), conf, corner, chars); break
            if chosen is not None:
                mid, conf, corner, chars = chosen
                info.update(corner=corner, top=chars[:2], bot=chars[2:], top_conf=conf, bot_conf=conf)
                results.append((mid, conf, info))
            else:
                corner, chars, conf, mn = first
                info.update(corner=corner, top=chars[:2], bot=chars[2:], top_conf=conf, bot_conf=conf)
                results.append((-1, 0.0, info))
        return results

    def read_debug(self, square: np.ndarray) -> tuple[int, float, dict]:
        order, dark = corner_ranking(square)
        info = {"ranking": order, "corner": order[0], "orient_ok": True,
                "orient_conf": (dark[order[0]] - dark[order[1]]) / (dark[order[0]] + 1e-6),
                "top": "", "bot": "", "top_conf": 0.0, "bot_conf": 0.0}
        first = None
        for corner in order[:self.max_orient]:
            rot = _ROT_TO_TL[corner]
            o = square if rot is None else cv2.rotate(square, rot)
            chars, conf, mn = self._classify(slice_cells(o))
            if first is None:
                first = (corner, chars, conf, o)
            if decode_id(chars) >= 0 and mn >= self.min_cell_conf:
                info.update(corner=corner, top=chars[:2], bot=chars[2:],
                            top_conf=conf, bot_conf=conf, _oriented=o)
                return decode_id(chars), conf, info
        c, chars, conf, o = first
        info.update(corner=c, top=chars[:2], bot=chars[2:],
                    top_conf=conf, bot_conf=conf, _oriented=o)
        return -1, 0.0, info

    def recognize(self, square: np.ndarray, min_conf: float = 0.5) -> tuple[int, float]:
        mid, conf, _ = self.read_debug(square)
        if mid < 0 or conf < min_conf:
            return -1, 0.0
        return mid, conf


def decode_markers_cached(entries, recognizer, tracker, decay: float = 0.6):
    """一帧多 marker:**每帧都重新识别**(批处理,便宜),用衰减投票平滑。

    不再按位置缓存跳过识别——否则 marker 被换/移动时旧位置会赖着旧 id(位置记忆 bug)。
    每帧 read_batch 重读所有 marker → 喂 tracker.update(decay) 衰减投票 → query_at 取稳定 id。
    这样 id 永远跟随"当前帧实际读到的",换 marker 几帧内接管;衰减投票仍消单帧 flicker。
    entries: list[(pts, square)]。返回 (items, n_cnn),item={pts,center,corner,id,info}。
    """
    squares = [sq for _, sq in entries]
    if squares and hasattr(recognizer, "read_batch"):
        reads = recognizer.read_batch(squares)
    else:
        reads = [recognizer.read_debug(sq) for sq in squares]
    n_cnn = len(squares)
    detections = []
    pend = []
    for (pts, _sq), (mid, conf, info) in zip(entries, reads):
        center = tuple(float(v) for v in pts.mean(axis=0))
        detections.append((mid, conf, center))
        pend.append((pts, center, info["corner"], info))
    tracker.update(detections, decay=decay)
    items = []
    for pts, center, corner, info in pend:
        items.append({"pts": pts, "center": center, "corner": corner,
                      "id": tracker.query_at(center), "info": info})
    return items, n_cnn


class _FrameGrabber:
    """后台线程持续抓帧,只保留最新一帧——把相机 read 延迟移出主循环,
    并丢弃缓冲旧帧(消除滞后)。主循环 read() 立即拿到最新帧。"""

    def __init__(self, src, exposure):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        self._lock = threading.Lock()
        self._frame = None
        self._stop = False
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while not self._stop:
            ok, f = self.cap.read()
            if ok:
                with self._lock:
                    self._frame = f

    def read(self):
        with self._lock:
            f = self._frame
        return (f is not None), (None if f is None else f.copy())

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self._stop = True
        self._t.join(timeout=1.0)
        self.cap.release()


def main() -> int:
    from ultralytics import YOLO
    from .digit_detect import (order_corners, warp_square, draw_corners,
                               marker_up_vector, id_anchor, MarkerTracker)

    parser = argparse.ArgumentParser(description="Digit marker (tri) decoder.")
    parser.add_argument("--source", default="0")
    parser.add_argument("--model", default="models/tri_marker_obb.pt")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--recognizer", choices=["cnn", "ocr"], default="cnn",
                        help="cnn=轻量数字分类器(默认,快); ocr=RapidOCR(对照)。")
    parser.add_argument("--min-cell-conf", type=float, default=0.9,
                        help="接受 id 所需的 4 格最低置信度门槛(高=宁缺毋滥,少错 id;低=多解出)。")
    parser.add_argument("--no-track", action="store_true", default=False,
                        help="关掉多帧投票+解码缓存(每帧都重新识别,慢但无状态)。")
    parser.add_argument("--no-thread", action="store_true", default=False,
                        help="关掉后台抓帧线程(相机 read 回到主循环,慢)。")
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
    if args.recognizer == "cnn":
        from pathlib import Path as _P
        if _P("models/tri_digit_cnn.pt").exists():
            recognizer = DigitClassifierTri(min_cell_conf=args.min_cell_conf)
            print(f"recognizer: CNN (tri_digit_cnn.pt), min_cell_conf={args.min_cell_conf}")
        else:
            print("WARN: models/tri_digit_cnn.pt 缺失，回退 RapidOCR")
            recognizer = DigitRecognizerTri()
    else:
        recognizer = DigitRecognizerTri()
        print("recognizer: RapidOCR")
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    win = "Digit Marker TRI"

    src = int(args.source) if args.source.isdigit() else args.source
    exposure = args.camera_exposure if args.camera_exposure is not None else -6
    if args.no_thread:
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    else:
        cap = _FrameGrabber(src, exposure)
        time.sleep(0.5)        # 等后台线程抓到第一帧
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("Digit Marker TRI decoder. Esc=quit.")

    tracker = None if args.no_track else MarkerTracker()
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
        t = time.perf_counter()
        # 收集本帧所有检出 marker
        entries = []
        for r in results:
            if r.obb is None:
                continue
            for i in range(len(r.obb)):
                pts = r.obb.xyxyxyxy[i].cpu().numpy().reshape(4, 2)
                square = warp_square(enhanced, pts, size=200)
                if square.size:
                    entries.append((pts, square))
        n_mk = len(entries)
        # 解码:有跟踪→已锁定的跳过 CNN(稳态 decode≈0);无跟踪→每帧重识别
        if tracker is not None:
            items, _ = decode_markers_cached(entries, recognizer, tracker)
        else:
            items = []
            for pts, square in entries:
                mid, _c, info = recognizer.read_debug(square)
                items.append({"pts": pts, "center": pts.mean(axis=0),
                              "corner": info["corner"], "id": mid, "info": info})
        t_dec = time.perf_counter() - t

        for it in items:
            pts, corner, marker_id = it["pts"], it["corner"], it["id"]
            center = pts.mean(axis=0)
            if dbg is not None and dbg_n < 80 and it.get("info"):
                info = it["info"]
                cv2.imwrite(str(dbg / f"m{dbg_n:03d}_orient.png"), info.get("_oriented", entries[0][1]))
                dbg_log.write(f"m{dbg_n:03d} ranking={info['ranking']} chosen={corner} "
                              f"top='{info['top']}' bot='{info['bot']}' id={marker_id}\n")
                dbg_log.flush(); dbg_n += 1
            # 三段解耦:1) 检出→绿框  2) 方向→箭头  3) 解出→id(总是各画各的)
            draw_corners(disp, pts, color=(0, 220, 0))
            mc, up = marker_up_vector(pts, corner)
            ordered = order_corners(pts)
            arrow_len = 0.6 * np.hypot(*(ordered[0] - mc))
            tip = (int(mc[0] + up[0] * arrow_len), int(mc[1] + up[1] * arrow_len))
            cv2.arrowedLine(disp, (int(mc[0]), int(mc[1])), tip,
                            (0, 0, 255), 1, cv2.LINE_AA, tipLength=0.3)
            if marker_id >= 0:
                n_ok += 1
                anchor = id_anchor(pts, corner)
                cv2.putText(disp, f"{marker_id:03d}", (int(anchor[0]) - 14, int(anchor[1]) + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 1, cv2.LINE_AA)
            else:
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
