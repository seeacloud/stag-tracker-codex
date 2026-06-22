# 符号 Marker 系统 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 tri marker 的 2×2 数字换成 11 个高区分度几何符号(参数化绘制:统一线宽/圆角线头/可调大小间距),沿用整条 tri 识别管线,只换"每格内容"。

**Architecture:** 新增 `symbol_set.py`(符号画法唯一真相源,参数驱动)+ `symbol_marker.py`(复刻 tri 外框+内角黑三角+2×2 布局,格内用 `draw_symbol`)+ `symbol_marker_ui.py`(Tkinter 调参,参数存 `symbol_marker_settings.json` 自动存取)。逻辑标签仍是 `0-9X`,故 `checksum_char`/`decode_id`/`slice_cells`/`DigitCNN`/`MarkerTracker`/校验/投票全部复用。训练数据切格用同一 `slice_cells`,重训出 `models/symbol_cnn.pt`,识别用 `DigitClassifierTri(model_path=...)`。

**Tech Stack:** OpenCV(参数化绘制,圆端点 cv2.LINE_AA)、PIL 不需要(符号纯几何,用 cv2)、PyTorch(复用 DigitCNN)、Tkinter(复用 digit_marker_tri_ui 模式)、现有 gen/train/decode 管线。

---

## File Structure

- `vision_fusion/symbol_set.py`(新建):`SYMBOLS="0123456789X"`;`draw_symbol(char, size, stroke, round_cap) -> 64x64 ndarray`(11 符号画法,统一线宽+圆角)。唯一真相源。
- `vision_fusion/symbol_marker.py`(新建):`render_marker_symbol(marker_id, **layout)` —— 复刻 `digit_marker_tri.generate_marker_tri` 的外框+内角黑三角+2×2 中心布局,但每格调 `draw_symbol`。复用 `checksum_char`。+ CLI 批量导出。
- `vision_fusion/symbol_marker_ui.py`(新建):Tkinter GUI,滑块控线宽/圆角/符号大小/列距/行距/padding/三角大小/边框,实时预览+区间导出,参数存取 `symbol_marker_settings.json`。
- `vision_fusion/nn_synth_symbol.py`(新建):渲染符号 marker → `slice_cells` 切格 → 退化+`--scene` → `datasets/symbol_crops/<char>/`。复用 `nn_augment`/`overlay_marker`/`slice_cells`。
- `vision_fusion/digit_detect_tri.py`(修改):`main()` 加 `--symbol`(用 `models/symbol_cnn.pt`)。`DigitClassifierTri` 已支持 `model_path`,无需改类。
- `vision_fusion/tests/test_symbol_marker.py`(新建):符号集/渲染/标签/端到端测试。
- 复用不改:`nn_train_tri_digit`(--out 指向 symbol_cnn.pt)、`digit_detect_tri.slice_cells/decode_id/DigitClassifierTri/decode_markers_cached`、`MarkerTracker`、`tri_marker_obb.pt`(检测对符号 marker 仍有效,因外框+三角不变;Task 6 验证,不行再补训)。

---
## Task 1: symbol_set.py — 参数化符号画法(统一线宽+圆角)

**Files:** Create `vision_fusion/symbol_set.py`; Test `vision_fusion/tests/test_symbol_marker.py`

- [ ] **Step 1: 失败测试**

```python
# vision_fusion/tests/test_symbol_marker.py
import numpy as np
from vision_fusion.symbol_set import SYMBOLS, draw_symbol

def test_symbols_11_classes():
    assert SYMBOLS == "0123456789X"

def test_draw_symbol_shape_and_ink():
    for ch in SYMBOLS:
        g = draw_symbol(ch, size=64, stroke=10)
        assert g.shape == (64, 64) and g.dtype == np.uint8
        assert int(g.min()) < 60      # 有墨
        assert int(g.max()) > 200     # 有白底

def test_draw_symbol_stroke_scales_ink():
    thin = (draw_symbol("4", 64, 6) < 128).sum()    # ⊢ 竖+横
    thick = (draw_symbol("4", 64, 16) < 128).sum()
    assert thick > thin               # 线宽变大→墨更多
```

- [ ] **Step 2: 跑测试确认失败** — `python -m pytest vision_fusion/tests/test_symbol_marker.py -q`,Expected: ImportError。

- [ ] **Step 3: 实现 symbol_set.py**

```python
"""符号集唯一真相源:v8 的 11 个高区分度几何符号,参数化绘制(统一线宽+圆角线头)。
逻辑标签沿用 0-9X,复用 checksum_char/decode_id。"""
from __future__ import annotations
import cv2
import numpy as np

SYMBOLS = "0123456789X"
# char -> 画法 key
_KIND = {"0": "dring", "1": "L", "2": "vbar", "3": "bslash", "4": "Tl",
         "5": "corner", "6": "Y", "7": "J", "8": "Tr", "9": "fslash", "X": "box"}


def draw_symbol(char: str, size: int = 64, stroke: int = 10, round_cap: bool = True) -> np.ndarray:
    """画单个符号(白底 255 墨 0)。size 画布边长;stroke 统一线宽;round_cap 圆角线头。
    坐标按 size 比例缩放(参考 64px)。"""
    g = np.full((size, size), 255, np.uint8)
    s = size / 64.0
    t = max(2, int(round(stroke)))
    cap = cv2.LINE_AA
    def P(x, y): return (int(x * s), int(y * s))
    def line(a, b):
        cv2.line(g, P(*a), P(*b), 0, t, lineType=cap)
        if round_cap:                                  # 圆头:端点画实心圆
            for pt in (a, b):
                cv2.circle(g, P(*pt), t // 2, 0, -1, lineType=cap)
    def bar(x0, y0, x1, y1):                            # 粗块(矩形,可圆角端)
        cv2.rectangle(g, P(x0, y0), P(x1, y1), 0, -1, lineType=cap)
    k = _KIND[char]
    if k == "dring":   cv2.circle(g, P(32, 32), int(22 * s), 0, -1, lineType=cap)
    elif k == "box":   cv2.rectangle(g, P(12, 12), P(52, 52), 0, t, lineType=cap)
    elif k == "vbar":  bar(32 - stroke/2, 6, 32 + stroke/2, 58)
    elif k == "bslash":line((12, 12), (52, 52))
    elif k == "fslash":line((52, 12), (12, 52))
    elif k == "L":     line((23, 8), (23, 50)); line((23, 50), (50, 50))
    elif k == "J":     line((41, 8), (41, 50)); line((41, 50), (14, 50))
    elif k == "Tl":    line((17, 8), (17, 56)); line((17, 32), (54, 32))
    elif k == "Tr":    line((47, 8), (47, 56)); line((47, 32), (10, 32))
    elif k == "corner":line((12, 15), (48, 15)); line((44, 15), (44, 44))   # ⌐
    elif k == "Y":     line((32, 56), (32, 33)); line((32, 33), (12, 10)); line((32, 33), (52, 10))
    return g
```

- [ ] **Step 4: 跑测试确认通过** — Expected: 3 passed。

- [ ] **Step 5: 区分度回归测试 + 渲染存档**

```python
# 追加到 test_symbol_marker.py
import itertools, cv2 as _cv2
def test_symbol_separability_beats_digits():
    from vision_fusion.eval_glyph_separability import degrade
    rng = np.random.default_rng(0)
    def feat(g):
        a = np.zeros((64, 64), np.float32)
        for _ in range(30):
            d = degrade(g, rng).astype(np.float32); d = (d-d.mean())/(d.std()+1e-6); a += d
        return (a/30).ravel()
    F = {c: feat(draw_symbol(c, 64, 10)) for c in SYMBOLS}
    mind = min(float(np.linalg.norm(F[a]-F[b])) for a, b in itertools.combinations(SYMBOLS, 2))
    assert mind > 30.0, f"区分度 {mind:.1f} 不及预期(数字基线 23.5)"
```
Run pytest;再存一张符号表 `python -c "import cv2,numpy as np;from vision_fusion.symbol_set import *;..."` 存 `docs/test-screenshots/symbolset_final.png` 供人工核对。

- [ ] **Step 6: 提交** — `git add vision_fusion/symbol_set.py vision_fusion/tests/test_symbol_marker.py && git commit -m "feat(symbol): symbol_set 参数化11符号画法(统一线宽+圆角)"`

---

## Task 2: symbol_marker.py — marker 渲染(底边黑条定向)

**Files:** Create `vision_fusion/symbol_marker.py`; Test 追加 `test_symbol_marker.py`

- [ ] **Step 1: 失败测试**

```python
def test_render_marker_symbol():
    from vision_fusion.symbol_marker import render_marker_symbol, marker_chars
    assert marker_chars(83) == "0833"        # 复用 checksum_char
    g = render_marker_symbol(83, pixels=300)
    assert g.shape == (300, 300) and g.dtype == np.uint8
    # 底边黑条:最底部一行应几乎全黑(定向标记),顶部一行不全黑
    bottom = g[int(300*0.93):, 20:280].mean()
    top    = g[:int(300*0.04), 20:280].mean()
    assert bottom < 80, f"底边应是黑条 mean={bottom:.0f}"
```

- [ ] **Step 2: 跑测试确认失败** — Expected: ImportError。

- [ ] **Step 3: 实现 symbol_marker.py**

```python
"""符号 marker 渲染:外框 + 底边加宽黑条(定向) + 2×2 符号格。复用 checksum_char。
布局复刻 generate_marker_tri,格内改用 symbol_set.draw_symbol。"""
from __future__ import annotations
import cv2
import numpy as np
from .digit_marker_tri import checksum_char
from .symbol_set import draw_symbol


def marker_chars(marker_id: int) -> str:
    return f"{marker_id:03d}" + checksum_char(marker_id)


def render_marker_symbol(marker_id, *, pixels=300, border_ratio=0.06, bottom_extra_ratio=0.06,
                         pad_ratio=0.08, col_gap_ratio=0.38, row_gap_ratio=0.42,
                         sym_size_ratio=0.22, stroke_ratio=0.16, round_cap=True) -> np.ndarray:
    p = pixels
    img = np.full((p, p), 255, np.uint8)
    b = int(p * border_ratio)
    cv2.rectangle(img, (0, 0), (p-1, p-1), 0, -1)               # 全黑
    cv2.rectangle(img, (b, b), (p-1-b, p-1-b), 255, -1)         # 挖白 → 黑边框
    # 底边加宽黑条(定向标记):底部再压一条更宽的黑带
    be = int(p * bottom_extra_ratio)
    cv2.rectangle(img, (b, p-1-b-be), (p-1-b, p-1-b), 0, -1)
    # 2×2 符号中心
    pad = int(p * pad_ratio); lo, hi = b+pad, p-1-b-be-pad
    cx = cy = (lo+hi)/2.0; col = p*col_gap_ratio; row = p*row_gap_ratio
    centers = [(cx-col/2, cy-row/2), (cx+col/2, cy-row/2),
               (cx-col/2, cy+row/2), (cx+col/2, cy+row/2)]
    chars = marker_chars(marker_id)
    cell = max(16, int(p * sym_size_ratio))
    stroke = max(2, int(stroke_ratio * cell))
    for ch, (gx, gy) in zip(chars, centers):
        sym = draw_symbol(ch, size=cell, stroke=stroke, round_cap=round_cap)
        x0, y0 = int(gx-cell/2), int(gy-cell/2)
        roi = img[y0:y0+cell, x0:x0+cell]
        if roi.shape == sym.shape:
            img[y0:y0+cell, x0:x0+cell] = np.minimum(roi, sym)   # 墨色叠加
    return img
```

- [ ] **Step 4: 跑测试确认通过** — Expected: pass。

- [ ] **Step 5: 加 CLI 批量导出**(复刻 digit_marker_tri 的 --range/--id/--output),`main()` 输出 `symbol_{id:03d}_{check}.png`。

- [ ] **Step 6: 提交** — `git commit -m "feat(symbol): symbol_marker 渲染(底边黑条定向+2x2符号)+CLI"`

---
## Task 3: 底边黑条定向 (edge_ranking / orient_by_edge)

**Files:** Modify `vision_fusion/digit_detect_tri.py`; Test 追加

- [ ] **Step 1: 失败测试**

```python
def test_edge_ranking_finds_bottom():
    import cv2
    from vision_fusion.symbol_marker import render_marker_symbol
    from vision_fusion.digit_detect_tri import edge_ranking
    g = render_marker_symbol(283, pixels=200)          # 底边黑条
    # 正放:最暗边应是 bottom(索引 2: 0上1右2下3左)
    order, _ = edge_ranking(g)
    assert order[0] == 2, f"正放最暗边应为底 2,实际 {order[0]}"
    # 旋 180:黑条到顶,最暗边应是 top(0)
    assert edge_ranking(cv2.rotate(g, cv2.ROTATE_180))[0][0] == 0
```

- [ ] **Step 2: 跑测试确认失败** — ImportError edge_ranking。

- [ ] **Step 3: 实现 edge_ranking + orient_by_edge**(加到 digit_detect_tri.py,复用 `_ensure_gray` 去平面思路)

```python
_EDGE_ROT_TO_BOTTOM = {2: None, 3: cv2.ROTATE_90_CLOCKWISE,
                       0: cv2.ROTATE_180, 1: cv2.ROTATE_90_COUNTERCLOCKWISE}

def _edge_band_darkness(gray):
    """去线性平面后,比 4 条边带(上/右/下/左)的暗度。返回 4 值,越大越暗。"""
    g = gray.astype(np.float32); h, w = g.shape
    xx, yy = _xy_grid(h, w)
    x = xx.ravel(); y = yy.ravel(); z = g.ravel(); n = z.size
    M = np.array([[x@x, x@y, x.sum()], [x@y, y@y, y.sum()], [x.sum(), y.sum(), n]], np.float64)
    rhs = np.array([x@z, y@z, z.sum()], np.float64)
    try:
        a, b, c = np.linalg.solve(M, rhs); r = g - (a*xx + b*yy + c)
    except np.linalg.LinAlgError:
        r = g - g.mean()
    s = h; t = int(s*0.12); m0, m1 = int(s*0.2), int(s*0.8)
    bands = [r[0:t, m0:m1], r[m0:m1, s-t:s], r[s-t:s, m0:m1], r[m0:m1, 0:t]]  # 上右下左
    return [float(-bnd.mean()) for bnd in bands]

def edge_ranking(square):
    gray = _ensure_gray(square)
    dk = _edge_band_darkness(gray)
    order = sorted(range(4), key=lambda i: dk[i], reverse=True)
    return order, dk

def orient_by_edge(square):
    """把最暗边旋到底部。返回旋正后的方图。"""
    order, _ = edge_ranking(square)
    rot = _EDGE_ROT_TO_BOTTOM[order[0]]
    return square if rot is None else cv2.rotate(square, rot)
```

- [ ] **Step 4: 跑测试确认通过**;再补一个抗梯度测试(叠线性梯度后仍判对底边),Expected: pass。

- [ ] **Step 5: slice_cells 加 `mask_tri=True` 开关**(符号路径传 False,跳过抹三角):

```python
def slice_cells(square, half=CELL_HALF, out=CELL_OUT, mask_tri=True):
    gray = _ensure_gray(square); gray = normalize_square(gray)
    if mask_tri:
        gray = _mask_triangle(gray)
    ...（其余不变）
```

- [ ] **Step 6: 提交** — `git commit -m "feat(symbol): 底边黑条定向 edge_ranking/orient_by_edge + slice_cells mask_tri 开关"`

---

## Task 4: 符号训练数据生成器 nn_synth_symbol.py

**Files:** Create `vision_fusion/nn_synth_symbol.py`

- [ ] **Step 1: 实现**(复刻 nn_synth_tri_digit:渲染 → orient_by_edge 旋正 → slice_cells(mask_tri=False) → 退化 + --scene → datasets/symbol_crops/<char>/)。marker_chars 来自 symbol_marker。clean+aug+scene 三源同 tri。

```python
# 关键差异:marker = render_marker_symbol(mid, **params); 切格用 slice_cells(sq, mask_tri=False)
# scene 路径:overlay_marker 小角度 → warp_square → orient_by_edge → slice_cells(mask_tri=False)
```

- [ ] **Step 2: 小样冒烟 + 切格存档**(--id-max 8 --aug 1 --archive),人工核对符号居中、未被边条蹭到。

- [ ] **Step 3: 全量生成** — `python -m vision_fusion.nn_synth_symbol --id-max 999 --aug 12 --scene 18 --output datasets/symbol_crops`,Expected: 各类计数正常。

- [ ] **Step 4: 提交** — `git commit -m "feat(symbol): 符号训练数据生成器(边定向+不抹三角)"`

---

## Task 5: 训练 symbol_cnn.pt

- [ ] **Step 1: 训练**(复用 nn_train_tri_digit):
  `python -m vision_fusion.nn_train_tri_digit --data datasets/symbol_crops --epochs 15 --out models/symbol_cnn.pt`
  Expected: val_acc >0.99。
- [ ] **Step 2: 提交**(模型 gitignore,提交不含 .pt;记录 val_acc 到 commit msg)。

---

## Task 6: 识别接入 (--symbol) + 端到端验证

**Files:** Modify `vision_fusion/digit_detect_tri.py`; Test 追加

- [ ] **Step 1: 失败测试(端到端读回)**

```python
import os, pytest, cv2
@pytest.mark.skipif(not os.path.exists("models/symbol_cnn.pt"), reason="symbol_cnn 未训练")
def test_symbol_recognize_roundtrip():
    from vision_fusion.symbol_marker import render_marker_symbol
    from vision_fusion.digit_detect_tri import DigitClassifierTri, edge_ranking
    rec = DigitClassifierTri(model_path="models/symbol_cnn.pt", orient=edge_ranking, mask_tri=False)
    for mid in (83, 7, 20, 99, 283):
        got, _ = rec.recognize(render_marker_symbol(mid, pixels=200), min_conf=0.0)
        assert got == mid, f"{mid} → {got}"
```

- [ ] **Step 2: 给 DigitClassifierTri 加 `orient`(默认 corner_ranking)+ `mask_tri`(默认 True)参数**,read_batch/read_debug 用注入的 orient 和把 mask_tri 传给 slice_cells。数字路径默认不变。

- [ ] **Step 3: 跑测试确认通过**(端到端读回正确,含 007X 的 X)。

- [ ] **Step 4: main() 加 `--symbol`**:置 `model_path=models/symbol_cnn.pt, orient=edge_ranking, mask_tri=False`;箭头方向改用 edge 的"指向上方"。

- [ ] **Step 5: A/B 闸门**:仿 eval_enhance,在符号 marker 的合成淡+梯度上测 误读=0、正确率;FPS profile 不回退。存档。

- [ ] **Step 6: 提交** — `git commit -m "feat(symbol): --symbol 接入(边定向+symbol_cnn),端到端+A/B闸门"`

---

## Task 7: 调参 GUI + 文档

**Files:** Create `vision_fusion/symbol_marker_ui.py`; Modify README + memory

- [ ] **Step 1: symbol_marker_ui.py**(复刻 digit_marker_tri_ui):滑块控线宽/圆角/符号大小/列距/行距/padding/边框/底边额外宽;实时预览 + 区间导出;参数存取 `symbol_marker_settings.json`(启动加载、WM_DELETE+按钮保存)。render/train/decode 读同一 json。
- [ ] **Step 2: 冒烟**(import + 渲染一张确认不崩;GUI 本身难自动测,导出函数可测)。
- [ ] **Step 3: README** 加符号 marker 段(生成/训练/识别/`--symbol`/GUI)。
- [ ] **Step 4: memory** 写 `symbol-marker-system.md`:符号集 v8、底边黑条定向、复用 tri 管线、与数字版并存、A/B 数字。
- [ ] **Step 5: 提交 + push**。

---

## 自检

- **Spec 覆盖**:符号集(T1)、底边黑条定向(T2 渲染+T3 检测)、参数化绘制线宽/圆角/大小间距(T1/T2/T7)、参数自动存取(T7)、训练数据(T4)、训练(T5)、识别接入(T6)、GUI(T7)、A/B 闸门守误读0(T6)、文档记忆(T7)。✓
- **类型一致**:`draw_symbol(char,size,stroke,round_cap)`、`render_marker_symbol(marker_id,**layout)`、`marker_chars`、`edge_ranking→(order,dk)`/`orient_by_edge`、`slice_cells(...,mask_tri)`、`DigitClassifierTri(model_path,orient,mask_tri)` 全程一致。✓
- **复用**:checksum_char/decode_id/DigitCNN/nn_train_tri_digit/MarkerTracker/decode_markers_cached 不改;数字路径默认参数不变,符号路径靠开关。✓
- **占位符**:每 code step 有完整代码;measurement step 有命令+期望。✓
- **风险**:① tri_marker_obb.pt 对符号 marker(外框+底条,无三角)检出可能略降——T6 验证,不行则补训 YOLO(加一任务);② 底边黑条 warp 后若被算进格——render 已把符号区限制在 be 之上,slice_cells 不抹三角但符号格不含底条。
