# 画面预处理优化方案 (digit_detect_tri 识别增强) 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不动检测、不增加错误 id(误读保持 0)、保持 ~30 FPS 的前提下,用针对"识别用 warp 小图"的画面增强(逐 marker 对比归一化 + 静止多帧时域平均),提升淡/糊 marker 的 id 解出率。

**Architecture:** 增强只作用于识别(YOLO 框定已 ~100%,不碰)。**铁律一**:任何改变格子外观的增强(归一化)必须烘进**共享的 `slice_cells`**——训练数据生成(`nn_synth_tri_digit`)和推理(`DigitClassifierTri`)都走它,自动同分布,再重训一次。**铁律二**:每项增强先用"淡+梯度"合成 ground-truth 量化,**只有"正确↑ 且 误读保持 0"才采纳**(守 [[id-precision-over-recall]])。时域平均只让图更干净、趋近训练分布,免重训。

**Tech Stack:** OpenCV(百分位拉伸 / 时域平均)、现有 DigitClassifierTri + nn_synth_tri_digit + nn_train_tri_digit 管线、PyTorch。

---

## File Structure

- `vision_fusion/digit_detect_tri.py`(修改):新增 `normalize_square()`;`slice_cells` 开头调用它(训练/推理同源);`decode_markers_cached` 增静止 marker 多帧平均(经 tracker 缓冲 warp 小图)。
- `vision_fusion/eval_enhance.py`(新建):离线 A/B 评测器——给定预处理开关,在合成"淡+梯度"ground-truth 上输出 (正确/误读/漏读),作为每项增强的**采纳闸门**。
- `vision_fusion/nn_synth_tri_digit.py`(不改代码):共用 `slice_cells` 自动获得归一化,只需重新生成数据。
- `vision_fusion/nn_train_tri_digit.py`(不改代码):重训。
- `vision_fusion/tests/test_tri_digit_cnn.py`(追加):`normalize_square` 不变量 + 端到端。

---
## Task 1: 离线 A/B 评测器(采纳闸门)

**Files:**
- Create: `vision_fusion/eval_enhance.py`

在合成"淡+梯度+透视"分布(有 ground-truth)上,对**当前模型**测每项预处理的 (正确/误读/漏读)。后续每个增强都要先过这个闸门:**正确↑ 且 误读=0 才采纳**。

- [ ] **Step 1: 实现评测器**

```python
# vision_fusion/eval_enhance.py
"""离线评测画面增强对识别的影响(合成淡+梯度 ground-truth)。

用法: python -m vision_fusion.eval_enhance              # 基线
评测器给定一个 square 预处理函数,输出 正确/误读/漏读。误读必须保持 0。
"""
from __future__ import annotations
import glob, os, random
import cv2, numpy as np
from .digit_detect_tri import DigitClassifierTri
from .digit_detect import warp_square
from .nn_synth_ui import overlay_marker, load_settings, CANVAS_W, CANVAS_H


def faint_samples(n_per=4, seed=9):
    """用 overlay_marker(透视+混合+缩放)+ 线性梯度造"淡"样本,返回 (true_id, square)。"""
    s = load_settings()
    params = dict(brightness=s.get("brightness", 100.0), contrast=s.get("contrast", 0.5),
                  blur_sigma=s.get("blur_sigma", 2.5), scatter_sigma=s.get("scatter_sigma", 5.0),
                  opacity=s.get("opacity", 0.8), levels_black=0.0, levels_white=255.0, levels_gamma=1.0)
    mk = [(int(os.path.basename(f).split("_")[1]), cv2.imread(f, cv2.IMREAD_GRAYSCALE))
          for f in sorted(glob.glob("digit_markers_tri/digit_*.png"))]
    rng = random.Random(seed)
    out = []
    for mid, g in mk:
        for _ in range(n_per):
            scene = np.full((CANVAS_H, CANVAS_W), rng.randint(125, 150), np.uint8)
            yy, xx = np.mgrid[0:CANVAS_H, 0:CANVAS_W]
            a = rng.uniform(0, 6.28); r = (np.cos(a) * xx + np.sin(a) * yy)
            r = (r - r.min()) / (r.max() - r.min())
            scene = np.clip(scene + (r - 0.5) * 70, 0, 255).astype(np.uint8)
            scene, cor, _ = overlay_marker(scene, g, CANVAS_W // 2, CANVAS_H // 2,
                                           rng.uniform(-180, 180), rng.uniform(0.2, 0.45), **params)
            sq = warp_square(scene, cor, 200)
            if sq.size:
                out.append((mid, sq))
    return out


def evaluate(pre=None, samples=None):
    """pre: 可选的 square→square 预处理(模拟"运行时也加这个增强")。返回 (正确,误读,漏读,总)。"""
    rec = DigitClassifierTri(min_cell_conf=0.9, max_orient=1)
    samples = samples or faint_samples()
    c = w = m = 0
    for mid, sq in samples:
        s2 = pre(sq) if pre else sq
        got, _ = rec.recognize(s2, min_conf=0.0)
        if got == mid: c += 1
        elif got >= 0: w += 1
        else: m += 1
    return c, w, m, c + w + m


def main() -> int:
    samples = faint_samples()
    c, w, m, n = evaluate(None, samples)
    print(f"基线(无额外增强): 正确 {c}/{n} ({100*c/n:.0f}%)  误读 {w}  漏 {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 跑基线**

Run: `python -m vision_fusion.eval_enhance`
Expected: 打印基线正确率 + **误读 0**(若误读非 0,先停下查识别管线,别加增强)。记下这条作对照。

- [ ] **Step 3: 提交**

```bash
git add vision_fusion/eval_enhance.py
git commit -m "test(nn): 画面增强离线 A/B 评测器(淡+梯度 ground-truth, 误读必须 0)"
```

---

## Task 2: 逐 marker 对比归一化函数 (normalize_square)

**Files:**
- Modify: `vision_fusion/digit_detect_tri.py`(新增函数,**暂不接入** slice_cells)
- Test: `vision_fusion/tests/test_tri_digit_cnn.py`

百分位拉伸:把 warp 小图挤在窄灰段的像素拉到满 0–255,让淡 marker 的数字"跳"出来。对整张 200 crop 做(有黑边+白底+数字作动态范围参照),不做逐格(空格会放大噪声)。

- [ ] **Step 1: 写失败测试**

```python
# 追加到 test_tri_digit_cnn.py
def test_normalize_square_stretches_contrast():
    from vision_fusion.digit_detect_tri import normalize_square
    # 一张挤在 [120,160] 窄灰段的图,拉伸后应接近铺满 [0,255]
    g = (np.linspace(120, 160, 200, dtype=np.float32)[None, :].repeat(200, 0)).astype(np.uint8)
    out = normalize_square(g)
    assert out.shape == g.shape and out.dtype == np.uint8
    assert int(out.min()) < 30 and int(out.max()) > 225      # 动态范围被拉开
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest vision_fusion/tests/test_tri_digit_cnn.py::test_normalize_square_stretches_contrast -q`
Expected: FAIL,`ImportError: cannot import name 'normalize_square'`。

- [ ] **Step 3: 实现 normalize_square**

加到 `vision_fusion/digit_detect_tri.py`(`_ensure_gray` 之后):

```python
def normalize_square(square: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0) -> np.ndarray:
    """逐 marker 百分位对比拉伸:把窄灰段拉到满 0-255,让淡 marker 数字更清晰。
    用百分位(非 min/max)抗个别极值。对整张 crop 做,不逐格(空格会放大噪声)。"""
    g = _ensure_gray(square)
    gf = g.astype(np.float32)
    lo, hi = np.percentile(gf, [lo_pct, hi_pct])
    if hi - lo < 1.0:
        return g
    return np.clip((gf - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest vision_fusion/tests/test_tri_digit_cnn.py::test_normalize_square_stretches_contrast -q`
Expected: PASS。

- [ ] **Step 5: A/B 闸门(关键决策点)**

Run:
```bash
python -c "from vision_fusion.eval_enhance import evaluate, faint_samples; from vision_fusion.digit_detect_tri import normalize_square; s=faint_samples(); import sys; \
print('基线', evaluate(None,s)); print('归一化', evaluate(normalize_square,s))"
```
Expected: 对比两行 (正确,误读,漏,总)。**决策:仅当"归一化"的正确↑ 且 误读仍为 0,才继续 Task 3 接入;否则放弃归一化(记录结论),直接跳到 Task 5 时域平均。**

- [ ] **Step 6: 提交**

```bash
git add vision_fusion/digit_detect_tri.py vision_fusion/tests/test_tri_digit_cnn.py
git commit -m "feat(digit_detect_tri): normalize_square 逐marker对比拉伸(暂未接入,待A/B闸门)"
```

---
## Task 3: 接入 slice_cells + 重新生成数据 + 重训(**仅当 Task 2 闸门通过**)

**Files:**
- Modify: `vision_fusion/digit_detect_tri.py:slice_cells`
- 重生成 `datasets/tri_digit_crops/`、重训 `models/tri_digit_cnn.pt`

烘进共享的 `slice_cells` → 训练生成(`nn_synth_tri_digit` 走 slice_cells)与推理自动同分布。

- [ ] **Step 1: slice_cells 开头加归一化**

把 `slice_cells` 里 `gray = _ensure_gray(square)` 之后改为先归一化:

```python
def slice_cells(square: np.ndarray, half: float = CELL_HALF, out: int = CELL_OUT):
    gray = _ensure_gray(square)
    gray = normalize_square(gray)          # 逐 marker 对比拉伸(训练/推理同源)
    gray = _mask_triangle(gray)
    s = gray.shape[0]
    ...（其余不变）
```

- [ ] **Step 2: 重新生成训练数据(自动带归一化)**

```bash
rm -rf datasets/tri_digit_crops
python -m vision_fusion.nn_synth_tri_digit --id-min 0 --id-max 999 --aug 12 --scene 18 --output datasets/tri_digit_crops
```
Expected: per-class counts 正常(各数字 ~1.2 万、X ~2800)。

- [ ] **Step 3: 重训**

```bash
python -m vision_fusion.nn_train_tri_digit --data datasets/tri_digit_crops --epochs 15
```
Expected: val_acc 收敛 >0.99,生成 `models/tri_digit_cnn.pt`。

- [ ] **Step 4: 全量测试(分布变了,确认没破)**

Run: `python -m pytest vision_fusion/tests/ -q`
Expected: 全过(干净 marker 归一化后仍高置信读出 → 真实/旋转/批处理测试照过)。

- [ ] **Step 5: 提交**

```bash
git add vision_fusion/digit_detect_tri.py
git commit -m "feat(digit_detect_tri): 把逐marker对比归一化烘进 slice_cells(训练/推理同源)+重训"
```

---

## Task 4: 归一化后端到端复测(守精度 + FPS)

**Files:**
- 用 `eval_enhance` + profiler,产出对照,**不改代码**

- [ ] **Step 1: 闸门复测(模型已重训)**

Run: `python -m vision_fusion.eval_enhance`
Expected: 正确率较 Task 1 基线 **↑**,**误读仍 0**。若误读>0 → 回退 Task 3(归一化引入了新误读,不可接受)。

- [ ] **Step 2: FPS 不回退确认**

Run: `python -m vision_fusion.profile_tri_pipeline --recognizer cnn --frames 6 --iters 8`
Expected: decode 仍 <15ms(归一化是 O(N) 廉价),整链估算仍 ~30 FPS。

- [ ] **Step 3: 存档对照**

```bash
TS=$(date +%Y%m%d-%H%M%S)
python -m vision_fusion.eval_enhance > "docs/test-screenshots/enhance-eval-$TS.txt"
git add "docs/test-screenshots/enhance-eval-$TS.txt"
git commit -m "docs: 归一化前后 faint 识别率对照存档"
```

---

## Task 5: 静止 marker 多帧时域平均(免重训,gated)

**Files:**
- Modify: `vision_fusion/digit_detect_tri.py:decode_markers_cached`
- Test: `vision_fusion/tests/test_tri_digit_cnn.py`

相机固定 → 同一静止位置的 warp 小图叠几帧求平均,噪声 ÷√K,淡数字浮出。移动的 marker(位置 key 变)自动不平均,避免拖影。这只是让图更干净(趋近训练分布),不重训。

- [ ] **Step 1: 写失败测试(平均降噪)**

```python
# 追加到 test_tri_digit_cnn.py
def test_temporal_average_denoises():
    from vision_fusion.digit_detect_tri import _TemporalAvg
    base = np.full((200, 200), 128, np.uint8)
    ta = _TemporalAvg(k=5)
    key = (3, 4)
    rng = np.random.default_rng(0)
    last = None
    for _ in range(5):
        noisy = np.clip(base.astype(np.float32) + rng.normal(0, 25, base.shape), 0, 255).astype(np.uint8)
        last = ta.push(key, noisy)
    # 5 帧平均后,噪声标准差应明显小于单帧(~÷√5)
    assert float(last.std()) < 12.0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest vision_fusion/tests/test_tri_digit_cnn.py::test_temporal_average_denoises -q`
Expected: FAIL,`ImportError: cannot import name '_TemporalAvg'`。

- [ ] **Step 3: 实现 _TemporalAvg + 接入 decode_markers_cached**

加到 `digit_detect_tri.py`:

```python
from collections import deque

class _TemporalAvg:
    """按位置 key 缓冲最近 k 帧的 warp 小图,返回其平均(降噪)。
    位置变了(marker 移动/换位)对应 key 不同 → 自然不跨位置平均、无拖影。"""
    def __init__(self, k: int = 5):
        self.k = k
        self.buf: dict = {}

    def push(self, key, square: np.ndarray) -> np.ndarray:
        d = self.buf.get(key)
        if d is None:
            d = deque(maxlen=self.k); self.buf[key] = d
        d.append(square.astype(np.float32))
        return (np.mean(d, axis=0)).astype(np.uint8)

    def prune(self, live_keys):
        for k in [k for k in self.buf if k not in live_keys]:
            del self.buf[k]
```

在 `decode_markers_cached` 里,读之前先按位置平均(key 用 40px 网格):

```python
def decode_markers_cached(entries, recognizer, tracker, decay: float = 0.6):
    ta = getattr(tracker, "_tavg", None)
    if ta is None:
        ta = _TemporalAvg(k=5); tracker._tavg = ta
    live = set()
    proc = []   # (pts, 平均后的 square)
    for pts, square in entries:
        c = pts.mean(axis=0); key = (round(c[0] / 40.0), round(c[1] / 40.0))
        live.add(key)
        proc.append((pts, ta.push(key, square)))
    ta.prune(live)
    squares = [sq for _, sq in proc]
    if squares and hasattr(recognizer, "read_batch"):
        reads = recognizer.read_batch(squares)
    else:
        reads = [recognizer.read_debug(sq) for sq in squares]
    n_cnn = len(squares)
    detections = []; pend = []
    for (pts, _sq), (mid, conf, info) in zip(proc, reads):
        center = tuple(float(v) for v in pts.mean(axis=0))
        detections.append((mid, conf, center))
        pend.append((pts, center, info["corner"], info))
    tracker.update(detections, decay=decay)
    items = []
    for pts, center, corner, info in pend:
        items.append({"pts": pts, "center": center, "corner": corner,
                      "id": tracker.query_at(center), "info": info})
    return items, n_cnn
```

- [ ] **Step 4: 跑测试确认通过 + 全量**

Run: `python -m pytest vision_fusion/tests/ -q`
Expected: 全过(含降噪测试;换-marker 跟随测试仍过——位置不变时平均同一张零图,行为不变)。

- [ ] **Step 5: 闸门(多帧噪声样本)**

扩展 `eval_enhance` 加一个"每个样本给 K 张带噪版本、先平均再读"的对照,确认 faint 正确率↑ 且误读=0。若无增益就回退本任务(只留归一化)。

- [ ] **Step 6: 提交**

```bash
git add vision_fusion/digit_detect_tri.py vision_fusion/tests/test_tri_digit_cnn.py
git commit -m "feat(digit_detect_tri): 静止marker多帧时域平均降噪(位置key隔离,移动不拖影)"
```

---

## Task 6: 文档与记忆

**Files:**
- Modify: `README.md`、memory `digit-detect-tri-decoder.md`

- [ ] **Step 1: README**

在解码器段补:画面增强(逐 marker 对比归一化烘进 slice_cells、静止多帧平均),用于提升淡 marker 识别;强调"增强必须训练/推理同源,改了要重训"。

- [ ] **Step 2: memory**

更新 `digit-detect-tri-decoder.md`:记录最终采纳了哪些增强(及各自 A/B 数字)、被否的(及原因),关联 [[id-precision-over-recall]]。

- [ ] **Step 3: 提交 + push**

```bash
git add README.md
git commit -m "docs(digit_detect_tri): 画面增强方案与 A/B 结论"
git push
```

---

## 自检

- **覆盖**:对比归一化(T2-T4)、时域平均(T5)、采纳闸门(T1,每项增强都过)、训练/推理同源(烘进 slice_cells + 重训 T3)、守精度(每步查误读=0)、FPS 不回退(T4)、文档记忆(T6)。✓
- **铁律**:任何改格子外观的增强都经 slice_cells(同源)+重训;时域平均只降噪免重训。✓
- **占位符扫描**:每个 code step 都有完整代码;measurement step 有确切命令 + 期望 + 决策门槛。✓
- **类型一致**:`normalize_square(square,lo_pct,hi_pct)`、`_TemporalAvg.push(key,square)/prune(live_keys)`、`decode_markers_cached(entries,recognizer,tracker,decay)` 全程签名一致。✓
- **风险**:① 归一化对空白/纯背景 crop 会放大噪声——但作用于含黑边+数字的整张 crop,有真实动态范围,且经闸门验证才采纳;② 时域平均若 warp 抖动会轻微糊——位置 key 隔离 + K=5 适中;③ 摄像头黑屏期间只能合成验证,真实确认留到设备恢复。
