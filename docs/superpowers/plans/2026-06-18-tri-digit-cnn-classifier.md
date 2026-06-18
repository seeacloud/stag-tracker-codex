# Tri Digit CNN Classifier 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用专训的轻量 CNN(0-9 + X，11 类)替换 `digit_detect_tri` 里的通用 RapidOCR，把识别从 ~66ms/次降到 <1ms 且更准。

**Architecture:** 复用已有的 `DigitCNN`(3 层卷积，64×64 灰度输入)。新增一个**共享切格函数** `slice_cells`，按生成器的字形中心(`0.5 ± col_gap/2`, `0.5 ± row_gap/2`)从拉正后的 marker 切出 4 个数字格——训练数据生成与实时识别用**同一个函数**，保证几何完全一致。用 marker 生成器(加载 GUI 保存的 Gothic 字体参数)渲染全量 ID 切格作训练数据，重训 11 类模型，再包一个与 `DigitRecognizerTri` 接口一致的 `DigitClassifierTri` 直接换上。

**Tech Stack:** PyTorch(`DigitCNN`)、OpenCV、PIL(生成器)、现有 `nn_augment.random_degrade`、`generate_marker_tri`、`checksum_char`、`decode_id`。

---

## File Structure

- `vision_fusion/digit_detect_tri.py`（修改）：新增 `CHARS`、`cell_centers()`、`slice_cells()`、`DigitClassifierTri` 类；`main()` 加 `--recognizer {cnn,ocr}`。
- `vision_fusion/nn_synth_tri_digit.py`（新建）：加载 `digit_marker_tri_settings.json`，渲染 ID → `slice_cells` → 退化增广 → 存 `datasets/tri_digit_crops/<char>/`，并存档切格校验图。
- `vision_fusion/nn_train_tri_digit.py`（新建）：复用 `DigitCNN(num_classes=11)`，从 `<char>/`(含 `X`)目录训练，存 `models/tri_digit_cnn.pt` + `models/tri_digit_labels.json`。
- `vision_fusion/tests/test_tri_digit_cnn.py`（新建）：`cell_centers`/`slice_cells`/`marker_chars`/`DigitClassifierTri` 测试。

---
## Task 1: 共享切格几何 + slicer

**Files:**
- Modify: `vision_fusion/digit_detect_tri.py`（顶部常量区 + 新函数）
- Test: `vision_fusion/tests/test_tri_digit_cnn.py`

切格几何锁定当前部署 marker 的参数(`digit_marker_tri_settings.json`：col_gap=0.3765, row_gap=0.4176)。中心为 `0.5 ± gap/2`，顺序 `[TL=d1, TR=d2, BL=d3, BR=check]`，与生成器 `generate_marker_tri` 的 `glyphs` 顺序一致。

- [ ] **Step 1: 写失败测试**

```python
# vision_fusion/tests/test_tri_digit_cnn.py
import cv2, numpy as np, pytest
from vision_fusion.digit_marker_tri import generate_marker_tri
from vision_fusion.digit_detect_tri import CHARS, cell_centers, slice_cells

def test_chars_is_11_classes():
    assert CHARS == "0123456789X"

def test_cell_centers_order_and_values():
    c = cell_centers(col_gap=0.3765, row_gap=0.4176)
    assert len(c) == 4
    # TL,TR,BL,BR
    assert c[0] == pytest.approx((0.5 - 0.3765/2, 0.5 - 0.4176/2))
    assert c[1] == pytest.approx((0.5 + 0.3765/2, 0.5 - 0.4176/2))
    assert c[2] == pytest.approx((0.5 - 0.3765/2, 0.5 + 0.4176/2))
    assert c[3] == pytest.approx((0.5 + 0.3765/2, 0.5 + 0.4176/2))

def test_slice_cells_shapes():
    g = generate_marker_tri(83, pixels=200)
    cells = slice_cells(g)
    assert len(cells) == 4
    for cell in cells:
        assert cell.shape == (64, 64)

def test_slice_cells_capture_ink():
    # 每个格子都应含黑色字形像素(白底 255, 字 0)
    g = generate_marker_tri(283, pixels=200)   # 2,8,3 + 校验, 四格都有字
    for cell in slice_cells(g):
        assert int(cell.min()) < 100
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest vision_fusion/tests/test_tri_digit_cnn.py -q`
Expected: FAIL，`ImportError: cannot import name 'CHARS'`（或 cell_centers/slice_cells 未定义）。

- [ ] **Step 3: 实现 CHARS / cell_centers / slice_cells**

加到 `vision_fusion/digit_detect_tri.py`（紧跟现有 `_TRI_L` 常量之后）：

```python
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest vision_fusion/tests/test_tri_digit_cnn.py -q`
Expected: 4 passed。若 `test_slice_cells_capture_ink` 失败，说明 `CELL_HALF` 偏小漏掉字形——调大到 0.23 再跑。

- [ ] **Step 5: 提交**

```bash
git add vision_fusion/digit_detect_tri.py vision_fusion/tests/test_tri_digit_cnn.py
git commit -m "feat(digit_detect_tri): shared cell slicer (CHARS/cell_centers/slice_cells)"
```

---

## Task 2: tri 数字训练数据生成器

**Files:**
- Create: `vision_fusion/nn_synth_tri_digit.py`
- Test: `vision_fusion/tests/test_tri_digit_cnn.py`（追加 `marker_chars` 测试）

用 GUI 保存的字体/参数渲染 marker，切 4 格，按 `f"{id:03d}" + checksum_char(id)` 自动打标签，退化增广后存标准分类目录。**关键**：切格用 Task 1 的 `slice_cells`，与推理同源。

- [ ] **Step 1: 写失败测试(标签映射)**

```python
# 追加到 test_tri_digit_cnn.py
from vision_fusion.nn_synth_tri_digit import marker_chars

def test_marker_chars_maps_cells():
    # ID 83 → "083" + 校验3 → 4 格字符 [0,8,3,3]
    assert marker_chars(83) == "0833"
    # ID 7 → "007" + 校验X
    assert marker_chars(7) == "007X"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest vision_fusion/tests/test_tri_digit_cnn.py::test_marker_chars_maps_cells -q`
Expected: FAIL，`ModuleNotFoundError: nn_synth_tri_digit`。

- [ ] **Step 3: 实现生成器**

```python
# vision_fusion/nn_synth_tri_digit.py
"""tri 数字分类器训练数据：用 GUI 保存的字体/参数渲染 marker，切 4 格自动打标签。

切格复用 digit_detect_tri.slice_cells(与推理同源)。退化沿用 nn_augment.random_degrade。
输出 datasets/tri_digit_crops/<char>/syn_xxxx.png，char ∈ {0-9, X}。

Usage:
    python -m vision_fusion.nn_synth_tri_digit --aug 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .digit_marker_tri import generate_marker_tri, checksum_char
from .digit_detect_tri import slice_cells, CHARS
from .nn_augment import random_degrade

SETTINGS = "digit_marker_tri_settings.json"


def marker_chars(marker_id: int) -> str:
    """ID → 4 格字符串，顺序 [d1,d2,d3,check]，与 cell_centers 对应。"""
    return f"{marker_id:03d}" + checksum_char(marker_id)


def _render_params() -> dict:
    s = json.loads(Path(SETTINGS).read_text(encoding="utf-8"))
    keys = ("pixels", "font_path", "font_size_ratio", "border_ratio",
            "chamfer_ratio", "pad_ratio", "col_gap_ratio", "row_gap_ratio",
            "stroke_ratio")
    return {k: s[k] for k in keys if k in s}


def main() -> int:
    ap = argparse.ArgumentParser(description="Synthesize tri digit-cell crops.")
    ap.add_argument("--output", default="datasets/tri_digit_crops")
    ap.add_argument("--id-min", type=int, default=0)
    ap.add_argument("--id-max", type=int, default=999)
    ap.add_argument("--aug", type=int, default=20, help="每格退化增广张数。")
    ap.add_argument("--archive", default=None, help="存几张切格校验图的目录。")
    args = ap.parse_args()

    params = _render_params()
    out = Path(args.output)
    for ch in CHARS:
        (out / ch).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    counts = {ch: 0 for ch in CHARS}
    archived = 0
    for mid in range(args.id_min, args.id_max + 1):
        chars = marker_chars(mid)
        marker = generate_marker_tri(mid, **params)
        cells = slice_cells(marker)
        for ch, cell in zip(chars, cells):
            # 原图 + 增广
            cv2.imwrite(str(out / ch / f"m{mid:03d}_{counts[ch]:05d}.png"), cell)
            counts[ch] += 1
            for _ in range(args.aug):
                deg = random_degrade(cell, rng)
                cv2.imwrite(str(out / ch / f"m{mid:03d}_{counts[ch]:05d}.png"), deg)
                counts[ch] += 1
        if args.archive and archived < 6:
            ad = Path(args.archive); ad.mkdir(parents=True, exist_ok=True)
            vis = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
            s = marker.shape[0]; h = int(0.21 * s)
            from .digit_detect_tri import cell_centers
            for (ncx, ncy), ch in zip(cell_centers(), chars):
                cx, cy = int(ncx * s), int(ncy * s)
                cv2.rectangle(vis, (cx - h, cy - h), (cx + h, cy + h), (0, 0, 255), 2)
                cv2.putText(vis, ch, (cx - 8, cy - h - 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)
            cv2.imwrite(str(ad / f"slice_{mid:03d}.png"), vis)
            archived += 1
    print("per-class counts:", {k: v for k, v in counts.items()})
    print(f"output: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest vision_fusion/tests/test_tri_digit_cnn.py::test_marker_chars_maps_cells -q`
Expected: PASS。

- [ ] **Step 5: 生成数据 + 存档切格校验图**

先确认 `nn_augment.random_degrade(img, rng)` 签名(下一步前必读)：
Run: `grep -nE "def random_degrade" vision_fusion/nn_augment.py`
若签名不同(如不收 rng)，相应调整 Step 3 的调用。然后：

```bash
TS=$(date +%Y%m%d-%H%M%S)
python -m vision_fusion.nn_synth_tri_digit --aug 20 \
  --archive "docs/test-screenshots/tri-slice-verify-$TS"
```
Expected: 打印 per-class counts(每类几千张，X 类 ~1800+)，`docs/test-screenshots/tri-slice-verify-*/slice_*.png` 里红框正好罩住每个数字。**让用户检查切格图**。

- [ ] **Step 6: 提交**

```bash
git add vision_fusion/nn_synth_tri_digit.py vision_fusion/tests/test_tri_digit_cnn.py
git commit -m "feat(nn): tri digit-cell training-data generator (correct GUI font, 0-9+X)"
```

---
## Task 3: 训练 11 类 tri 数字分类器

**Files:**
- Create: `vision_fusion/nn_train_tri_digit.py`

复用 `nn_train_digit.DigitCNN`。从 `datasets/tri_digit_crops/<char>/` 加载(char ∈ `CHARS`，索引 = `CHARS.index(char)`)。归一化与现有一致：`img/255.0`，单通道。

- [ ] **Step 1: 实现训练脚本**

```python
# vision_fusion/nn_train_tri_digit.py
"""训练 tri 数字分类器(0-9 + X，11 类)。复用 DigitCNN。

数据：datasets/tri_digit_crops/<char>/*.png，char ∈ "0123456789X"。
存：models/tri_digit_cnn.pt + models/tri_digit_labels.json。

Usage:
    python -m vision_fusion.nn_train_tri_digit --data datasets/tri_digit_crops --epochs 15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from .nn_train_digit import DigitCNN
from .digit_detect_tri import CHARS


class CellDataset(Dataset):
    def __init__(self, root: str):
        self.samples = []
        for ch in CHARS:
            for p in (Path(root) / ch).glob("*.png"):
                self.samples.append((str(p), CHARS.index(ch)))
        if not self.samples:
            raise SystemExit(f"no crops under {root}; run nn_synth_tri_digit first")
        print(f"{len(self.samples)} crops, {len(CHARS)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((64, 64), np.uint8)
        if img.shape != (64, 64):
            img = cv2.resize(img, (64, 64))
        return torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0), label


def main() -> int:
    ap = argparse.ArgumentParser(description="Train tri digit CNN (0-9+X).")
    ap.add_argument("--data", default="datasets/tri_digit_crops")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default="models/tri_digit_cnn.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = CellDataset(args.data)
    n_val = max(1, int(len(ds) * 0.1))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val],
                                    generator=torch.Generator().manual_seed(7))
    tl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    vl = DataLoader(val_ds, batch_size=args.batch, num_workers=0)

    model = DigitCNN(num_classes=len(CHARS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    best = 0.0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = lossf(model(x), y); loss.backward(); opt.step()
        model.eval(); correct = total = 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item(); total += y.numel()
        acc = correct / max(1, total)
        print(f"epoch {ep+1}/{args.epochs}  val_acc={acc:.4f}")
        if acc >= best:
            best = acc
            torch.save(model.state_dict(), args.out)
    Path("models/tri_digit_labels.json").write_text(json.dumps(list(CHARS)), encoding="ascii")
    print(f"best val_acc={best:.4f}  saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 训练**

```bash
python -m vision_fusion.nn_train_tri_digit --data datasets/tri_digit_crops --epochs 15
```
Expected: `val_acc` 收敛到 >0.99(干净渲染 + 退化增广，11 类应很高)，生成 `models/tri_digit_cnn.pt`。若 X 类样本过少导致其 acc 低，回 Task 2 调大 `--id-max` 或单独补 X。

- [ ] **Step 3: 提交**

```bash
git add vision_fusion/nn_train_tri_digit.py
git commit -m "feat(nn): train 11-class (0-9+X) tri digit CNN"
```

---

## Task 4: DigitClassifierTri 识别器(替换 OCR)

**Files:**
- Modify: `vision_fusion/digit_detect_tri.py`（新增类）
- Test: `vision_fusion/tests/test_tri_digit_cnn.py`（追加端到端测试）

与 `DigitRecognizerTri` 接口一致：`read_debug(square) -> (id, conf, info)`、`recognize(square, min_conf)`。沿用"暗度排序 + 逐朝向 + 首个过校验"策略——但识别用 CNN，4 格一次 batch，全 4 朝向也才 16 次微推理。

- [ ] **Step 1: 写失败测试**

```python
# 追加到 test_tri_digit_cnn.py
import os
@pytest.mark.skipif(not os.path.exists("models/tri_digit_cnn.pt"),
                    reason="tri_digit_cnn.pt 未训练")
def test_classifier_reads_rendered_ids():
    from vision_fusion.digit_detect_tri import DigitClassifierTri
    rec = DigitClassifierTri()
    for mid in (83, 7, 20, 99, 283):      # 含校验 X 的 7(007X)
        got, conf = rec.recognize(generate_marker_tri(mid, pixels=200), min_conf=0.0)
        assert got == mid, f"id {mid} read as {got}"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest vision_fusion/tests/test_tri_digit_cnn.py::test_classifier_reads_rendered_ids -q`
Expected: FAIL，`ImportError: cannot import name 'DigitClassifierTri'`。

- [ ] **Step 3: 实现 DigitClassifierTri**

加到 `vision_fusion/digit_detect_tri.py`（`DigitRecognizerTri` 之后）：

```python
class DigitClassifierTri:
    """轻量 CNN(0-9+X)读 2×2 数字 + 加权 mod11 校验。替代 RapidOCR。

    接口同 DigitRecognizerTri：read_debug/recognize。识别用 DigitCNN，
    4 格一次 batch；按角暗度排序逐朝向试，首个过 decode_id 校验的采纳。
    """

    def __init__(self, model_path: str = "models/tri_digit_cnn.pt"):
        import torch
        from .nn_train_digit import DigitCNN
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = DigitCNN(num_classes=len(CHARS)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def _classify(self, cells: list) -> tuple[str, float]:
        """4 格 → (4 字符串, 平均置信度)。一次 batch 推理。"""
        x = self.torch.from_numpy(
            np.stack([c.astype(np.float32) / 255.0 for c in cells])[:, None]
        ).to(self.device)
        with self.torch.no_grad():
            prob = self.torch.softmax(self.model(x), dim=1)
            conf, idx = prob.max(dim=1)
        chars = "".join(CHARS[i] for i in idx.tolist())
        return chars, float(conf.mean().item())

    def read_debug(self, square: np.ndarray) -> tuple[int, float, dict]:
        order, dark = corner_ranking(square)
        info = {"ranking": order, "corner": order[0], "orient_ok": True,
                "orient_conf": (dark[order[0]] - dark[order[1]]) / (dark[order[0]] + 1e-6),
                "top": "", "bot": "", "top_conf": 0.0, "bot_conf": 0.0}
        first = None
        for corner in order:
            rot = _ROT_TO_TL[corner]
            o = square if rot is None else cv2.rotate(square, rot)
            chars, conf = self._classify(slice_cells(o))
            if first is None:
                first = (corner, chars, conf, o)
            mid = decode_id(chars)
            if mid >= 0:
                info.update(corner=corner, top=chars[:2], bot=chars[2:],
                            top_conf=conf, bot_conf=conf, _oriented=o)
                return mid, conf, info
        c, chars, conf, o = first
        info.update(corner=c, top=chars[:2], bot=chars[2:],
                    top_conf=conf, bot_conf=conf, _oriented=o)
        return -1, 0.0, info

    def recognize(self, square: np.ndarray, min_conf: float = 0.5) -> tuple[int, float]:
        mid, conf, _ = self.read_debug(square)
        if mid < 0 or conf < min_conf:
            return -1, 0.0
        return mid, conf
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest vision_fusion/tests/test_tri_digit_cnn.py -q`
Expected: 全 PASS（含端到端 5 个 ID 正确，含 007X 的 X）。

- [ ] **Step 5: 提交**

```bash
git add vision_fusion/digit_detect_tri.py vision_fusion/tests/test_tri_digit_cnn.py
git commit -m "feat(digit_detect_tri): DigitClassifierTri (CNN 0-9+X) replacing RapidOCR"
```

---

## Task 5: 接入实时管线(--recognizer)

**Files:**
- Modify: `vision_fusion/digit_detect_tri.py:main()`

加 `--recognizer {cnn,ocr}`(默认 cnn)；模型缺失时回退 ocr 并告警。

- [ ] **Step 1: 改 main 选择识别器**

把 `main()` 里 `recognizer = DigitRecognizerTri()` 一行替换为：

```python
    if args.recognizer == "cnn":
        from pathlib import Path as _P
        if _P("models/tri_digit_cnn.pt").exists():
            recognizer = DigitClassifierTri()
            print("recognizer: CNN (tri_digit_cnn.pt)")
        else:
            print("WARN: models/tri_digit_cnn.pt 缺失，回退 RapidOCR")
            recognizer = DigitRecognizerTri()
    else:
        recognizer = DigitRecognizerTri()
        print("recognizer: RapidOCR")
```

并在 argparse 区加：

```python
    parser.add_argument("--recognizer", choices=["cnn", "ocr"], default="cnn",
                        help="cnn=轻量数字分类器(默认,快); ocr=RapidOCR(对照)。")
```

- [ ] **Step 2: 冒烟(无相机，--source 指向合成图)**

Run: `python -c "import vision_fusion.digit_detect_tri as m; print('main ok', hasattr(m,'DigitClassifierTri'))"`
Expected: `main ok True`，且 `python -m pytest vision_fusion/tests/ -q` 全过。

- [ ] **Step 3: 提交**

```bash
git add vision_fusion/digit_detect_tri.py
git commit -m "feat(digit_detect_tri): --recognizer cnn|ocr, default cnn with ocr fallback"
```

---

## Task 6: 验证、提速实测、文档与记忆

**Files:**
- Modify: `vision_fusion/profile_tri_pipeline.py`（支持 `--recognizer`）、`README.md`
- Memory: 更新 `digit-detect-tri-decoder.md`

- [ ] **Step 1: profiler 支持选识别器并实测对比**

在 `profile_tri_pipeline.py` 的 argparse 加 `--recognizer {cnn,ocr}`，按选择实例化 `DigitClassifierTri`/`DigitRecognizerTri`(CNN 无 `_ocr`，计数包裹改为包裹 `read_debug`)。然后：

```bash
TS=$(date +%Y%m%d-%H%M%S)
python -m vision_fusion.profile_tri_pipeline --recognizer cnn --frames 6 --iters 8 \
  --archive "docs/test-screenshots/cnn-profile-$TS"
```
Expected: `decode` 从 ~5591ms 掉到 <50ms，每帧合计由 YOLO(~74ms) 主导 → **>10 FPS**。存档对比。

- [ ] **Step 2: 端到端识别准确率(合成 + 真实帧)**

在含 marker 的合成帧与真实帧上跑 `main --debug-dir`，统计 decode 成功率，与 RapidOCR 对照，存档到 `docs/test-screenshots/`。让用户检查。

- [ ] **Step 3: 更新 README 与 memory**

README「切角数字 Marker 解码器」段补充：默认 `--recognizer cnn`(轻量 0-9+X 分类器，比 RapidOCR 快百倍)，`--recognizer ocr` 留作对照；数据生成 `nn_synth_tri_digit`、训练 `nn_train_tri_digit`。更新 memory `digit-detect-tri-decoder.md`：识别已由 RapidOCR 换为专训 CNN(11 类)，附速度/准确率数字与模型路径。

- [ ] **Step 4: 提交**

```bash
git add README.md vision_fusion/profile_tri_pipeline.py
git commit -m "docs+perf(digit_detect_tri): CNN recognizer profile, README, accuracy archive"
```

---

## 自检

- **覆盖**：数据(T2)用对的 Gothic 字体重生 0-9+X；重训(T3)；切格训练/推理同源(T1 `slice_cells`);替换 OCR(T4)；接入(T5)；提速+准确率实测+文档(T6)。✓
- **类型一致**：`CHARS`/`cell_centers`/`slice_cells`/`marker_chars`/`DigitClassifierTri.read_debug` 在各任务签名一致；`read_debug` 返回 `(id,conf,info)` 与 `DigitRecognizerTri` 同，main 无需改判别逻辑。✓
- **依赖前置**：T2 Step5 先验 `random_degrade` 签名；T3 复用 `DigitCNN`;T4 复用 `corner_ranking/_ROT_TO_TL/decode_id`。✓
- **风险**：切格半边长 `CELL_HALF` 若不准→T2 存档图人工核对后调整;X 类样本量→T3 监控该类 acc。
