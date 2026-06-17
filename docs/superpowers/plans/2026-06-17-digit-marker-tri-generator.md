# 数字 Marker 生成器 `digit_marker_tri` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新建一版数字 marker 生成器：左上切角(chamfer)定向 + 2×2 大数字 + 加权 mod 11(X 兜底)校验，CLI 批量导出 + Tkinter 实时调参 GUI，所有排版参数(字体/字号/列距/行距/stroke 仿粗/padding/切角大小)可调。

**Architecture:** 纯核心函数(无 IO、可单测) ← CLI(单个/区间批量到指定目录) 与 GUI(实时预览+导出)。三个新文件，不碰 `digit_marker` v1/v2/`digit_marker3` 及现有 UI(守住 working state)。

**Tech Stack:** Python 3.11、Pillow(PIL) 绘制、numpy、Tkinter+cv2+ImageTk(GUI 预览)、pytest。

参考 spec：`docs/superpowers/specs/2026-06-17-digit-marker-tri-generator-design.md`

---

## File Structure

- Create: `vision_fusion/digit_marker_tri.py` — 核心 `checksum_char()` + `generate_marker_tri()` + CLI(`parse_args`/`_ids_from_args`/`main`)。
- Create: `vision_fusion/digit_marker_tri_ui.py` — Tkinter GUI，import 核心(不复制绘制逻辑)。
- Create: `vision_fusion/tests/test_digit_marker_tri.py` — pytest 单测(校验位 + 布局不变量 + CLI ID 解析 + GUI 复用)。

文件名约定沿用现有：CLI 单字符墨色灰度 PNG，输出 `digit_{id:03d}_{check}.png`。

---

### Task 1: `checksum_char` 加权 mod 11 + X 校验位（TDD）

**Files:**
- Create: `vision_fusion/digit_marker_tri.py`
- Test: `vision_fusion/tests/test_digit_marker_tri.py`

- [ ] **Step 1: Write the failing test**

写入 `vision_fusion/tests/test_digit_marker_tri.py`：

```python
import numpy as np
import pytest

from vision_fusion.digit_marker_tri import checksum_char


def test_checksum_known_values():
    # 083: 1*0 + 2*8 + 3*3 = 25, 25 % 11 = 3
    assert checksum_char(83) == "3"
    # 000: 0 → "0"
    assert checksum_char(0) == "0"


def test_checksum_emits_X_when_value_is_10():
    # 022: 1*0 + 2*2 + 3*2 = 10 → "X"
    assert checksum_char(22) == "X"
    # 999: 9 + 18 + 27 = 54, 54 % 11 = 10 → "X"
    assert checksum_char(999) == "X"


def test_checksum_detects_transposition():
    # 换位必被抓到：123 与 213 校验不同
    assert checksum_char(123) != checksum_char(213)


def test_checksum_all_ids_single_legal_char():
    legal = set("0123456789X")
    for mid in range(1000):
        c = checksum_char(mid)
        assert len(c) == 1 and c in legal
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest vision_fusion/tests/test_digit_marker_tri.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vision_fusion.digit_marker_tri'`。

- [ ] **Step 3: Write minimal implementation**

写入 `vision_fusion/digit_marker_tri.py`（先只放 imports + checksum）：

```python
"""数字 Marker 生成器：左上切角定向 + 2×2 数字 + 加权 mod 11(X) 校验。

设计见 docs/superpowers/specs/2026-06-17-digit-marker-tri-generator-design.md
不复用旧的 digit_marker.checksum(求和 mod 10)——本版用加权 mod 11 抓换位错。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_FONT = "C:/Windows/Fonts/consolab.ttf"


def checksum_char(marker_id: int) -> str:
    """加权 mod 11 校验位；结果 10 按 ISBN-10 风格返回 'X'。"""
    d = f"{marker_id:03d}"
    c = (1 * int(d[0]) + 2 * int(d[1]) + 3 * int(d[2])) % 11
    return "X" if c == 10 else str(c)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest vision_fusion/tests/test_digit_marker_tri.py -v`
Expected: PASS — 4 个 test 全绿。

- [ ] **Step 5: Commit**

```bash
git add vision_fusion/digit_marker_tri.py vision_fusion/tests/test_digit_marker_tri.py
git commit -m "feat(digit_marker_tri): add weighted mod-11 checksum with X fallback"
```

---

### Task 2: `generate_marker_tri` 切角边框 + 2×2 数字（TDD）

**Files:**
- Modify: `vision_fusion/digit_marker_tri.py`
- Test: `vision_fusion/tests/test_digit_marker_tri.py`

布局不变量靠像素断言验证：输出尺寸/类型、左上切角处变白(方向标记真切了)、
右上角黑框完好(只切左上)、四格附近有数字暗像素、增大 stroke 墨量不减。

- [ ] **Step 1: Write the failing test**

在 `vision_fusion/tests/test_digit_marker_tri.py` 末尾追加：

```python
def test_generate_shape_and_dtype():
    arr = generate_marker_tri(83, pixels=600)
    assert arr.shape == (600, 600)
    assert arr.dtype == np.uint8


def test_generate_chamfer_only_top_left():
    p = 600
    arr = generate_marker_tri(83, pixels=p, border_ratio=0.07, chamfer_ratio=0.18)
    b = int(p * 0.07)
    q = b // 2  # 落在边框带内、且在切角三角内(2q < chamfer)
    assert arr[q, q] == 255            # 左上被切掉 → 白
    assert arr[q, p - 1 - q] == 0      # 右上黑框完好 → 黑
    assert arr[p - 1 - q, q] == 0      # 左下黑框完好 → 黑


def test_generate_digits_have_ink_near_cells():
    p = 600
    arr = generate_marker_tri(83, pixels=p)
    b = int(p * 0.07); pad = int(p * 0.06)
    c = (b + pad + p - 1 - b - pad) / 2.0
    col = p * 0.34; row = p * 0.34; w = int(p * 0.28) // 2
    centers = [(c - col / 2, c - row / 2), (c + col / 2, c - row / 2),
               (c - col / 2, c + row / 2), (c + col / 2, c + row / 2)]
    for gx, gy in centers:
        win = arr[int(gy - w):int(gy + w), int(gx - w):int(gx + w)]
        assert win.min() < 128  # 该格窗口内存在墨色像素


def test_stroke_ratio_adds_ink():
    base = generate_marker_tri(888, pixels=400, stroke_ratio=0.0)
    bold = generate_marker_tri(888, pixels=400, stroke_ratio=0.08)
    assert int((bold < 128).sum()) >= int((base < 128).sum())
```

并把文件顶部 import 行补上 `generate_marker_tri`：

```python
from vision_fusion.digit_marker_tri import checksum_char, generate_marker_tri
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest vision_fusion/tests/test_digit_marker_tri.py -v`
Expected: FAIL — `ImportError: cannot import name 'generate_marker_tri'`。

- [ ] **Step 3: Write minimal implementation**

在 `vision_fusion/digit_marker_tri.py` 的 `checksum_char` 之后追加：

```python
def generate_marker_tri(
    marker_id: int,
    *,
    pixels: int = 600,
    border_ratio: float = 0.07,
    chamfer_ratio: float = 0.18,
    pad_ratio: float = 0.06,
    col_gap_ratio: float = 0.34,
    row_gap_ratio: float = 0.34,
    font_path: str = DEFAULT_FONT,
    font_size_ratio: float = 0.28,
    stroke_ratio: float = 0.0,
) -> np.ndarray:
    """切角边框 + 2×2 数字 marker（灰度 ndarray，背景 255 墨色 0）。"""
    p = pixels
    img = Image.new("L", (p, p), 255)
    draw = ImageDraw.Draw(img)

    b = int(p * border_ratio)
    draw.rectangle([0, 0, p - 1, p - 1], fill=0)              # 全黑
    draw.rectangle([b, b, p - 1 - b, p - 1 - b], fill=255)    # 挖白内部 → 黑边框
    cut = int(p * chamfer_ratio)
    draw.polygon([(0, 0), (cut, 0), (0, cut)], fill=255)      # 左上切角

    pad = int(p * pad_ratio)
    lo, hi = b + pad, p - 1 - b - pad
    cx = cy = (lo + hi) / 2.0
    col = p * col_gap_ratio
    row = p * row_gap_ratio
    centers = [
        (cx - col / 2, cy - row / 2),  # TL d1
        (cx + col / 2, cy - row / 2),  # TR d2
        (cx - col / 2, cy + row / 2),  # BL d3
        (cx + col / 2, cy + row / 2),  # BR check
    ]

    text = f"{marker_id:03d}{checksum_char(marker_id)}"
    size = max(8, int(p * font_size_ratio))
    try:
        font = ImageFont.truetype(font_path, size)
    except OSError:
        font = ImageFont.load_default()
    stroke_w = max(0, round(stroke_ratio * size))

    for glyph, (gx, gy) in zip(text, centers):
        bb = font.getbbox(glyph, stroke_width=stroke_w)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        draw.text((gx - tw / 2 - bb[0], gy - th / 2 - bb[1]), glyph,
                  fill=0, font=font, stroke_width=stroke_w, stroke_fill=0)

    return np.array(img)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest vision_fusion/tests/test_digit_marker_tri.py -v`
Expected: PASS — 共 8 个 test 全绿（4 校验 + 4 布局）。

- [ ] **Step 5: Commit**

```bash
git add vision_fusion/digit_marker_tri.py vision_fusion/tests/test_digit_marker_tri.py
git commit -m "feat(digit_marker_tri): add chamfer-border 2x2 marker renderer"
```

---

### Task 3: CLI — 单个/区间批量导出到指定目录（TDD）

**Files:**
- Modify: `vision_fusion/digit_marker_tri.py`
- Test: `vision_fusion/tests/test_digit_marker_tri.py`

可单测的纯逻辑是"ID 解析"(`--id` / `--range` / 默认)。`main()` 的文件写入用
tmp_path 做一次小批量集成验证。

- [ ] **Step 1: Write the failing test**

在测试文件末尾追加（并补 import）：

```python
import argparse
from vision_fusion.digit_marker_tri import _ids_from_args, main


def test_ids_from_args_single():
    ns = argparse.Namespace(id=83, range=None)
    assert _ids_from_args(ns) == [83]


def test_ids_from_args_inclusive_range():
    ns = argparse.Namespace(id=None, range=[5, 25])
    ids = _ids_from_args(ns)
    assert ids == list(range(5, 26))
    assert len(ids) == 21  # 闭区间


def test_ids_from_args_default_is_0_99():
    ns = argparse.Namespace(id=None, range=None)
    assert _ids_from_args(ns) == list(range(0, 100))


def test_main_batch_exports_range(tmp_path, monkeypatch):
    out = tmp_path / "markers_5_25"
    monkeypatch.setattr(
        "sys.argv",
        ["digit_marker_tri", "--range", "5", "25", "--output", str(out),
         "--pixels", "120"],
    )
    rc = main()
    assert rc == 0
    pngs = sorted(out.glob("*.png"))
    assert len(pngs) == 21
    # 文件名带校验位：5 → 005 + checksum
    assert (out / f"digit_005_{checksum_char(5)}.png").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest vision_fusion/tests/test_digit_marker_tri.py -v`
Expected: FAIL — `ImportError: cannot import name '_ids_from_args'`。

- [ ] **Step 3: Write minimal implementation**

在 `vision_fusion/digit_marker_tri.py` 末尾追加：

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate chamfer-oriented 2x2 digit markers (weighted mod-11 checksum).")
    parser.add_argument("--id", type=int, default=None, help="Single marker ID (0-999).")
    parser.add_argument("--range", nargs=2, type=int, default=None,
                        metavar=("START", "END"), help="Inclusive ID range to batch-export.")
    parser.add_argument("--output", type=str, default="digit_markers_tri",
                        help="Output directory (created if missing).")
    parser.add_argument("--pixels", type=int, default=600)
    parser.add_argument("--border-ratio", type=float, default=0.07)
    parser.add_argument("--chamfer-ratio", type=float, default=0.18)
    parser.add_argument("--pad-ratio", type=float, default=0.06)
    parser.add_argument("--col-gap-ratio", type=float, default=0.34)
    parser.add_argument("--row-gap-ratio", type=float, default=0.34)
    parser.add_argument("--font-path", type=str, default=DEFAULT_FONT)
    parser.add_argument("--font-size-ratio", type=float, default=0.28)
    parser.add_argument("--stroke-ratio", type=float, default=0.0)
    return parser.parse_args()


def _ids_from_args(args: argparse.Namespace) -> list[int]:
    if args.id is not None:
        return [args.id]
    if args.range is not None:
        return list(range(args.range[0], args.range[1] + 1))
    return list(range(0, 100))


def main() -> int:
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ids = _ids_from_args(args)
    for mid in ids:
        arr = generate_marker_tri(
            mid, pixels=args.pixels, border_ratio=args.border_ratio,
            chamfer_ratio=args.chamfer_ratio, pad_ratio=args.pad_ratio,
            col_gap_ratio=args.col_gap_ratio, row_gap_ratio=args.row_gap_ratio,
            font_path=args.font_path, font_size_ratio=args.font_size_ratio,
            stroke_ratio=args.stroke_ratio,
        )
        Image.fromarray(arr).save(str(out / f"digit_{mid:03d}_{checksum_char(mid)}.png"))
    print(f"Generated {len(ids)} markers in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest vision_fusion/tests/test_digit_marker_tri.py -v`
Expected: PASS — 共 12 个 test 全绿。

- [ ] **Step 5: Commit**

```bash
git add vision_fusion/digit_marker_tri.py vision_fusion/tests/test_digit_marker_tri.py
git commit -m "feat(digit_marker_tri): add CLI with single/range batch export"
```

---

### Task 4: Tkinter 调参 GUI（复用核心，不复制绘制）

**Files:**
- Create: `vision_fusion/digit_marker_tri_ui.py`
- Test: `vision_fusion/tests/test_digit_marker_tri.py`

GUI 依赖显示器，无法 headless 单测；单测只锁"GUI 复用核心函数、未复制绘制逻辑"，
可见行为靠 Task 5 的手动冒烟 + 导出 PNG 存档验证。

- [ ] **Step 1: Write the failing test**

在测试文件末尾追加：

```python
def test_ui_reuses_core_not_copies():
    from vision_fusion import digit_marker_tri_ui as ui
    from vision_fusion import digit_marker_tri as core
    assert ui.generate_marker_tri is core.generate_marker_tri
    assert ui.checksum_char is core.checksum_char
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest vision_fusion/tests/test_digit_marker_tri.py::test_ui_reuses_core_not_copies -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vision_fusion.digit_marker_tri_ui'`。

- [ ] **Step 3: Write minimal implementation**

写入 `vision_fusion/digit_marker_tri_ui.py`：

```python
"""数字 Marker 生成器 GUI（digit_marker_tri）：实时调参 + 单个/区间批量导出。

Usage:
    python -m vision_fusion.digit_marker_tri_ui
"""
from __future__ import annotations

import json
import tkinter as tk
from tkinter import ttk
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

from .digit_marker_tri import checksum_char, generate_marker_tri

SETTINGS_FILE = Path("digit_marker_tri_settings.json")


def load_settings() -> dict:
    if SETTINGS_FILE.is_file():
        try:
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_settings(params: dict) -> None:
    SETTINGS_FILE.write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")


def list_system_fonts() -> list[str]:
    fonts_dir = Path("C:/Windows/Fonts")
    return [str(f) for f in sorted(fonts_dir.glob("*.ttf"))]


class DigitMarkerTriUI:
    # (标签, 参数键, 滑块下限, 上限, 默认值)
    SLIDERS = [
        ("字体大小", "font_size_ratio", 0.10, 0.45, 0.28),
        ("边框粗细", "border_ratio", 0.03, 0.15, 0.07),
        ("切角大小", "chamfer_ratio", 0.05, 0.35, 0.18),
        ("四周padding", "pad_ratio", 0.0, 0.20, 0.06),
        ("列距", "col_gap_ratio", 0.15, 0.55, 0.34),
        ("行距", "row_gap_ratio", 0.15, 0.55, 0.34),
        ("粗细stroke", "stroke_ratio", 0.0, 0.15, 0.0),
    ]

    def __init__(self):
        saved = load_settings()
        self.root = tk.Tk()
        self.root.title("数字 Marker 生成器 (tri)")
        self.marker_id = tk.IntVar(value=saved.get("marker_id", 83))
        self.pixels = tk.IntVar(value=saved.get("pixels", 600))
        self.font_path = tk.StringVar(value=saved.get("font_path", "C:/Windows/Fonts/consolab.ttf"))
        self.start_id = tk.IntVar(value=saved.get("start_id", 5))
        self.end_id = tk.IntVar(value=saved.get("end_id", 25))
        self.output_dir = tk.StringVar(value=saved.get("output_dir", "digit_markers_tri"))
        self.ratios = {key: tk.DoubleVar(value=saved.get(key, dflt))
                       for _, key, _, _, dflt in self.SLIDERS}
        self.fonts = list_system_fonts()
        self._build_ui()
        self._refresh()

    def _params(self) -> dict:
        return {key: var.get() for key, var in self.ratios.items()}

    def _gen(self, mid: int) -> np.ndarray:
        return generate_marker_tri(mid, pixels=self.pixels.get(),
                                   font_path=self.font_path.get(), **self._params())

    def _build_ui(self):
        left = ttk.Frame(self.root)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        f = ttk.Frame(left); f.pack(fill=tk.X, pady=3)
        ttk.Label(f, text="ID:").pack(side=tk.LEFT)
        ttk.Spinbox(f, from_=0, to=999, textvariable=self.marker_id, width=5,
                    command=self._refresh).pack(side=tk.LEFT, padx=5)
        self.marker_id.trace_add("write", lambda *a: self._refresh())

        f = ttk.Frame(left); f.pack(fill=tk.X, pady=3)
        ttk.Label(f, text="字体:").pack(side=tk.LEFT)
        combo = ttk.Combobox(f, textvariable=self.font_path, width=30, values=self.fonts)
        combo.pack(side=tk.LEFT, padx=5)
        combo.bind("<<ComboboxSelected>>", lambda e: self._refresh())

        for label, key, lo, hi, _ in self.SLIDERS:
            f = ttk.Frame(left); f.pack(fill=tk.X, pady=3)
            ttk.Label(f, text=label, width=12).pack(side=tk.LEFT)
            var = self.ratios[key]
            ttk.Scale(f, variable=var, from_=lo, to=hi, length=150, orient=tk.HORIZONTAL,
                      command=lambda e: self._refresh()).pack(side=tk.LEFT, padx=5)
            lbl = ttk.Label(f, text=""); lbl.pack(side=tk.LEFT)
            var.trace_add("write", lambda *a, v=var, l=lbl: l.config(text=f"{v.get():.3f}"))

        f = ttk.Frame(left); f.pack(fill=tk.X, pady=3)
        ttk.Label(f, text="像素:").pack(side=tk.LEFT)
        ttk.Spinbox(f, from_=120, to=1200, textvariable=self.pixels, width=6,
                    command=self._refresh).pack(side=tk.LEFT, padx=5)

        ttk.Button(left, text="保存当前 marker", command=self._save_one).pack(fill=tk.X, pady=(8, 3))

        f = ttk.Frame(left); f.pack(fill=tk.X, pady=3)
        ttk.Label(f, text="区间:").pack(side=tk.LEFT)
        ttk.Spinbox(f, from_=0, to=999, textvariable=self.start_id, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(f, text="–").pack(side=tk.LEFT)
        ttk.Spinbox(f, from_=0, to=999, textvariable=self.end_id, width=5).pack(side=tk.LEFT, padx=2)

        f = ttk.Frame(left); f.pack(fill=tk.X, pady=3)
        ttk.Label(f, text="输出:").pack(side=tk.LEFT)
        ttk.Entry(f, textvariable=self.output_dir, width=22).pack(side=tk.LEFT, padx=5)
        ttk.Button(left, text="批量导出区间", command=self._save_batch).pack(fill=tk.X, pady=3)

        self.status = tk.StringVar(value="就绪")
        ttk.Label(left, textvariable=self.status).pack(pady=5)
        self.canvas = tk.Canvas(self.root, width=400, height=400, bg="gray")
        self.canvas.pack(side=tk.RIGHT, padx=10, pady=10)

    def _refresh(self, *_):
        try:
            mid = self.marker_id.get()
        except (tk.TclError, ValueError):
            return
        disp = cv2.resize(self._gen(mid), (400, 400), interpolation=cv2.INTER_AREA)
        self._tk_img = ImageTk.PhotoImage(Image.fromarray(disp))
        self.canvas.delete("all")
        self.canvas.create_image(200, 200, image=self._tk_img)
        c = checksum_char(mid)
        self.status.set(f'ID={mid:03d} 校验={c} 文本="{mid:03d}{c}"')

    def _save_one(self):
        mid = self.marker_id.get()
        out = Path(self.output_dir.get()); out.mkdir(parents=True, exist_ok=True)
        path = out / f"digit_{mid:03d}_{checksum_char(mid)}.png"
        Image.fromarray(self._gen(mid)).save(str(path))
        self._persist(); self.status.set(f"已保存: {path}")

    def _save_batch(self):
        out = Path(self.output_dir.get()); out.mkdir(parents=True, exist_ok=True)
        lo, hi = self.start_id.get(), self.end_id.get()
        if lo > hi:
            lo, hi = hi, lo
        for mid in range(lo, hi + 1):
            Image.fromarray(self._gen(mid)).save(
                str(out / f"digit_{mid:03d}_{checksum_char(mid)}.png"))
        self._persist(); self.status.set(f"已导出 {hi - lo + 1} 张到 {out}")

    def _persist(self):
        params = {"marker_id": self.marker_id.get(), "pixels": self.pixels.get(),
                  "font_path": self.font_path.get(), "start_id": self.start_id.get(),
                  "end_id": self.end_id.get(), "output_dir": self.output_dir.get()}
        params.update(self._params())
        save_settings(params)

    def run(self):
        self.root.mainloop()


def main() -> int:
    DigitMarkerTriUI().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest vision_fusion/tests/test_digit_marker_tri.py -v`
Expected: PASS — 共 13 个 test 全绿。

- [ ] **Step 5: Commit**

```bash
git add vision_fusion/digit_marker_tri_ui.py vision_fusion/tests/test_digit_marker_tri.py
git commit -m "feat(digit_marker_tri): add tkinter tuning GUI with range export"
```

---

### Task 5: 全量测试 + 可视存档 + README

**Files:**
- Create: `docs/test-screenshots/<timestamp>/` (导出的样张 PNG)
- Modify: `README.md`（追加运行说明）

marker PNG 本身就是可视产物，导出几张到时间戳目录即为存档（无需浏览器/Playwright，
这是桌面 GUI + PNG，不是 web app）。

- [ ] **Step 1: 跑全部单测**

Run: `python -m pytest vision_fusion/tests/test_digit_marker_tri.py -v`
Expected: PASS — 13 个 test 全绿。

- [ ] **Step 2: CLI 批量导出样张到时间戳目录**

挑能覆盖关键情形的 ID：`0`(全 0)、`83`(普通)、`22` 与 `999`(校验=X)，外加你举的
`5-25` 区间。时间戳用当天，例如 `20260617-HHMMSS`：

```bash
python -m vision_fusion.digit_marker_tri --range 5 25 \
  --output docs/test-screenshots/digit-tri-$(date +%Y%m%d-%H%M%S)/
python -m vision_fusion.digit_marker_tri --id 0   --output docs/test-screenshots/digit-tri-samples/
python -m vision_fusion.digit_marker_tri --id 22  --output docs/test-screenshots/digit-tri-samples/
python -m vision_fusion.digit_marker_tri --id 999 --output docs/test-screenshots/digit-tri-samples/
```

人工核查点：左上有切角(五边形轮廓)；四格数字清晰不重叠不出框；`22`/`999` 第 4 格
是 `X`；`5-25` 目录有 21 张、文件名带校验位。把目录路径报给用户。

- [ ] **Step 3: 手动冒烟 GUI（需显示器）**

Run: `python -m vision_fusion.digit_marker_tri_ui`
拖动各滑块看实时预览：字号/列距/行距/padding/切角大小/stroke 仿粗都生效；改 ID 切换
预览；填区间 + 输出目录点"批量导出区间"，确认目录里出现对应张数。截一张窗口图存到
上面的时间戳目录（可选）。`Q`/关窗退出。

- [ ] **Step 4: 更新 README**

在 README.md 现有 marker 生成说明附近追加：

````markdown
### 切角数字 Marker 生成器 (digit_marker_tri)

左上切角定向 + 2×2 大数字 + 加权 mod 11(X 兜底)校验。CLI 批量导出：

```bash
# 导出 5–25 号到指定目录（闭区间，目录自动建）
python -m vision_fusion.digit_marker_tri --range 5 25 --output out/markers_5_25/
python -m vision_fusion.digit_marker_tri --id 83 --output digit_markers_tri/
```

实时调参（字体/字号/列距/行距/粗细 stroke/padding/切角大小）+ 区间导出 GUI：

```bash
python -m vision_fusion.digit_marker_tri_ui
```

校验位 = `(1·d₁+2·d₂+3·d₃) mod 11`，值为 10 时印 `X`（ISBN-10 风格），1000 个 ID 全可用。
````

- [ ] **Step 5: Commit**

```bash
git add README.md docs/test-screenshots/
git commit -m "docs(digit_marker_tri): add run instructions and sample markers"
```

---

## 自检清单（spec 覆盖）

- spec §3 架构三文件 → Task 1/2(核心) + Task 3(CLI) + Task 4(GUI)
- spec §4 加权 mod 11 + X 校验 → Task 1
- spec §5 切角边框 + 2×2 + 7 旋钮 → Task 2
- spec §6 GUI 实时调参 + ID 可选 + 区间到指定目录 → Task 4
- spec §7 CLI 单个/区间批量到指定目录 → Task 3
- spec §8 测试(校验/布局不变量/CLI/GUI 复用) + 可视存档 → Task 1-4 单测 + Task 5
- spec §9 不碰 v1/v2/v3 → 全程仅新建文件






