"""符号 marker 调参 GUI:实时预览 + 区间批量导出 + 参数自动存取。

可调:统一线宽、圆角线头、符号大小、列距、行距、边框、底边黑条宽。
参数存 symbol_marker_settings.json,启动自动加载、改动/关闭自动保存。
渲染调 symbol_marker.render_marker_symbol,训练/识别读同一 json,保证一致。
"""
from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

from .symbol_marker import render_marker_symbol, checksum_char

SETTINGS_FILE = Path("symbol_marker_settings.json")

_RATIOS = [
    ("line_ratio", 0.08, 0.04, 0.16),       # 黑线宽(外框=分隔,同步)
    ("padding_ratio", 0.04, 0.0, 0.12),     # 裁切框相对白格内缩
    ("sym_fill", 0.72, 0.45, 0.95),         # 符号占裁切框比例
    ("stroke_ratio", 0.18, 0.08, 0.30),     # 符号线宽
]


def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_settings(params: dict) -> None:
    SETTINGS_FILE.write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")


class SymbolMarkerUI:
    def __init__(self):
        saved = load_settings()
        self.root = tk.Tk()
        self.root.title("Symbol Marker 调参")
        self.marker_id = tk.IntVar(value=saved.get("marker_id", 283))
        self.pixels = tk.IntVar(value=saved.get("pixels", 600))
        self.round_cap = tk.BooleanVar(value=saved.get("round_cap", False))
        self.show_cells = tk.BooleanVar(value=saved.get("show_cells", True))
        self.start_id = tk.IntVar(value=saved.get("start_id", 0))
        self.end_id = tk.IntVar(value=saved.get("end_id", 100))
        self.output_dir = tk.StringVar(value=saved.get("output_dir", "symbol_markers"))
        self.ratios = {k: tk.DoubleVar(value=saved.get(k, d)) for k, d, _, _ in _RATIOS}
        self.status = tk.StringVar(value="就绪")
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._refresh()

    def _params(self) -> dict:
        return {k: round(v.get(), 4) for k, v in self.ratios.items()}

    def _gen(self, mid: int) -> np.ndarray:
        return render_marker_symbol(mid, pixels=self.pixels.get(),
                                    round_cap=self.round_cap.get(), **self._params())

    def _build_ui(self):
        left = ttk.Frame(self.root, padding=8)
        left.pack(side=tk.LEFT, fill=tk.Y)
        row = ttk.Frame(left); row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="marker_id").pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=0, to=999, textvariable=self.marker_id, width=6,
                    command=self._refresh).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(left, text="圆角线头", variable=self.round_cap,
                        command=self._refresh).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(left, text="显示字符格范围(切格)", variable=self.show_cells,
                        command=self._refresh).pack(anchor=tk.W, pady=2)
        for key, _d, lo, hi in _RATIOS:
            f = ttk.Frame(left); f.pack(fill=tk.X, pady=2)
            ttk.Label(f, text=key, width=18).pack(side=tk.LEFT)
            ttk.Scale(f, variable=self.ratios[key], from_=lo, to=hi, length=150,
                      orient=tk.HORIZONTAL, command=self._refresh).pack(side=tk.LEFT)
        ttk.Button(left, text="保存当前 marker", command=self._save_one).pack(fill=tk.X, pady=(8, 3))
        rng = ttk.Frame(left); rng.pack(fill=tk.X, pady=3)
        ttk.Label(rng, text="区间").pack(side=tk.LEFT)
        ttk.Spinbox(rng, from_=0, to=999, textvariable=self.start_id, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Spinbox(rng, from_=0, to=999, textvariable=self.end_id, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Entry(left, textvariable=self.output_dir).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="批量导出区间", command=self._save_batch).pack(fill=tk.X, pady=3)
        ttk.Button(left, text="保存参数", command=self._persist).pack(fill=tk.X, pady=3)
        ttk.Label(left, textvariable=self.status, wraplength=200,
                  foreground="green").pack(fill=tk.X, pady=6)
        self.canvas = ttk.Label(self.root); self.canvas.pack(side=tk.LEFT, padx=8, pady=8)

    def _refresh(self, *_):
        mid = self.marker_id.get()
        img = self._gen(mid)
        disp = Image.fromarray(img).resize((360, 360), Image.NEAREST).convert("L")
        if self.show_cells.get():
            from .symbol_marker import cell_boxes
            import PIL.ImageDraw as _D
            d = _D.Draw(disp); W = 360
            for x0, y0, x1, y1 in cell_boxes(**self._params()):
                d.rectangle([int(x0 * W), int(y0 * W), int(x1 * W), int(y1 * W)],
                            outline=0, width=1)
        self._tk = ImageTk.PhotoImage(disp)
        self.canvas.configure(image=self._tk)
        self.status.set(f"预览 {mid:03d}{checksum_char(mid)}")

    def _save_one(self):
        mid = self.marker_id.get()
        out = Path(self.output_dir.get()); out.mkdir(parents=True, exist_ok=True)
        path = out / f"symbol_{mid:03d}_{checksum_char(mid)}.png"
        Image.fromarray(self._gen(mid)).save(str(path))
        self._persist(); self.status.set(f"已保存: {path}")

    def _save_batch(self):
        lo, hi = self.start_id.get(), self.end_id.get()
        out = Path(self.output_dir.get()); out.mkdir(parents=True, exist_ok=True)
        for mid in range(lo, hi + 1):
            Image.fromarray(self._gen(mid)).save(
                str(out / f"symbol_{mid:03d}_{checksum_char(mid)}.png"))
        self._persist(); self.status.set(f"已导出 {hi - lo + 1} 张到 {out}")

    def _persist(self):
        params = {"marker_id": self.marker_id.get(), "pixels": self.pixels.get(),
                  "round_cap": self.round_cap.get(), "show_cells": self.show_cells.get(),
                  "start_id": self.start_id.get(),
                  "end_id": self.end_id.get(), "output_dir": self.output_dir.get()}
        params.update(self._params())
        save_settings(params)
        self.status.set("参数已保存")

    def _on_close(self):
        self._persist(); self.root.destroy()

    def run(self):
        self.root.mainloop()


def main() -> int:
    SymbolMarkerUI().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
