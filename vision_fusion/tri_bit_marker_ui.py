"""Tri-Bit Marker 调参 GUI:实时预览 + 区间批量导出 + 参数自动存取。

可调:外框宽(border_ratio)、格间距(gap_ratio)、填充比(fill_ratio)。
参数存 tri_bit_marker_settings.json,detector 读同一 json 保证一致。
"""
from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

from .tri_bit_marker import render_tri_bit_marker, encode_id, MAX_ID, N_DATA

SETTINGS_FILE = Path("tri_bit_marker_settings.json")

_RATIOS = [
    ("border_ratio", 0.08, 0.02, 0.16),     # 外框宽
    ("gap_ratio", 0.0, 0.0, 0.06),          # 格间距(格与格之间的黑线)
    ("fill_ratio", 1.0, 0.5, 1.0),          # 格内填充比(1=填满,<1留白)
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


class TriBitMarkerUI:
    def __init__(self):
        saved = load_settings()
        self.root = tk.Tk()
        self.root.title("Tri-Bit Marker 调参")
        self.marker_id = tk.IntVar(value=saved.get("marker_id", 0))
        self.pixels = tk.IntVar(value=saved.get("pixels", 600))
        self.show_grid = tk.BooleanVar(value=saved.get("show_grid", True))
        self.start_id = tk.IntVar(value=saved.get("start_id", 0))
        self.end_id = tk.IntVar(value=saved.get("end_id", 100))
        self.output_dir = tk.StringVar(value=saved.get("output_dir", "tri_bit_markers"))
        self.ratios = {k: tk.DoubleVar(value=saved.get(k, d)) for k, d, _, _ in _RATIOS}
        self._val_labels = {}
        self.status = tk.StringVar(value="就绪")
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._refresh()

    def _params(self) -> dict:
        return {k: round(v.get(), 4) for k, v in self.ratios.items()}

    def _gen(self, mid: int) -> np.ndarray:
        return render_tri_bit_marker(mid, pixels=self.pixels.get(), **self._params())

    def _build_ui(self):
        left = ttk.Frame(self.root, padding=8); left.pack(side=tk.LEFT, fill=tk.Y)
        row = ttk.Frame(left); row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="marker_id").pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=0, to=MAX_ID, textvariable=self.marker_id, width=7,
                    command=self._refresh).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(left, text="显示格线", variable=self.show_grid,
                        command=self._refresh).pack(anchor=tk.W, pady=2)
        for key, _d, lo, hi in _RATIOS:
            f = ttk.Frame(left); f.pack(fill=tk.X, pady=2)
            ttk.Label(f, text=key, width=16).pack(side=tk.LEFT)
            ttk.Scale(f, variable=self.ratios[key], from_=lo, to=hi, length=150,
                      orient=tk.HORIZONTAL, command=self._refresh).pack(side=tk.LEFT)
            val = ttk.Label(f, width=6); val.pack(side=tk.LEFT, padx=4)
            self._val_labels[key] = val
        ttk.Button(left, text="保存当前 marker", command=self._save_one).pack(fill=tk.X, pady=(8, 3))
        rng = ttk.Frame(left); rng.pack(fill=tk.X, pady=3)
        ttk.Label(rng, text="区间").pack(side=tk.LEFT)
        ttk.Spinbox(rng, from_=0, to=MAX_ID, textvariable=self.start_id, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Spinbox(rng, from_=0, to=MAX_ID, textvariable=self.end_id, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Entry(left, textvariable=self.output_dir).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="批量导出区间", command=self._save_batch).pack(fill=tk.X, pady=3)
        ttk.Button(left, text="保存参数", command=self._persist).pack(fill=tk.X, pady=3)
        ttk.Label(left, textvariable=self.status, wraplength=220, foreground="green").pack(fill=tk.X, pady=6)
        self.canvas = ttk.Label(self.root); self.canvas.pack(side=tk.LEFT, padx=8, pady=8)

    def _refresh(self, *_):
        for k, lbl in self._val_labels.items():
            lbl.config(text=f"{self.ratios[k].get():.4f}")
        mid = self.marker_id.get()
        img = self._gen(mid)
        disp = Image.fromarray(img).resize((400, 400), Image.NEAREST).convert("L")
        if self.show_grid.get():
            import PIL.ImageDraw as _D
            d = _D.Draw(disp); W = 400
            params = self._params()
            br = params["border_ratio"]; gap = params["gap_ratio"]
            inner = 1.0 - 2 * br; cell = inner / 4
            for r in range(4):
                for c in range(4):
                    x0 = int((br + c * cell) * W); y0 = int((br + r * cell) * W)
                    x1 = int((br + (c + 1) * cell) * W); y1 = int((br + (r + 1) * cell) * W)
                    d.rectangle([x0, y0, x1, y1], outline=128, width=1)
        self._tk = ImageTk.PhotoImage(disp)
        self.canvas.configure(image=self._tk)
        bits = encode_id(mid)
        self.status.set(f"id={mid}  data={bits[:N_DATA]}  check={bits[N_DATA:]}")

    def _save_one(self):
        mid = self.marker_id.get()
        out = Path(self.output_dir.get()); out.mkdir(parents=True, exist_ok=True)
        path = out / f"tribit_{mid:05d}.png"
        Image.fromarray(self._gen(mid)).save(str(path))
        self._persist(); self.status.set(f"已保存: {path}")

    def _save_batch(self):
        lo, hi = self.start_id.get(), self.end_id.get()
        out = Path(self.output_dir.get()); out.mkdir(parents=True, exist_ok=True)
        for mid in range(lo, hi + 1):
            Image.fromarray(self._gen(mid)).save(str(out / f"tribit_{mid:05d}.png"))
        self._persist(); self.status.set(f"已导出 {hi - lo + 1} 张到 {out}")

    def _persist(self):
        params = {"marker_id": self.marker_id.get(), "pixels": self.pixels.get(),
                  "show_grid": self.show_grid.get(), "start_id": self.start_id.get(),
                  "end_id": self.end_id.get(), "output_dir": self.output_dir.get()}
        params.update(self._params())
        save_settings(params)
        self.status.set("参数已保存")

    def _on_close(self):
        self._persist(); self.root.destroy()

    def run(self):
        self.root.mainloop()


def main() -> int:
    TriBitMarkerUI().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
