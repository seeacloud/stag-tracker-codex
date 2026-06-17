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
        ("三角大小", "chamfer_ratio", 0.05, 0.35, 0.18),
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
