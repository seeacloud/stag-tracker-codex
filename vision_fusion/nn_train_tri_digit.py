"""训练 tri 数字分类器(0-9 + X，11 类)。复用 nn_train_digit.DigitCNN。

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
    # 类不均衡(X 远少于数字)→ 反频率加权交叉熵。
    counts = np.bincount([lbl for _, lbl in ds.samples], minlength=len(CHARS)).astype(np.float32)
    w = (counts.sum() / (len(CHARS) * np.clip(counts, 1, None)))
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"device={device}  params={sum(p.numel() for p in model.parameters()):,}  class_w={np.round(w,2)}")

    best = 0.0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); lossf(model(x), y).backward(); opt.step()
        model.eval()
        per_correct = np.zeros(len(CHARS)); per_total = np.zeros(len(CHARS))
        with torch.no_grad():
            for x, y in vl:
                x = x.to(device)
                pred = model(x).argmax(1).cpu().numpy()
                for yi, pi in zip(y.numpy(), pred):
                    per_total[yi] += 1; per_correct[yi] += (yi == pi)
        acc = per_correct.sum() / max(1, per_total.sum())
        x_acc = per_correct[CHARS.index("X")] / max(1, per_total[CHARS.index("X")])
        print(f"epoch {ep+1}/{args.epochs}  val_acc={acc:.4f}  X_acc={x_acc:.4f}")
        if acc >= best:
            best = acc
            torch.save(model.state_dict(), args.out)
    Path("models/tri_digit_labels.json").write_text(json.dumps(list(CHARS)), encoding="ascii")
    print(f"best val_acc={best:.4f}  saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
