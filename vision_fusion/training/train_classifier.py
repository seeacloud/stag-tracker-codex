"""Train MobileNetV3-Small classifier for STag HD17 marker identification.

Usage:
    python -m vision_fusion.training.train_classifier --data training_data/ --output models/stag_hd17.pt

Directory structure expected:
    training_data/
        0/          <- clear + augmented patches for marker ID 0
        0_blurry/   <- blurry patches for marker ID 0 (optional, merged into class 0)
        1/
        1_blurry/
        ...
        unknown/    <- negative samples (optional, becomes class 157)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class MarkerDataset(Dataset):
    def __init__(self, data_dir: Path, num_classes: int = 158):
        self.samples: list[tuple[Path, int]] = []
        self.num_classes = num_classes

        for subdir in sorted(data_dir.iterdir()):
            if not subdir.is_dir():
                continue
            name = subdir.name
            if name == "unknown":
                label = num_classes - 1
            elif name.endswith("_blurry"):
                label = int(name.replace("_blurry", ""))
            elif name.isdigit():
                label = int(name)
            else:
                continue

            for img_path in subdir.glob("*.png"):
                self.samples.append((img_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((128, 128), dtype=np.uint8)
        # Resize to 128x128 if needed
        if img.shape != (128, 128):
            img = cv2.resize(img, (128, 128))

        # Normalize to [0, 1] and replicate to 3 channels (pretrained expects RGB)
        tensor = torch.from_numpy(img).float() / 255.0
        tensor = tensor.unsqueeze(0).expand(3, -1, -1)  # (3, 128, 128)

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std

        return tensor, label


def build_model(num_classes: int = 158, pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    # Replace classifier head
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(1024, num_classes),
    )
    return model


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader,
               criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def main() -> int:
    parser = argparse.ArgumentParser(description="Train MobileNetV3-Small marker classifier.")
    parser.add_argument("--data", required=True, help="Training data directory.")
    parser.add_argument("--output", default="models/stag_hd17.pt", help="Output model path.")
    parser.add_argument("--epochs-frozen", type=int, default=5, help="Epochs with frozen backbone.")
    parser.add_argument("--epochs-finetune", type=int, default=20, help="Epochs with full fine-tuning.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-4)
    parser.add_argument("--num-classes", type=int, default=158, help="157 markers + 1 unknown.")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = MarkerDataset(Path(args.data), num_classes=args.num_classes)
    print(f"Total samples: {len(dataset)}")
    if len(dataset) == 0:
        print("ERROR: No training data found.", file=sys.stderr)
        return 1

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    print(f"Train: {train_size}, Val: {val_size}")

    model = build_model(num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    # Phase 1: frozen backbone
    for param in model.features.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)

    print(f"\n--- Phase 1: Frozen backbone ({args.epochs_frozen} epochs) ---")
    for epoch in range(args.epochs_frozen):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        print(f"  Epoch {epoch+1}/{args.epochs_frozen}: "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    # Phase 2: fine-tune all layers
    for param in model.features.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_finetune)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_finetune)

    print(f"\n--- Phase 2: Fine-tune all ({args.epochs_finetune} epochs) ---")
    best_val_acc = 0.0
    for epoch in range(args.epochs_finetune):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        print(f"  Epoch {epoch+1}/{args.epochs_finetune}: "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": args.num_classes,
                "val_acc": val_acc,
            }, output_path)
            print(f"    -> Saved best model (val_acc={val_acc:.4f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
