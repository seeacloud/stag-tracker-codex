"""Train YOLOv8-nano marker detector from labeled data.

Usage:
    python -m vision_fusion.training.train_yolo --data yolo_data/ --output models/yolo_marker.pt

Supports incremental training: re-run after adding more labeled frames.
Each run trains from scratch on ALL available data (not fine-tuning previous model).
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path


def prepare_dataset(data_dir: Path, work_dir: Path, val_split: float = 0.15) -> Path:
    """Organize data into YOLO dataset structure and write data.yaml."""
    img_dir = data_dir / "images"
    lbl_dir = data_dir / "labels"

    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        raise SystemExit(f"No images found in {img_dir}")

    # Split into train/val
    random.shuffle(images)
    val_count = max(1, int(len(images) * val_split))
    val_images = images[:val_count]
    train_images = images[val_count:]

    # Create YOLO directory structure
    for split in ("train", "val"):
        (work_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (work_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    def copy_pair(img_path: Path, split: str):
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        shutil.copy2(img_path, work_dir / split / "images" / img_path.name)
        if lbl_path.exists():
            shutil.copy2(lbl_path, work_dir / split / "labels" / lbl_path.name)

    for img in train_images:
        copy_pair(img, "train")
    for img in val_images:
        copy_pair(img, "val")

    # Write data.yaml
    yaml_path = work_dir / "data.yaml"
    yaml_path.write_text(
        f"path: {work_dir.resolve()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"nc: 1\n"
        f"names: ['stag_marker']\n"
    )

    print(f"Dataset: {len(train_images)} train, {len(val_images)} val")
    return yaml_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train YOLOv8-nano marker detector.")
    parser.add_argument("--data", required=True, help="Labeled data directory (with images/ and labels/).")
    parser.add_argument("--output", default="models/yolo_marker.pt", help="Output model path.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model", default="yolov8n.pt", help="Base model (nano by default).")
    parser.add_argument("--device", default="0", help="CUDA device.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    data_dir = Path(args.data)
    work_dir = data_dir / "_yolo_dataset"
    if work_dir.exists():
        shutil.rmtree(work_dir)

    yaml_path = prepare_dataset(data_dir, work_dir)

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(data_dir / "_runs"),
        name="train",
        exist_ok=True,
        seed=args.seed,
        verbose=True,
    )

    # Copy best model to output path
    best_path = Path(results.save_dir) / "weights" / "best.pt"
    if best_path.exists():
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, output_path)
        print(f"\nModel saved to: {args.output}")
    else:
        print("WARNING: best.pt not found, check training output", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
