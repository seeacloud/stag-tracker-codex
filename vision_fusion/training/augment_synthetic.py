"""Generate synthetic augmented training data from clear marker templates.

Usage:
    python -m vision_fusion.training.augment_synthetic --input training_data/ --output training_data/ --samples-per-id 200

Takes clear patches from {input}/{marker_id}/ and generates augmented versions
(blur, perspective, noise, gamma) saved to {output}/{marker_id}/ alongside originals.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def disc_kernel(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = (x * x + y * y <= radius * radius).astype(np.float32)
    return mask / mask.sum()


def augment_patch(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = patch.shape[:2]
    out = patch.astype(np.float32)

    # Random gamma (0.4 - 1.4)
    gamma = rng.uniform(0.4, 1.4)
    out = np.power(out / 255.0, gamma) * 255.0

    # Random contrast jitter
    alpha = rng.uniform(0.6, 1.4)
    mean = out.mean()
    out = (out - mean) * alpha + mean

    # Random blur (defocus or Gaussian)
    blur_type = rng.choice(["disc", "gaussian", "none"], p=[0.4, 0.3, 0.3])
    if blur_type == "disc":
        radius = rng.integers(2, 8)
        kernel = disc_kernel(int(radius))
        out = cv2.filter2D(out.clip(0, 255).astype(np.uint8), -1, kernel).astype(np.float32)
    elif blur_type == "gaussian":
        sigma = rng.uniform(1.0, 4.0)
        ksize = int(sigma * 4) | 1
        out = cv2.GaussianBlur(out.clip(0, 255).astype(np.uint8), (ksize, ksize), sigma).astype(np.float32)

    # Random perspective warp (small)
    if rng.random() < 0.7:
        max_shift = int(h * 0.08)
        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        dst = src + rng.uniform(-max_shift, max_shift, (4, 2)).astype(np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(out.clip(0, 255).astype(np.uint8), M, (w, h),
                                  borderMode=cv2.BORDER_REFLECT).astype(np.float32)

    # Random Gaussian noise
    if rng.random() < 0.5:
        noise_std = rng.uniform(5, 25)
        noise = rng.normal(0, noise_std, out.shape)
        out = out + noise

    # Random rotation (0, 90, 180, 270)
    if rng.random() < 0.3:
        k = rng.integers(1, 4)
        out_uint = out.clip(0, 255).astype(np.uint8)
        out = np.rot90(out_uint, k).astype(np.float32)

    return out.clip(0, 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic augmented training data.")
    parser.add_argument("--input", required=True, help="Input directory with clear patches (e.g. training_data/).")
    parser.add_argument("--output", required=True, help="Output directory (can be same as input).")
    parser.add_argument("--samples-per-id", type=int, default=200, help="Target augmented samples per marker ID.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    rng = np.random.default_rng(args.seed)

    # Find all marker IDs from both clear (numeric) and blurry (*_blurry) directories
    marker_ids: set[int] = set()
    for d in input_dir.iterdir():
        if not d.is_dir():
            continue
        if d.name.isdigit():
            marker_ids.add(int(d.name))
        elif d.name.endswith("_blurry"):
            try:
                marker_ids.add(int(d.name.replace("_blurry", "")))
            except ValueError:
                pass

    if not marker_ids:
        print(f"ERROR: No marker directories found in {input_dir}", file=sys.stderr)
        return 1

    total_generated = 0
    for marker_id in sorted(marker_ids):
        # Gather source patches from both clear and blurry dirs
        source_patches: list[Path] = []
        clear_dir = input_dir / str(marker_id)
        blurry_dir = input_dir / f"{marker_id}_blurry"
        if clear_dir.is_dir():
            source_patches.extend(clear_dir.glob("*.png"))
        if blurry_dir.is_dir():
            source_patches.extend(blurry_dir.glob("*.png"))
        if not source_patches:
            continue

        out_dir = output_dir / str(marker_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(out_dir.glob("*.png")))
        needed = max(0, args.samples_per_id - existing)

        if needed == 0:
            print(f"  ID {marker_id}: already has {existing} patches, skipping")
            continue

        for i in range(needed):
            src_path = rng.choice(source_patches)
            patch = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
            if patch is None:
                continue
            augmented = augment_patch(patch, rng)
            filename = f"aug_{i:04d}.png"
            cv2.imwrite(str(out_dir / filename), augmented)
            total_generated += 1

        print(f"  ID {marker_id}: generated {needed} augmented patches (from {len(source_patches)} sources)")

    print(f"\nDone. Generated {total_generated} augmented patches total.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
