"""AUIE Gate A — chip extraction + geographic train/val/test split.

Usage:
    python make_chips.py --image s2_bbmp_2024_janmar.tif \
                         --labels dw4_bbmp_2024_janmar.tif \
                         --out chips/ --chip-size 512 --grid 6 6

What it does, and why:
  * Aligns labels to the image grid (nearest-neighbour) if grids differ —
    defensive; GEE exports on one CRS/scale should already align.
  * Extracts non-overlapping chip pairs, skipping chips with >20% nodata or
    >40% of a single... no—skipping only on nodata; class balance is reported,
    not enforced, so the paper can state the real distribution.
  * GEOGRAPHIC split: the AOI is divided into grid_rows x grid_cols blocks;
    whole blocks are assigned to train/val/test (~70/15/15, seeded shuffle).
    Adjacent chips never straddle splits -> no spatial leakage. This is the
    split reviewers will ask about; the manifest records block assignments.
  * Writes manifest.json (per-chip: file, split, window, class histogram),
    stats.json (per-band mean/std computed on TRAIN ONLY — use these for
    normalization in every backbone), and prints a per-split class report,
    warning if any split is missing a class.

Outputs GeoTIFF chips (georeferencing preserved):
    chips/{split}/images/chip_rXXXX_cXXXX.tif   uint16, 7 bands
    chips/{split}/masks/chip_rXXXX_cXXXX.tif    uint8, 1 band
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window

SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42
NODATA_MAX_FRACTION = 0.20
NUM_CLASSES = 4


def align_labels(img_ds: rasterio.DatasetReader, lbl_ds: rasterio.DatasetReader) -> np.ndarray:
    """Return label array on the image grid (nearest neighbour)."""
    same = (
        lbl_ds.crs == img_ds.crs
        and lbl_ds.transform == img_ds.transform
        and lbl_ds.shape == img_ds.shape
    )
    if same:
        return lbl_ds.read(1)
    print("labels grid differs from image grid -> reprojecting (nearest)")
    dst = np.zeros(img_ds.shape, dtype=np.uint8)
    reproject(
        source=lbl_ds.read(1),
        destination=dst,
        src_transform=lbl_ds.transform,
        src_crs=lbl_ds.crs,
        dst_transform=img_ds.transform,
        dst_crs=img_ds.crs,
        resampling=Resampling.nearest,
    )
    return dst


def assign_blocks(grid_rows: int, grid_cols: int) -> dict[tuple[int, int], str]:
    """Seeded shuffle of spatial blocks into train/val/test by whole blocks."""
    blocks = [(r, c) for r in range(grid_rows) for c in range(grid_cols)]
    random.Random(SEED).shuffle(blocks)
    n = len(blocks)
    if n < 3:
        raise ValueError("grid must have at least 3 blocks for a 3-way split")
    n_train = max(1, min(round(n * SPLIT_FRACTIONS["train"]), n - 2))
    n_val = max(1, min(round(n * SPLIT_FRACTIONS["val"]), n - n_train - 1))
    assignment: dict[tuple[int, int], str] = {}
    for i, blk in enumerate(blocks):
        assignment[blk] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return assignment


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chip-size", type=int, default=512)
    ap.add_argument("--grid", type=int, nargs=2, default=[6, 6],
                    metavar=("ROWS", "COLS"),
                    help="geographic split blocks across the AOI")
    args = ap.parse_args()

    out = Path(args.out)
    cs = args.chip_size
    with rasterio.open(args.image) as img_ds, rasterio.open(args.labels) as lbl_ds:
        labels = align_labels(img_ds, lbl_ds)
        height, width = img_ds.shape
        n_rows, n_cols = height // cs, width // cs
        grid_rows, grid_cols = args.grid
        block_of = assign_blocks(grid_rows, grid_cols)

        for split in SPLIT_FRACTIONS:
            (out / split / "images").mkdir(parents=True, exist_ok=True)
            (out / split / "masks").mkdir(parents=True, exist_ok=True)

        manifest, kept, skipped = [], Counter(), 0
        # accumulators for train-only normalization stats (Welford-lite)
        band_sum = np.zeros(img_ds.count, dtype=np.float64)
        band_sqsum = np.zeros(img_ds.count, dtype=np.float64)
        train_pixels = 0
        class_pixels: dict[str, Counter] = {s: Counter() for s in SPLIT_FRACTIONS}

        for r in range(n_rows):
            for c in range(n_cols):
                win = Window(c * cs, r * cs, cs, cs)
                chip = img_ds.read(window=win)  # (bands, cs, cs)
                nodata_frac = float((chip.sum(axis=0) == 0).mean())
                if nodata_frac > NODATA_MAX_FRACTION:
                    skipped += 1
                    continue
                mask = labels[r * cs:(r + 1) * cs, c * cs:(c + 1) * cs]

                blk = (min(r * grid_rows // n_rows, grid_rows - 1),
                       min(c * grid_cols // n_cols, grid_cols - 1))
                split = block_of[blk]
                name = f"chip_r{r:04d}_c{c:04d}.tif"
                transform = img_ds.window_transform(win)

                with rasterio.open(
                    out / split / "images" / name, "w", driver="GTiff",
                    height=cs, width=cs, count=img_ds.count,
                    dtype=chip.dtype, crs=img_ds.crs, transform=transform,
                    compress="deflate",
                ) as dst:
                    dst.write(chip)
                with rasterio.open(
                    out / split / "masks" / name, "w", driver="GTiff",
                    height=cs, width=cs, count=1, dtype="uint8",
                    crs=img_ds.crs, transform=transform, compress="deflate",
                ) as dst:
                    dst.write(mask[np.newaxis].astype(np.uint8))

                hist = {int(k): int(v) for k, v in zip(*np.unique(mask, return_counts=True))}
                class_pixels[split].update(hist)
                manifest.append({
                    "file": name, "split": split, "block": list(blk),
                    "window": [c * cs, r * cs, cs, cs], "class_histogram": hist,
                })
                kept[split] += 1
                if split == "train":
                    f = chip.astype(np.float64)
                    band_sum += f.sum(axis=(1, 2))
                    band_sqsum += (f ** 2).sum(axis=(1, 2))
                    train_pixels += cs * cs

        mean = band_sum / max(train_pixels, 1)
        std = np.sqrt(np.maximum(band_sqsum / max(train_pixels, 1) - mean ** 2, 0))
        (out / "manifest.json").write_text(json.dumps({
            "seed": SEED, "chip_size": cs, "grid": [grid_rows, grid_cols],
            "block_assignment": {f"{r},{c}": s for (r, c), s in block_of.items()},
            "chips": manifest,
        }, indent=1))
        (out / "stats.json").write_text(json.dumps({
            "computed_on": "train split only",
            "band_mean": mean.round(3).tolist(),
            "band_std": std.round(3).tolist(),
        }, indent=1))

        print(f"\nchips kept: {dict(kept)}   skipped(nodata): {skipped}")
        for split, counter in class_pixels.items():
            total = sum(counter.values()) or 1
            pct = {k: f"{100 * v / total:.1f}%" for k, v in sorted(counter.items())}
            print(f"{split:5s} class pixel share: {pct}")
            missing = set(range(NUM_CLASSES)) - set(counter)
            if missing:
                print(f"  WARNING: {split} is missing class(es) {sorted(missing)} — "
                      f"consider a different --grid or seed")
        print(f"\nnormalization stats (train only) -> {out/'stats.json'}")


if __name__ == "__main__":
    main()
