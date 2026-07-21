"""Chip dataset + class weights — shared by EVERY backbone (Gate C fairness).

Framework-light by design: core logic returns numpy; the torch Dataset is a
thin wrapper at the bottom. All backbones consume the SAME normalization
(train-only stats.json), the SAME crops, the SAME weights — so Gate C
compares architectures, not data pipelines.

Band order on disk (7 bands): B02 B03 B04 B08 B8A B11 B12
  RGB models (SegFormer 3-ch experiments):  indices [2, 1, 0]   (R,G,B)
  All-band models (SegFormer-7ch, U-Net):   indices [0..6]
  Prithvi-EO-2.0 (6-band HLS convention):   indices [0, 1, 2, 4, 5, 6]
                                            (B02 B03 B04 B8A B11 B12)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import rasterio

PRITHVI_BANDS = [0, 1, 2, 4, 5, 6]
ALL_BANDS = list(range(7))
RGB_BANDS = [2, 1, 0]
NUM_CLASSES = 4
IGNORE_INDEX = 255


def load_stats(stats_json: str | Path) -> tuple[np.ndarray, np.ndarray]:
    s = json.loads(Path(stats_json).read_text())
    return (np.array(s["band_mean"], np.float32),
            np.array(s["band_std"], np.float32))


def class_weights_from_manifest(manifest_json: str | Path,
                                split: str = "train") -> np.ndarray:
    """Inverse-frequency weights, normalized to mean 1.0.

    With the real Gate A distribution (~70/25/2/2) built-up gets ~0.05x and
    water/background ~9x — the anti-collapse measure the U-Net bug taught us.
    """
    m = json.loads(Path(manifest_json).read_text())
    counts = np.zeros(NUM_CLASSES, np.float64)
    for chip in m["chips"]:
        if chip["split"] != split:
            continue
        for cid, n in chip["class_histogram"].items():
            cid = int(cid)
            if 0 <= cid < NUM_CLASSES:
                counts[cid] += n
    freq = counts / counts.sum()
    w = 1.0 / np.maximum(freq, 1e-6)
    return (w / w.mean()).astype(np.float32)


class ChipSamples:
    """Numpy-level chip reader: normalize, band-select, crop, augment."""

    def __init__(self, chips_dir: str | Path, split: str, stats_json: str | Path,
                 band_indices: list[int] = ALL_BANDS, crop: int = 224,
                 augment: bool = False, samples_per_chip: int = 4,
                 seed: int = 42):
        base = Path(chips_dir) / split
        self.images = sorted((base / "images").glob("*.tif"))
        if not self.images:
            raise FileNotFoundError(f"No chips under {base}/images")
        self.masks_dir = base / "masks"
        mean, std = load_stats(stats_json)
        self.mean = mean[band_indices].reshape(-1, 1, 1)
        self.std = np.maximum(std[band_indices], 1e-6).reshape(-1, 1, 1)
        self.bands = band_indices
        self.crop = crop
        self.augment = augment
        # train: several random crops per 512-chip; eval: deterministic grid
        self.samples_per_chip = samples_per_chip if augment else \
            max(1, (512 // crop) ** 2)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.images) * self.samples_per_chip

    def _read(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        img_path = self.images[idx]
        with rasterio.open(img_path) as ds:
            img = ds.read().astype(np.float32)
        with rasterio.open(self.masks_dir / img_path.name) as ds:
            mask = ds.read(1).astype(np.int64)
        return img, mask

    def get(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        chip_idx, sample_idx = divmod(i, self.samples_per_chip)
        img, mask = self._read(chip_idx)
        c, size = self.crop, img.shape[1]
        if self.augment:
            y = self.rng.randint(0, size - c)
            x = self.rng.randint(0, size - c)
        else:  # deterministic tiling for val/test reproducibility
            per_row = max(1, size // c)
            y = (sample_idx // per_row) * c
            x = (sample_idx % per_row) * c
            y, x = min(y, size - c), min(x, size - c)
        img, mask = img[:, y:y + c, x:x + c], mask[y:y + c, x:x + c]
        mask = np.where(img.sum(axis=0) == 0, 255, mask)   # nodata -> ignore
        img = (img[self.bands] - self.mean) / self.std
        if self.augment:
            if self.rng.random() < 0.5:
                img, mask = img[:, :, ::-1], mask[:, ::-1]
            if self.rng.random() < 0.5:
                img, mask = img[:, ::-1, :], mask[::-1, :]
            k = self.rng.randint(0, 3)
            if k:
                img = np.rot90(img, k, axes=(1, 2))
                mask = np.rot90(mask, k)
        return np.ascontiguousarray(img), np.ascontiguousarray(mask)


try:  # thin torch wrapper — kit stays importable without torch installed
    import torch
    from torch.utils.data import Dataset

    class ChipDataset(Dataset):
        def __init__(self, *args, **kwargs):
            self.samples = ChipSamples(*args, **kwargs)

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, i):
            img, mask = self.samples.get(i)
            return torch.from_numpy(img), torch.from_numpy(mask)
except ImportError:  # pragma: no cover
    pass
