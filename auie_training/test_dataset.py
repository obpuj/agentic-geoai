"""Offline tests for the shared dataset (numpy level — no torch needed)."""
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from dataset import ChipSamples, class_weights_from_manifest, PRITHVI_BANDS


@pytest.fixture(scope="module")
def chips(tmp_path_factory):
    root = tmp_path_factory.mktemp("chips")
    t = from_origin(700000, 1430000, 10, 10)
    rng = np.random.default_rng(1)
    for split, n in [("train", 3), ("val", 1)]:
        (root / split / "images").mkdir(parents=True)
        (root / split / "masks").mkdir(parents=True)
        for i in range(n):
            img = rng.integers(100, 4000, (7, 512, 512)).astype("uint16")
            mask = rng.integers(0, 4, (512, 512)).astype("uint8")
            for sub, arr, dt, cnt in [("images", img, "uint16", 7),
                                      ("masks", mask[None], "uint8", 1)]:
                with rasterio.open(root / split / sub / f"chip_{i}.tif", "w",
                                   driver="GTiff", height=512, width=512,
                                   count=cnt, dtype=dt, crs="EPSG:32643",
                                   transform=t) as ds:
                    ds.write(arr if cnt > 1 else arr)
    (root / "stats.json").write_text(json.dumps(
        {"band_mean": [2000.0] * 7, "band_std": [900.0] * 7}))
    (root / "manifest.json").write_text(json.dumps({"chips": [
        {"split": "train",
         "class_histogram": {"0": 19, "1": 253, "2": 22, "3": 706}}]}))
    return root


def test_shapes_normalization_and_bands(chips):
    s = ChipSamples(chips, "train", chips / "stats.json",
                    band_indices=PRITHVI_BANDS, crop=224, augment=True)
    img, mask = s.get(0)
    assert img.shape == (6, 224, 224) and mask.shape == (224, 224)
    assert abs(float(img.mean())) < 0.5  # roughly zero-centered
    assert img.dtype == np.float32 and mask.dtype == np.int64


def test_eval_tiling_is_deterministic(chips):
    s1 = ChipSamples(chips, "val", chips / "stats.json", crop=224, augment=False)
    s2 = ChipSamples(chips, "val", chips / "stats.json", crop=224, augment=False)
    a, b = s1.get(3), s2.get(3)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert len(s1) == 1 * 4  # 512//224 = 2 -> 2x2 grid per chip


def test_class_weights_fight_the_imbalance(chips):
    w = class_weights_from_manifest(chips / "manifest.json")
    assert w.shape == (4,)
    assert w[3] < 0.2          # built-up (70%) heavily down-weighted
    assert w[0] > 1 and w[2] > 1  # rare classes up-weighted
    assert abs(float(w.mean()) - 1.0) < 1e-5
