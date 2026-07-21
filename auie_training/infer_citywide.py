"""Wall-to-wall precompute: composite GeoTIFFs -> landcover/ndvi/ndbi rasters.

Runs AFTER Gate C, once per year, with whichever backbone won:
    python infer_citywide.py --composite s2_bbmp_2024_janmar.tif --year 2024 \
        --model /path/to/best --stats chips/stats.json --out data/precomputed \
        --arch segformer --bands all

Produces exactly the layout the Vision Agent reads:
    landcover_{year}.tif  (uint8 0..3, 255 nodata)
    ndvi_{year}.tif, ndbi_{year}.tif  (float32, from the composite directly —
                                       model-free, defensible for all years)
Also updates meta.json with model identity for provenance.

Sliding window with overlap + center-cropped stitching to kill tile-edge
artifacts. ~50 km x ~40 km at 10 m runs in minutes on a Kaggle GPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import torch

from dataset import ALL_BANDS, PRITHVI_BANDS, RGB_BANDS, load_stats

TILE, OVERLAP = 512, 64
NODATA_OUT = 255
BANDS = {"all": ALL_BANDS, "rgb": RGB_BANDS, "prithvi": PRITHVI_BANDS}
# composite band order: B02 B03 B04 B08 B8A B11 B12
B04, B08, B11 = 2, 3, 5


def load_model(arch: str, path: str, n_bands: int, device: str):
    if arch == "segformer":
        from transformers import SegformerForSemanticSegmentation
        m = SegformerForSemanticSegmentation.from_pretrained(path)
        m.to(device).eval()

        def forward(x):
            logits = m(pixel_values=x).logits
            return torch.nn.functional.interpolate(
                logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return forward
    if arch == "prithvi":
        # TerraTorch checkpoint: exported as torchscript for portability
        m = torch.jit.load(path, map_location=device).eval()
        return lambda x: m(x)
    raise ValueError(arch)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--composite", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", choices=["segformer", "prithvi"], required=True)
    ap.add_argument("--bands", choices=list(BANDS), default="all")
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--model-version", default="gate_c_winner")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    band_idx = BANDS[args.bands]
    mean, std = load_stats(args.stats)
    mean = mean[band_idx].reshape(-1, 1, 1)
    std = np.maximum(std[band_idx], 1e-6).reshape(-1, 1, 1)
    forward = load_model(args.arch, args.model, len(band_idx), device)

    with rasterio.open(args.composite) as ds:
        H, W = ds.height, ds.width
        profile = ds.profile
        img = ds.read().astype(np.float32)  # (7,H,W); ~1-2GB, fine on Kaggle

    valid = img.sum(axis=0) > 0
    pred = np.full((H, W), NODATA_OUT, np.uint8)
    step = TILE - 2 * OVERLAP
    with torch.no_grad():
        for y0 in range(0, H, step):
            for x0 in range(0, W, step):
                ys, xs = max(0, y0 - OVERLAP), max(0, x0 - OVERLAP)
                ye, xe = min(H, ys + TILE), min(W, xs + TILE)
                ys, xs = max(0, ye - TILE), max(0, xe - TILE)
                tile = (img[band_idx, ys:ye, xs:xe] - mean) / std
                t = torch.from_numpy(tile[None]).to(device)
                logits = forward(t)[0].cpu().numpy()
                classes = logits.argmax(0).astype(np.uint8)
                # write only the interior (center crop) of each tile
                iy0 = y0 if y0 == 0 else y0
                cy0, cx0 = y0 - ys, x0 - xs
                cy1 = min(cy0 + step, ye - ys)
                cx1 = min(cx0 + step, xe - xs)
                pred[y0:y0 + (cy1 - cy0), x0:x0 + (cx1 - cx0)] = \
                    classes[cy0:cy1, cx0:cx1]
    pred[~valid] = NODATA_OUT

    lc_profile = dict(profile, count=1, dtype="uint8", nodata=NODATA_OUT,
                      compress="deflate")
    with rasterio.open(out / f"landcover_{args.year}.tif", "w", **lc_profile) as ds:
        ds.write(pred[None])

    # model-free indices straight from the composite (reflectance x10000)
    def ratio(a, b):
        num, den = img[a] - img[b], img[a] + img[b]
        r = np.where(den != 0, num / np.maximum(den, 1e-6), np.nan)
        r[~valid] = np.nan
        return r.astype(np.float32)

    idx_profile = dict(profile, count=1, dtype="float32", nodata=np.nan,
                       compress="deflate")
    with rasterio.open(out / f"ndvi_{args.year}.tif", "w", **idx_profile) as ds:
        ds.write(ratio(B08, B04)[None])
    with rasterio.open(out / f"ndbi_{args.year}.tif", "w", **idx_profile) as ds:
        ds.write(ratio(B11, B08)[None])

    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({
        "model": args.model_name or f"{args.arch}",
        "model_version": args.model_version,
        "composite_window": "Jan-Mar",
        "label_source": "Dynamic World-derived supervision",
    })
    meta.setdefault("years_processed", [])
    if args.year not in meta["years_processed"]:
        meta["years_processed"].append(args.year)
    meta_path.write_text(json.dumps(meta, indent=1))
    covered = 100 * float(valid.mean())
    print(f"{args.year}: landcover/ndvi/ndbi written to {out} "
          f"({covered:.1f}% valid pixels)")


if __name__ == "__main__":
    main()
