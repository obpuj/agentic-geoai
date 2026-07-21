"""Build the knowledge corpus from real project artifacts (run once, rerun on
changes):

    python -m auie.build_corpus --out data/kb

Sources (all things that actually exist in the repo/data by now):
  * precomputed/meta.json               -> model + composite provenance chunk
  * prithvi_test_metrics.json           -> model accuracy chunk (per-class IoU)
  * segformer_test_metrics.json         -> comparison/benchmark chunk
  * BBMP_oldWards.geojson               -> one chunk per ward (name, area, zone)
  * gazetteer ALIASES                   -> one chunk per locality alias
  * docs/*.md (optional)                -> report sections, chunked by heading

Everything the reporter cites must trace to one of these chunks or to the
evidence bundle — that is the whole point of the Knowledge Agent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config
from .gazetteer import ALIASES
from .knowledge import Chunk, KnowledgeBase


def build_chunks(precomputed_dir: str, wards_geojson: str,
                 prithvi_metrics: str | None = None,
                 segformer_metrics: str | None = None,
                 docs_dir: str | None = None) -> list[Chunk]:
    chunks: list[Chunk] = []

    meta_path = Path(precomputed_dir) / "meta.json"
    if meta_path.exists():
        m = json.loads(meta_path.read_text())
        chunks.append(Chunk(
            "provenance-model",
            f"Analysis rasters were produced by {m.get('model')} "
            f"(version {m.get('model_version')}) from Sentinel-2 "
            f"{m.get('composite_window')} dry-season median composites, "
            f"years {sorted(m.get('years_processed', []))}. Supervision: "
            f"{m.get('label_source')}. Class scheme: 0 background/bare, "
            f"1 vegetation, 2 waterbody, 3 built-up, at 10 m resolution.",
            {"kind": "provenance"}))

    if prithvi_metrics and Path(prithvi_metrics).exists():
        t = json.loads(Path(prithvi_metrics).read_text())
        chunks.append(Chunk(
            "accuracy-prithvi",
            "Deployed model accuracy on the geographic holdout (agreement "
            "with Dynamic World-derived labels): "
            f"mIoU {t.get('test/mIoU', 0):.3f}; per-class IoU background "
            f"{t.get('test/IoU_0', 0):.3f}, vegetation {t.get('test/IoU_1', 0):.3f}, "
            f"water {t.get('test/IoU_2', 0):.3f}, built-up {t.get('test/IoU_3', 0):.3f}. "
            "These measure agreement with product-derived labels, not ground truth.",
            {"kind": "accuracy", "model": "prithvi"}))

    if segformer_metrics and Path(segformer_metrics).exists():
        s = json.loads(Path(segformer_metrics).read_text())
        iou = s.get("per_class_iou", [0, 0, 0, 0])
        chunks.append(Chunk(
            "accuracy-segformer",
            f"Benchmark comparison: SegFormer MiT-B0 (7-band, ImageNet init) "
            f"scored mIoU {s.get('miou', 0):.3f} on the same holdout "
            f"(background {iou[0]:.3f}, vegetation {iou[1]:.3f}, water "
            f"{iou[2]:.3f}, built-up {iou[3]:.3f}); the deployed Prithvi "
            f"model scored higher on all four classes.",
            {"kind": "accuracy", "model": "segformer"}))

    chunks.append(Chunk(
        "coverage",
        f"Spatial coverage: the 198 BBMP wards of Bengaluru. Temporal "
        f"coverage: annual epochs {config.AVAILABLE_YEARS[0]}-"
        f"{config.AVAILABLE_YEARS[-1]} (Sentinel-2 era; earlier years are "
        f"not servable). Electronic City lies largely outside the BBMP "
        f"boundary and is not covered.",
        {"kind": "coverage"}))

    import geopandas as gpd
    gdf = gpd.read_file(wards_geojson)
    for _, row in gdf.iterrows():
        name = row[config.WARD_NAME_COLUMN]
        area = row.get("AREA_SQ_KM")
        zone = row.get("ASS_CONST1")
        chunks.append(Chunk(
            f"ward-{str(name).lower().replace(' ', '-')}",
            f"Ward {name}: official area "
            f"{area:.2f} sq km, assembly constituency {zone}." if area
            else f"Ward {name}, assembly constituency {zone}.",
            {"kind": "ward", "ward": str(name)}))

    for alias, wards in ALIASES.items():
        chunks.append(Chunk(
            f"alias-{alias.replace(' ', '-')}",
            f"The locality '{alias.title()}' is interpreted as ward(s): "
            f"{', '.join(wards)}. Statistics for it aggregate those wards.",
            {"kind": "alias"}))

    if docs_dir and Path(docs_dir).exists():
        for md in sorted(Path(docs_dir).glob("*.md")):
            section, lines = "intro", []
            for line in md.read_text().splitlines():
                if line.startswith("#"):
                    if lines:
                        chunks.append(Chunk(f"doc-{md.stem}-{section}"[:80],
                                            "\n".join(lines)[:1200],
                                            {"kind": "doc", "file": md.name}))
                    section, lines = line.strip("# ").lower()[:40], []
                elif line.strip():
                    lines.append(line)
            if lines:
                chunks.append(Chunk(f"doc-{md.stem}-{section}"[:80],
                                    "\n".join(lines)[:1200],
                                    {"kind": "doc", "file": md.name}))
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/kb")
    ap.add_argument("--precomputed", default=None)
    ap.add_argument("--wards", default=None)
    ap.add_argument("--prithvi-metrics", default="prithvi_test_metrics.json")
    ap.add_argument("--segformer-metrics", default="segformer_test_metrics.json")
    ap.add_argument("--docs", default="docs")
    args = ap.parse_args()

    import os
    pre = args.precomputed or os.environ.get("AUIE_PRECOMPUTED_DIR", "data/precomputed")
    wards = args.wards or config.WARDS_GEOJSON
    chunks = build_chunks(pre, wards, args.prithvi_metrics,
                          args.segformer_metrics, args.docs)
    print(f"{len(chunks)} chunks")
    kb = KnowledgeBase(chunks)  # default embedder: all-MiniLM-L6-v2
    kb.save(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
