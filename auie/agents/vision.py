"""Vision Agent — clip-and-count over precomputed annual rasters.

No inference at query time (standing rule 2): wall-to-wall predictions are
precomputed per epoch; this agent clips them by the resolved ward geometry
and computes deterministic statistics.

Expected layout (AUIE_PRECOMPUTED_DIR, default data/precomputed/):
    landcover_{year}.tif   uint8, classes 0=background 1=vegetation
                           2=water 3=built-up, 255=nodata
    ndvi_{year}.tif        float32, nodata=nan   (optional but expected)
    ndbi_{year}.tif        float32, nodata=nan   (optional)
    meta.json              {"model": ..., "model_version": ...,
                            "composite_window": "Jan-Mar", ...}

Every per-year stat carries valid_pixel_fraction: cloud-gap honesty is a
provenance obligation, not an implementation detail.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask

from ..gazetteer import Gazetteer
from .base import AgentResult, BaseAgent

if TYPE_CHECKING:
    from ..pipeline import ResolvedTask

CLASS_NAMES = {0: "background", 1: "vegetation", 2: "water", 3: "built_up"}
NODATA = 255
PIXEL_KM2 = (10 * 10) / 1_000_000  # 10 m pixels -> km^2


class VisionAgent(BaseAgent):
    name = "vision"

    def __init__(self, gazetteer: Gazetteer, precomputed_dir: str | None = None):
        self.gazetteer = gazetteer
        self.dir = Path(precomputed_dir
                        or os.environ.get("AUIE_PRECOMPUTED_DIR", "data/precomputed"))
        meta_path = self.dir / "meta.json"
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    # -- internals --------------------------------------------------------
    def _path(self, product: str, year: int) -> Path:
        p = self.dir / f"{product}_{year}.tif"
        if not p.exists():
            raise FileNotFoundError(
                f"Precomputed raster missing: {p}. The temporal resolver "
                f"offered year {year}; either precompute it or remove it "
                f"from config.AVAILABLE_YEARS.")
        return p

    def _clip(self, path: Path, geom_wgs84) -> tuple[np.ndarray, float]:
        """Clip raster by geometry; return (array, valid_fraction_in_geom)."""
        with rasterio.open(path) as ds:
            import geopandas as gpd
            g = gpd.GeoSeries([geom_wgs84], crs=self.gazetteer.gdf.crs
                              ).to_crs(ds.crs).iloc[0]
            arr, _ = rio_mask(ds, [g], crop=True, nodata=NODATA, filled=True)
        arr = arr[0]
        in_geom = arr != NODATA  # outside-geometry fill is also NODATA
        # NB: with a single NODATA sentinel, cloud-gap nodata inside the
        # geometry and outside-geometry fill are indistinguishable here;
        # valid_fraction is therefore relative to the clip bounding box's
        # in-geometry estimate computed from geometry area instead.
        geom_km2 = self._geom_area_km2(g)
        valid_km2 = float(in_geom.sum()) * PIXEL_KM2
        valid_fraction = min(valid_km2 / geom_km2, 1.0) if geom_km2 > 0 else 0.0
        return arr, round(valid_fraction, 4)

    @staticmethod
    def _geom_area_km2(geom_projected) -> float:
        return float(geom_projected.area) / 1_000_000

    def _landcover_stats(self, year: int, geom) -> dict:
        arr, valid_fraction = self._clip(self._path("landcover", year), geom)
        valid = arr[arr != NODATA]
        counts = {name: int((valid == cid).sum()) for cid, name in CLASS_NAMES.items()}
        total = max(valid.size, 1)
        return {
            "year": year,
            "area_km2": {n: round(c * PIXEL_KM2, 4) for n, c in counts.items()},
            "share": {n: round(c / total, 4) for n, c in counts.items()},
            "valid_pixel_fraction": valid_fraction,
        }

    def _index_mean(self, product: str, year: int, geom) -> float | None:
        try:
            arr, _ = self._clip(self._path(product, year), geom)
        except FileNotFoundError:
            return None
        vals = arr[np.isfinite(arr) & (arr != NODATA)]
        return round(float(vals.mean()), 4) if vals.size else None

    # -- contract ---------------------------------------------------------
    def run(self, resolved: "ResolvedTask") -> AgentResult:
        geom = self.gazetteer.geometry_for(resolved.location)
        years = resolved.years.years
        per_year = {y: self._landcover_stats(y, geom) for y in years}
        stats: dict = {"per_year": per_year}

        analysis = resolved.task.analysis_type.value
        if len(years) >= 2:
            first, last = per_year[years[0]], per_year[years[-1]]
            b0 = first["area_km2"]["built_up"]
            b1 = last["area_km2"]["built_up"]
            stats["built_up_change_km2"] = round(b1 - b0, 4)
            stats["built_up_change_pct"] = (
                round(100 * (b1 - b0) / b0, 2) if b0 > 0 else None)
            v0 = first["area_km2"]["vegetation"]
            v1 = last["area_km2"]["vegetation"]
            stats["vegetation_change_km2"] = round(v1 - v0, 4)
        if analysis in ("vegetation_change", "trend"):
            stats["ndvi_mean_by_year"] = {
                y: self._index_mean("ndvi", y, geom) for y in years}
        if analysis in ("change_detection", "urban_expansion", "trend"):
            stats["ndbi_mean_by_year"] = {
                y: self._index_mean("ndbi", y, geom) for y in years}

        return AgentResult(
            agent=self.name,
            layers=[{"type": "raster_ref",
                     "product": "landcover",
                     "years": years,
                     "wards": resolved.location.ward_names}],
            stats=stats,
            provenance={
                "model": self.meta.get("model", "unknown"),
                "model_version": self.meta.get("model_version", "unknown"),
                "composite_window": self.meta.get("composite_window", "Jan-Mar"),
                "label_source": self.meta.get(
                    "label_source", "Dynamic World-derived supervision"),
                "valid_pixel_fraction_by_year": {
                    y: per_year[y]["valid_pixel_fraction"] for y in years},
            },
        )
