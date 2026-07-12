"""AUIE configuration — v2 (city-wide, multi-year).

Locations are no longer an enum: the gazetteer (gazetteer.py) resolves free-
text place names against the BBMP ward file. Years are annual epochs matching
the precomputed raster series.
"""
import os

# Annual epochs with precomputed rasters (Jan-Mar dry-season composites).
# Extend as wall-to-wall precompute completes; the temporal resolver snaps
# requests to this list.
AVAILABLE_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

# Path to the BBMP ward boundaries (198-ward file, WARD_NAME column).
WARDS_GEOJSON = os.environ.get("AUIE_WARDS_GEOJSON", "data/BBMP_oldWards.geojson")
WARD_NAME_COLUMN = "WARD_NAME"
WARD_NO_COLUMN = "WARD_NO"

ANALYSIS_TYPES: dict[str, str] = {
    "built_up_mapping": "Built-up / impervious surface map for one year",
    "land_cover": "4-class land cover map for one year",
    "change_detection": "Built-up change between two years",
    "urban_expansion": "Urban expansion statistics between two years",
    "vegetation_change": "Vegetation (NDVI) change between two years",
    "trend": "Year-by-year trajectory across all available epochs",
}

BITEMPORAL = {"change_detection", "urban_expansion", "vegetation_change"}
SINGLE_YEAR = {"built_up_mapping", "land_cover"}
MULTI_YEAR = {"trend"}
