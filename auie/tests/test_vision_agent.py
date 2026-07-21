"""Vision Agent against synthetic precomputed rasters with known geometry:
a 'ward' polygon covering a region whose class layout we control exactly,
so every km^2 figure is verifiable by hand."""
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from auie.agents.vision import VisionAgent
from auie.gazetteer import Gazetteer
from auie.pipeline import ResolvedTask
from auie.schema import TaskSpec
from auie.temporal import ResolvedYears

# raster: EPSG:32643, 10 m pixels, 400x400 (4 km x 4 km)
X0, Y0, N = 700000, 1430000, 400


@pytest.fixture(scope="module")
def precomputed(tmp_path_factory):
    d = tmp_path_factory.mktemp("precomp")
    t = from_origin(X0, Y0, 10, 10)

    def write(name, arr, dtype):
        with rasterio.open(d / name, "w", driver="GTiff", height=N, width=N,
                           count=1, dtype=dtype, crs="EPSG:32643",
                           transform=t, nodata=255 if dtype == "uint8" else None) as ds:
            ds.write(arr[np.newaxis])

    # 2019: left half vegetation(1), right half built(3)
    lc19 = np.ones((N, N), np.uint8)
    lc19[:, N // 2:] = 3
    # 2025: everything built except top-left 100x100 water(2)
    lc25 = np.full((N, N), 3, np.uint8)
    lc25[:100, :100] = 2
    write("landcover_2019.tif", lc19, "uint8")
    write("landcover_2025.tif", lc25, "uint8")
    write("ndvi_2019.tif", np.full((N, N), 0.6, np.float32), "float32")
    write("ndvi_2025.tif", np.full((N, N), 0.2, np.float32), "float32")
    (d / "meta.json").write_text(json.dumps(
        {"model": "segformer-test", "model_version": "9.9"}))
    return str(d)


@pytest.fixture(scope="module")
def gaz(tmp_path_factory, precomputed):
    """One 'ward' square covering the WHOLE raster, in EPSG:4326."""
    import geopandas as gpd
    from shapely.geometry import box
    ward = gpd.GeoDataFrame(
        {"WARD_NAME": ["Testward"], "WARD_NO": [1]},
        geometry=[box(X0, Y0 - N * 10, X0 + N * 10, Y0)], crs="EPSG:32643"
    ).to_crs("EPSG:4326")
    p = tmp_path_factory.mktemp("g") / "wards.geojson"
    ward.to_file(p, driver="GeoJSON")
    return Gazetteer(str(p))


def _resolved(gaz_, analysis, years):
    loc = gaz_.resolve("Testward").location
    task = TaskSpec(location="Testward", analysis_type=analysis,
                    **({"year": years[0]} if len(years) == 1
                       else {"start_year": years[0], "end_year": years[-1]}))
    return ResolvedTask(task=task, location=loc,
                        years=ResolvedYears(list(years)))


def test_single_year_areas(gaz, precomputed):
    agent = VisionAgent(gaz, precomputed)
    out = agent.run(_resolved(gaz, "built_up_mapping", [2019]))
    a = out.stats["per_year"][2019]["area_km2"]
    # 400x400 px = 16 km^2; half veg half built in 2019
    assert a["vegetation"] == pytest.approx(8.0, abs=0.1)
    assert a["built_up"] == pytest.approx(8.0, abs=0.1)
    assert out.stats["per_year"][2019]["valid_pixel_fraction"] > 0.98
    assert out.provenance["model"] == "segformer-test"


def test_change_between_years(gaz, precomputed):
    agent = VisionAgent(gaz, precomputed)
    out = agent.run(_resolved(gaz, "urban_expansion", [2019, 2025]))
    # built: 8 -> 15 km^2 (16 minus 1 km^2 water corner)
    assert out.stats["built_up_change_km2"] == pytest.approx(7.0, abs=0.1)
    assert out.stats["built_up_change_pct"] == pytest.approx(87.5, abs=2)
    assert out.stats["vegetation_change_km2"] == pytest.approx(-8.0, abs=0.1)


def test_ndvi_series_for_vegetation_change(gaz, precomputed):
    agent = VisionAgent(gaz, precomputed)
    out = agent.run(_resolved(gaz, "vegetation_change", [2019, 2025]))
    nd = out.stats["ndvi_mean_by_year"]
    assert nd[2019] == pytest.approx(0.6, abs=0.01)
    assert nd[2025] == pytest.approx(0.2, abs=0.01)


def test_missing_year_fails_loudly(gaz, precomputed):
    agent = VisionAgent(gaz, precomputed)
    with pytest.raises(FileNotFoundError):
        agent.run(_resolved(gaz, "built_up_mapping", [2023]))
