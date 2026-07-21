"""Dashboard API tests: query flow with scripted LLM + fixture wards,
layer rendering against a synthetic precomputed raster."""
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from auie.agents.base import StubGISAgent, StubVisionAgent
from auie.gazetteer import Gazetteer
from auie.knowledge import Chunk, KnowledgeBase
from auie.llm import ScriptedClient
from auie.router import Router
from auie.webapp import create_app

X0, Y0, N = 700000, 1430000, 200


@pytest.fixture()
def synth_precomputed(tmp_path):
    t = from_origin(X0, Y0, 10, 10)
    arr = np.ones((N, N), np.uint8) * 3
    arr[:50, :50] = 2
    with rasterio.open(tmp_path / "landcover_2024.tif", "w", driver="GTiff",
                       height=N, width=N, count=1, dtype="uint8",
                       crs="EPSG:32643", transform=t, nodata=255) as ds:
        ds.write(arr[np.newaxis])
    return str(tmp_path)


@pytest.fixture()
def raster_gaz(tmp_path_factory):
    import geopandas as gpd
    from shapely.geometry import box
    ward = gpd.GeoDataFrame(
        {"WARD_NAME": ["Testward"], "WARD_NO": [1]},
        geometry=[box(X0, Y0 - N * 10, X0 + N * 10, Y0)],
        crs="EPSG:32643").to_crs("EPSG:4326")
    p = tmp_path_factory.mktemp("gw") / "wards.geojson"
    ward.to_file(p, driver="GeoJSON")
    return Gazetteer(str(p))


def fake_embedder(texts):
    out = np.zeros((len(texts), 32), np.float32)
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            out[i, hash(tok) % 32] += 1
    return out


def _app(wards_file, precomputed="/nonexistent", planner_json=None,
         answer="Grounded answer text."):
    spec = planner_json or {
        "status": "ok",
        "task": {"location": "Jayanagar", "analysis_type": "built_up_mapping",
                 "year": 2024, "start_year": None, "end_year": None,
                 "parameters": {}},
        "message": None}
    client = ScriptedClient([json.dumps(spec), answer])
    gaz = Gazetteer(wards_file)
    router = Router({"vision": StubVisionAgent(), "gis": StubGISAgent()})
    kb = KnowledgeBase([Chunk("accuracy-prithvi", "model accuracy miou")],
                       embed_fn=fake_embedder)
    return create_app(client=client, gazetteer=gaz, router=router, kb=kb,
                      precomputed_dir=precomputed)


def test_query_ok_returns_spec_answer_geometry(wards_file):
    app = _app(wards_file)
    r = app.test_client().post("/api/query",
                               json={"question": "buildings in jayanagar 2024"})
    d = r.get_json()
    assert r.status_code == 200 and d["status"] == "ok"
    assert d["spec"]["location"] == "Jayanagar"
    assert d["spec"]["years"] == [2024]
    assert d["answer"] == "Grounded answer text."
    assert d["cited"] == ["accuracy-prithvi"]
    assert d["geometry"]["type"] in ("Polygon", "MultiPolygon")


def test_query_rejection_passthrough(wards_file):
    oos = {"status": "out_of_scope", "task": None,
           "message": "Flood risk is not supported."}
    app = _app(wards_file, planner_json=oos)
    d = app.test_client().post("/api/query",
                               json={"question": "flood?"}).get_json()
    assert d["status"] == "rejected" and "Flood" in d["message"]


def test_query_empty_400(wards_file):
    app = _app(wards_file)
    assert app.test_client().post("/api/query", json={}).status_code == 400


def test_layer_renders_png_with_bounds(raster_gaz, synth_precomputed):
    app = create_app(client=ScriptedClient([]), gazetteer=raster_gaz,
                     router=Router({"vision": StubVisionAgent(),
                                    "gis": StubGISAgent()}),
                     kb=KnowledgeBase([Chunk("x", "y")], embed_fn=fake_embedder),
                     precomputed_dir=synth_precomputed)
    r = app.test_client().get("/api/layer?year=2024&loc=Testward")
    d = r.get_json()
    assert r.status_code == 200
    assert d["image"].startswith("data:image/png;base64,")
    (s, w), (n, e) = d["bounds"]
    assert s < n and w < e and 12 < s < 14 and 76 < w < 79  # sane 4326 box


def test_layer_missing_year_404(raster_gaz, synth_precomputed):
    app = create_app(client=ScriptedClient([]), gazetteer=raster_gaz,
                     router=Router({"vision": StubVisionAgent(),
                                    "gis": StubGISAgent()}),
                     kb=KnowledgeBase([Chunk("x", "y")], embed_fn=fake_embedder),
                     precomputed_dir=synth_precomputed)
    assert app.test_client().get(
        "/api/layer?year=2019&loc=Testward").status_code == 404
