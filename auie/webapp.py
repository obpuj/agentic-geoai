"""AUIE dashboard (pipeline stage 10) — Flask app.

Run:  python -m auie.webapp        (port 5050, matching the existing app)

Endpoints:
  GET  /            single-page dashboard
  POST /api/query   {"question": ...} -> spec, stats, answer, cited, geometry
  GET  /api/layer   ?year=&loc=       -> {"image": dataURL png, "bounds": [[s,w],[n,e]]}

All dependencies are injectable via create_app() so tests run with scripted
LLMs, fixture wards, and synthetic rasters. The layer endpoint re-resolves
the location text through the gazetteer (stateless, deterministic) rather
than trusting client-supplied ward lists.
"""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request

# class id -> RGBA (built amber, vegetation green, water blue, bare grey)
PALETTE = {
    0: (138, 143, 152, 200),
    1: (87, 167, 115, 210),
    2: (74, 144, 194, 220),
    3: (224, 168, 60, 210),
}
NODATA = 255


def _geometry_geojson(gazetteer, loc) -> dict:
    """Dissolved, simplified geometry for the map outline (EPSG:4326)."""
    geom = gazetteer.geometry_for(loc).simplify(0.0004)
    import shapely.geometry as sg
    return sg.mapping(geom)


def _render_layer(precomputed_dir: str, year: int, gazetteer, loc) -> dict:
    import geopandas as gpd
    import rasterio
    from PIL import Image
    from rasterio.mask import mask as rio_mask
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds as window_from_bounds

    path = Path(precomputed_dir) / f"landcover_{year}.tif"
    if not path.exists():
        raise FileNotFoundError(f"landcover_{year}.tif not in precomputed dir")
    geom = gazetteer.geometry_for(loc)
    with rasterio.open(path) as ds:
        g = gpd.GeoSeries([geom], crs=gazetteer.gdf.crs).to_crs(ds.crs).iloc[0]
        arr, transform = rio_mask(ds, [g], crop=True, nodata=NODATA, filled=True)
        # arr = arr[0]
        # h, w = arr.shape
        # left, top = transform * (0, 0)
        # right, bottom = transform * (w, h)
        # s, w4, n, e4 = None, None, None, None
        # w4, s, e4, n = transform_bounds(ds.crs, "EPSG:4326",
        #                                 left, bottom, right, top)
        arr = arr[0]
        h, w = arr.shape
        left, top = transform * (0, 0)
        right, bottom = transform * (w, h)
        w4, s, e4, n = transform_bounds(ds.crs, "EPSG:4326",
                                        left, bottom, right, top)
    step = max(1, -(-max(arr.shape) // 1400))   # ceil division
    if step > 1:
        arr = arr[::step, ::step]
    h, w = arr.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for cid, color in PALETTE.items():
        rgba[arr == cid] = color
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG", optimize=True)
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return {"image": data_url, "bounds": [[s, w4], [n, e4]]}


def create_app(client=None, gazetteer=None, router=None, kb=None,
               precomputed_dir: str | None = None) -> Flask:
    app = Flask(__name__)
    pre_dir = precomputed_dir or os.environ.get("AUIE_PRECOMPUTED_DIR",
                                                "data/precomputed")

    # lazy real dependencies (tests inject their own)
    def deps():
        nonlocal client, gazetteer, router, kb
        if gazetteer is None:
            from .gazetteer import Gazetteer
            gazetteer = Gazetteer()
        if router is None:
            from .agents.base import StubGISAgent
            from .agents.vision import VisionAgent
            from .router import Router
            router = Router({"vision": VisionAgent(gazetteer, pre_dir),
                             "gis": StubGISAgent()})
        if client is None:
            from .llm import default_client
            client = default_client()
        if kb is None:
            from .knowledge import KnowledgeBase
            kb = KnowledgeBase.load(os.environ.get("AUIE_KB_DIR", "data/kb"))
        return client, gazetteer, router, kb

    @app.get("/")
    def index():
        return render_template("dashboard.html")

    @app.post("/api/query")
    def api_query():
        question = (request.get_json(silent=True) or {}).get("question", "").strip()
        if not question:
            return jsonify({"status": "clarify",
                            "message": "Type a question to begin."}), 400
        c, g, r, k = deps()
        from .pipeline import handle_query
        res = handle_query(question, c, g, r)
        if res.status != "ok":
            return jsonify({"status": res.status, "message": res.message,
                            "candidates": res.candidates})
        from .reporter import report
        try:
            answer, cited = report(question, res.bundle, k, c)
        except Exception as e:  # answer must never take the map down with it
            answer, cited = f"(explanation unavailable: {type(e).__name__})", []
        rt = res.resolved
        return jsonify({
            "status": "ok",
            "spec": {
                "location": rt.location.name,
                "kind": rt.location.kind,
                "wards": rt.location.ward_names,
                "analysis": rt.task.analysis_type.value,
                "years": rt.years.years,
                "notes": rt.notes,
            },
            "stats": res.bundle.stats,
            "provenance": res.bundle.provenance,
            "answer": answer,
            "cited": cited,
            "geometry": _geometry_geojson(g, rt.location),
            "loc_query": ("bengaluru" if rt.location.kind == "region"
                          else rt.location.name),
        })

    @app.get("/api/layer")
    def api_layer():
        year = request.args.get("year", type=int)
        loc_text = request.args.get("loc", "")
        if not year or not loc_text:
            return jsonify({"error": "year and loc required"}), 400
        _, g, _, _ = deps()
        resolution = g.resolve(loc_text)
        if resolution.status != "ok":
            return jsonify({"error": resolution.message}), 400
        try:
            payload = _render_layer(pre_dir, year, g, resolution.location)
        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 404
        return jsonify(payload)

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5050, debug=False)
