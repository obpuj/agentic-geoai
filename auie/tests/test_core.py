"""v2 tests: schema shape rules, gazetteer traps, temporal snapping,
planner re-prompt loop, and the full pipeline with a scripted LLM."""
import json

import pytest
from pydantic import ValidationError

from auie.agents.base import StubGISAgent, StubVisionAgent
from auie.gazetteer import Gazetteer
from auie.llm import ScriptedClient
from auie.pipeline import handle_query
from auie.planner import plan
from auie.router import Router
from auie.schema import PlannerOutput, TaskSpec
from auie.temporal import resolve_years

VALID = {
    "status": "ok",
    "task": {"location": "Koramangala", "analysis_type": "urban_expansion",
             "year": None, "start_year": 2020, "end_year": 2025, "parameters": {}},
    "message": None,
}


# ---------------- schema ----------------

def test_valid_spec():
    assert PlannerOutput.model_validate(VALID).task.start_year == 2020


def test_free_text_location_accepted():
    TaskSpec(location="Anywhere At All", analysis_type="trend")


def test_unknown_analysis_rejected():
    bad = json.loads(json.dumps(VALID))
    bad["task"]["analysis_type"] = "flood_risk"
    with pytest.raises(ValidationError):
        PlannerOutput.model_validate(bad)


def test_bitemporal_needs_range_and_order():
    with pytest.raises(ValidationError):
        TaskSpec(location="x", analysis_type="change_detection", year=2024)
    with pytest.raises(ValidationError):
        TaskSpec(location="x", analysis_type="change_detection",
                 start_year=2025, end_year=2020)


def test_single_year_rejects_range():
    with pytest.raises(ValidationError):
        TaskSpec(location="x", analysis_type="land_cover",
                 start_year=2020, end_year=2025)


def test_ok_requires_task_and_rejection_requires_message():
    with pytest.raises(ValidationError):
        PlannerOutput.model_validate({"status": "ok", "task": None, "message": None})
    with pytest.raises(ValidationError):
        PlannerOutput.model_validate(
            {"status": "out_of_scope", "task": None, "message": None})


# ---------------- gazetteer ----------------

def test_exact_ward(wards_file):
    r = Gazetteer(wards_file).resolve("Jayanagar")
    assert r.status == "ok" and r.location.ward_names == ["Jayanagar"]


def test_substring_trap_no_vijayanagar_bleed(wards_file):
    """'Jayanagar' must resolve to Jayanagar, never Vijayanagar."""
    r = Gazetteer(wards_file).resolve("jayanagar")
    assert r.status == "ok" and r.location.name == "Jayanagar"


def test_transliteration_variant(wards_file):
    r = Gazetteer(wards_file).resolve("Bellandur")   # file has 'Bellanduru'
    assert r.status == "ok" and r.location.ward_names == ["Bellanduru"]


def test_alias_locality_whitefield(wards_file):
    r = Gazetteer(wards_file).resolve("Whitefield")
    assert r.status == "ok"
    assert set(r.location.ward_names) == {
        "Kadugodi",
        "Hagadur",
        "Garudachar Playa",
        "Hudi",
    }
    assert r.location.note  # interpretation must be disclosed


def test_known_outside_electronic_city(wards_file):
    r = Gazetteer(wards_file).resolve("Electronic City")
    assert r.status == "out_of_region" and "BBMP" in r.message


def test_region_terms(wards_file):
    r = Gazetteer(wards_file).resolve("Bengaluru")
    assert r.status == "ok" and r.location.kind == "region"
    assert len(r.location.ward_names) == 14  # all fixture wards


def test_misspelling_fuzzy(wards_file):
    r = Gazetteer(wards_file).resolve("Kaddugodi")
    assert r.status == "ok" and r.location.ward_names == ["Kadugodi"]


def test_out_of_region_city(wards_file):
    assert Gazetteer(wards_file).resolve("Mysuru").status == "out_of_region"


# ---------------- temporal ----------------

def test_years_passthrough():
    t = resolve_years("urban_expansion", None, 2020, 2025)
    assert t.status == "ok" and t.resolved.years == [2020, 2025]
    assert t.resolved.notes == []


def test_pre_era_start_snapped_with_note():
    t = resolve_years("vegetation_change", None, 2010, 2024)
    assert t.status == "ok" and t.resolved.years == [2019, 2024]
    assert any("2019" in n for n in t.resolved.notes)


def test_wholly_pre_era_rejected():
    t = resolve_years("change_detection", None, 2010, 2015)
    assert t.status == "rejected" and "Sentinel-2" in t.message


def test_future_year_capped():
    t = resolve_years("built_up_mapping", 2030, None, None)
    assert t.status == "ok" and t.resolved.years == [2025]


def test_trend_defaults_to_all_epochs():
    t = resolve_years("trend", None, None, None)
    assert t.resolved.years == [2019, 2020, 2021, 2022, 2023, 2024, 2025]


# ---------------- planner loop ----------------

def test_planner_recovers_then_degrades():
    bad = json.dumps({"status": "ok", "task": None, "message": None})
    ok = plan("q", ScriptedClient([bad, json.dumps(VALID)]))
    assert ok.status == "ok"
    dead = plan("q", ScriptedClient(["nope", "{broken"]))
    assert dead.status == "needs_clarification" and dead.message


# ---------------- full pipeline ----------------

def _router():
    return Router({"vision": StubVisionAgent(), "gis": StubGISAgent()})


def test_pipeline_ok_with_alias_and_snapped_year(wards_file):
    spec = json.loads(json.dumps(VALID))
    spec["task"].update(location="Whitefield", start_year=2010, end_year=2024)
    res = handle_query("growth in whitefield since 2010",
                       ScriptedClient([json.dumps(spec)]),
                       Gazetteer(wards_file), _router())
    assert res.status == "ok"
    assert set(res.resolved.location.ward_names) == {
        "Kadugodi",
        "Hagadur",
        "Garudachar Playa",
        "Hudi",
    }
    assert res.resolved.years.years == [2019, 2024]
    assert len(res.resolved.notes) >= 2  # alias note + snap note
    assert "resolution_notes" in res.bundle.provenance


def test_pipeline_rejects_outside_location(wards_file):
    spec = json.loads(json.dumps(VALID))
    spec["task"]["location"] = "Mysuru"
    res = handle_query("growth in mysuru", ScriptedClient([json.dumps(spec)]),
                       Gazetteer(wards_file), _router())
    assert res.status == "rejected" and "198" in res.message


def test_pipeline_surfaces_planner_rejection(wards_file):
    oos = {"status": "out_of_scope", "task": None, "message": "Flood risk is not supported."}
    res = handle_query("flood risk?", ScriptedClient([json.dumps(oos)]),
                       Gazetteer(wards_file), _router())
    assert res.status == "rejected" and "Flood" in res.message

# ---------------- llm retry ----------------

def test_retry_on_429_then_success(monkeypatch):
    import auie.llm as llm

    class Resp:
        def __init__(self, code, payload=None, retry_after=None):
            self.status_code = code
            self.headers = {"Retry-After": retry_after} if retry_after else {}
            self._payload = payload or {}
        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")
        def json(self):
            return self._payload

    calls = {"n": 0}
    def fake_post(url, **kw):
        calls["n"] += 1
        if calls["n"] < 3:
            return Resp(429, retry_after="0")
        return Resp(200, {"ok": True})

    monkeypatch.setattr(llm.requests, "post", fake_post)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    resp = llm._post_with_retry("http://x")
    assert resp.json() == {"ok": True} and calls["n"] == 3


def test_retry_gives_up_after_max(monkeypatch):
    import pytest as _pytest
    import auie.llm as llm

    class Resp:
        status_code = 429
        headers = {}
        def raise_for_status(self):
            raise RuntimeError("HTTP 429")
        def json(self):
            return {}

    monkeypatch.setattr(llm.requests, "post", lambda url, **kw: Resp())
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    with _pytest.raises(RuntimeError):
        llm._post_with_retry("http://x")
