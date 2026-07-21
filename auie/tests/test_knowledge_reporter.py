"""Offline tests: KnowledgeBase with a fake embedder, corpus builder against
the fixture ward file, reporter prompt discipline with a scripted LLM."""
import json

import numpy as np
import pytest

from auie.build_corpus import build_chunks
from auie.knowledge import Chunk, KnowledgeBase
from auie.llm import ScriptedClient
from auie.reporter import report
from auie.agents.base import StubGISAgent, StubVisionAgent
from auie.gazetteer import Gazetteer
from auie.pipeline import handle_query


def fake_embedder(texts):
    """Deterministic bag-of-chars embedding — enough to rank by overlap."""
    dim = 64
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            out[i, hash(tok) % dim] += 1.0
    return out


def test_kb_retrieves_relevant_chunk():
    kb = KnowledgeBase([
        Chunk("a", "prithvi model accuracy miou holdout"),
        Chunk("b", "ward jayanagar official area sq km"),
        Chunk("c", "sentinel composite window january march"),
    ], embed_fn=fake_embedder)
    top = kb.retrieve("what is the model accuracy miou", k=1)
    assert top[0].id == "a"


def test_kb_save_load_roundtrip(tmp_path):
    kb = KnowledgeBase([Chunk("x", "hello world"), Chunk("y", "goodbye moon")],
                       embed_fn=fake_embedder)
    kb.save(tmp_path / "kb")
    kb2 = KnowledgeBase.load(tmp_path / "kb", embed_fn=fake_embedder)
    assert [c.id for c in kb2.retrieve("hello world", k=1)] == ["x"]


def test_corpus_builder_from_fixtures(tmp_path, wards_file):
    pre = tmp_path / "precomputed"
    pre.mkdir()
    (pre / "meta.json").write_text(json.dumps({
        "model": "prithvi-eo-2.0-300m-frozen", "model_version": "v1-clean",
        "composite_window": "Jan-Mar",
        "label_source": "Dynamic World-derived supervision",
        "years_processed": [2019, 2025]}))
    pm = tmp_path / "prithvi.json"
    pm.write_text(json.dumps({"test/mIoU": 0.726, "test/IoU_0": 0.431,
                              "test/IoU_1": 0.808, "test/IoU_2": 0.780,
                              "test/IoU_3": 0.885}))
    chunks = build_chunks(str(pre), wards_file, str(pm), None, None)
    ids = [c.id for c in chunks]
    assert "provenance-model" in ids and "accuracy-prithvi" in ids
    assert any(i.startswith("ward-jayanagar") for i in ids)
    assert any(i.startswith("alias-whitefield") for i in ids)
    kinds = {c.meta.get("kind") for c in chunks}
    assert {"provenance", "accuracy", "ward", "alias", "coverage"} <= kinds


def test_reporter_prompt_contains_evidence_and_notes(wards_file):
    # run the real pipeline with stub agents to get a genuine bundle
    spec = {"status": "ok", "task": {
        "location": "Whitefield", "analysis_type": "urban_expansion",
        "year": None, "start_year": 2010, "end_year": 2024, "parameters": {}},
        "message": None}
    res = handle_query("growth in whitefield since 2010",
                       ScriptedClient([json.dumps(spec)]),
                       Gazetteer(wards_file),
                       __import__("auie.router", fromlist=["Router"]).Router(
                           {"vision": StubVisionAgent(), "gis": StubGISAgent()}))
    assert res.status == "ok"

    kb = KnowledgeBase([
        Chunk("accuracy-prithvi", "model accuracy miou 0.726 holdout"),
        Chunk("ward-kadugodi", "ward kadugodi area"),
    ], embed_fn=fake_embedder)
    client = ScriptedClient(["The area grew, per the evidence."])
    answer, chunk_ids = report("How much did Whitefield grow?",
                               res.bundle, kb, client)
    assert answer == "The area grew, per the evidence."
    assert len(chunk_ids) == 2
    system, user = client.calls[0]
    # prompt discipline: notes disclosed, stats present, context labeled
    assert "interpreted as ward(s)" in user          # alias note surfaced
    assert "Start year moved" in user                # snap note surfaced
    assert "urban_expansion" in user
    assert "[accuracy-prithvi]" in user
    assert "Never compute" in system
