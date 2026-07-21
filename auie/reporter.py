"""Evidence-grounded reporter (pipeline stage 9) — the second LLM call.

Contract, matching the abstract's promises exactly:
  * The LLM writes prose using ONLY figures present in the evidence bundle
    and retrieved chunks. It computes nothing.
  * Provenance is cited inline: imagery window, model + version, accuracy.
  * Resolution notes (alias interpretation, year snapping) MUST be surfaced —
    disclosed corrections are a design guarantee, not decoration.
"""
from __future__ import annotations

import json

from .knowledge import KnowledgeBase
from .llm import LLMClient
from .router import EvidenceBundle

SYSTEM_PROMPT = """You write a short, factual answer to an urban-monitoring question using ONLY the evidence provided. Hard rules:
1. Copy numbers exactly as they appear in EVIDENCE or CONTEXT. Never compute, convert, re-round, or re-derive any figure — percentages are already provided.
2. Cite provenance inline once: imagery window, model name, and (if present in CONTEXT) its accuracy, phrased as agreement with Dynamic World-derived labels.
3. If NOTES are present, state them plainly at the start (e.g. which wards a locality was interpreted as, or year adjustments). Never hide an interpretation.
4. 4-8 sentences of plain prose. No markdown, bullets, or headers. Never mention internal field names (like vision.per_year); express shares as percentages (e.g. 84.9%).
5. If the evidence seems insufficient for the question, say what is missing instead of guessing."""

USER_TEMPLATE = """QUESTION: {question}

NOTES (must be disclosed): {notes}

EVIDENCE (deterministic pipeline outputs):
{evidence}

CONTEXT (retrieved project knowledge):
{context}

Write the answer."""


# def _format_evidence(bundle: EvidenceBundle) -> str:
#     lines = [f"location: {bundle.resolved.location.name} "
#              f"({len(bundle.resolved.location.ward_names)} ward(s))",
#              f"analysis: {bundle.resolved.task.analysis_type.value}",
#              f"years: {bundle.resolved.years.years}"]
#     for key, val in bundle.stats.items():
#         lines.append(f"{key}: {json.dumps(val, default=str)}")
#     prov = bundle.provenance.get("vision", {})
#     for k in ("model", "model_version", "composite_window", "label_source"):
#         if k in prov:
#             lines.append(f"{k}: {prov[k]}")
#     return "\n".join(lines)


def _format_evidence(bundle: EvidenceBundle) -> str:
    lines = [f"location: {bundle.resolved.location.name} "
             f"({len(bundle.resolved.location.ward_names)} ward(s))",
             f"analysis: {bundle.resolved.task.analysis_type.value}",
             f"years: {bundle.resolved.years.years}"]
    for key, val in bundle.stats.items():
        if key.endswith("per_year"):
            for year, d in val.items():
                parts = [f"{cls} {d['area_km2'][cls]:.2f} km2 "
                         f"({100 * d['share'][cls]:.1f}%)"
                         for cls in ("built_up", "vegetation", "water", "background")]
                lines.append(f"{year}: " + ", ".join(parts) +
                             f", valid pixels {100 * d['valid_pixel_fraction']:.1f}%")
        else:
            lines.append(f"{key.split('.', 1)[-1]}: {json.dumps(val, default=str)}")
    prov = bundle.provenance.get("vision", {})
    for k in ("model", "model_version", "composite_window", "label_source"):
        if k in prov:
            lines.append(f"{k}: {prov[k]}")
    return "\n".join(lines)


def report(question: str, bundle: EvidenceBundle, kb: KnowledgeBase,
           client: LLMClient, k: int = 4) -> tuple[str, list[str]]:
    """Returns (answer_text, retrieved_chunk_ids) — log the ids per query;
    they are the paper's transparency evidence."""
    # loc = bundle.resolved.location
    # query = (f"{bundle.resolved.task.analysis_type.value} {loc.name} "
    #          f"{' '.join(loc.ward_names[:4])} model accuracy provenance")
    loc = bundle.resolved.location
    loc_terms = ("city-wide BBMP coverage all wards" if loc.kind == "region"
                 else f"{loc.name} {' '.join(loc.ward_names[:4])}")
    query = (f"{bundle.resolved.task.analysis_type.value} {loc_terms} "
             "model accuracy provenance")
    chunks = kb.retrieve(query, k=k)
    context = "\n".join(f"[{c.id}] {c.text}" for c in chunks)
    notes = "; ".join(bundle.resolved.notes) or "none"
    prompt = USER_TEMPLATE.format(question=question, notes=notes,
                                  evidence=_format_evidence(bundle),
                                  context=context)
    # answer = client.complete(SYSTEM_PROMPT, prompt)
    answer = client.complete(SYSTEM_PROMPT, prompt, json_output=False)
    return answer.strip(), [c.id for c in chunks]
