"""Formal evaluation run (paper Section: Evaluation).

    python -m auie.run_evaluation            # writes eval_results.json + eval_table.md

30 queries across four categories. Two automatic checks per query:
  1. STATUS: did the pipeline land in the expected {ok, clarify, rejected}?
  2. NUMERIC GROUNDING (ok-queries only): every number in the generated answer
     must appear in the evidence shown to the LLM (bundle stats + retrieved
     chunks), up to rounding. This mechanically tests the paper's core claim —
     the reporter cites, never computes.
Manual rubric columns (fill by hand in eval_table.md): notes_disclosed,
provenance_cited, unsupported_claims. Keep the filled table in the repo;
it IS the paper's Table 3.
"""
from __future__ import annotations

import json
import re
import time

from .agents.base import StubGISAgent
from .agents.vision import VisionAgent
from .gazetteer import Gazetteer
from .knowledge import KnowledgeBase
from .llm import default_client
from .pipeline import handle_query
from .reporter import _format_evidence, report
from .router import Router

# (query, expected_status)
CASES: list[tuple[str, str]] = [
    # A. in-scope: wards, exact / transliterated / misspelt (8)
    ("How much did built-up area grow in Jayanagar between 2020 and 2025?", "ok"),
    ("Show the built-up map of Bellandur as of 2024.", "ok"),
    ("Did Kaddugodi lose vegetation from 2019 to 2025?", "ok"),
    ("Land cover of HSR Layout in 2024.", "ok"),
    ("Built-up change in Hudi between 2019 and 2023.", "ok"),
    ("How much vegetation does Varthur have in 2025?", "ok"),
    ("Urban expansion in Yelahanka Satellite Town from 2019 to 2025.", "ok"),
    ("How did Vijnanapura change between 2020 and 2024?", "ok"),
    # B. in-scope: localities + city + trends + snapping (10)
    ("Urban expansion in Whitefield from 2019 to 2025.", "ok"),
    ("How has Sarjapur Road changed over the years?", "ok"),
    ("How has Bellandur changed over the years?", "ok"),
    ("How much did Bengaluru's built-up area grow between 2019 and 2025?", "ok"),
    ("Show the city's land cover for 2024.", "ok"),
    ("Did the city lose vegetation from 2010 to 2024?", "ok"),          # snap
    ("Built-up map of HSR Layout in 2030.", "ok"),                       # cap
    ("How has the whole city changed over time?", "ok"),                 # trend
    ("Land cover of Whitefield, latest imagery.", "ok"),
    ("Vegetation change in Hagadur since 2016.", "ok"),                  # snap
    # C. clarification (5)
    ("Show me the changes in Hebbal.", "clarify"),
    ("Analyze urban growth.", "clarify"),
    ("How fast is the city growing?", "clarify"),
    ("Compare two wards.", "clarify"),
    ("What happened in 2024?", "clarify"),
    # D. rejection: coverage + era + capability (7)
    ("Compare urban expansion in Mysuru between 2020 and 2025.", "rejected"),
    ("Built-up map of Electronic City in 2024.", "rejected"),
    ("Land cover of Mumbai in 2024.", "rejected"),
    ("How did Jayanagar change between 2005 and 2012?", "rejected"),
    ("What's the flood risk in Whitefield?", "rejected"),
    ("How bad is traffic on Sarjapur Road?", "rejected"),
    ("Property price trends in HSR Layout.", "rejected"),
]

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _evidence_numbers(evidence_text: str, chunks_text: str) -> set[str]:
    """All numbers the LLM was shown, in several rounding variants."""
    out: set[str] = set()
    # for tok in _NUM.findall(evidence_text + " " + chunks_text):
    #     v = float(tok)
    #     for nd in (0, 1, 2, 3, 4):
    #         out.add(f"{v:.{nd}f}".rstrip("0").rstrip("."))
    #         out.add(f"{round(v, nd)}")
    for tok in _NUM.findall(evidence_text + " " + chunks_text):
        v = float(tok)
        for val in (v, abs(v)):
            for nd in (0, 1, 2, 3, 4):
                out.add(f"{val:.{nd}f}".rstrip("0").rstrip("."))
                out.add(f"{round(val, nd)}")
    return out


def grounding_check(answer: str, evidence_text: str, chunks_text: str
                    ) -> tuple[bool, list[str]]:
    """True if every number in the answer traces to the evidence."""
    shown = _evidence_numbers(evidence_text, chunks_text)
    offenders = []
    for tok in _NUM.findall(answer):
        norm = tok.rstrip("0").rstrip(".") if "." in tok else tok
        if norm not in shown and tok not in shown:
            # years are structural, not derived figures
            if tok.isdigit() and 1900 <= int(tok) <= 2100:
                continue
            offenders.append(tok)
    return (not offenders), offenders


def main() -> None:
    client = default_client()
    gaz = Gazetteer()
    router = Router({"vision": VisionAgent(gaz), "gis": StubGISAgent()})
    kb = KnowledgeBase.load("data/kb")

    rows = []
    # for query, want in CASES:
    #     t0 = time.time()
    #     res = handle_query(query, client, gaz, router)
    for query, want in CASES:
        time.sleep(4.0)  # free-tier pacing
        t0 = time.time()
        try:
            res = handle_query(query, client, gaz, router)
        except Exception as e:
            rows.append({"query": query, "expected": want, "status": "error",
                         "status_ok": False,
                         "message": f"{type(e).__name__}: {str(e)[:200]}",
                         "latency_s": round(time.time() - t0, 2)})
            print(f"[ERR ] {query} -> {type(e).__name__}")
            continue
        row = {"query": query, "expected": want, "status": res.status,
               "status_ok": res.status == want}
        if res.status == "ok":
            chunks = kb.retrieve(
                f"{res.resolved.task.analysis_type.value} "
                f"{res.resolved.location.name} model accuracy provenance", k=4)
            chunks_text = " ".join(c.text for c in chunks)
            try:
                answer, cited = report(query, res.bundle, kb, client)
            except Exception as e:
                answer, cited = f"(reporter error: {e})", []
            grounded, offenders = grounding_check(
                answer, _format_evidence(res.bundle), chunks_text)
            row.update(answer=answer, cited=cited,
                       numbers_grounded=grounded,
                       ungrounded_numbers=offenders,
                       spec={"location": res.resolved.location.name,
                             "wards": len(res.resolved.location.ward_names),
                             "analysis": res.resolved.task.analysis_type.value,
                             "years": res.resolved.years.years,
                             "notes": res.resolved.notes})
        else:
            row["message"] = res.message
        row["latency_s"] = round(time.time() - t0, 2)
        rows.append(row)
        mark = "PASS" if row["status_ok"] else "FAIL"
        extra = ("" if res.status != "ok"
                 else f" | grounded={row['numbers_grounded']}")
        print(f"[{mark}] {query} -> {res.status}{extra}")

    n = len(rows)
    status_acc = sum(r["status_ok"] for r in rows) / n
    ok_rows = [r for r in rows if r["status"] == "ok" and "numbers_grounded" in r]
    grounded_rate = (sum(r["numbers_grounded"] for r in ok_rows) / len(ok_rows)
                     if ok_rows else 0)
    summary = {"n": n, "status_accuracy": round(status_acc, 3),
               "numeric_grounding_rate": round(grounded_rate, 3),
               "mean_latency_s": round(sum(r["latency_s"] for r in rows) / n, 2)}
    print("\nSUMMARY:", json.dumps(summary, indent=1))

    json.dump({"summary": summary, "rows": rows},
              open("eval_results.json", "w"), indent=1)

    with open("eval_table.md", "w") as f:
        f.write("| # | query | expected | got | status | grounded | "
                "notes_disclosed | provenance_cited | unsupported_claims |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for i, r in enumerate(rows, 1):
            f.write(f"| {i} | {r['query'][:60]} | {r['expected']} | "
                    f"{r['status']} | {'P' if r['status_ok'] else 'F'} | "
                    f"{r.get('numbers_grounded', '-')} |  |  |  |\n")
    print("wrote eval_results.json + eval_table.md "
          "(fill the three manual columns by hand)")


if __name__ == "__main__":
    main()
