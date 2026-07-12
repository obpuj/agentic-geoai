"""Gate B1 (live, target 15 July): 20 queries against the real planner +
gazetteer + temporal resolver. Needs GEMINI_API_KEY or GROQ_API_KEY and the
real ward file (AUIE_WARDS_GEOJSON env var or data/BBMP_oldWards.geojson).

Usage:  python -m auie.run_gate_b1
Categories now include the city-wide design's new cases: localities,
misspellings, transliterations, pre-era years, future years, and trends.
"""
from __future__ import annotations

import sys
import time

from .agents.base import StubGISAgent, StubVisionAgent
from .gazetteer import Gazetteer
from .llm import default_client
from .pipeline import handle_query
from .router import Router

# (query, expected QueryResult.status)
CASES: list[tuple[str, str]] = [
    # in-scope: wards, exact and fuzzy
    ("How much did built-up area grow in Jayanagar between 2020 and 2025?", "ok"),
    ("Show the built-up map of Bellandur as of 2024.", "ok"),          # translit
    ("Did Kaddugodi lose vegetation from 2019 to 2025?", "ok"),        # misspelt
    ("Land cover of HSR Layout in 2024.", "ok"),
    # in-scope: localities via alias table
    ("Urban expansion in Whitefield from 2019 to 2025.", "ok"),
    ("How has Sarjapur Road changed over the years?", "ok"),           # trend
    # in-scope: city-wide
    ("How much did Bengaluru's built-up area grow between 2019 and 2025?", "ok"),
    ("Show the city's land cover for 2024.", "ok"),
    # temporal snapping (ok WITH notes) and capping
    ("Did the city lose vegetation from 2010 to 2024?", "ok"),
    ("Built-up map of Yelahanka Satellite Town in 2030.", "ok"),
    # clarification
    ("Show me the changes in Hebbal.", "clarify"),
    ("Analyze urban growth.", "clarify"),
    ("How fast is the city growing?", "clarify"),
    # rejected: location outside coverage
    ("Compare urban expansion in Mysuru between 2020 and 2025.", "rejected"),
    ("Built-up map of Electronic City in 2024.", "rejected"),
    ("Land cover of Mumbai in 2024.", "rejected"),
    # rejected: wholly pre-Sentinel-2-era
    ("How did Jayanagar change between 2005 and 2012?", "rejected"),
    # rejected: unsupported analyses
    ("What's the flood risk in Whitefield?", "rejected"),
    ("How bad is traffic on Sarjapur Road?", "rejected"),
    ("Property price trends in HSR Layout.", "rejected"),
]


def main() -> int:
    client = default_client()
    gaz = Gazetteer()
    router = Router({"vision": StubVisionAgent(), "gis": StubGISAgent()})
    failures = 0
    for query, want in CASES:
        time.sleep(2.5)  # pace for free-tier rate limits
        res = handle_query(query, client, gaz, router)
        ok = res.status == want
        failures += 0 if ok else 1
        detail = ""
        if res.status == "ok":
            r = res.resolved
            detail = (f" -> {r.task.analysis_type.value} | "
                      f"{r.location.name} ({len(r.location.ward_names)} ward(s)) | "
                      f"years {r.years.years}"
                      + (f" | notes: {r.notes}" if r.notes else ""))
        else:
            detail = f" -> {res.message}"
        print(f"[{'PASS' if ok else 'FAIL'}] ({want}) {query}\n        {res.status}{detail}\n")
    print(f"Gate B1: {len(CASES) - failures}/{len(CASES)} passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
