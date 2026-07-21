# AUIE core — v2 (city-wide, multi-year)

Supersedes the Sprint B v1 package. Locations: all 198 BBMP wards + locality
aliases + city-wide, resolved deterministically. Years: 2019-2025 epochs
with disclosed snapping.

## Layout
```
auie/
  config.py      # AVAILABLE_YEARS, analysis types, ward-file path
  schema.py      # TaskSpec (free-text location, integer years) + envelope
  gazetteer.py   # location firewall: aliases, transliteration, fuzzy, traps
  temporal.py    # year firewall: snapping with notes, pre-era rejection
  planner.py     # LLM -> task spec; emits locations VERBATIM
  pipeline.py    # handle_query(): plan -> gazetteer -> temporal -> route
  router.py      # dispatch + EvidenceBundle (notes ride in provenance)
  agents/base.py # AgentResult contract + stubs
  tests/         # 23 offline tests incl. fixture ward file with real traps
  run_gate_b1.py # 20 live queries (needs API key + real ward file)
```

## Setup
```bash
pip install pydantic requests pytest geopandas shapely
python -m pytest auie/tests/ -q            # expect: 23 passed
export AUIE_WARDS_GEOJSON=/path/to/BBMP_oldWards.geojson
export GEMINI_API_KEY=...                  # or GROQ_API_KEY
python -m auie.run_gate_b1                 # live Gate B1 (target 15 Jul)
```

## MUST DO before trusting Gate B1 results
The alias table in gazetteer.py (ALIASES) is SEEDED FROM GENERAL KNOWLEDGE
and must be verified in QGIS against the ward map:
  - "whitefield" -> Kadugodi + Hagadur      (check; likely needs more wards)
  - "sarjapur road" -> Bellanduru + Varthuru (check)
Wrong aliases produce confidently wrong statistics — this is the one place
in the deterministic layer where a human decides ground truth. Extend the
table freely; unknown-ward entries fail loudly by design.

## Design notes
- Planner NEVER judges locations; out_of_scope = unsupported analysis only.
- Every silent-correction risk is a disclosed note: alias interpretation,
  year snapping, future-year capping. Notes flow into
  EvidenceBundle.provenance["resolution_notes"] -> the reporter must
  surface them in the answer.
- Electronic City is rejected with an honest boundary explanation. To
  support it, add a custom polygon (planned: extra_geometries dir).
