"""LLM planner v2 — emit what the user said; resolvers decide validity.

Division of labour (the moved firewall):
  planner  : intent -> analysis_type + verbatim location text + years as said
  gazetteer: is the location real/covered?   (deterministic)
  temporal : are the years servable?         (deterministic)
So 'out_of_scope' from the planner now means UNSUPPORTED ANALYSIS only
(flood risk, traffic, prices...). Unknown locations are NOT the planner's
call — it must pass them through verbatim.
"""
from __future__ import annotations

import json

from pydantic import ValidationError

from . import config
from .llm import LLMClient
from .schema import PlannerOutput

_ANALYSIS_LINES = "\n".join(
    f'- "{aid}": {desc}'
    + ("  [needs start_year+end_year]" if aid in config.BITEMPORAL
       else "  [needs year]" if aid in config.SINGLE_YEAR
       else "  [years optional]")
    for aid, desc in config.ANALYSIS_TYPES.items()
)

SYSTEM_PROMPT = f"""You convert a user's question about urban monitoring in the Bengaluru region into a JSON task specification. You NEVER answer the question and NEVER judge whether a location is covered — copy the place name exactly as the user wrote it; a downstream resolver validates it.

Respond with ONLY a JSON object, no markdown fences:
{{
  "status": "ok" | "needs_clarification" | "out_of_scope",
  "task": {{"location": "<verbatim place text>", "analysis_type": "<analysis_id>",
            "year": <int|null>, "start_year": <int|null>, "end_year": <int|null>,
            "parameters": {{}}}} | null,
  "message": "<string, required unless status is ok>"
}}

Valid analysis_id values (the ONLY analyses that exist):
{_ANALYSIS_LINES}

Rules:
1. "out_of_scope" is ONLY for unsupported analysis types (flood risk, air quality, traffic, property prices, population...). It is NEVER about the location.
2. Copy locations verbatim: "Koramangala" stays "Koramangala", "Mysuru" stays "Mysuru", "Bengaluru"/"the city" stays as written. Do not correct spelling.
3. "needs_clarification" when the analysis or required years are missing and not implied. Do not invent years.
4. Year phrases: "between X and Y"/"from X to Y" -> start_year/end_year. "as of X"/"in X" -> year. "currently"/"latest"/"now" -> year {max(config.AVAILABLE_YEARS)}. "over the years"/"trend"/"how has it changed over time" -> analysis "trend" with years null unless bounds are stated.
5. Emit the years the user said, even old ones like 2010 — the resolver handles availability.

Examples:

Q: How much did built-up area grow in Koramangala between 2020 and 2025?
A: {{"status":"ok","task":{{"location":"Koramangala","analysis_type":"urban_expansion","year":null,"start_year":2020,"end_year":2025,"parameters":{{}}}},"message":null}}

Q: Show the built-up map of Whitefield as of 2024.
A: {{"status":"ok","task":{{"location":"Whitefield","analysis_type":"built_up_mapping","year":2024,"start_year":null,"end_year":null,"parameters":{{}}}},"message":null}}

Q: How has Bellandur changed over the years?
A: {{"status":"ok","task":{{"location":"Bellandur","analysis_type":"trend","year":null,"start_year":null,"end_year":null,"parameters":{{}}}},"message":null}}

Q: How did Koramangala change between 2019 and 2024?
A: {{"status":"ok","task":{{"location":"Koramangala","analysis_type":"change_detection","year":null,"start_year":2019,"end_year":2024,"parameters":{{}}}},"message":null}}

Q: Did the city lose vegetation from 2010 to 2024?
A: {{"status":"ok","task":{{"location":"the city","analysis_type":"vegetation_change","year":null,"start_year":2010,"end_year":2024,"parameters":{{}}}},"message":null}}

Q: Land cover of Mysuru in 2024.
A: {{"status":"ok","task":{{"location":"Mysuru","analysis_type":"land_cover","year":2024,"start_year":null,"end_year":null,"parameters":{{}}}},"message":null}}

Q: Show me the changes in Hebbal.
A: {{"status":"needs_clarification","task":null,"message":"Which years should I compare for Hebbal? For example 2019 to 2025."}}

Q: What's the flood risk in Whitefield?
A: {{"status":"out_of_scope","task":null,"message":"Flood risk analysis is not supported. Available: built-up mapping, land cover, change detection, urban expansion, vegetation change, and multi-year trends."}}

Q: Analyze urban growth.
A: {{"status":"needs_clarification","task":null,"message":"Please name a place (a BBMP ward, a locality like Whitefield, or the whole city) and the years, e.g. 'urban growth in Yelahanka from 2019 to 2025'."}}"""

_REPROMPT_TEMPLATE = (
    "Your previous JSON failed validation with this error:\n{error}\n\n"
    "Original user query: {query}\n"
    "Return ONLY a corrected JSON object matching the schema."
)


def _parse(raw: str) -> PlannerOutput:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return PlannerOutput.model_validate(json.loads(text))


def plan(query: str, client: LLMClient) -> PlannerOutput:
    raw = client.complete(SYSTEM_PROMPT, query)
    try:
        return _parse(raw)
    except (json.JSONDecodeError, ValidationError) as first_error:
        retry = client.complete(
            SYSTEM_PROMPT, _REPROMPT_TEMPLATE.format(error=first_error, query=query)
        )
        try:
            return _parse(retry)
        except (json.JSONDecodeError, ValidationError):
            return PlannerOutput(
                status="needs_clarification",
                message=("I could not confidently interpret that request. Please "
                         "state a place, an analysis, and the years, e.g. "
                         "'urban expansion in Whitefield between 2019 and 2025'."),
            )
