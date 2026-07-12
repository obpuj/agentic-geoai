"""Query pipeline — the one entry point the dashboard calls.

    plan (LLM) -> gazetteer (deterministic) -> temporal (deterministic)
    -> router -> EvidenceBundle

Every early exit returns a QueryResult with a user-facing message, never an
exception. Resolver notes (snapped years, alias interpretation) ride along
so the reporter can disclose them — silent corrections are forbidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from .gazetteer import Gazetteer, ResolvedLocation
from .llm import LLMClient
from .planner import plan
from .router import EvidenceBundle, Router
from .schema import TaskSpec
from .temporal import ResolvedYears, resolve_years


@dataclass
class ResolvedTask:
    task: TaskSpec
    location: ResolvedLocation
    years: ResolvedYears

    @property
    def notes(self) -> list[str]:
        notes = list(self.years.notes)
        if self.location.note:
            notes.insert(0, self.location.note)
        return notes


@dataclass
class QueryResult:
    status: Literal["ok", "rejected", "clarify"]
    message: Optional[str] = None
    candidates: list[str] = field(default_factory=list)
    resolved: Optional[ResolvedTask] = None
    bundle: Optional[EvidenceBundle] = None


def handle_query(query: str, client: LLMClient, gazetteer: Gazetteer,
                 router: Router) -> QueryResult:
    p = plan(query, client)
    if p.status == "out_of_scope":
        return QueryResult("rejected", message=p.message)
    if p.status == "needs_clarification":
        return QueryResult("clarify", message=p.message)

    loc = gazetteer.resolve(p.task.location)
    if loc.status == "out_of_region":
        return QueryResult("rejected", message=loc.message)
    if loc.status == "ambiguous":
        return QueryResult("clarify", message=loc.message,
                           candidates=loc.candidates)

    t = resolve_years(p.task.analysis_type.value, p.task.year,
                      p.task.start_year, p.task.end_year)
    if t.status == "rejected":
        return QueryResult("rejected", message=t.message)

    resolved = ResolvedTask(task=p.task, location=loc.location,
                            years=t.resolved)
    bundle = router.execute(resolved)
    return QueryResult("ok", resolved=resolved, bundle=bundle)
