"""Task router v2 — dispatches ResolvedTask (post-gazetteer, post-temporal).

Still a plain dict, still deliberately not a framework.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .agents.base import AgentResult, BaseAgent
from .schema import AnalysisType

if TYPE_CHECKING:  # avoid circular import; pipeline imports router
    from .pipeline import ResolvedTask

ROUTES: dict[AnalysisType, list[str]] = {
    AnalysisType.BUILT_UP_MAPPING: ["vision", "gis"],
    AnalysisType.LAND_COVER: ["vision", "gis"],
    AnalysisType.CHANGE_DETECTION: ["vision", "gis"],
    AnalysisType.URBAN_EXPANSION: ["vision", "gis"],
    AnalysisType.VEGETATION_CHANGE: ["vision", "gis"],
    AnalysisType.TREND: ["vision", "gis"],
}


@dataclass
class EvidenceBundle:
    resolved: "ResolvedTask"
    results: list[AgentResult] = field(default_factory=list)

    @property
    def stats(self) -> dict:
        merged: dict = {}
        for r in self.results:
            merged.update({f"{r.agent}.{k}": v for k, v in r.stats.items()})
        return merged

    @property
    def provenance(self) -> dict:
        prov = {r.agent: r.provenance for r in self.results}
        prov["resolution_notes"] = self.resolved.notes
        return prov


class Router:
    def __init__(self, agents: dict[str, BaseAgent]):
        self.agents = agents

    def execute(self, resolved: "ResolvedTask") -> EvidenceBundle:
        bundle = EvidenceBundle(resolved=resolved)
        for agent_name in ROUTES[resolved.task.analysis_type]:
            if agent_name not in self.agents:
                raise KeyError(
                    f"Route for {resolved.task.analysis_type.value} needs agent "
                    f"'{agent_name}', which is not registered")
            bundle.results.append(self.agents[agent_name].run(resolved))
        return bundle
