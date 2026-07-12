"""Agent contract v2 — agents receive the fully resolved task."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..pipeline import ResolvedTask


@dataclass
class AgentResult:
    agent: str
    layers: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)


class BaseAgent(ABC):
    name: str = "base"

    @abstractmethod
    def run(self, resolved: "ResolvedTask") -> AgentResult: ...


class StubVisionAgent(BaseAgent):
    """Replace with the raster clip-and-count agent once precompute lands."""

    name = "vision"

    def run(self, resolved: "ResolvedTask") -> AgentResult:
        return AgentResult(
            agent=self.name,
            layers=[{"type": "raster_ref",
                     "years": resolved.years.years,
                     "wards": resolved.location.ward_names}],
            stats={"built_up_km2_by_year": {y: None for y in resolved.years.years}},
            provenance={"model": "stub", "model_version": "0.0"},
        )


class StubGISAgent(BaseAgent):
    name = "gis"

    def run(self, resolved: "ResolvedTask") -> AgentResult:
        return AgentResult(
            agent=self.name,
            layers=[{"type": "geojson_ref",
                     "wards": resolved.location.ward_names}],
            stats={"osm_footprint_agreement": None},
            provenance={"source": "stub"},
        )
