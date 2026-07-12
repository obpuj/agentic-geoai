"""Gazetteer — deterministic location resolution (the moved firewall).

With open locations, the LLM emits the place name AS TEXT and this module
decides validity. The LLM never judges coverage.

Resolution order (informed by the real BBMP_oldWards.geojson exploration):
  1. Region terms ("Bengaluru", "city-wide")           -> all 198 wards
  2. Alias table (localities that aren't ward names:
     "Whitefield", "Sarjapur Road")                    -> ward list
  3. Known-outside-BBMP localities ("Electronic City") -> honest rejection
  4. Exact match on normalized ward name               -> single ward
  5. Fuzzy match (difflib) with transliteration
     normalization (Bellandur -> Bellanduru)           -> single ward
  6. Multiple close candidates                         -> ambiguous
  7. Nothing close                                     -> out_of_region

NEVER substring 'contains' matching: "Jayanagar" must not match
"Vijayanagar" (real trap found in the ward file on 11 Jul).
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Literal, Optional

import geopandas as gpd

from . import config

# Localities people say that are NOT ward names, mapped to ward-name lists.
# TODO(Obana): verify/extend these ward lists against the map — they are
# seeded from local knowledge and MUST be checked in QGIS before Gate B.
ALIASES: dict[str, list[str]] = {
    "whitefield": [
        "Kadugodi",
        "Hagadur",
        "Garudachar Playa",
        "Hudi",
    ],
    "sarjapur road": ["Bellanduru"],
    "hsr": ["HSR Layout"],
    "hsr layout": ["HSR Layout"],
    "hal": ["HAL Airport"],
}

# Localities users will ask about that lie (largely) outside the 198-ward
# BBMP boundary. Rejected with an honest, specific message. To support one,
# add a custom polygon via extra_geometries instead.
KNOWN_OUTSIDE: dict[str, str] = {
    "electronic city": (
        "Electronic City lies largely outside BBMP's 198-ward boundary "
        "(Anekal taluk). It is not covered by the ward-based analysis."
    ),
    "e-city": (
        "Electronic City lies largely outside BBMP's 198-ward boundary "
        "(Anekal taluk). It is not covered by the ward-based analysis."
    ),
}

REGION_TERMS = {
    "bengaluru", "bangalore", "bbmp", "the city", "city", "whole city",
    "city-wide", "citywide", "all wards",
}

_FUZZY_CUTOFF = 0.82
_AMBIGUITY_MARGIN = 0.04


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _translit_variants(norm_name: str) -> set[str]:
    """Index Kannada-transliteration variants: bellanduru <-> bellandur."""
    out = {norm_name}
    for word in [norm_name]:
        if word.endswith("u") and len(word) > 5:
            out.add(word[:-1])
        else:
            out.add(word + "u")
    return out


@dataclass
class ResolvedLocation:
    kind: Literal["ward", "locality", "region"]
    name: str                      # display name
    ward_names: list[str]          # official WARD_NAME values covered
    matched_via: str               # exact | alias | fuzzy | region
    note: Optional[str] = None     # surfaced to the user in the answer


@dataclass
class Resolution:
    status: Literal["ok", "ambiguous", "out_of_region"]
    location: Optional[ResolvedLocation] = None
    candidates: list[str] = field(default_factory=list)
    message: Optional[str] = None


class Gazetteer:
    def __init__(self, wards_path: str | None = None,
                 aliases: dict[str, list[str]] | None = None):
        path = wards_path or config.WARDS_GEOJSON
        self.gdf = gpd.read_file(path)
        col = config.WARD_NAME_COLUMN
        self.ward_names: list[str] = self.gdf[col].tolist()
        self.aliases = {_norm(k): v for k, v in (aliases or ALIASES).items()}
        # normalized (incl. transliteration variants) -> official name
        self._index: dict[str, str] = {}
        for official in self.ward_names:
            for variant in _translit_variants(_norm(official)):
                self._index[variant] = official

    # -- public ----------------------------------------------------------
    def resolve(self, text: str) -> Resolution:
        q = _norm(text)
        if not q:
            return Resolution("out_of_region", message="Empty location.")

        if q in REGION_TERMS:
            return Resolution("ok", ResolvedLocation(
                "region", "BBMP (all 198 wards)", list(self.ward_names), "region"))

        if q in KNOWN_OUTSIDE:
            return Resolution("out_of_region", message=KNOWN_OUTSIDE[q])

        if q in self.aliases:
            wards = self.aliases[q]
            missing = [w for w in wards if w not in self.ward_names]
            if missing:  # alias table drifted from ward file — fail loudly
                raise ValueError(f"Alias '{text}' references unknown wards {missing}")
            return Resolution("ok", ResolvedLocation(
                "locality", text.strip().title(), wards, "alias",
                note=f"'{text.strip().title()}' interpreted as ward(s): {', '.join(wards)}"))

        for variant in _translit_variants(q):
            if variant in self._index:
                official = self._index[variant]
                return Resolution("ok", ResolvedLocation(
                    "ward", official, [official], "exact"))

        # fuzzy over normalized official names
        scored = sorted(
            ((difflib.SequenceMatcher(None, q, _norm(n)).ratio(), n)
             for n in self.ward_names),
            reverse=True,
        )
        best_score, best = scored[0]
        if best_score < _FUZZY_CUTOFF:
            return Resolution(
                "out_of_region",
                message=(f"'{text}' is not a BBMP ward, known locality, or "
                         f"region term. Coverage is the 198 BBMP wards."))
        close = [n for s, n in scored if best_score - s <= _AMBIGUITY_MARGIN]
        if len(close) > 1:
            return Resolution("ambiguous", candidates=close,
                              message=f"Did you mean: {', '.join(close)}?")
        return Resolution("ok", ResolvedLocation(
            "ward", best, [best], "fuzzy",
            note=f"Interpreted '{text}' as ward '{best}'."))

    def geometry_for(self, loc: ResolvedLocation):
        """Dissolved geometry for clipping (used by the Vision/GIS agents)."""
        col = config.WARD_NAME_COLUMN
        sel = self.gdf[self.gdf[col].isin(loc.ward_names)]
        return sel.geometry.union_all()
