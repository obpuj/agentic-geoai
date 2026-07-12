"""Temporal resolver — years counterpart of the gazetteer.

The planner emits the years the user said; this module decides validity
against config.AVAILABLE_YEARS. Snapping is allowed and always disclosed via
notes (which the reporter must surface); wholly pre-era requests are
rejected with the Sentinel-2 explanation, never silently served.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from . import config


@dataclass
class ResolvedYears:
    years: list[int]                 # 1 (single), 2 (bitemporal), or all (trend)
    notes: list[str] = field(default_factory=list)


@dataclass
class TemporalResolution:
    status: Literal["ok", "rejected"]
    resolved: Optional[ResolvedYears] = None
    message: Optional[str] = None


def _snap(year: int, notes: list[str]) -> int:
    avail = config.AVAILABLE_YEARS
    if year in avail:
        return year
    nearest = min(avail, key=lambda y: abs(y - year))
    notes.append(f"No {year} epoch; using nearest available year {nearest}.")
    return nearest


def resolve_years(analysis_type: str, year: Optional[int],
                  start_year: Optional[int], end_year: Optional[int]
                  ) -> TemporalResolution:
    avail = config.AVAILABLE_YEARS
    lo, hi = min(avail), max(avail)
    era_msg = (
        f"Consistent Sentinel-2 coverage for Bengaluru begins in {lo}; "
        f"the system covers {lo}-{hi}. Requests entirely before {lo} "
        f"cannot be served from Sentinel-2 data."
    )
    notes: list[str] = []

    if analysis_type in config.MULTI_YEAR:
        years = list(avail)
        if start_year or end_year:
            s = start_year or lo
            e = end_year or hi
            if e < lo:
                return TemporalResolution("rejected", message=era_msg)
            if s < lo:
                notes.append(f"Trend start moved from {s} to {lo} ({era_msg})")
                s = lo
            years = [y for y in avail if s <= y <= min(e, hi)]
            if e > hi:
                notes.append(f"Trend end capped at {hi} (latest processed epoch).")
        return TemporalResolution("ok", ResolvedYears(years, notes))

    if analysis_type in config.BITEMPORAL:
        if start_year is None or end_year is None:
            return TemporalResolution("rejected",
                message="This analysis needs a start and end year.")
        if end_year < lo:
            return TemporalResolution("rejected", message=era_msg)
        if start_year < lo:
            notes.append(f"Start year moved from {start_year} to {lo}. {era_msg}")
            start_year = lo
        s, e = _snap(start_year, notes), _snap(min(end_year, hi), notes)
        if end_year > hi:
            notes.append(f"End year capped at {hi} (latest processed epoch).")
        if s >= e:
            return TemporalResolution("rejected",
                message=f"After resolving to available epochs ({s}, {e}), the "
                        f"range is empty. Choose years between {lo} and {hi}.")
        return TemporalResolution("ok", ResolvedYears([s, e], notes))

    # single-year analyses
    if year is None:
        return TemporalResolution("rejected", message="This analysis needs a year.")
    if year < lo:
        return TemporalResolution("rejected", message=era_msg)
    y = _snap(min(year, hi), notes)
    if year > hi:
        notes.append(f"{year} not yet processed; using {hi}.")
    return TemporalResolution("ok", ResolvedYears([y], notes))
