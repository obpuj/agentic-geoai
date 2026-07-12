"""Schema v2 — city-wide, multi-year task specification.

Location is free text (the gazetteer decides validity). Years are integers
(the temporal resolver decides validity). Pydantic still enforces the
structural rules: which analyses need which temporal fields, and the
status/task/message envelope consistency.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from . import config

AnalysisType = Enum(
    "AnalysisType", {k.upper(): k for k in config.ANALYSIS_TYPES}, type=str
)


class TaskSpec(BaseModel):
    """Planner output payload. location is VERBATIM user text by design."""

    location: str = Field(min_length=1, max_length=120)
    analysis_type: AnalysisType
    year: Optional[int] = Field(default=None, ge=1900, le=2100)
    start_year: Optional[int] = Field(default=None, ge=1900, le=2100)
    end_year: Optional[int] = Field(default=None, ge=1900, le=2100)
    parameters: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _temporal_shape(self) -> "TaskSpec":
        a = self.analysis_type.value
        if a in config.BITEMPORAL:
            if self.start_year is None or self.end_year is None:
                raise ValueError(f"{a} requires start_year and end_year")
            if self.start_year >= self.end_year:
                raise ValueError("start_year must be before end_year")
            if self.year is not None:
                raise ValueError(f"{a} takes start/end years, not a single year")
        elif a in config.SINGLE_YEAR:
            if self.year is None:
                raise ValueError(f"{a} requires a single year")
            if self.start_year or self.end_year:
                raise ValueError(f"{a} takes a single year, not a range")
        # MULTI_YEAR (trend): all temporal fields optional
        return self


class PlannerOutput(BaseModel):
    status: Literal["ok", "needs_clarification", "out_of_scope"]
    task: Optional[TaskSpec] = None
    message: Optional[str] = None

    @model_validator(mode="after")
    def _envelope(self) -> "PlannerOutput":
        if self.status == "ok" and self.task is None:
            raise ValueError("status 'ok' requires a task")
        if self.status != "ok":
            if self.task is not None:
                raise ValueError(f"status '{self.status}' must not include a task")
            if not self.message:
                raise ValueError(f"status '{self.status}' requires a message")
        return self
