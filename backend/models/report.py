from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from typing import Any


class ReportRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str | None = Field(default=None, alias="_id")
    session_id: str
    user_id: str
    totals: dict[str, int | float] = Field(default_factory=dict)
    accuracy_pct: float | None = Field(default=None, ge=0, le=100)
    body_area_breakdown: dict[str, int] = Field(default_factory=dict)
    defect_counts: dict[str, int] = Field(default_factory=dict)
    fatigue: dict[str, int | float | str] = Field(default_factory=dict)
    worst_rep: dict[str, int | float | str | None] | None = None
    recommendations: list[str] = Field(default_factory=list)
    compared_to_last: dict[str, int | float | str | None] = Field(default_factory=dict)
    alignment: dict[str, Any] = Field(default_factory=dict)