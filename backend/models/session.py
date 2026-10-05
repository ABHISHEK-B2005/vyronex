from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SessionRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str | None = Field(default=None, alias="_id")
    user_id: str
    activity: str = Field(min_length=1, max_length=100)
    source: Literal["live", "upload"] | str = "live"
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    avg_form_score: float | None = Field(default=None, ge=0, le=100)
    overall_risk: Literal["low", "moderate", "high", "LOW", "MODERATE", "HIGH"] | str | None = None


class RepRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str | None = Field(default=None, alias="_id")
    user_id: str | None = None
    session_id: str
    rep_number: int = Field(ge=1)
    timestamp_sec: float = Field(ge=0)
    status: Literal["correct", "defective", "high_risk"] | str
    defect_type: str | None = None
    body_area: str | None = None
    risk_level: Literal["low", "moderate", "high", "LOW", "MODERATE", "HIGH"] | str
    form_scores: dict[str, float] = Field(default_factory=dict)
    features: dict[str, float] = Field(default_factory=dict)
    feedback: str = ""