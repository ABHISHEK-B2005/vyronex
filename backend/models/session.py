from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SessionRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str | None = Field(default=None, alias="_id")
    user_id: str
    activity: str = Field(min_length=1, max_length=100)
    source: Literal["live", "upload"] = "live"
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    avg_form_score: float | None = Field(default=None, ge=0, le=100)
    overall_risk: Literal["low", "moderate", "high"] | None = None


class RepRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str | None = Field(default=None, alias="_id")
    user_id: str
    session_id: str
    rep_number: int = Field(ge=1)
    timestamp_sec: float = Field(ge=0)
    status: Literal["correct", "defective", "high_risk"]
    defect_type: str | None = None
    body_area: str | None = None
    risk_level: Literal["LOW", "MODERATE", "HIGH"]
    form_scores: dict[str, float] = Field(default_factory=dict)
    features: dict[str, float] = Field(default_factory=dict)
    feedback: str = ""
    model_version: str | None = "v1.0.0"
    model_confidence: float | None = 0.85
    user_feedback: str | None = None
    disputed: bool = False
    expert_reviewed: bool = False
    expert_label: str | None = None
    expert_defect_type: str | None = None
    expert_id: str | None = None
    expert_reviewed_at: datetime | None = None