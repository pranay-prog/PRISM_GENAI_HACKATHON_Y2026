"""Pydantic request/response models (shown in Swagger at /docs)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MessageIn(BaseModel):
    text: str = Field(..., min_length=1, examples=["The router has a red LOS light."])
    final: bool = Field(False, description="True when the utterance/turn is complete")


class RunScenarioIn(BaseModel):
    scenario_id: str = Field(..., examples=["outage_wifi_compensation"])
    speed: float = Field(1.0, gt=0, le=20, description="Playback speed multiplier for scripted chunk times")


class SessionOut(BaseModel):
    session_id: str
    state: dict[str, Any]
    scenario_running: bool | None = None


class MessageOut(BaseModel):
    session_id: str
    decision: dict[str, Any]
    answer: dict[str, Any] | None = None


class EventOut(BaseModel):
    event_id: str
    session_id: str
    timestamp: float
    type: str
    payload: dict[str, Any]


class EventsOut(BaseModel):
    session_id: str
    count: int
    events: list[EventOut]
