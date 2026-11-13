"""
Shared agent contract.

Every agent — Patient Intelligence, Smart Scheduling, Imaging, etc. — returns
the SAME shape: Insights -> Recommendations -> Actions (the deck's slide-17
UI pattern). The frontend renders any agent's output with one component
because they all speak this contract. This file is the integration seam:
if you reimplement an agent, keep this shape and the UI keeps working.
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class Insight(BaseModel):
    """Something the agent observed from the data. Read-only, no side effects."""
    label: str
    detail: str
    severity: Literal["info", "watch", "urgent"] = "info"


class Recommendation(BaseModel):
    """A suggested next step. Advisory — becomes an Action only if the clinician acts."""
    text: str
    rationale: Optional[str] = None


class Action(BaseModel):
    """
    A proposed side effect that a human must approve before it runs.
    `type` tells the backend which handler executes it on approval.
    """
    id: Optional[str] = None                      # assigned when persisted
    label: str                                    # button text, e.g. "Add summary to record"
    type: str                                     # handler key, e.g. "save_summary"
    payload: dict = Field(default_factory=dict)   # data the handler needs
    status: Literal["proposed", "approved", "executed", "dismissed"] = "proposed"


class AgentResult(BaseModel):
    """The full envelope one agent run returns."""
    agent: str                                    # agent key, e.g. "patient_intelligence"
    subject: str                                  # what it ran on, e.g. "patient:12"
    summary: str                                  # short human-readable headline
    insights: list[Insight] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    actions: list[Action] = Field(default_factory=list)
    source: Literal["llm", "demo"] = "llm"        # was this a real model call or offline fallback
