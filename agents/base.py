"""
BaseAgent — the lifecycle every agent shares.

    trigger  ->  gather_context  ->  reason  ->  AgentResult  ->  persist proposed actions

An agent only ever PROPOSES actions. Nothing with a side effect runs until a
human approves it through the API. Subclasses implement gather_context() and
reason(); everything else (persisting proposed actions, exposing metadata) is
handled here so all seven agents behave identically.
"""
from __future__ import annotations

from contracts import AgentResult
import db


class BaseAgent:
    key: str = "base"          # stable identifier, e.g. "patient_intelligence"
    name: str = "Base Agent"   # display name
    role: str = ""             # deck's "Role" column, e.g. "Doctor"

    def gather_context(self, subject_id) -> dict:
        """Pull whatever the agent needs from the data layer. Override."""
        raise NotImplementedError

    def reason(self, context: dict) -> AgentResult:
        """Turn context into insights/recommendations/actions. Override."""
        raise NotImplementedError

    def run(self, subject_id) -> AgentResult:
        context = self.gather_context(subject_id)
        result = self.reason(context)
        # Persist each proposed action so approval has something to reference.
        for action in result.actions:
            action_id = db.save_action(
                agent=result.agent, subject=result.subject,
                label=action.label, type_=action.type, payload=action.payload)
            action.id = str(action_id)
        return result
