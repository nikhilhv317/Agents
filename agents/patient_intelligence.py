"""
Patient Intelligence Agent  (deck slide: Role = Doctor / Front Desk)

Wakes up when a patient record is opened. Reads the patient + their visit
history and produces a synthesised clinical brief so the clinician doesn't
hand-write it: a summary, risk flags (insights), suggested next steps
(recommendations), and approvable actions.

Prediction/synthesis here is done by the LLM. If the model is unavailable it
falls back to a deterministic brief built from the same data, so the flow is
always demonstrable.
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel

from agents.base import BaseAgent
from contracts import AgentResult, Insight, Recommendation, Action
import llm
import db

# --- Agentic pattern: retrieval before reasoning / context selection -----------
# Dumping a patient's ENTIRE visit history into the prompt doesn't scale (a
# long-term patient can have dozens of visits) and dilutes the model's attention
# with irrelevant old visits. Instead we select a small, relevant context window:
# the most recent visits (recency) plus any older visit that shares a keyword
# with those recent visits (continuity — e.g. a recurring complaint or an
# ongoing course of treatment). This is a simple stand-in for retrieval; the
# same slot is where a real embedding-similarity search would plug in later.
MAX_CONTEXT_VISITS = 5
RECENT_ANCHOR = 2  # always-included most recent visits, used to score relevance

_STOPWORDS = {"the", "a", "an", "and", "or", "for", "of", "to", "with", "on",
              "in", "at", "is", "was", "review", "follow-up", "check-up"}


def _keywords(text: str) -> set[str]:
    return {w.strip(".,()") for w in (text or "").lower().split()} - _STOPWORDS


def _select_relevant_visits(visits: list[dict], max_visits: int = MAX_CONTEXT_VISITS) -> tuple[list[dict], int]:
    """Return (selected visits in chronological order, total visit count).
    Recency-anchored, then backfilled with older visits that share vocabulary
    with the recent ones — a cheap proxy for 'topically related to now'."""
    total = len(visits)
    if total <= max_visits:
        return visits, total

    recent = visits[-RECENT_ANCHOR:]
    anchor_kw = set().union(*(_keywords(f"{v['complaint']} {v['diagnosis']}") for v in recent))

    older = visits[:-RECENT_ANCHOR]
    scored = sorted(
        older,
        key=lambda v: len(_keywords(f"{v['complaint']} {v['diagnosis']}") & anchor_kw),
        reverse=True,
    )
    fill_count = max(max_visits - len(recent), 0)
    selected = scored[:fill_count] + recent
    # restore chronological order (selection scrambled it)
    selected_dates = {(v["date"], v["diagnosis"]) for v in selected}
    ordered = [v for v in visits if (v["date"], v["diagnosis"]) in selected_dates]
    return ordered, total


# The shape we ask the local model to fill. Passed to Ollama as a JSON schema so
# generation is constrained to exactly this — no stray keys, no prose, valid every
# time. It's the LLM's raw output; reason() maps it onto the shared AgentResult.
class _BriefInsight(BaseModel):
    label: str
    detail: str
    severity: Literal["info", "watch", "urgent"]


class _BriefRec(BaseModel):
    text: str
    rationale: Optional[str] = None


class _BriefAction(BaseModel):
    label: str
    type: Literal["save_summary", "flag_followup", "draft_message"]


class _BriefResponse(BaseModel):
    summary: str
    insights: list[_BriefInsight]
    recommendations: list[_BriefRec]
    actions: list[_BriefAction]

SYSTEM = """You are the Patient Intelligence Agent in a dental practice management system.
You assist a dentist by turning a patient's record and past visits into a concise clinical brief.
You synthesise only from the data given — never invent findings, allergies, or history.
Respond with ONLY a JSON object (no markdown, no prose) of this exact shape:
{
  "summary": "2-3 sentence brief of who this patient is clinically",
  "insights": [{"label": "...", "detail": "...", "severity": "info|watch|urgent"}],
  "recommendations": [{"text": "...", "rationale": "..."}],
  "actions": [{"label": "...", "type": "save_summary|flag_followup|draft_message"}]
}
Prioritise safety-relevant insights (allergies, systemic conditions, drug interactions) as "urgent" or "watch"."""


class PatientIntelligenceAgent(BaseAgent):
    key = "patient_intelligence"
    name = "Patient Intelligence Agent"
    role = "Doctor / Front Desk"

    def gather_context(self, subject_id) -> dict:
        patient = db.get_patient(int(subject_id))
        if patient is None:
            raise ValueError(f"patient {subject_id} not found")
        return patient

    def reason(self, context: dict) -> AgentResult:
        subject = f"patient:{context['id']}"
        selected, total = _select_relevant_visits(context["visits"])
        try:
            data = llm.complete_json(SYSTEM, self._prompt(context, selected),
                                     schema=_BriefResponse.model_json_schema(),
                                     task="patient_brief")
            result = AgentResult(
                agent=self.key, subject=subject, source="llm",
                summary=data.get("summary", ""),
                insights=[Insight(**i) for i in data.get("insights", [])],
                recommendations=[Recommendation(**r) for r in data.get("recommendations", [])],
                actions=[Action(**a) for a in data.get("actions", [])],
            )
        except llm.LLMUnavailable:
            result = self._demo(context, subject, selected)
        # context selection is part of the agent's reasoning, so it's visible,
        # not a silent truncation the clinician can't see
        if total > len(selected):
            result.insights.append(Insight(
                label="Context window",
                detail=f"Showing {len(selected)} of {total} visits — most recent "
                       "plus older visits related to the current ones.",
                severity="info"))
        # so "Save brief" actually persists the brief to the record on approval
        for a in result.actions:
            if a.type == "save_summary":
                a.payload = {"patient_id": context["id"], "content": result.summary}
        return result

    def _prompt(self, p: dict, visits: list[dict]) -> str:
        visit_lines = "\n".join(
            f"- {v['date']}: {v['complaint']} -> {v['diagnosis']} ({v['treatment']}). {v['notes']}"
            for v in visits) or "No prior visits recorded."
        return (
            f"Patient: {p['name']}, {p['age']}y {p['sex']}, MRN {p['mrn']}\n"
            f"Allergies: {p['allergies'] or 'none recorded'}\n"
            f"Medical history: {p['conditions'] or 'none recorded'}\n"
            f"Visit history:\n{visit_lines}\n\n"
            "Produce the clinical brief."
        )

    def _demo(self, p: dict, subject: str, visits: list[dict]) -> AgentResult:
        """Deterministic fallback built from the record — no model needed."""
        insights, recs, actions = [], [], []
        if p["allergies"]:
            insights.append(Insight(label="Allergy on file", severity="urgent",
                                    detail=f"{p['allergies']}. Verify before prescribing."))
        if p["conditions"]:
            insights.append(Insight(label="Systemic history", severity="watch",
                                    detail=f"{p['conditions']}. May affect healing and treatment planning."))
        n = len(visits)
        insights.append(Insight(label="Visit history", severity="info",
                                detail=f"{n} recorded visit(s) in context. Last: "
                                       f"{visits[-1]['diagnosis'] if n else 'none'}."))
        recs.append(Recommendation(text="Review latest diagnosis before this appointment",
                                   rationale="Continuity with prior treatment."))
        if p["conditions"] and "diabet" in p["conditions"].lower():
            recs.append(Recommendation(text="Monitor healing and infection risk",
                                       rationale="Diabetes slows wound healing."))
        actions = [
            Action(label="Add brief to record", type="save_summary"),
            Action(label="Flag for follow-up", type="flag_followup"),
        ]
        summary = (f"{p['name']} ({p['age']}y {p['sex']}), {n} prior visit(s). "
                   f"{'Allergies: ' + p['allergies'] + '. ' if p['allergies'] else ''}"
                   f"{'History: ' + p['conditions'] + '.' if p['conditions'] else 'No significant history.'}")
        return AgentResult(agent=self.key, subject=subject, source="demo",
                           summary=summary, insights=insights,
                           recommendations=recs, actions=actions)


def run(subject_id) -> AgentResult:
    """Registry entry point (see agents/__init__.py)."""
    return PatientIntelligenceAgent().run(subject_id)
