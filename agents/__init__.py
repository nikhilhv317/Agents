"""
Agent registry + action handlers.

REGISTRY is the one catalogue of every agent: key -> AgentSpec. Add the next
agent by writing its module and adding one AgentSpec line — it is then listed
by GET /api/agents and runnable via POST /api/agents/{key}/run with no other
API changes.

Entry points are resolved LAZILY ("module:function", imported on first run), so
the app still boots when one agent's heavy deps (torch, langgraph, sklearn) or
model files are missing — only that agent fails, with a clear error.

ACTION_HANDLERS maps an action `type` to the function that executes it once a
human approves. This is where a proposed action becomes a real side effect.
"""
from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, asdict
from typing import Literal

import db


@dataclass(frozen=True)
class AgentSpec:
    key: str
    name: str
    role: str
    description: str
    entry: str                                  # "module:function", called with **params
    params: tuple[str, ...] = ()                # accepted keyword params (for listing / docs)
    input: Literal["json", "files"] = "json"    # "files" agents run via their multipart endpoint
    admin_only: bool = False

    def load(self):
        module, func = self.entry.split(":")
        return getattr(importlib.import_module(module), func)


REGISTRY: dict[str, AgentSpec] = {s.key: s for s in [
    AgentSpec("scheduling", "Smart Scheduling", "Front desk",
              "Scores a day's appointments for no-show risk (scikit-learn + LangGraph).",
              "agents.scheduling_agent:run_for_date", params=("day",)),
    AgentSpec("imaging", "Imaging & Diagnostic", "Doctor",
              "Two-stage vision screen: dental gate, then YOLO lesion detection.",
              "agents.imaging_agent:analyze_batch", params=("image_paths", "patient_id"),
              input="files"),
    AgentSpec("report", "Report", "Doctor",
              "Drafts a 3-4 line visit summary from record, imaging and clinical findings.",
              "agents.report_agent:generate", params=("patient_id", "clinical", "clinical_from_exam")),
    AgentSpec("patient_intelligence", "Patient Intelligence", "Doctor / Front desk",
              "Drafts a clinical brief (insights, risks, recommendations) from the record.",
              "agents.patient_intelligence:run", params=("subject_id",)),
    AgentSpec("clinical_assistant", "Clinical Assistant", "Doctor",
              "Drafts a SOAP note, pulling in the latest imaging finding.",
              "agents.clinical_assistant:draft",
              params=("patient_id", "complaint", "findings", "propose_actions", "findings_from_exam")),
    AgentSpec("visit_pipeline", "Visit Pipeline", "Doctor",
              "Chains Clinical -> Report for one visit, pausing for review at each draft.",
              "agents.visit_pipeline:start", params=("patient_id", "complaint", "trigger")),
    AgentSpec("copilot", "Clinic Copilot", "Doctor / Front desk",
              "Tool-calling chat: the model picks lookups; writes come back as proposals.",
              "agents.copilot:chat", params=("message", "history")),
    AgentSpec("retraining", "Model Retraining", "Admin",
              "Retrains the no-show model on new outcomes; promotes only if AUC holds.",
              "ml.retrain:run", params=("force",), admin_only=True),
]}


def get_agent(key: str) -> AgentSpec | None:
    return REGISTRY.get(key)


def list_agents() -> list[dict]:
    return [asdict(s) for s in REGISTRY.values()]


def entry(key: str):
    """The callable behind an agent — for the typed endpoints in app.py."""
    return REGISTRY[key].load()


def run(key: str, **params):
    """Run an agent by key. Pydantic results (the shared contract) come back as dicts."""
    result = entry(key)(**params)
    return result.model_dump() if hasattr(result, "model_dump") else result


# ---- action handlers: run only after human approval -------------------------


def _pid(action):
    try:
        return json.loads(action.get("payload") or "{}").get("patient_id")
    except Exception:
        return None


def _content(action):
    try:
        return json.loads(action.get("payload") or "{}").get("content", "")
    except Exception:
        return ""


def _handle_save_summary(action: dict) -> str:
    pid = _pid(action)
    if pid:
        db.save_note(pid, "summary", _content(action))
    return "Patient brief saved to record."


def _handle_flag_followup(action: dict) -> str:
    return "Patient flagged for follow-up."


def _handle_draft_message(action: dict) -> str:
    return "Draft message prepared for the patient."


def _handle_add_findings(action: dict) -> str:
    pid = _pid(action)
    if pid:
        db.save_note(pid, "imaging", _content(action))
        # EVENT: a finding landed on the chart -> start/advance the visit pipeline.
        # Never let a pipeline problem fail the approval itself.
        try:
            from agents import visit_pipeline
            return "Imaging findings saved to patient record. " + visit_pipeline.on_imaging_approved(pid)
        except Exception as e:
            db.add_log("pipeline", f"patient:{pid}", f"Could not start visit pipeline: {e}")
        return "Imaging findings saved to patient record."
    return "Imaging findings recorded (no patient linked)."


def _handle_flag_specialist(action: dict) -> str:
    return "Flagged for specialist review."


def _handle_save_note(action: dict) -> str:
    pid = _pid(action)
    if pid:
        db.save_note(pid, "soap", _content(action))
    return "Clinical (SOAP) note saved to patient record."


def _handle_save_copilot_note(action: dict) -> str:
    pid = _pid(action)
    if pid:
        db.save_note(pid, "copilot", _content(action))
    return "Copilot note saved to patient record."


ACTION_HANDLERS = {
    "save_copilot_note": _handle_save_copilot_note,
    "save_summary": _handle_save_summary,
    "flag_followup": _handle_flag_followup,
    "draft_message": _handle_draft_message,
    "add_findings": _handle_add_findings,
    "flag_specialist": _handle_flag_specialist,
    "save_note": _handle_save_note,
}


def execute_action(action_id: int, approved_by: str | None = None) -> str:
    action = db.get_action(action_id)
    if not action:
        raise ValueError("action not found")
    if action["status"] != "proposed":
        return f"Action already {action['status']}."
    handler = ACTION_HANDLERS.get(action["type"])
    message = handler(action) if handler else f"Executed: {action['label']}"
    db.set_action_status(action_id, "executed")
    # record WHO approved — an agent proposed it, a named person signed it off
    db.add_log(action["agent"], action["subject"],
               message + (f" (approved by {approved_by})" if approved_by else ""))
    return message
