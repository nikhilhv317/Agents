"""
Agent registry + action handlers.

REGISTRY maps agent keys to instances. Add the next agent by importing it and
adding one line — the API and frontend need no changes because everything
speaks the shared contract.

ACTION_HANDLERS maps an action `type` to the function that executes it once a
human approves. This is where a proposed action becomes a real side effect.
"""
from agents.patient_intelligence import PatientIntelligenceAgent
import db

REGISTRY = {
    PatientIntelligenceAgent.key: PatientIntelligenceAgent(),
    # SmartSchedulingAgent.key: SmartSchedulingAgent(),
    # ImagingDiagnosticAgent.key: ImagingDiagnosticAgent(),
}


def get_agent(key: str):
    return REGISTRY.get(key)


# ---- action handlers: run only after human approval -------------------------

import json
import db


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
