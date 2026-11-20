"""
Visit Pipeline — an event-driven, durable multi-agent workflow (LangGraph).

Until now the agents only connected when a person clicked through each screen:
approve a scan, open Create Report, type findings, generate, finalize. This
chains them, starts on an EVENT, and PAUSES for a human at each decision:

    [event: imaging finding approved]
        -> gather (all of today's imaging)
        -> draft_soap   (Clinical Assistant agent writes the SOAP note)
        -> review_soap  ⏸ interrupt: clinician approves / edits / redrafts / rejects
        -> draft_report (Report agent writes the summary, embeds the scans)
        -> review_report ⏸ interrupt: approve / edit / redraft / reject
        -> finalize  (report finalized + visit added to history)

What makes it agentic infrastructure rather than a script:
  - interrupt(): the graph stops mid-run and waits — minutes or days — for a
    human. The clinician's decision is passed back in with Command(resume=...).
  - SqliteSaver checkpointer: every step is saved to pipeline_checkpoints.db,
    keyed by thread_id. A paused run survives a server restart, and a run that
    was mid-step when the server died is continued from its last checkpoint on
    startup (resume_incomplete).
  - Event-driven: approving an imaging finding starts the pipeline. A second
    image approved while the SOAP note is still in draft triggers a redraft so
    the note covers every image — nobody has to remember to regenerate it.

LangGraph rule that shapes the code: when a run resumes, the node that called
interrupt() RE-RUNS FROM ITS START. So review nodes do nothing before
interrupt() and put all side effects (saving the note, finalizing) after it,
where they run exactly once. Drafting lives in separate nodes for that reason.

All model calls happen on ONE background worker thread: the HTTP request that
approves a scan returns immediately, and LLM work is serialized (one small
local model, one CPU).
"""
from __future__ import annotations

import os
import re
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from typing_extensions import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

import db
from agents.safety import safety_check  # noqa: F401  (also imported from here by older callers)

CHECKPOINT_PATH = os.path.join(os.path.dirname(db.DB_PATH) or ".", "pipeline_checkpoints.db")
NO_IMAGING = "No imaging findings on record."


class VisitState(TypedDict, total=False):
    patient_id: int
    complaint: str
    imaging: str
    soap: dict
    soap_source: str
    soap_drafts: int
    soap_decision: str      # approve | redraft | reject
    report_id: int
    report_summary: str
    report_source: str
    report_decision: str    # approve | redraft | reject
    outcome: str            # done | rejected_soap | rejected_report


def _soap_text(soap: dict) -> str:
    return (f"S: {soap.get('subjective', '')}\nO: {soap.get('objective', '')}\n"
            f"A: {soap.get('assessment', '')}\nP: " + "; ".join(soap.get("plan") or []))


def _log(state: VisitState, msg: str):
    db.add_log("pipeline", f"patient:{state['patient_id']}", msg)



def _grounding_soap(state: VisitState, soap: dict) -> list[dict]:
    """Same check the Clinical Assistant runs on its own drafts; re-run here so a
    clinician's edit or a redraft is checked too."""
    from agents import clinical_assistant
    findings = "" if state.get("imaging") == NO_IMAGING else state.get("imaging", "")
    return clinical_assistant.ground_soap(db.get_patient(state["patient_id"]),
                                          state.get("complaint", ""), findings, soap)


def _grounding_report(state: VisitState, summary: str) -> list[dict]:
    """Same check the Report agent runs on its own drafts; re-run so edits are checked."""
    from agents import report_agent
    return report_agent.ground_report(db.get_patient(state["patient_id"]), state.get("imaging", ""),
                                      _soap_text(state["soap"]), summary)


def _record(patient_id: int) -> dict:
    p = db.get_patient(patient_id)
    return {"allergies": p["allergies"] or "none recorded",
            "conditions": p["conditions"] or "none recorded"}


# ---- nodes ------------------------------------------------------------------

def gather(state: VisitState) -> dict:
    return {"imaging": db.imaging_summary(state["patient_id"])}


def draft_soap(state: VisitState) -> dict:
    from agents import clinical_assistant
    findings = "" if state["imaging"] == NO_IMAGING else state["imaging"]
    r = clinical_assistant.draft(state["patient_id"], state.get("complaint", ""),
                                 findings, propose_actions=False,
                                 findings_from_exam=False)
    n = state.get("soap_drafts", 0) + 1
    _log(state, f"Clinical agent drafted SOAP note (draft {n}) — waiting for review")
    return {"soap": r["soap"], "soap_source": r["source"], "soap_drafts": n}


def review_soap(state: VisitState) -> dict:
    # ⏸ nothing above this line has side effects — it re-runs on resume
    pid = state["patient_id"]
    d = interrupt({"stage": "soap", "soap": state["soap"], "source": state["soap_source"],
                   "imaging": state["imaging"], "drafts": state.get("soap_drafts", 1),
                   "record": _record(pid),
                   "warnings": safety_check(db.get_patient(pid), _soap_text(state["soap"])),
                   "unsupported": _grounding_soap(state, state["soap"])})
    action = d.get("action")
    if action == "approve":
        soap = d.get("soap") or state["soap"]
        db.save_note(pid, "soap", _soap_text(soap))
        left = safety_check(db.get_patient(pid), _soap_text(soap))
        ungrounded = _grounding_soap(state, soap)
        _log(state, f"SOAP note approved by {d.get('by', '?')} and saved to chart"
                    + (f" — despite {len(left)} safety warning(s)" if left else "")
                    + (f" — with {len(ungrounded)} unsourced finding(s) accepted by the clinician"
                       if ungrounded else ""))
        return {"soap": soap, "soap_decision": "approve"}
    if action == "redraft":
        _log(state, f"SOAP redraft requested ({d.get('reason', 'by ' + d.get('by', '?'))})")
        upd = {"soap_decision": "redraft"}
        if d.get("complaint") is not None:
            upd["complaint"] = d["complaint"]
        return upd
    _log(state, f"SOAP note rejected by {d.get('by', '?')} — pipeline stopped")
    return {"soap_decision": "reject", "outcome": "rejected_soap"}


def draft_report(state: VisitState) -> dict:
    from agents import report_agent
    if state.get("report_id"):   # a redraft replaces the previous draft
        db.update_report(state["report_id"], status="discarded")
    r = report_agent.generate(state["patient_id"], _soap_text(state["soap"]), clinical_from_exam=False)
    _log(state, f"Report agent drafted report #{r['report_id']} — waiting for review")
    return {"report_id": r["report_id"], "report_summary": r["ai_summary"],
            "report_source": r["source"]}


def review_report(state: VisitState) -> dict:
    pid = state["patient_id"]
    d = interrupt({"stage": "report", "report_id": state["report_id"],
                   "summary": state["report_summary"], "source": state["report_source"],
                   "record": _record(pid),
                   "warnings": safety_check(db.get_patient(pid), state["report_summary"]),
                   "unsupported": _grounding_report(state, state["report_summary"])})
    action = d.get("action")
    if action == "approve":
        from agents import report_agent
        summary = d.get("summary") or state["report_summary"]
        if d.get("summary"):
            db.update_report(state["report_id"], ai_summary=summary)
        report_agent.finalize(state["report_id"])
        left = safety_check(db.get_patient(pid), summary)
        ungrounded = _grounding_report(state, summary)
        _log(state, f"Report #{state['report_id']} approved by {d.get('by', '?')} and finalized"
                    + (f" — despite {len(left)} safety warning(s)" if left else "")
                    + (f" — with {len(ungrounded)} unsourced finding(s) accepted by the clinician"
                       if ungrounded else ""))
        return {"report_decision": "approve", "outcome": "done"}
    if action == "redraft":
        _log(state, f"Report redraft requested by {d.get('by', '?')}")
        return {"report_decision": "redraft"}
    db.update_report(state["report_id"], status="discarded")
    _log(state, f"Report rejected by {d.get('by', '?')} — pipeline stopped")
    return {"report_decision": "reject", "outcome": "rejected_report"}


def _after_soap(state: VisitState) -> str:
    return {"approve": "draft_report", "redraft": "gather"}.get(state["soap_decision"], END)


def _after_report(state: VisitState) -> str:
    return {"redraft": "draft_report"}.get(state["report_decision"], END)


def build_graph(checkpointer):
    g = StateGraph(VisitState)
    for name, fn in [("gather", gather), ("draft_soap", draft_soap), ("review_soap", review_soap),
                     ("draft_report", draft_report), ("review_report", review_report)]:
        g.add_node(name, fn)
    g.add_edge(START, "gather")
    g.add_edge("gather", "draft_soap")
    g.add_edge("draft_soap", "review_soap")
    g.add_conditional_edges("review_soap", _after_soap,
                            {"draft_report": "draft_report", "gather": "gather", END: END})
    g.add_edge("draft_report", "review_report")
    g.add_conditional_edges("review_report", _after_report,
                            {"draft_report": "draft_report", END: END})
    return g.compile(checkpointer=checkpointer)


_conn = sqlite3.connect(CHECKPOINT_PATH, check_same_thread=False)
_graph = build_graph(SqliteSaver(_conn))
_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="visit-pipeline")


def _config(run: dict) -> dict:
    return {"configurable": {"thread_id": run["thread_id"]}}


def pending(run: dict) -> Optional[dict]:
    """The interrupt payload a paused run is waiting on (the draft to review)."""
    snap = _graph.get_state(_config(run))
    return snap.interrupts[0].value if snap.interrupts else None


def _sync_status(run_id: int) -> dict:
    """Derive the run's status from the checkpoint — the graph is the source of truth."""
    run = db.get_pipeline_run(run_id)
    snap = _graph.get_state(_config(run))
    if snap.interrupts:
        status = f"awaiting_{snap.interrupts[0].value['stage']}"
    elif not snap.next:
        status = "done" if snap.values.get("outcome") == "done" else "rejected"
    else:
        status = "running"
    db.update_pipeline_run(run_id, status=status)
    return db.get_pipeline_run(run_id)


def _execute(run_id: int, graph_input):
    """Worker-thread body: advance the graph until it pauses or ends."""
    run = db.get_pipeline_run(run_id)
    try:
        db.update_pipeline_run(run_id, status="running", error=None)
        _graph.invoke(graph_input, _config(run))
        run = _sync_status(run_id)
        # an image was approved while we were drafting: redraft so the note covers it
        while run["status"] == "awaiting_soap" and run["rerun_requested"]:
            db.update_pipeline_run(run_id, rerun_requested=0)
            _graph.invoke(Command(resume={"action": "redraft", "reason": "new imaging approved"}),
                          _config(run))
            run = _sync_status(run_id)
    except Exception as e:
        db.update_pipeline_run(run_id, status="failed", error=str(e)[:500])
        db.add_log("pipeline", f"patient:{run['patient_id']}", f"Pipeline failed: {e}")


# ---- public API -------------------------------------------------------------

def start(patient_id: int, complaint: str = "", trigger: str = "manual") -> dict:
    p = db.get_patient(int(patient_id))
    if p is None:
        raise ValueError("patient not found")
    active = db.active_pipeline_run(p["id"])
    if active:
        raise ValueError(f"patient already has an unfinished visit pipeline (run #{active['id']})")
    thread_id = f"visit-{p['id']}-{uuid.uuid4().hex[:8]}"
    run_id = db.create_pipeline_run(thread_id, p["id"], p["name"], trigger)
    db.add_log("pipeline", f"patient:{p['id']}", f"Visit pipeline started ({trigger})")
    _worker.submit(_execute, run_id, {"patient_id": p["id"], "complaint": complaint})
    return db.get_pipeline_run(run_id)


def resume(run_id: int, decision: dict) -> dict:
    """Pass a human decision into a paused run. decision['stage'] must match
    what the run is actually waiting on, so a stale browser tab can't approve
    the wrong step."""
    run = db.get_pipeline_run(run_id)
    if run is None:
        raise ValueError("run not found")
    if run["status"] != f"awaiting_{decision.get('stage')}":
        raise ValueError(f"run is '{run['status']}', not awaiting {decision.get('stage')}")
    if decision.get("action") not in ("approve", "redraft", "reject"):
        raise ValueError("action must be approve, redraft or reject")
    db.update_pipeline_run(run_id, status="running")
    _worker.submit(_execute, run_id, Command(resume=decision))
    return db.get_pipeline_run(run_id)


def on_imaging_approved(patient_id: int) -> str:
    """EVENT handler: an imaging finding was approved onto a patient's chart."""
    active = db.active_pipeline_run(patient_id)
    if active is None:
        start(patient_id, trigger="imaging finding approved")
        return "Visit pipeline started: SOAP note is being drafted."
    if active["status"] == "awaiting_soap":
        db.update_pipeline_run(active["id"], status="running")
        _worker.submit(_execute, active["id"],
                       Command(resume={"action": "redraft", "reason": "new imaging approved"}))
        return "SOAP draft is being updated to include the new finding."
    if active["status"] == "running":
        db.update_pipeline_run(active["id"], rerun_requested=1)
        return "Pipeline is drafting; it will redraft to include this finding."
    db.add_log("pipeline", f"patient:{patient_id}",
               "New imaging approved after the SOAP note was signed off — not included in this visit's report")
    return "SOAP note already approved; this finding is not in the current report."


def list_runs(limit: int = 30) -> list[dict]:
    runs = db.list_pipeline_runs(limit)
    for r in runs:
        r["pending"] = pending(r) if r["status"].startswith("awaiting_") else None
    return runs


def resume_incomplete():
    """On startup: continue runs that were mid-step when the server stopped.
    invoke(None) picks up from the last saved checkpoint."""
    for r in db.list_pipeline_runs(limit=200, statuses=["running"]):
        db.add_log("pipeline", f"patient:{r['patient_id']}",
                   f"Resuming run #{r['id']} from its last checkpoint after restart")
        _worker.submit(_execute, r["id"], None)
