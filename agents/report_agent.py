"""
Report Agent — the end of the pipeline.

Takes a patient + their collected imaging findings + the clinician's findings and
produces a short (3-4 line) AI summary with a TINY local model (fast). The report
is saved as a draft, then finalized on approval.

Speed is the point: qwen3:1.7b with think=False, a compact schema, one short call.

--- Agentic pattern: agent-to-agent handoff -------------------------------------
This agent does not talk to the Imaging & Diagnostic Agent directly. It reads
the imaging agent's PERSISTED OUTPUT (patient_notes, kind='imaging') as its own
input context — the handoff is the shared record, not a function call between
agents. This is a deliberate choice: it keeps agents decoupled (report_agent has
no import-time dependency on imaging_agent) while still composing their work.
`_receive_imaging_handoff()` below names and logs that handoff explicitly so it's
visible in the Agent Activity feed, not just an implicit SQL join.
"""
from __future__ import annotations

from pydantic import BaseModel
import db
import llm

SYSTEM = """You are a dental Report Assistant. From the numbered sources only, write a
concise clinical summary of 3-4 short lines. No headings, no preamble, no bullet points.
Never invent findings, examinations, measurements or tooth numbers.
Always state each recorded allergy and condition exactly as written in [R].
End every sentence with the tag(s) of the source(s) it uses, e.g. "Advanced caries on imaging [I]."
Respond with ONLY JSON: {"summary": "the 3-4 line summary"}."""


class _Summary(BaseModel):
    summary: str


def _receive_imaging_handoff(patient_id: int) -> str:
    """Pull the Imaging & Diagnostic Agent's output for this patient and log the
    handoff. ALL imaging findings from this visit are pulled, not just the most
    recent one — multiple images can each detect different conditions, and the
    report agent's context should reflect all of them."""
    imaging = db.imaging_summary(patient_id)
    if imaging and imaging != "No imaging findings on record.":
        db.add_log("report", f"patient:{patient_id}",
                   "Handoff received from imaging agent: incorporating imaging "
                   "findings into report context.")
    return imaging


def ground_report(patient: dict, imaging: str, clinical: str, summary: str,
                  exam_given: bool = False, require_citations: bool = False) -> list[dict]:
    """Every finding in the summary must come from what the model was given:
    the record, imaging and clinician findings. Visit history is NOT in the
    prompt, so a finding found only there is flagged as copied from a past visit.
    Citations are checked when present (always, for model drafts)."""
    from agents import grounding
    sources = (f"Imaging: {imaging}\nClinical: {clinical}\n" + grounding.record_sources(patient))
    found = grounding.check(summary, sources,
                            exam_findings_given=exam_given or grounding.mentions_exam(sources),
                            history=grounding.history_sources(patient))
    if require_citations:
        found += grounding.no_citations_issue(summary)
    if grounding.has_citations(summary):
        found += grounding.check_citations(summary, grounding.report_sources(patient, imaging, clinical),
                                           require=require_citations)
    return found


def _prompt(patient: dict, imaging: str, clinical: str) -> str:
    """The user prompt (also used verbatim to build the LoRA training set, ml/lora/)."""
    from agents import grounding
    from agents.untrusted import defuse   # the name can come from the public booking page
    return (
        f"Patient: {defuse(patient['name'])}, {patient['age']}y {patient['sex']}\n"
        f"Allergies: {patient['allergies'] or 'none recorded'}\n"
        f"Medical conditions: {patient['conditions'] or 'none recorded'}\n\n"
        f"Sources:\n{grounding.format_sources(grounding.report_sources(patient, imaging, clinical))}\n\n"
        "Write the 3-4 line summary."
    )


def generate(patient_id: int, clinical: str = "", clinical_from_exam: bool = True) -> dict:
    """clinical_from_exam: the clinician typed these findings, so the summary may
    describe an examination. The pipeline passes False (its 'clinical' is the SOAP note)."""
    patient = db.get_patient(int(patient_id))
    if patient is None:
        raise ValueError(f"patient {patient_id} not found")

    imaging = _receive_imaging_handoff(patient_id)

    from agents import grounding
    prompt = _prompt(patient, imaging, clinical)
    try:
        data = llm.complete_json(SYSTEM, prompt, schema=_Summary.model_json_schema(),
                                 task="report_summary")
        ai_summary = data.get("summary", "").strip()
        source = "llm"
    except llm.LLMUnavailable:
        bits = [f"{patient['name']} ({patient['age']}y {patient['sex']})."]
        if patient["conditions"]:
            bits.append(f"History: {patient['conditions']}.")
        bits.append(f"Imaging: {imaging}")
        if clinical:
            bits.append(f"Clinical: {clinical}")
        bits.append("Clinician review recommended before finalising.")
        ai_summary = " ".join(bits)
        source = "demo"

    # attach the patient's recent scan images so the printed report can show them
    import json as _json
    imgs = db.get_scan_images(patient_id, limit=100)   # ALL images from the visit
    images_json = _json.dumps(imgs) if imgs else None

    rid = db.create_report(patient_id, patient["name"], imaging, clinical,
                           ai_summary, status="draft", images=images_json)
    unsupported = ground_report(patient, imaging, clinical, ai_summary,
                                exam_given=clinical_from_exam and bool(clinical.strip()),
                                require_citations=(source == "llm"))
    if unsupported:
        db.add_log("report", f"patient:{patient_id}",
                   f"Grounding check: {len(unsupported)} unsourced finding(s) in report draft #{rid}")
    return {"report_id": rid, "patient_id": patient_id, "patient": patient["name"],
            "imaging": imaging, "clinical": clinical, "ai_summary": ai_summary,
            "source": source, "status": "draft", "images": imgs, "unsupported": unsupported,
            "sources": grounding.report_sources(patient, imaging, clinical)}


def finalize(report_id: int) -> dict:
    """Finalize a draft and record the encounter in the visit history. Shared by
    the Reports screen and the visit pipeline so both do exactly the same thing."""
    from datetime import date as _date
    db.finalize_report(report_id)
    r = db.get_report(report_id)
    db.add_visit(r["patient_id"], _date.today().isoformat(),
                 (r["clinical"] or "Visit")[:80], "See report", "",
                 r["ai_summary"] or "")
    db.add_log("report", f"patient:{r['patient_id']}", f"Report finalized for {r['patient_name']}")
    return r
