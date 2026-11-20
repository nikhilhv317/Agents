"""
Clinical Assistant Agent (deck: Role = Doctor, "Reports / SOAP notes").

Drafts a structured SOAP note (Subjective / Objective / Assessment / Plan) from
the patient's record + visit history + today's chief complaint and findings.
The local LLM writes the note; a clinician reviews and approves before it's saved.

Safety flags (allergies, systemic conditions) are read straight from the record
— never left to the model to remember. Falls back to a deterministic templated
note if the model is unavailable, so it always produces something.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel

import db
import llm

SYSTEM = """You are a dental Clinical Assistant. Draft a concise, professional SOAP note
from the patient record and today's visit. Synthesise ONLY from the data given — never
invent findings, measurements, or history. Respond with ONLY a JSON object of this shape:
{
  "subjective": "what the patient reports today, with relevant history context",
  "objective": "clinical/exam/imaging findings stated for today",
  "assessment": "clinical impression",
  "plan": ["concrete next step", "..."]
}
Keep each section a few sentences; the plan is 2-4 short items."""


class _Soap(BaseModel):
    subjective: str
    objective: str
    assessment: str
    plan: list[str]


def draft(patient_id: int, complaint: str = "", findings: str = "",
          propose_actions: bool = True, findings_from_exam: Optional[bool] = None) -> dict:
    """propose_actions=False: the caller owns approval (the visit pipeline
    pauses its graph for review instead of creating Approve buttons).
    findings_from_exam: may the note describe an examination? Default: yes if the
    clinician typed findings, no if they were auto-pulled from imaging."""
    patient = db.get_patient(int(patient_id))
    if patient is None:
        raise ValueError(f"patient {patient_id} not found")

    # auto-pull the most recent imaging finding for this patient if the clinician
    # didn't type findings — this is the Scan -> Note loop.
    if findings_from_exam is None:
        findings_from_exam = bool(findings.strip())
    if not findings.strip():
        li = db.latest_note(patient_id, "imaging")
        if li:
            findings = li["content"]

    prompt = _prompt(patient, complaint, findings)
    try:
        data = llm.complete_json(SYSTEM, prompt, schema=_Soap.model_json_schema(),
                                 task="soap_note")
        soap = {"subjective": data.get("subjective", ""),
                "objective": data.get("objective", ""),
                "assessment": data.get("assessment", ""),
                "plan": data.get("plan", []) or []}
        source = "llm"
    except llm.LLMUnavailable:
        soap = _demo_soap(patient, complaint, findings)
        source = "demo"

    # safety flags come from the record, not the model
    insights = []
    if patient["allergies"]:
        insights.append({"label": "Allergy on file", "severity": "urgent",
                         "detail": f"{patient['allergies']}. Verify before prescribing."})
    if patient["conditions"]:
        insights.append({"label": "Systemic history", "severity": "watch",
                         "detail": f"{patient['conditions']}. Consider in treatment planning."})

    unsupported = ground_soap(patient, complaint, findings, soap, findings_from_exam)
    if unsupported:
        insights.append({"label": f"Grounding check: {len(unsupported)} unsourced finding(s)",
                         "severity": "watch",
                         "detail": "; ".join(f"{u['claim']}: {u['detail']}" for u in unsupported)
                                   + " Verify or remove before saving."})

    recommendations = [{"text": p, "rationale": None} for p in soap["plan"]]

    note_text = (f"S: {soap['subjective']}\nO: {soap['objective']}\n"
                 f"A: {soap['assessment']}\nP: " + "; ".join(soap["plan"]))
    actions = []
    for label, atype in ([("Save note to record", "save_note"),
                          ("Flag for follow-up", "flag_followup")] if propose_actions else []):
        payload = {"assessment": soap["assessment"]}
        if atype == "save_note":
            payload = {"patient_id": patient_id, "content": note_text}
        aid = db.save_action("clinical", f"patient:{patient_id}", label, atype, payload)
        actions.append({"id": str(aid), "label": label, "type": atype, "status": "proposed"})

    return {
        "agent": "clinical", "subject": f"patient:{patient_id}", "source": source,
        "patient": patient["name"], "soap": soap, "used_finding": bool(findings.strip()),
        "summary": f"SOAP note drafted for {patient['name']}"
                   + (f" — {soap['assessment']}" if soap["assessment"] else ""),
        "insights": insights, "recommendations": recommendations, "actions": actions,
        "unsupported": unsupported,
    }


def ground_soap(patient: dict, complaint: str, findings: str, soap: dict,
                exam_given: bool = False) -> list[dict]:
    """Every finding in the note must come from what the model was given.
    Objective = what is observed TODAY: only today's complaint/findings (+ the
    record) count, past visits don't. Subjective and Assessment may also draw on
    history. The plan is recommendations, not findings, so it isn't checked."""
    from agents import grounding
    today = f"Complaint: {complaint}\nFindings: {findings}\n" + grounding.record_sources(patient)
    exam = exam_given or grounding.mentions_exam(today)
    found = grounding.check(soap.get("objective", ""), today, exam_findings_given=exam,
                            history=grounding.history_sources(patient))
    found += grounding.check(f"{soap.get('subjective', '')}\n{soap.get('assessment', '')}",
                             today + "\n" + grounding.history_sources(patient),
                             exam_findings_given=exam)
    seen, out = set(), []
    for f in found:
        if f["claim"] not in seen:
            seen.add(f["claim"]); out.append(f)
    return out


def _prompt(p: dict, complaint: str, findings: str) -> str:
    from agents.untrusted import defuse   # chart text can come from the public booking page
    visits = "\n".join(
        defuse(f"- {v['date']}: {v['complaint']} -> {v['diagnosis']} ({v['treatment']}). {v['notes']}")
        for v in p["visits"]) or "No prior visits recorded."
    return (
        f"Patient: {defuse(p['name'])}, {p['age']}y {p['sex']}, MRN {p['mrn']}\n"
        f"Allergies: {p['allergies'] or 'none recorded'}\n"
        f"Medical history: {p['conditions'] or 'none recorded'}\n"
        f"Visit history:\n{visits}\n\n"
        f"Today's chief complaint: {complaint or '(not provided)'}\n"
        f"Today's exam / imaging findings: {findings or '(not provided)'}\n\n"
        "Draft the SOAP note."
    )


def _demo_soap(p: dict, complaint: str, findings: str) -> dict:
    last = p["visits"][-1] if p["visits"] else None
    subjective = (complaint or "Patient presents for review.")
    if last:
        subjective += f" History notable for {last['diagnosis']} ({last['date']})."
    objective = findings or "Clinical examination performed; findings as charted."
    assessment = (f"Assessment pending clinician review"
                  + (f"; consider continuity with prior {last['diagnosis']}." if last else "."))
    plan = ["Confirm findings on examination",
            "Discuss options with patient"]
    if p["conditions"] and "diabet" in p["conditions"].lower():
        plan.append("Monitor healing given diabetic status")
    return {"subjective": subjective, "objective": objective,
            "assessment": assessment, "plan": plan}
