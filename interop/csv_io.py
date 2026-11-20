"""
CSV import — the generic connector for practice-management systems: almost all
of them can export patients and appointments as CSV. Column names are matched
loosely (case, spaces, common synonyms). Each row is validated on its own; a bad
row is reported and skipped, never half-imported.

Patients:      name, phone, age, sex, allergies, conditions
Appointments:  name (or mrn) + phone, date (YYYY-MM-DD or DD/MM/YYYY), time (HH:MM), type
"""
from __future__ import annotations

import csv
import io
from datetime import datetime

import db
from interop.fhir import _upsert

_SYNONYMS = {"name": {"name", "patient", "patient name", "full name"},
             "phone": {"phone", "mobile", "phone number", "contact", "tel"},
             "age": {"age"}, "sex": {"sex", "gender"},
             "allergies": {"allergies", "allergy"}, "conditions": {"conditions", "medical history", "history"},
             "mrn": {"mrn", "patient id", "chart number"},
             "date": {"date", "appointment date", "appt date"}, "time": {"time", "appointment time", "slot"},
             "type": {"type", "appointment type", "procedure", "reason"}}
APPT_TYPES = {"checkup", "cleaning", "filling", "root_canal", "extraction", "emergency", "followup"}


def _rows(text: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    out = []
    for raw in reader:
        row = {}
        for k, v in raw.items():
            key = (k or "").strip().lower()
            canon = next((c for c, names in _SYNONYMS.items() if key in names), None)
            if canon:
                row[canon] = (v or "").strip()
        out.append(row)
    return out


def import_patients(text: str) -> dict:
    created = updated = 0
    errors = []
    for i, r in enumerate(_rows(text), start=2):          # row 1 is the header
        if not r.get("name"):
            errors.append({"row": i, "error": "name is empty"}); continue
        try:
            age = int(r["age"]) if r.get("age") else None
        except ValueError:
            errors.append({"row": i, "error": f"age '{r['age']}' is not a number"}); continue
        sex = {"male": "M", "m": "M", "female": "F", "f": "F"}.get((r.get("sex") or "").lower(), "")
        res = _upsert(r["name"], r.get("phone", ""), age, sex, r.get("allergies", ""), r.get("conditions", ""))
        created += res["created"]; updated += not res["created"]
    return {"created": created, "updated": updated, "errors": errors}


def _parse_date(s: str) -> str:
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            pass
    raise ValueError(f"date '{s}' not recognised (use YYYY-MM-DD)")


def import_appointments(text: str) -> dict:
    booked = duplicates = 0
    errors = []
    for i, r in enumerate(_rows(text), start=2):
        try:
            pid = None
            if r.get("mrn"):
                conn = db.connect()
                hit = conn.execute("SELECT id FROM patients WHERE mrn=? AND active=1", (r["mrn"],)).fetchone()
                conn.close()
                pid = hit["id"] if hit else None
            if pid is None:
                if not r.get("name"):
                    raise ValueError("needs a name or a known mrn")
                pid = _upsert(r["name"], r.get("phone", ""), None, "", "", "")["patient_id"]
            appt_type = (r.get("type") or "checkup").strip().lower().replace(" ", "_")
            if appt_type not in APPT_TYPES:
                raise ValueError(f"type '{r.get('type')}' not one of {sorted(APPT_TYPES)}")
            time_ = r.get("time") or ""
            if not time_ or ":" not in time_:
                raise ValueError("time must be HH:MM")
            _, dup = db.create_appointment(pid, _parse_date(r.get("date", "")), time_, appt_type)
            duplicates += dup; booked += not dup
        except ValueError as e:
            errors.append({"row": i, "error": str(e)})
    return {"booked": booked, "duplicates_skipped": duplicates, "errors": errors}
