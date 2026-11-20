"""API-level tests: auth, registry, patients, booking, scheduling decisions, reports."""
from datetime import date, timedelta

import db


# ---- auth -----------------------------------------------------------------------

def test_login_rejects_bad_password(client):
    assert client.post("/api/auth/login", json={"username": "admin", "password": "nope"}).status_code == 401


def test_protected_route_needs_token(client):
    assert client.get("/api/patients").status_code == 401


def test_staff_cannot_create_staff(client, staff):
    r = client.post("/api/staff", json={"username": "x", "display_name": "X", "password": "longenough"},
                    headers=staff)
    assert r.status_code == 403


def test_staff_cannot_change_someone_elses_password(client, staff):
    r = client.put("/api/staff/admin/password", json={"new_password": "whatever123"}, headers=staff)
    assert r.status_code == 403


# ---- agent registry ----------------------------------------------------------------

def test_registry_lists_all_agents(client, admin):
    keys = {a["key"] for a in client.get("/api/agents", headers=admin).json()}
    assert {"scheduling", "imaging", "report", "patient_intelligence", "clinical_assistant",
            "visit_pipeline", "copilot", "retraining"} <= keys


def test_registry_rejects_unknown_agent_and_params(client, admin):
    assert client.post("/api/agents/nope/run", json={}, headers=admin).status_code == 404
    r = client.post("/api/agents/report/run", json={"bogus": 1}, headers=admin)
    assert r.status_code == 400 and "bogus" in r.json()["detail"]


def test_registry_sends_file_agents_to_upload_endpoint(client, admin):
    assert client.post("/api/agents/imaging/run", json={}, headers=admin).status_code == 400


def test_registry_admin_only(client, staff):
    assert client.post("/api/agents/retraining/run", json={}, headers=staff).status_code == 403


# ---- patients + public booking -------------------------------------------------------

def test_patient_create_update_soft_delete(client, admin):
    r = client.post("/api/patients", json={"name": "Test Person", "phone": "9000000001"}, headers=admin)
    pid = r.json()["id"]
    assert client.put(f"/api/patients/{pid}", json={"name": "Test Person", "allergies": "Latex"},
                      headers=admin).status_code == 200
    assert db.get_patient(pid)["allergies"] == "Latex"
    assert client.delete(f"/api/patients/{pid}", headers=admin).status_code == 200
    assert pid not in {p["id"] for p in client.get("/api/patients", headers=admin).json()}
    assert db.get_patient(pid) is not None          # soft delete: record kept


def test_public_booking_validates_input(client):
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    bad_phone = {"name": "A", "phone": "123", "appt_date": tomorrow, "appt_time": "10:00"}
    assert client.post("/api/public/book", json=bad_phone).status_code == 400
    past = {"name": "A", "phone": "9000000002", "appt_date": "2020-01-01", "appt_time": "10:00"}
    assert client.post("/api/public/book", json=past).status_code == 400
    ok = {"name": "Booked Person", "phone": "9000000002", "appt_date": tomorrow, "appt_time": "10:00"}
    r = client.post("/api/public/book", json=ok)
    assert r.status_code == 200 and "patient_id" not in r.json()   # never leaks ids


# ---- scheduling decisions --------------------------------------------------------------

def test_decision_resubmit_replaces_instead_of_duplicating(client, admin):
    aid, _ = db.create_appointment(1, date.today().isoformat(), "10:00", "cleaning")
    for d in ("approve", "approve", "hold"):
        body = {"decisions": [{"appt_id": aid, "patient": "R", "appt_type": "cleaning",
                               "appt_hour": 10, "p_attend": 0.5, "decision": d}]}
        assert client.post("/api/scheduling/decisions", json=body, headers=admin).status_code == 200
    conn = db.connect()
    rows = conn.execute("SELECT decision FROM decisions WHERE appt_id=?", (aid,)).fetchall()
    conn.close()
    assert [r[0] for r in rows] == ["hold"]
    assert aid not in {q["appt_id"] for q in db.approved_today()}


# ---- reports (works with or without Ollama: falls back to a templated summary) ----------

def test_report_generate_and_finalize_adds_visit(client, admin):
    before = len(db.get_patient(4)["visits"])
    r = client.post("/api/reports/generate", json={"patient_id": 4, "clinical": "Fluoride applied."},
                    headers=admin)
    assert r.status_code == 200 and r.json()["status"] == "draft"
    assert client.post(f"/api/reports/{r.json()['report_id']}/finalize", headers=admin).status_code == 200
    assert len(db.get_patient(4)["visits"]) == before + 1
