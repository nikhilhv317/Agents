"""Role-based access: front desk vs clinician vs admin."""
import db


def test_login_returns_role(client):
    r = client.post("/api/auth/login", json={"username": "admin", "password": "admin-test-pw"}).json()
    assert r["role"] == "admin"


def test_front_desk_cannot_take_clinical_actions(client, staff, admin):
    aid = db.save_action("imaging", "patient:2", "Add findings", "add_findings",
                         {"patient_id": 2, "content": "Healthy — no findings"})
    assert client.post(f"/api/actions/{aid}/approve", headers=staff).status_code == 403
    assert client.delete("/api/patients/2", headers=staff).status_code == 403
    assert client.post("/api/reports/generate", json={"patient_id": 2}, headers=staff).status_code == 403
    assert client.post("/api/agents/clinical_assistant/run", json={"patient_id": 2}, headers=staff).status_code == 403
    # but the front desk's own work is allowed
    assert client.get("/api/patients", headers=staff).status_code == 200
    assert client.post("/api/agents/scheduling/run", json={}, headers=staff).status_code == 200


def test_clinician_can_approve_but_not_manage_staff(client, clinician):
    aid = db.save_action("imaging", "patient:3", "Add findings", "add_findings",
                         {"patient_id": 3, "content": "Healthy — no findings"})
    assert client.post(f"/api/actions/{aid}/approve", headers=clinician).status_code == 200
    r = client.post("/api/staff", json={"username": "z", "display_name": "Z", "password": "longenough1"},
                    headers=clinician)
    assert r.status_code == 403


def test_role_change_and_deactivation_apply_immediately(client, admin):
    db.create_staff("temp", "Temp", "temp-test-pw", role="clinician")
    tok = {"Authorization": "Bearer " + client.post(
        "/api/auth/login", json={"username": "temp", "password": "temp-test-pw"}).json()["token"]}
    sid = next(s["id"] for s in db.list_staff() if s["username"] == "temp")
    assert client.put(f"/api/staff/{sid}/role", json={"role": "front_desk"}, headers=admin).status_code == 200
    aid = db.save_action("imaging", "patient:4", "x", "flag_specialist", {})
    assert client.post(f"/api/actions/{aid}/approve", headers=tok).status_code == 403   # same token, new role
    client.delete(f"/api/staff/{sid}", headers=admin)
    assert client.get("/api/patients", headers=tok).status_code == 401                 # deactivated
