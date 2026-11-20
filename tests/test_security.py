"""Encryption at rest and clinic separation."""
import db


def test_sensitive_fields_are_ciphertext_on_disk_and_readable_in_app(client, admin):
    pid = client.post("/api/patients", json={"name": "Crypto Test", "phone": "9123456780",
                                             "allergies": "Penicillin", "conditions": "Asthma"},
                      headers=admin).json()["id"]
    db.save_note(pid, "soap", "Tender to percussion on #46.")
    conn = db.connect()
    raw = conn.execute("SELECT name, phone, allergies, conditions FROM patients WHERE id=?", (pid,)).fetchone()
    note = conn.execute("SELECT content FROM patient_notes WHERE patient_id=?", (pid,)).fetchone()[0]
    conn.close()
    assert raw["name"] == "Crypto Test"                       # names stay searchable
    for v in (raw["phone"], raw["allergies"], raw["conditions"], note):
        assert v.startswith(db.ENC_PREFIX) and "Penicillin" not in v and "percussion" not in v
    p = client.get(f"/api/patients/{pid}", headers=admin).json()
    assert p["allergies"] == "Penicillin" and p["phone"] == "9123456780"
    assert db.get_notes(pid)[0]["content"] == "Tender to percussion on #46."


def test_plaintext_rows_from_old_databases_get_encrypted():
    conn = db.connect()
    conn.execute("INSERT INTO patient_notes (patient_id, kind, content) VALUES (2, 'soap', 'old plaintext note')")
    conn.commit()
    assert db.migrate_encryption(conn) >= 1
    v = conn.execute("SELECT content FROM patient_notes WHERE content LIKE ?", (db.ENC_PREFIX + "%",)).fetchall()
    conn.close()
    assert any(db.dec(r[0]) == "old plaintext note" for r in v)


def test_other_clinic_cannot_see_or_touch_patients(client, admin):
    db.create_staff("otherclinic", "Other Clinic Dr", "other-test-pw", role="clinician", clinic_id=2)
    tok = {"Authorization": "Bearer " + client.post(
        "/api/auth/login", json={"username": "otherclinic", "password": "other-test-pw"}).json()["token"]}
    assert client.get("/api/patients", headers=tok).json() == []
    assert client.get("/api/patients/1", headers=tok).status_code == 404
    assert client.get("/api/patients/1/fhir", headers=tok).status_code == 404
    assert client.post("/api/reports/generate", json={"patient_id": 1}, headers=tok).status_code == 404
    assert client.post("/api/agents/patient_intelligence/run", json={"subject_id": 1}, headers=tok).status_code == 404
    mine = client.post("/api/patients", json={"name": "Clinic Two Patient"}, headers=tok).json()["id"]
    assert [p["id"] for p in client.get("/api/patients", headers=tok).json()] == [mine]
    assert mine not in {p["id"] for p in client.get("/api/patients", headers=admin).json()}
