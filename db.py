"""
SQLite data layer.

One unified patient identity: the `patients` table holds the named roster that
Appointments schedules, AI Summary briefs, and Reports documents — so "Start
visit" from a schedule row lands on a real chart.

Tables:
  patients      - the patient record (the spine)
  visits        - past clinical visits (history the agents read)
  appointments  - schedule rows (today = attended NULL; history = attended 0/1)
  agent_actions - proposed actions awaiting approval
  agent_log     - executed actions, timestamped
  decisions     - finalized scheduling decisions
  patient_notes - PERSISTED outputs written back to the record (imaging findings,
                  SOAP notes) — this is what closes the loop between visits.
"""
import sqlite3
import json
import os

DB_PATH = os.getenv("DPMS_DB_PATH", os.path.join(os.path.dirname(__file__), "dpms.db"))
os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)  # safe no-op locally; matters when
                                                              # DPMS_DB_PATH points at a fresh volume mount


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY, mrn TEXT UNIQUE, name TEXT NOT NULL,
    age INTEGER, sex TEXT, phone TEXT, allergies TEXT, conditions TEXT, active INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS visits (
    id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL REFERENCES patients(id),
    date TEXT NOT NULL, complaint TEXT, diagnosis TEXT, treatment TEXT, notes TEXT
);
CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL, patient_name TEXT,
    booked_on TEXT NOT NULL, appt_date TEXT NOT NULL, appt_hour INTEGER NOT NULL, dow INTEGER NOT NULL,
    appt_type TEXT NOT NULL, lead_days INTEGER NOT NULL, confirmed INTEGER NOT NULL,
    prior_appts INTEGER NOT NULL, prior_noshows INTEGER NOT NULL, is_new INTEGER NOT NULL, attended INTEGER
);
CREATE TABLE IF NOT EXISTS agent_actions (
    id INTEGER PRIMARY KEY, agent TEXT NOT NULL, subject TEXT NOT NULL, label TEXT NOT NULL,
    type TEXT NOT NULL, payload TEXT, status TEXT NOT NULL DEFAULT 'proposed',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS agent_log (
    id INTEGER PRIMARY KEY, agent TEXT NOT NULL, subject TEXT NOT NULL, message TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY, appt_id INTEGER, patient TEXT, appt_type TEXT, appt_hour INTEGER,
    p_attend REAL, decision TEXT NOT NULL, reminder TEXT,
    decided_on TEXT NOT NULL, decided_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS patient_notes (
    id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL, kind TEXT NOT NULL,
    content TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

# id, mrn, name, age, sex, phone, allergies, conditions  — the unified roster
SEED_PATIENTS = [
    (1,  "VDC-0001", "Ramesh Kulkarni", 54, "M", "+91 98861 20034", "Penicillin", "Type 2 diabetes, Hypertension"),
    (2,  "VDC-0002", "Anita Desai",     32, "F", "+91 99012 55210", "", ""),
    (3,  "VDC-0003", "Suresh Rao",      60, "M", "+91 98450 11882", "", "Hypertension"),
    (4,  "VDC-0004", "Kavya Nair",      29, "F", "+91 99023 77120", "", ""),
    (5,  "VDC-0005", "Arjun Menon",     45, "M", "+91 90080 34551", "Ibuprofen", ""),
    (6,  "VDC-0006", "Priya Shetty",    38, "F", "+91 98867 90233", "", ""),
    (7,  "VDC-0007", "Vijay Kumar",     52, "M", "+91 99456 21098", "", "Type 2 diabetes"),
    (8,  "VDC-0008", "Deepa Iyer",      34, "F", "+91 90084 55317", "Latex", ""),
    (9,  "VDC-0009", "Karthik Reddy",   41, "M", "+91 98861 33445", "", "Asthma"),
    (10, "VDC-0010", "Lakshmi Rao",     47, "F", "+91 99012 88760", "", ""),
    (11, "VDC-0011", "Sneha Gowda",     26, "F", "+91 90083 12094", "", "Pregnant (2nd trimester)"),
    (12, "VDC-0012", "Manoj Pillai",    55, "M", "+91 98450 66231", "Penicillin", "Hypertension, Bleeding disorder"),
]

# patient_id, date, complaint, diagnosis, treatment, notes
SEED_VISITS = [
    (1, "2024-11-02", "Bleeding gums", "Generalised chronic gingivitis", "Scaling + polishing", "Smoker; advised twice-daily brushing."),
    (1, "2025-03-18", "Pain lower right molar", "Deep caries #46", "Composite restoration", "Diabetic — monitor healing."),
    (1, "2025-08-09", "Sensitivity follow-up", "Reversible pulpitis #46", "Desensitising, review", "Improving."),
    (2, "2025-01-20", "Routine check-up", "Sound dentition", "Prophylaxis", "No active caries."),
    (2, "2025-07-11", "Wisdom tooth ache", "Impacted #38", "Referred for extraction", "OPG advised."),
    (3, "2025-02-14", "Loose tooth", "Chronic periodontitis", "Deep scaling", "Monitor mobility #31."),
    (4, "2025-05-03", "Check-up", "Early caries #16", "Fluoride application", "Advised diet review."),
    (5, "2025-04-22", "Broken filling", "Fractured restoration #24", "Replacement", ""),
    (6, "2025-06-30", "Cleaning", "Mild gingivitis", "Scaling", "Good response."),
    (7, "2025-03-05", "Toothache", "Advanced caries #36", "Root canal started", "Diabetic — infection watch."),
    (8, "2025-07-02", "Swollen gum", "Localised abscess #45", "Drainage + antibiotics", "Latex-free gloves."),
    (9, "2025-05-19", "Bad breath", "Plaque accumulation", "Scaling + counselling", "Asthmatic — no aerosol issues noted."),
    (10, "2025-02-27", "Sensitivity", "Exposed dentine #15", "Desensitising", ""),
    (11, "2025-06-11", "Swollen gums", "Pregnancy gingivitis", "Gentle scaling", "Avoid elective X-rays; 2nd trimester."),
    (12, "2025-01-09", "Extraction consult", "Non-restorable #26", "Extraction planned", "On blood thinner — bleeding precautions."),
]


def init_db(reset: bool = False):
    if reset and os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = connect()
    conn.executescript(SCHEMA)
    try:
        conn.execute("ALTER TABLE patients ADD COLUMN active INTEGER NOT NULL DEFAULT 1")
        conn.commit()
    except Exception:
        pass  # column already exists
    if conn.execute("SELECT COUNT(*) AS n FROM patients").fetchone()["n"] == 0:
        conn.executemany("INSERT INTO patients (id,mrn,name,age,sex,phone,allergies,conditions) "
                         "VALUES (?,?,?,?,?,?,?,?)", SEED_PATIENTS)
        conn.executemany("INSERT INTO visits (patient_id,date,complaint,diagnosis,treatment,notes) "
                         "VALUES (?,?,?,?,?,?)", SEED_VISITS)
        conn.commit()
    conn.close()


# ---- patients ----
def list_patients():
    conn = connect()
    rows = conn.execute("SELECT id, mrn, name, age, sex FROM patients WHERE active = 1 ORDER BY name").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_patient(patient_id: int):
    conn = connect()
    p = conn.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
    if p is None:
        conn.close()
        return None
    visits = conn.execute("SELECT date, complaint, diagnosis, treatment, notes FROM visits "
                          "WHERE patient_id = ? ORDER BY date", (patient_id,)).fetchall()
    conn.close()
    patient = dict(p)
    patient["visits"] = [dict(v) for v in visits]
    return patient


# ---- actions + log ----
def save_action(agent, subject, label, type_, payload):
    conn = connect()
    cur = conn.execute("INSERT INTO agent_actions (agent,subject,label,type,payload) VALUES (?,?,?,?,?)",
                       (agent, subject, label, type_, json.dumps(payload)))
    conn.commit(); aid = cur.lastrowid; conn.close()
    return aid


def get_action(action_id: int):
    conn = connect()
    row = conn.execute("SELECT * FROM agent_actions WHERE id = ?", (action_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def set_action_status(action_id: int, status: str):
    conn = connect()
    conn.execute("UPDATE agent_actions SET status = ? WHERE id = ?", (status, action_id))
    conn.commit(); conn.close()


def add_log(agent, subject, message):
    conn = connect()
    conn.execute("INSERT INTO agent_log (agent,subject,message) VALUES (?,?,?)", (agent, subject, message))
    conn.commit(); conn.close()


def get_log(limit: int = 30):
    conn = connect()
    rows = conn.execute("SELECT agent,subject,message,created_at FROM agent_log ORDER BY id DESC LIMIT ?",
                        (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---- scheduling decisions + outcome ----
def save_decision(appt_id, patient, appt_type, appt_hour, p_attend, decision, reminder=None, decided_on=None):
    from datetime import date as _date
    decided_on = decided_on or _date.today().isoformat()
    conn = connect()
    conn.execute("INSERT INTO decisions (appt_id,patient,appt_type,appt_hour,p_attend,decision,reminder,decided_on) "
                 "VALUES (?,?,?,?,?,?,?,?)", (appt_id, patient, appt_type, appt_hour, p_attend, decision, reminder, decided_on))
    conn.commit(); conn.close()


def list_finalized():
    conn = connect()
    rows = conn.execute("SELECT d.appt_id, d.patient, d.appt_type, d.appt_hour, d.p_attend, d.decision, "
                        "d.decided_on, d.decided_at, a.attended "
                        "FROM decisions d LEFT JOIN appointments a ON a.id = d.appt_id "
                        "ORDER BY d.decided_on DESC, d.appt_hour ASC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def decided_ids_for(day):
    conn = connect()
    rows = conn.execute("SELECT appt_id FROM decisions WHERE decided_on = ?", (day,)).fetchall()
    conn.close()
    return {r["appt_id"] for r in rows}


def set_attended(appt_id: int, attended: int):
    """Outcome capture: record whether the patient actually showed (0/1). The
    retraining signal — the decision plus the real result."""
    conn = connect()
    conn.execute("UPDATE appointments SET attended = ? WHERE id = ?", (attended, appt_id))
    conn.commit(); conn.close()


# ---- persisted notes (the record) ----
def save_note(patient_id: int, kind: str, content: str):
    conn = connect()
    conn.execute("INSERT INTO patient_notes (patient_id,kind,content) VALUES (?,?,?)",
                 (patient_id, kind, content))
    conn.commit(); conn.close()


def get_notes(patient_id: int, kind: str = None):
    conn = connect()
    if kind:
        rows = conn.execute("SELECT kind,content,created_at FROM patient_notes WHERE patient_id=? AND kind=? "
                            "ORDER BY id DESC", (patient_id, kind)).fetchall()
    else:
        rows = conn.execute("SELECT kind,content,created_at FROM patient_notes WHERE patient_id=? "
                            "ORDER BY id DESC", (patient_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def latest_note(patient_id: int, kind: str):
    notes = get_notes(patient_id, kind)
    return notes[0] if notes else None


# ---- reports (image collection + clinical findings + AI summary) ----
def _reports_schema():
    conn = connect()
    conn.execute("""CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL, patient_name TEXT,
        imaging TEXT, clinical TEXT, ai_summary TEXT, images TEXT,
        status TEXT NOT NULL DEFAULT 'draft',
        created_at TEXT NOT NULL DEFAULT (datetime('now')), finalized_at TEXT )""")
    conn.commit(); conn.close()


def create_report(patient_id, patient_name, imaging, clinical, ai_summary, status="draft", images=None):
    _reports_schema()
    conn = connect()
    try:
        conn.execute("ALTER TABLE reports ADD COLUMN images TEXT")
        conn.commit()
    except Exception:
        pass
    cur = conn.execute("INSERT INTO reports (patient_id,patient_name,imaging,clinical,ai_summary,status,images) "
                       "VALUES (?,?,?,?,?,?,?)", (patient_id, patient_name, imaging, clinical, ai_summary, status, images))
    conn.commit(); rid = cur.lastrowid; conn.close()
    return rid


def finalize_report(report_id):
    conn = connect()
    conn.execute("UPDATE reports SET status='finalized', finalized_at=datetime('now') WHERE id=?", (report_id,))
    conn.commit(); conn.close()


def update_report(report_id, ai_summary=None, status=None):
    conn = connect()
    if ai_summary is not None:
        conn.execute("UPDATE reports SET ai_summary=? WHERE id=?", (ai_summary, report_id))
    if status is not None:
        conn.execute("UPDATE reports SET status=? WHERE id=?", (status, report_id))
    conn.commit(); conn.close()


# ---- visit pipeline runs (graph state itself lives in the LangGraph checkpointer) ----
def _pipeline_schema():
    conn = connect()
    conn.execute("""CREATE TABLE IF NOT EXISTS pipeline_runs (
        id INTEGER PRIMARY KEY, thread_id TEXT UNIQUE NOT NULL,
        patient_id INTEGER NOT NULL, patient_name TEXT, trigger TEXT,
        status TEXT NOT NULL,              -- running | awaiting_soap | awaiting_report | done | rejected | failed
        rerun_requested INTEGER NOT NULL DEFAULT 0,
        error TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now')))""")
    conn.commit(); conn.close()


def create_pipeline_run(thread_id, patient_id, patient_name, trigger):
    _pipeline_schema()
    conn = connect()
    cur = conn.execute("INSERT INTO pipeline_runs (thread_id,patient_id,patient_name,trigger,status) "
                       "VALUES (?,?,?,?, 'running')", (thread_id, patient_id, patient_name, trigger))
    conn.commit(); rid = cur.lastrowid; conn.close()
    return rid


def update_pipeline_run(run_id, **fields):
    allowed = {"status", "rerun_requested", "error"}
    sets = [f"{k}=?" for k in fields if k in allowed]
    if not sets:
        return
    conn = connect()
    conn.execute(f"UPDATE pipeline_runs SET {','.join(sets)}, updated_at=datetime('now') WHERE id=?",
                 [v for k, v in fields.items() if k in allowed] + [run_id])
    conn.commit(); conn.close()


def get_pipeline_run(run_id):
    _pipeline_schema()
    conn = connect()
    r = conn.execute("SELECT * FROM pipeline_runs WHERE id=?", (run_id,)).fetchone()
    conn.close()
    return dict(r) if r else None


def list_pipeline_runs(limit=30, statuses=None):
    _pipeline_schema()
    conn = connect()
    if statuses:
        q = ",".join("?" * len(statuses))
        rows = conn.execute(f"SELECT * FROM pipeline_runs WHERE status IN ({q}) ORDER BY id DESC LIMIT ?",
                            (*statuses, limit)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM pipeline_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def active_pipeline_run(patient_id):
    """The patient's unfinished run, if any — one visit pipeline per patient at a time."""
    _pipeline_schema()
    conn = connect()
    r = conn.execute("SELECT * FROM pipeline_runs WHERE patient_id=? AND status IN "
                     "('running','awaiting_soap','awaiting_report') ORDER BY id DESC LIMIT 1",
                     (patient_id,)).fetchone()
    conn.close()
    return dict(r) if r else None


def get_report(report_id):
    _reports_schema()
    conn = connect()
    r = conn.execute("SELECT * FROM reports WHERE id=?", (report_id,)).fetchone()
    conn.close()
    return dict(r) if r else None


def list_reports(status=None):
    _reports_schema()
    conn = connect()
    if status:
        rows = conn.execute("SELECT id,patient_id,patient_name,ai_summary,status,created_at,finalized_at "
                            "FROM reports WHERE status=? ORDER BY id DESC", (status,)).fetchall()
    else:
        rows = conn.execute("SELECT id,patient_id,patient_name,ai_summary,status,created_at,finalized_at "
                            "FROM reports ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def approved_today(day=None):
    """Patients whose appointment was approved today — the queue that flows to Scan."""
    from datetime import date as _date
    day = day or _date.today().isoformat()
    conn = connect()
    rows = conn.execute(
        "SELECT d.appt_id, d.appt_hour, d.appt_type, a.patient_id, a.patient_name "
        "FROM decisions d JOIN appointments a ON a.id = d.appt_id "
        "WHERE d.decided_on = ? AND d.decision = 'approve' ORDER BY d.appt_hour", (day,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---- patient CRUD ----
import re


def normalize_phone(phone: str) -> str:
    """Digits only, last 10 kept. '+91 98861 20034', '9886120034' and
    '098861-20034' all normalize to the same value, so formatting differences
    don't cause a returning patient to be treated as a new one."""
    digits = re.sub(r"\D", "", phone or "")
    return digits[-10:] if len(digits) >= 10 else digits


def find_duplicate_candidates(name: str = "", phone: str = ""):
    """Find existing ACTIVE patients who look like the same person, so the
    caller can reuse that record — and its visit/note/scan history — instead
    of creating a duplicate. Phone match is the strong signal (two people
    rarely share a mobile number); name match is weaker and returned so a
    human can confirm rather than being auto-merged."""
    norm_phone = normalize_phone(phone)
    name_norm = (name or "").strip().lower()
    if not norm_phone and not name_norm:
        return []
    conn = connect()
    rows = conn.execute(
        "SELECT id, mrn, name, age, sex, phone FROM patients WHERE active=1").fetchall()
    conn.close()
    matches = []
    for r in rows:
        r = dict(r)
        phone_match = bool(norm_phone) and normalize_phone(r["phone"]) == norm_phone
        name_match = bool(name_norm) and r["name"].strip().lower() == name_norm
        if phone_match or name_match:
            r["match_reason"] = "phone" if phone_match else "name"
            matches.append(r)
    # phone matches first — they're the reliable signal
    matches.sort(key=lambda m: m["match_reason"] != "phone")
    return matches


def next_mrn():
    conn = connect()
    rows = conn.execute("SELECT mrn FROM patients WHERE mrn LIKE 'VDC-%'").fetchall()
    conn.close()
    n = 0
    for r in rows:
        try:
            n = max(n, int(r["mrn"].split("-")[1]))
        except Exception:
            pass
    return f"VDC-{n+1:04d}"


def create_patient(name, age, sex, phone="", allergies="", conditions=""):
    mrn = next_mrn()
    conn = connect()
    cur = conn.execute("INSERT INTO patients (mrn,name,age,sex,phone,allergies,conditions) "
                       "VALUES (?,?,?,?,?,?,?)", (mrn, name, age, sex, phone, allergies, conditions))
    conn.commit(); pid = cur.lastrowid; conn.close()
    return pid, mrn


def update_patient(pid, fields: dict):
    allowed = {"name", "age", "sex", "phone", "allergies", "conditions"}
    sets, vals = [], []
    for k, v in fields.items():
        if k in allowed and v is not None:
            sets.append(f"{k}=?"); vals.append(v)
    if not sets:
        return
    vals.append(pid)
    conn = connect(); conn.execute(f"UPDATE patients SET {','.join(sets)} WHERE id=?", vals)
    conn.commit(); conn.close()


def deactivate_patient(pid):
    """Soft delete — never erase a medical record; just hide it from the roster."""
    conn = connect(); conn.execute("UPDATE patients SET active=0 WHERE id=?", (pid,))
    conn.commit(); conn.close()


def add_visit(patient_id, date, complaint, diagnosis, treatment, notes):
    conn = connect()
    conn.execute("INSERT INTO visits (patient_id,date,complaint,diagnosis,treatment,notes) "
                 "VALUES (?,?,?,?,?,?)", (patient_id, date, complaint, diagnosis, treatment, notes))
    conn.commit(); conn.close()


# ---- appointment booking (computes the no-show features so the agent can score it) ----
def create_appointment(patient_id, appt_date, appt_time, appt_type):
    from datetime import date as _date, datetime as _dt
    p = get_patient(patient_id)
    name = p["name"] if p else f"Patient {patient_id}"
    try:
        hour = int(str(appt_time).split(":")[0])
    except Exception:
        hour = 10
    try:
        d = _dt.strptime(appt_date, "%Y-%m-%d").date()
    except Exception:
        d = _date.today()
    lead = max(0, (d - _date.today()).days)
    conn = connect()
    # Duplicate guard: the same patient booking the same date/hour/type while
    # a prior booking is still pending (attended IS NULL) is almost always an
    # accidental repeat submission (double-click, resubmitted form) — reuse
    # the existing row instead of stacking up identical entries. A patient
    # can still have several DIFFERENT appointments the same day (different
    # hour or type); only exact repeats are collapsed.
    dup = conn.execute(
        "SELECT id FROM appointments WHERE patient_id=? AND appt_date=? AND appt_hour=? "
        "AND appt_type=? AND attended IS NULL", (patient_id, d.isoformat(), hour, appt_type)).fetchone()
    if dup:
        conn.close()
        return dup["id"], True
    row = conn.execute("SELECT COUNT(*) AS a, COALESCE(SUM(1-attended),0) AS ns FROM appointments "
                       "WHERE patient_id=? AND attended IS NOT NULL", (patient_id,)).fetchone()
    prior_appts = row["a"]; prior_ns = int(row["ns"] or 0)
    cur = conn.execute(
        "INSERT INTO appointments (patient_id,patient_name,booked_on,appt_date,appt_hour,dow,"
        "appt_type,lead_days,confirmed,prior_appts,prior_noshows,is_new,attended) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (patient_id, name, _date.today().isoformat(), d.isoformat(), hour, d.weekday(),
         appt_type, lead, 0, prior_appts, prior_ns, 1 if prior_appts == 0 else 0, None))
    conn.commit(); aid = cur.lastrowid; conn.close()
    return aid, False


# ---- model registry + retrain log ----
def _ml_schema():
    conn = connect()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS model_registry (
        id INTEGER PRIMARY KEY, version INTEGER NOT NULL, path TEXT NOT NULL,
        auc REAL, trained_on INTEGER, status TEXT NOT NULL DEFAULT 'candidate',
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS retrain_log (
        id INTEGER PRIMARY KEY, triggered_at TEXT NOT NULL DEFAULT (datetime('now')),
        new_rows INTEGER, old_auc REAL, new_auc REAL, promoted INTEGER NOT NULL DEFAULT 0,
        reason TEXT
    );
    """)
    conn.commit(); conn.close()


def get_active_model():
    """Return the active model registry row, or None if none exists."""
    _ml_schema()
    conn = connect()
    row = conn.execute(
        "SELECT * FROM model_registry WHERE status='active' ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    return dict(row) if row else None


def register_model(version, path, auc, trained_on, promote=False):
    _ml_schema()
    conn = connect()
    if promote:
        conn.execute("UPDATE model_registry SET status='retired' WHERE status='active'")
    status = "active" if promote else "candidate"
    cur = conn.execute(
        "INSERT INTO model_registry (version,path,auc,trained_on,status) VALUES (?,?,?,?,?)",
        (version, path, auc, trained_on, status))
    conn.commit(); rid = cur.lastrowid; conn.close()
    return rid


def log_retrain(new_rows, old_auc, new_auc, promoted, reason):
    _ml_schema()
    conn = connect()
    conn.execute(
        "INSERT INTO retrain_log (new_rows,old_auc,new_auc,promoted,reason) VALUES (?,?,?,?,?)",
        (new_rows, old_auc, new_auc, 1 if promoted else 0, reason))
    conn.commit(); conn.close()


def retrain_history(limit=20):
    _ml_schema()
    conn = connect()
    rows = conn.execute(
        "SELECT * FROM retrain_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def count_new_outcomes(since_at=None):
    """How many labelled outcomes exist since the last retrain (or ever)."""
    conn = connect()
    if since_at:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM appointments WHERE attended IS NOT NULL "
            "AND booked_on > ?", (since_at,)).fetchone()["n"]
    else:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM appointments WHERE attended IS NOT NULL").fetchone()["n"]
    conn.close()
    return n


# ---- staff auth ----
def _auth_schema():
    conn = connect()
    conn.execute("""CREATE TABLE IF NOT EXISTS staff (
        id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL,
        display_name TEXT NOT NULL, password_hash TEXT NOT NULL,
        active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL DEFAULT (datetime('now')))""")
    conn.commit(); conn.close()


def seed_staff():
    """Make sure the `admin` account never uses a known password.

    - First run: create admin with ADMIN_PASSWORD, or a random password that is
      printed to the console once.
    - ADMIN_PASSWORD set on a later run: admin's password is reset to it (the
      recovery path if the admin password is lost).
    - An existing admin still on the old 'admin123' default is rotated to a
      random password, printed once.
    """
    import bcrypt as _bc
    import secrets as _secrets
    _auth_schema()
    env_pw = os.getenv("ADMIN_PASSWORD", "")
    if env_pw and len(env_pw) < 8:
        print("[auth] ADMIN_PASSWORD is shorter than 8 characters — ignoring it.")
        env_pw = ""
    conn = connect()
    admin = conn.execute("SELECT password_hash FROM staff WHERE username='admin'").fetchone()
    first_run = conn.execute("SELECT COUNT(*) AS n FROM staff").fetchone()["n"] == 0
    new_pw = None
    if env_pw:
        if admin is None or not _bc.checkpw(env_pw.encode(), admin["password_hash"].encode()):
            new_pw = env_pw
    elif first_run or (admin is not None
                       and _bc.checkpw(b"admin123", admin["password_hash"].encode())):
        new_pw = _secrets.token_urlsafe(12)
        print("\n" + "=" * 64 +
              f"\n[auth] admin password set to:  {new_pw}"
              "\n[auth] Log in and change it, or set ADMIN_PASSWORD to choose your own."
              "\n" + "=" * 64 + "\n", flush=True)
    if new_pw:
        h = _bc.hashpw(new_pw.encode(), _bc.gensalt()).decode()
        if admin is None:
            conn.execute("INSERT INTO staff (username,display_name,password_hash) VALUES (?,?,?)",
                         ("admin", "Dr. Admin", h))
        else:
            conn.execute("UPDATE staff SET password_hash=? WHERE username='admin'", (h,))
        conn.commit()
    conn.close()


def get_staff_by_username(username: str):
    _auth_schema()
    conn = connect()
    row = conn.execute("SELECT * FROM staff WHERE username=? AND active=1",
                       (username,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_staff():
    _auth_schema()
    conn = connect()
    rows = conn.execute("SELECT id,username,display_name,active,created_at "
                        "FROM staff ORDER BY id").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_staff(username, display_name, password):
    import bcrypt as _bc
    _auth_schema()
    h = _bc.hashpw(password.encode(), _bc.gensalt()).decode()
    conn = connect()
    try:
        conn.execute("INSERT INTO staff (username,display_name,password_hash) VALUES (?,?,?)",
                     (username, display_name, h))
        conn.commit()
    except Exception as e:
        conn.close()
        raise ValueError(f"username already exists") from e
    conn.close()


def change_password(username, new_password):
    import bcrypt as _bc
    h = _bc.hashpw(new_password.encode(), _bc.gensalt()).decode()
    conn = connect()
    conn.execute("UPDATE staff SET password_hash=? WHERE username=?", (h, username))
    conn.commit(); conn.close()


def deactivate_staff(staff_id):
    conn = connect()
    conn.execute("UPDATE staff SET active=0 WHERE id=?", (staff_id,))
    conn.commit(); conn.close()


# ---- scan images stored per patient (data URIs, for report embedding) ----
def save_scan_image(patient_id: int, filename: str, data_uri: str, verdict: str = "",
                    detections: str = None):
    conn = connect()
    conn.execute("""CREATE TABLE IF NOT EXISTS scan_images (
        id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL, filename TEXT,
        data_uri TEXT NOT NULL, verdict TEXT, detections TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')))""")
    try:
        conn.execute("ALTER TABLE scan_images ADD COLUMN detections TEXT")
        conn.commit()
    except Exception:
        pass
    conn.execute("INSERT INTO scan_images (patient_id,filename,data_uri,verdict,detections) "
                 "VALUES (?,?,?,?,?)", (patient_id, filename, data_uri, verdict, detections))
    conn.commit(); conn.close()


def get_scan_images(patient_id: int, limit: int = 100):
    conn = connect()
    conn.execute("""CREATE TABLE IF NOT EXISTS scan_images (
        id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL, filename TEXT,
        data_uri TEXT NOT NULL, verdict TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now')))""")
    try:
        conn.execute("ALTER TABLE scan_images ADD COLUMN detections TEXT")
        conn.commit()
    except Exception:
        pass
    rows = conn.execute("SELECT filename,data_uri,verdict,detections,created_at FROM scan_images "
                        "WHERE patient_id=? ORDER BY id ASC LIMIT ?", (patient_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_imaging_findings(patient_id: int, same_day_only: bool = True):
    """
    All imaging findings for a patient (optionally just today's), newest first.
    Reports aggregate ALL of them — a visit can involve several images, each with
    its own detected conditions, and the report must show every one.
    """
    from datetime import date as _date
    conn = connect()
    if same_day_only:
        rows = conn.execute(
            "SELECT content, created_at FROM patient_notes WHERE patient_id=? AND kind='imaging' "
            "AND date(created_at)=date('now','localtime') ORDER BY id ASC", (patient_id,)).fetchall()
        if not rows:  # fall back to all if nothing today
            rows = conn.execute(
                "SELECT content, created_at FROM patient_notes WHERE patient_id=? AND kind='imaging' "
                "ORDER BY id ASC", (patient_id,)).fetchall()
    else:
        rows = conn.execute(
            "SELECT content, created_at FROM patient_notes WHERE patient_id=? AND kind='imaging' "
            "ORDER BY id ASC", (patient_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def imaging_summary(patient_id: int) -> str:
    """Combine every imaging finding into one multi-line block for the report."""
    findings = get_imaging_findings(patient_id)
    if not findings:
        return "No imaging findings on record."
    lines = []
    for i, f in enumerate(findings, 1):
        lines.append(f"Image {i}: {f['content']}")
    return "\n".join(lines)
