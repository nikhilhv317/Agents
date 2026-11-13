"""
Synthetic appointment history for training the no-show model.

The point of this file is honesty: we deliberately inject enough randomness that
the features only PARTLY explain who shows up, so a model trained on it lands
around AUC ~0.75 — where real no-show models sit — not a fake 0.95. If it looked
near-perfect, it would collapse on real data.

Key discipline (same as you'd need on real data):
  - appointments are generated in chronological order per patient, so a patient's
    "prior no-show rate" only ever counts appointments BEFORE the current one.
    That avoids leaking the outcome into its own feature.

Writes rows into the `appointments` table (created here if needed) with the
`attended` outcome filled in. train.py reads them back out.
"""
from __future__ import annotations

import os
import random
import math
import sqlite3
from datetime import date, timedelta
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # so `import db` works from D:\Agents
import db  # reuse the same database

N_PATIENTS = 300
APPTS_PER_PATIENT = (4, 22)      # min, max historical appointments per patient
START = date(2023, 1, 2)
DAYS_SPAN = 900

APPT_TYPES = ["checkup", "cleaning", "filling", "root_canal", "extraction", "emergency", "followup"]
# Emergencies/pain are kept; routine cleanings are easy to skip. This is a base
# tendency, NOT a hard rule — noise below still moves individuals.
TYPE_ATTEND_BIAS = {
    "checkup": 0.00, "cleaning": -0.35, "filling": 0.25, "root_canal": 0.55,
    "extraction": 0.45, "emergency": 0.95, "followup": -0.10,
}

rng = random.Random(42)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _extend_schema(conn: sqlite3.Connection):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS appointments (
        id            INTEGER PRIMARY KEY,
        patient_id    INTEGER NOT NULL,
        patient_name  TEXT,
        booked_on     TEXT NOT NULL,
        appt_date     TEXT NOT NULL,
        appt_hour     INTEGER NOT NULL,
        dow           INTEGER NOT NULL,        -- 0=Mon .. 6=Sun
        appt_type     TEXT NOT NULL,
        lead_days     INTEGER NOT NULL,
        confirmed     INTEGER NOT NULL,        -- reminder acknowledged (0/1)
        prior_appts   INTEGER NOT NULL,        -- count before this one
        prior_noshows INTEGER NOT NULL,        -- count before this one
        is_new        INTEGER NOT NULL,
        attended      INTEGER                  -- outcome: 1 showed, 0 no-show (NULL = future)
    );
    """)


def generate(reset: bool = True):
    db.init_db()
    conn = db.connect()
    if reset:
        conn.execute("DROP TABLE IF EXISTS appointments")
    _extend_schema(conn)

    rows = []
    for pid in range(1, N_PATIENTS + 1):
        # Latent reliability trait: the part of behaviour features can't see.
        # This is what keeps the problem realistically hard.
        reliability = rng.gauss(0, 1)
        name = f"Patient {pid:03d}"
        n = rng.randint(*APPTS_PER_PATIENT)

        prior_appts = 0
        prior_noshows = 0
        cursor_day = rng.randint(0, 120)

        for k in range(n):
            cursor_day += rng.randint(7, 90)
            if cursor_day >= DAYS_SPAN:
                break
            appt_dt = START + timedelta(days=cursor_day)
            lead = rng.choice([0, 1, 1, 2, 3, 5, 7, 10, 14, 21, 30, 45])
            booked = appt_dt - timedelta(days=lead)
            hour = rng.choice([9, 9, 10, 11, 12, 14, 15, 16, 17, 18])
            dow = appt_dt.weekday()
            atype = rng.choice(APPT_TYPES)
            confirmed = 1 if rng.random() < 0.6 else 0
            is_new = 1 if prior_appts == 0 else 0
            prior_rate = (prior_noshows / prior_appts) if prior_appts else 0.25  # base rate for new

            # ----- the "true" attend-probability (unknown to the model) -------
            # Weighted sum of BEHAVIOURAL signals only (no demographics), + the
            # latent trait, + noise. Coefficients tuned so features explain part,
            # not all, of the outcome.
            z = 2.05                                    # baseline: most people show (~18% no-show target)
            z += -1.8 * prior_rate                      # own past misses: strongest signal
            z += -0.045 * lead                          # longer lead -> more drift
            z += TYPE_ATTEND_BIAS[atype]                # appointment type
            z += 0.6 * confirmed                        # acknowledged reminder helps
            z += -0.35 if dow == 0 else 0.0             # Monday dip
            z += -0.30 if hour <= 9 else 0.0            # early-morning dip
            z += -0.25 * is_new                         # new patients slightly flakier
            z += 0.9 * reliability                      # latent trait (features can't see)
            z += rng.gauss(0, 0.9)                       # irreducible noise: traffic, life

            attended = 1 if _sigmoid(z) > rng.random() else 0

            rows.append((pid, name, booked.isoformat(), appt_dt.isoformat(), hour, dow,
                         atype, lead, confirmed, prior_appts, prior_noshows, is_new, attended))

            prior_appts += 1
            if attended == 0:
                prior_noshows += 1

    conn.executemany(
        "INSERT INTO appointments (patient_id, patient_name, booked_on, appt_date, appt_hour, dow, "
        "appt_type, lead_days, confirmed, prior_appts, prior_noshows, is_new, attended) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()

    total = len(rows)
    noshows = sum(1 for r in rows if r[-1] == 0)
    conn.close()
    print(f"Generated {total} appointments for {N_PATIENTS} patients.")
    print(f"No-show rate: {noshows/total:.1%}  (realistic clinic range is ~15-25%)")
    seed_today()


def seed_today():
    """
    Today's schedule for the REAL roster patients (ids 1-12 in the patients
    table), each with a hand-set no-show feature profile so the model produces a
    realistic green/yellow/red spread. Because appt.patient_id == patients.id,
    "Start visit" from a schedule row opens that patient's real chart.

    Finalized starts empty — it fills only with decisions you submit.
    """
    from datetime import date as _date
    today = _date.today(); dow = today.weekday()
    # patient_id -> (prior_appts, prior_noshows, lead_days, appt_type, confirmed, hour)
    PROFILES = {
        1:  (10, 0, 1,  "root_canal", 1, 9),
        2:  (1,  0, 14, "cleaning",   0, 10),
        3:  (6,  4, 21, "cleaning",   0, 9),
        4:  (5,  1, 7,  "checkup",    1, 10),
        5:  (8,  3, 30, "checkup",    0, 11),
        6:  (12, 1, 2,  "filling",    1, 11),
        7:  (4,  2, 10, "followup",   0, 12),
        8:  (9,  0, 3,  "emergency",  1, 14),
        9:  (7,  5, 45, "cleaning",   0, 14),
        10: (3,  1, 5,  "filling",    1, 15),
        11: (0,  0, 21, "checkup",    0, 16),
        12: (6,  2, 14, "extraction", 1, 17),
    }
    conn = db.connect()
    conn.execute("DELETE FROM appointments WHERE attended IS NULL")
    names = {r["id"]: r["name"] for r in conn.execute("SELECT id, name FROM patients").fetchall()}
    rows = []
    for pid, (appts, ns, lead, atype, confirmed, hour) in PROFILES.items():
        rows.append((pid, names.get(pid, f"Patient {pid}"), today.isoformat(), today.isoformat(),
                     hour, dow, atype, lead, confirmed, appts, ns, 1 if appts == 0 else 0, None))
    conn.executemany(
        "INSERT INTO appointments (patient_id, patient_name, booked_on, appt_date, appt_hour, dow, "
        "appt_type, lead_days, confirmed, prior_appts, prior_noshows, is_new, attended) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit(); conn.close()
    print(f"Seeded {len(rows)} appointments for today ({today.isoformat()}) — real roster patients.")


if __name__ == "__main__":
    generate()
