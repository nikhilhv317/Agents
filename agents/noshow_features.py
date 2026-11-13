"""
Feature engineering — the ONE definition of a feature row, used by both
train.py and the live agent. Keeping it in one place is what prevents
training/serving skew (the classic bug where the model is trained on features
computed differently from how they're computed at prediction time).

FEATURES is the exact column order the model expects. NUMERIC / CATEGORICAL
tell the sklearn pipeline how to treat each.
"""
from __future__ import annotations

NUMERIC = ["lead_days", "appt_hour", "dow", "confirmed", "prior_appts",
           "prior_noshow_rate", "is_new"]
CATEGORICAL = ["appt_type"]
FEATURES = NUMERIC + CATEGORICAL


def row_to_features(row: dict) -> dict:
    """
    Turn one appointment record (dict-like, as stored in the DB) into the
    feature dict the model consumes. `prior_noshow_rate` is derived here so the
    same rule applies in training and serving.
    """
    prior_appts = int(row.get("prior_appts", 0) or 0)
    prior_noshows = int(row.get("prior_noshows", 0) or 0)
    prior_rate = (prior_noshows / prior_appts) if prior_appts else 0.25  # base rate for new patients
    return {
        "lead_days": int(row.get("lead_days", 0) or 0),
        "appt_hour": int(row.get("appt_hour", 12) or 12),
        "dow": int(row.get("dow", 0) or 0),
        "confirmed": int(row.get("confirmed", 0) or 0),
        "prior_appts": prior_appts,
        "prior_noshow_rate": round(prior_rate, 4),
        "is_new": int(row.get("is_new", 1) or 0),
        "appt_type": row.get("appt_type", "checkup"),
    }


def band(p_attend: float) -> str:
    """
    Colour band by probability of ATTENDING:
      > 70%  -> low   (green)  — likely to attend
      30-70% -> medium(yellow) — uncertain
      < 30%  -> high  (red)    — likely no-show
    """
    if p_attend > 0.70:
        return "low"
    if p_attend >= 0.30:
        return "medium"
    return "high"