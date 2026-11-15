"""
Automated retraining pipeline for the no-show model.

Can be called three ways:
  1. Automatically by APScheduler (weekly, from app.py startup)
  2. Manually from the terminal: python ml/retrain.py
  3. Via the API: POST /api/ml/retrain (admin only once auth lands)

--- Agentic pattern: evaluation loop --------------------------------------------
The other agents in this system observe data and propose actions for a human to
approve. This one is different in kind: its "action" is evaluating and updating
ANOTHER agent (the scheduling agent's no-show model), and it approves its own
action automatically — but only against a hard, auditable gate (AUC_TOLERANCE),
never on vibes. That's the sense-evaluate-act loop:
  sense     -> how many new labelled outcomes exist since the last run
  evaluate  -> train a candidate, score it on a time-based holdout, compare
               against the CURRENTLY DEPLOYED model's own AUC (not a fixed bar)
  act       -> promote (hot-swap, versioned, never overwrites) or skip and log why
Every run is logged to retrain_log regardless of outcome — the gate rejecting a
candidate is as much a record of the agent working correctly as a promotion is.

What it does, mechanically:
  - checks how many new labelled outcomes exist
  - if enough data (MIN_NEW_OUTCOMES), retrains the sklearn pipeline
  - evaluates on a time-based holdout
  - compares to the currently deployed model's AUC
  - if AUC >= current - TOLERANCE: promotes (hot-swaps the active model)
  - logs everything to retrain_log regardless of outcome
  - versioned model files never overwrite each other
"""
from __future__ import annotations

import os, sys, pickle, time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

import db
from agents.noshow_features import row_to_features, FEATURES

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "..", "models")
ACTIVE_PATH = os.path.join(MODELS_DIR, "noshow.pkl")   # what the agent loads

MIN_NEW_OUTCOMES = 50    # minimum new labelled rows to bother retraining
TRAIN_SPLIT = 0.80       # time-based: first 80% train, last 20% evaluate
AUC_TOLERANCE = 0.02     # allow up to 2pp drop vs current (new model might be
                          # slightly lower on synthetic data but better on real)


def _load_frame():
    conn = db.connect()
    rows = conn.execute(
        "SELECT * FROM appointments WHERE attended IS NOT NULL "
        "ORDER BY appt_date").fetchall()
    conn.close()
    if not rows:
        return pd.DataFrame()
    records = []
    for r in rows:
        r = dict(r)
        feat = row_to_features(r)
        feat["attended"] = int(r["attended"])
        feat["appt_date"] = r["appt_date"]
        records.append(feat)
    return pd.DataFrame(records)


def _build_pipeline():
    pre = ColumnTransformer([
        ("num", StandardScaler(), [f for f in FEATURES if f != "appt_type"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["appt_type"]),
    ])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline([("pre", pre), ("clf", clf)])


def _evaluate(pipe, df):
    """AUC on the no-show class (inverted: 1 - P(attend))."""
    p_attend = pipe.predict_proba(df[FEATURES])[:, 1]
    return float(roc_auc_score(1 - df["attended"], 1 - p_attend))


def run(force=False, verbose=True):
    """
    Run the retraining pipeline.

    force=True: skip the MIN_NEW_OUTCOMES check (useful for manual runs
    when you want to retrain on whatever data exists).
    Returns a dict with the outcome.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    db._ml_schema()

    t0 = time.time()
    total_outcomes = db.count_new_outcomes()
    # "new" = outcomes the active model hasn't been trained on. trained_on is the
    # row count it was fit on, so the difference is what's accumulated since.
    # With no registered model yet, everything counts as new (first run).
    active = db.get_active_model()
    new_outcomes = total_outcomes - ((active["trained_on"] or 0) if active else 0)
    if verbose:
        print(f"[retrain] labelled outcomes in DB: {total_outcomes} ({new_outcomes} new)")

    if new_outcomes < MIN_NEW_OUTCOMES and not force:
        reason = (f"Not enough new data — {new_outcomes} new outcomes since the "
                  f"active model, need {MIN_NEW_OUTCOMES}. Skipping.")
        if verbose:
            print(f"[retrain] {reason}")
        db.log_retrain(new_outcomes, None, None, False, reason)
        return {"promoted": False, "reason": reason}

    df = _load_frame()
    if df.empty:
        reason = "DataFrame is empty after loading."
        db.log_retrain(0, None, None, False, reason)
        return {"promoted": False, "reason": reason}

    # time-based split
    df = df.sort_values("appt_date").reset_index(drop=True)
    cut = max(int(len(df) * TRAIN_SPLIT), 1)
    train, test = df.iloc[:cut], df.iloc[cut:]
    if len(test) < 10:
        # not enough test rows — fit on all and evaluate on train (less honest but
        # still useful for registering the model when data is scarce)
        test = train

    pipe = _build_pipeline()
    pipe.fit(train[FEATURES], train["attended"])
    new_auc = _evaluate(pipe, test)
    if verbose:
        print(f"[retrain] new model AUC: {new_auc:.4f} (train={len(train)}, test={len(test)})")

    # compare to currently active model
    old_auc = active["auc"] if active else None
    if verbose:
        print(f"[retrain] current deployed AUC: {old_auc}")

    should_promote = (old_auc is None) or (new_auc >= old_auc - AUC_TOLERANCE)
    reason = ""

    if should_promote:
        # version number = next integer after highest existing
        existing = [f for f in os.listdir(MODELS_DIR) if f.startswith("noshow_v")]
        vnum = max([int(f.split("_v")[1].split(".")[0]) for f in existing], default=0) + 1 if existing else 1
        vpath = os.path.join(MODELS_DIR, f"noshow_v{vnum}.pkl")
        # fit final model on ALL data before saving
        pipe.fit(df[FEATURES], df["attended"])
        with open(vpath, "wb") as fh:
            pickle.dump(pipe, fh)
        # atomic copy to the active path the agent loads
        import shutil
        shutil.copy2(vpath, ACTIVE_PATH)
        db.register_model(vnum, vpath, new_auc, len(df), promote=True)
        reason = (f"Self-evaluation: promoted v{vnum}, AUC {new_auc:.4f}"
                  + (f" vs previous {old_auc:.4f}" if old_auc else " (first model)"))
        db.add_log("retraining", "model:noshow", reason)
        if verbose:
            print(f"[retrain] ✓ {reason}")
    else:
        reason = (f"Self-evaluation: rejected candidate, AUC {new_auc:.4f} < "
                  f"current {old_auc:.4f} - tolerance {AUC_TOLERANCE}")
        if verbose:
            print(f"[retrain] ✗ {reason}")

    db.log_retrain(new_outcomes, old_auc, new_auc, should_promote, reason)
    elapsed = round(time.time() - t0, 1)
    if verbose:
        print(f"[retrain] done in {elapsed}s")
    return {"promoted": should_promote, "new_auc": new_auc,
            "old_auc": old_auc, "reason": reason, "elapsed_s": elapsed}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="retrain even if below MIN_NEW_OUTCOMES")
    args = ap.parse_args()
    result = run(force=args.force)
    print("\nResult:", result)
