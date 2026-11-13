"""
Train the no-show model.

  - reads the synthetic appointments from SQLite
  - splits by TIME, not randomly (train on the past, test on the future) — the
    only honest way to estimate how it'll do on tomorrow's schedule
  - fits a LogisticRegression pipeline (one-hot categoricals, scale numerics,
    class_weight balanced for the ~18% no-show imbalance)
  - reports AUC + precision/recall on the no-show class + a calibration check.
    NOT accuracy — "predict everyone shows" scores ~82% and is useless.
  - pickles the whole pipeline to models/noshow.pkl

The model predicts P(attend). Expect AUC ~0.75 — that's the target, not a
disappointment: it means the pipeline is being tested at realistic difficulty.
"""
from __future__ import annotations

import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # so `import db` / `agents.*` work

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, brier_score_loss

import db
from agents.noshow_features import row_to_features, NUMERIC, CATEGORICAL, FEATURES

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "..", "models", "noshow.pkl")


def load_frame() -> pd.DataFrame:
    conn = db.connect()
    rows = conn.execute(
        "SELECT * FROM appointments WHERE attended IS NOT NULL ORDER BY appt_date").fetchall()
    conn.close()
    if not rows:
        raise SystemExit("No appointments found. Run: python ml/generate_data.py first.")
    records = []
    for r in rows:
        r = dict(r)
        feat = row_to_features(r)
        feat["attended"] = int(r["attended"])
        feat["appt_date"] = r["appt_date"]
        records.append(feat)
    return pd.DataFrame(records)


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline([("pre", pre), ("clf", clf)])


def main():
    df = load_frame()

    # Time-based split: first 80% of dates train, last 20% test.
    df = df.sort_values("appt_date").reset_index(drop=True)
    cut = int(len(df) * 0.8)
    train, test = df.iloc[:cut], df.iloc[cut:]

    X_cols = FEATURES
    y_train, y_test = train["attended"], test["attended"]

    pipe = build_pipeline()
    pipe.fit(train[X_cols], y_train)

    # P(attend) = probability of class 1
    p_attend = pipe.predict_proba(test[X_cols])[:, 1]
    pred_show = (p_attend >= 0.5).astype(int)

    print(f"\nRows: {len(df)}  train: {len(train)}  test: {len(test)}")
    print(f"No-show rate (test): {1 - y_test.mean():.1%}\n")

    # AUC on the no-show class: invert since model outputs P(attend).
    auc = roc_auc_score(1 - y_test, 1 - p_attend)
    print(f"AUC (no-show detection): {auc:.3f}   <- target ~0.75, honest range")
    print(f"Brier score (calibration, lower=better): {brier_score_loss(y_test, p_attend):.3f}\n")

    print("Classification report (0 = no-show, the class we care about):")
    print(classification_report(y_test, pred_show, target_names=["no-show", "showed"], digits=3))

    # Refit on ALL data before saving — use every row for the deployed model.
    pipe.fit(df[X_cols], df["attended"])
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipe, f)
    print(f"Saved model -> {os.path.relpath(MODEL_PATH)}")


if __name__ == "__main__":
    main()
