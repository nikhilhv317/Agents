"""
Smart Scheduling Agent — a LangGraph analysis pipeline.

    load_schedule -> featurize -> predict -> explain -> END

Division of labour:
  - predict : scikit-learn LogisticRegression -> P(attend) per appointment. The
              NUMBER. Instant, reproducible, explainable.
  - explain : templated one-line reason from the model's own coefficients. No LLM,
              so the table loads with zero model latency.
  - the LLM is called ONLY when a reminder is actually approved
    (generate_reminder), so nothing waits on it up front. Its model tag is
    resolved through llm.py's routing table (task="scheduling_reminder"), not
    hardcoded here — see llm.py for why that's a routing table and not just a
    constant.

The human decision (approve / hold / reject) is made per-row in the UI and
recorded via the app's /decision endpoint — not inside the graph — so there is
no parked run to go stale.
"""
from __future__ import annotations

import os
import pickle
from datetime import date
from typing import Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

import db
from agents.noshow_features import row_to_features, FEATURES, band
import llm as _llm  # reuse the one place the model name is defined — see llm.py

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "..", "models", "noshow.pkl")
# No separate SCHEDULING_MODEL setting — reuse llm.py's MODEL so there is only
# ONE place that names the model. Two independent env vars that happen to
# default to the same value today is exactly how model tags drift apart over
# time (this project already had three different tags mentioned across three
# files before this fix).
LLM_MODEL = _llm.resolve_model("scheduling_reminder")


class SchedulingState(TypedDict, total=False):
    date: str
    appointments: list
    predictions: list
    expected: dict


# ---- nodes ------------------------------------------------------------------

def load_schedule(state: SchedulingState) -> dict:
    day = state.get("date") or date.today().isoformat()
    conn = db.connect()
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM appointments WHERE appt_date = ? AND attended IS NULL "
        "ORDER BY appt_hour", (day,)).fetchall()]
    conn.close()
    return {"date": day, "appointments": rows}


def featurize(state: SchedulingState) -> dict:
    for a in state["appointments"]:
        a["_features"] = row_to_features(a)
    return {"appointments": state["appointments"]}


def _load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _top_reasons(model, feat: dict, k: int = 2) -> list[str]:
    """Which features pushed this patient toward no-show — from the LR coefficients."""
    try:
        import numpy as np, pandas as pd
        pre, clf = model.named_steps["pre"], model.named_steps["clf"]
        X = pre.transform(pd.DataFrame([feat])[FEATURES])
        X = X.toarray()[0] if hasattr(X, "toarray") else X[0]
        names = pre.get_feature_names_out()
        contrib = clf.coef_[0] * X
        order = np.argsort(contrib)          # most negative (risk-increasing) first
        return [names[i].split("__")[-1] for i in order[:k]]
    except Exception:
        return []


def predict(state: SchedulingState) -> dict:
    import pandas as pd
    model = _load_model()
    preds, p_sum, var_sum = [], 0.0, 0.0
    for a in state["appointments"]:
        feat = a["_features"]
        p = float(model.predict_proba(pd.DataFrame([feat])[FEATURES])[0, 1])
        preds.append({"id": a["id"], "patient_id": a["patient_id"], "patient": a["patient_name"],
                      "hour": a["appt_hour"], "type": a["appt_type"],
                      "p_attend": round(p, 3), "band": band(p),
                      "is_new": bool(feat.get("is_new")),
                      "drivers": _top_reasons(model, feat)})
        # First-time patients still count toward the day's expected-attendance
        # math (using the same base-rate assumption row_to_features already
        # applies) — only the PER-ROW risk chip is suppressed for them, since
        # a % framed as "risk" is misleading when there's no history behind it.
        p_sum += p
        var_sum += p * (1 - p)
    n = len(state["appointments"])
    std = var_sum ** 0.5
    expected = {"n": n, "attend_mean": round(p_sum, 1),
                "low": max(0, round(p_sum - std)), "high": min(n, round(p_sum + std))}
    return {"predictions": preds, "expected": expected}


def explain(state: SchedulingState) -> dict:
    """Templated reason line per row — no LLM, so the table is instant."""
    for p in state["predictions"]:
        if p["is_new"]:
            p["reason"] = "First appointment — no attendance history yet."
            continue
        drivers = ", ".join(p["drivers"]) or "booking pattern"
        p["reason"] = f"{int((1 - p['p_attend']) * 100)}% no-show risk — driven by {drivers}."
    return {"predictions": state["predictions"]}


# ---- graph ------------------------------------------------------------------

def build_graph():
    g = StateGraph(SchedulingState)
    g.add_node("load_schedule", load_schedule)
    g.add_node("featurize", featurize)
    g.add_node("predict", predict)
    g.add_node("explain", explain)
    g.add_edge(START, "load_schedule")
    g.add_edge("load_schedule", "featurize")
    g.add_edge("featurize", "predict")
    g.add_edge("predict", "explain")
    g.add_edge("explain", END)
    return g.compile()


_graph = build_graph()


def run_for_date(day: Optional[str] = None) -> dict:
    """Score `day`'s schedule (default today). Fast — pure sklearn, no LLM."""
    day = day or date.today().isoformat()
    state = _graph.invoke({"date": day})
    decided = db.decided_ids_for(day)
    for p in state["predictions"]:
        p["decided"] = p["id"] in decided
    return {"date": day, "expected": state["expected"], "predictions": state["predictions"]}


# ---- on-demand reminder (the only LLM call) ---------------------------------

def _get_llm():
    try:
        from langchain_ollama import ChatOllama
        return ChatOllama(model=LLM_MODEL, temperature=0.3, reasoning=False, keep_alive="30m")
    except Exception:
        return None


def templated_reminder(patient: str, hour: int) -> str:
    """Instant reminder text — no LLM, used by the fast batch submit."""
    return (f"Hi, a reminder of your dental appointment today at {hour}:00. "
            "Please reply to confirm, or call us to reschedule.")


def generate_reminder(patient: str, hour: int) -> str:
    """Draft one SMS reminder. LLM if available, templated fallback otherwise."""
    llm = _get_llm()
    if llm:
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            msg = [SystemMessage(content="Write a warm, brief SMS reminder for a dental "
                                         "appointment. One or two sentences. Ask them to confirm "
                                         "or reschedule. No placeholders."),
                   HumanMessage(content=f"Appointment today at {hour}:00 for {patient}.")]
            return llm.invoke(msg).content.strip()
        except Exception:
            pass
    return (f"Hi, a reminder of your dental appointment today at {hour}:00. "
            "Please reply to confirm, or call us to reschedule.")
