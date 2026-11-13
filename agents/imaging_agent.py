"""
Imaging & Diagnostic Agent — a LangGraph pipeline over the two-stage vision models.

    gate --(dental)--> detect --> verdict --> explain
         --(non-dental)--------> verdict --> explain     (detector skipped)

Division of labour (same idea as the other agents):
  - the VISION MODELS (your .pt gate + detector) are the reasoning core — they
    produce the finding. Not an LLM.
  - the graph sequences gate -> detect -> verdict, applying YOUR verdict logic.
  - the result is the shared Insights -> Recommendations -> Actions contract, so
    the Scans page renders it with the same component, and the clinician approves
    what goes to the record.

Verdict logic (from the project handoff):
  non-dental                -> "Non-dental — rejected"    (detector skipped)
  dental, 0 detections      -> "Healthy — no findings"
  only abnormal box(es)     -> "Abnormal — needs review"
  any lesion box (0..6)     -> "Disease Detected"
"""
from __future__ import annotations

import os
from typing import Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

import db
from vision import pipeline as vp

# per-class clinical next step (keys = data.yaml names)
RECOMMENDATIONS = {
    "early_caries":    "Early caries — recommend fluoride / monitoring; confirm with bitewing.",
    "advanced_caries": "Advanced caries — recommend restorative treatment; radiograph for depth.",
    "gingivitis":      "Gingival inflammation — recommend scaling and oral-hygiene counselling.",
    "hypodontia":      "Missing teeth — assess for space management / prosthetic options.",
    "plaque":          "Plaque accumulation — recommend professional cleaning.",
    "discoloration":   "Discoloration — assess extrinsic vs intrinsic; review cleaning / whitening.",
    "ulcer":           "Ulcer present — review; if it persists beyond 2 weeks, refer for evaluation.",
    "abnormal":        "Abnormal finding not matching a specific class — clinician review recommended.",
}
SEVERITY = {
    "advanced_caries": "urgent", "ulcer": "urgent",
    "early_caries": "watch", "gingivitis": "watch", "plaque": "watch",
    "hypodontia": "watch", "abnormal": "watch",
    "discoloration": "info",
}


class ImagingState(TypedDict, total=False):
    image_path: str
    p_dental: float
    threshold: float
    gated: bool
    detections: list
    verdict: str
    conditions: list


def gate_node(state: ImagingState) -> dict:
    from PIL import Image
    p, thr = vp.gate_p_dental(Image.open(state["image_path"]))
    return {"p_dental": p, "threshold": thr, "gated": p >= thr}


def detect_node(state: ImagingState) -> dict:
    from PIL import Image
    return {"detections": vp.detect(Image.open(state["image_path"]))}


def _verdict_and_conditions(gated: bool, dets: list):
    if not gated:
        return "Non-dental — rejected", []
    if not dets:
        return "Healthy — no findings", []
    agg = {}
    for d in dets:
        a = agg.setdefault(d["name"], {"name": d["name"], "display": d["display"],
                                       "count": 0, "max_conf": 0.0})
        a["count"] += 1
        a["max_conf"] = max(a["max_conf"], d["conf"])
    conditions = list(agg.values())
    lesions = [c for c in conditions if c["name"] != "abnormal"]
    verdict = "Abnormal — needs review" if not lesions else "Disease Detected"
    return verdict, conditions


def verdict_node(state: ImagingState) -> dict:
    verdict, conditions = _verdict_and_conditions(state["gated"], state.get("detections", []))
    return {"verdict": verdict, "conditions": conditions}


def _route(state: ImagingState) -> str:
    return "detect" if state.get("gated") else "verdict"


def build_graph():
    g = StateGraph(ImagingState)
    g.add_node("gate", gate_node)
    g.add_node("detect", detect_node)
    g.add_node("verdict", verdict_node)
    g.add_edge(START, "gate")
    g.add_conditional_edges("gate", _route, {"detect": "detect", "verdict": "verdict"})
    g.add_edge("detect", "verdict")
    g.add_edge("verdict", END)
    return g.compile()


_graph = build_graph()


def analyze(image_path: str, patient_id: int = None) -> dict:
    """Run the pipeline on one image (via the graph). Returns the shared contract."""
    state = _graph.invoke({"image_path": image_path})
    return _build_payload(state["p_dental"], state["threshold"], state["gated"],
                          state.get("detections", []), patient_id)


def analyze_batch(image_paths: list[str], patient_id: int = None) -> list[dict]:
    """Full-speed path: batched gate over all images, batched detector over only
    the ones that pass the gate, then per-image verdict + contract. Findings
    attach to `patient_id` if given, so Approve writes them to that chart."""
    from PIL import Image
    imgs = [Image.open(p) for p in image_paths]
    gates = vp.gate_p_dental_batch(imgs)                      # one batched forward pass
    dental_idx = [i for i, (p, thr) in enumerate(gates) if p >= thr]
    det_lists = vp.detect_batch([imgs[i] for i in dental_idx])  # one batched detector pass
    dets_by_i = {i: det_lists[k] for k, i in enumerate(dental_idx)}
    return [_build_payload(p, thr, p >= thr, dets_by_i.get(i, []), patient_id)
            for i, (p, thr) in enumerate(gates)]


def _build_payload(p_dental: float, threshold: float, gated: bool, detections: list,
                   patient_id: int = None) -> dict:
    verdict, conditions = _verdict_and_conditions(gated, detections)
    subject = f"patient:{patient_id}" if patient_id else "scan"

    insights = []
    if not gated:
        insights.append({"label": "Not a dental image", "severity": "info",
                         "detail": f"Gate P(dental)={p_dental:.3f} < threshold "
                                   f"{threshold:.3f}. Detector skipped."})
    elif not conditions:
        insights.append({"label": "No findings", "severity": "info",
                         "detail": f"Gate passed (P(dental)={p_dental:.3f}); "
                                   "detector found no lesions."})
    else:
        for c in sorted(conditions, key=lambda x: -x["max_conf"]):
            insights.append({"label": f"{c['display']} ×{c['count']}",
                             "severity": SEVERITY.get(c["name"], "watch"),
                             "detail": f"max confidence {c['max_conf']:.2f}"})

    recommendations = [{"text": RECOMMENDATIONS.get(c["name"], "Clinician review recommended."),
                        "rationale": f"{c['display']} detected"} for c in conditions]

    if not gated:
        action_specs = []
    elif not conditions:
        action_specs = [("Record as healthy screen", "add_findings")]
    else:
        action_specs = [("Add findings to record", "add_findings"),
                        ("Flag for specialist review", "flag_specialist")]
    findings_text = f"{verdict}: " + (", ".join(f"{c['display']} (x{c['count']}, "
                    f"{int(c['max_conf']*100)}%)" for c in conditions) if conditions else "no lesions")
    actions = []
    for label, atype in action_specs:
        payload = {"verdict": verdict}
        if patient_id and atype == "add_findings":
            payload = {"patient_id": patient_id, "content": findings_text}
        aid = db.save_action("imaging", subject, label, atype, payload)
        actions.append({"id": str(aid), "label": label, "type": atype, "status": "proposed"})

    summary = {
        "Non-dental — rejected": "This image was not recognised as dental and was rejected before detection.",
        "Healthy — no findings": "Dental image confirmed; no lesions detected at the screening threshold.",
    }.get(verdict, f"{verdict}: {', '.join(c['display'] for c in conditions)}.")

    return {
        "agent": "imaging", "subject": "scan", "source": "model",
        "verdict": verdict, "p_dental": round(p_dental, 3), "gated": gated,
        "summary": summary, "insights": insights,
        "recommendations": recommendations, "actions": actions,
        "detections": detections,
    }


def analyze_video(video_path: str, patient_id: int = None, sample_fps: float = None) -> dict:
    """Sample frames from a video and run the same gate+detect pipeline as
    analyze_batch — but aggregate PER CLASS PER FRAME, not per raw detection
    box. A video sweep re-detects the same physical lesion across many
    consecutive frames; counting every box the way analyze() does for
    distinct photos would report one real cavity, seen in 40 frames, as "40
    cavities" — misleading, not just noisy. Instead each condition is
    reported as the number of SAMPLED FRAMES it appeared in (out of how many
    were dental), with the single highest-confidence frame kept as a
    representative image, similar in spirit to a photo upload's thumbnail."""
    import base64
    import io

    frames = vp.extract_frames(video_path, sample_fps=sample_fps)
    if not frames:
        raise vp.PipelineUnavailable("no frames could be extracted from the video")

    gates = vp.gate_p_dental_batch(frames)
    dental_idx = [i for i, (p, thr) in enumerate(gates) if p >= thr]
    det_lists = vp.detect_batch([frames[i] for i in dental_idx])
    dets_by_i = {i: det_lists[k] for k, i in enumerate(dental_idx)}
    n_dental = len(dental_idx)

    # aggregate per class ACROSS FRAMES — at most one count per frame per
    # class, even if that class had several boxes in the same frame
    agg = {}
    for i in dental_idx:
        seen_this_frame = set()
        for d in dets_by_i.get(i, []):
            if d["name"] in seen_this_frame:
                continue
            seen_this_frame.add(d["name"])
            a = agg.setdefault(d["name"], {"name": d["name"], "display": d["display"],
                                           "frame_count": 0, "max_conf": 0.0, "best_frame_idx": i})
            a["frame_count"] += 1
            if d["conf"] > a["max_conf"]:
                a["max_conf"] = d["conf"]
                a["best_frame_idx"] = i

    conditions = list(agg.values())
    lesions = [c for c in conditions if c["name"] != "abnormal"]
    if n_dental == 0:
        verdict = "Non-dental — rejected"
    elif not conditions:
        verdict = "Healthy — no findings"
    elif not lesions:
        verdict = "Abnormal — needs review"
    else:
        verdict = "Disease Detected"

    subject = f"patient:{patient_id}" if patient_id else "scan"
    insights = []
    if n_dental == 0:
        insights.append({"label": "No dental frames found", "severity": "info",
                         "detail": f"None of the {len(frames)} sampled frames passed the dental gate."})
    elif not conditions:
        insights.append({"label": "No findings", "severity": "info",
                         "detail": f"{n_dental} of {len(frames)} sampled frames were dental; "
                                   "detector found no lesions in any of them."})
    else:
        for c in sorted(conditions, key=lambda x: -x["max_conf"]):
            insights.append({"label": f"{c['display']} — seen in {c['frame_count']}/{n_dental} dental frames",
                             "severity": SEVERITY.get(c["name"], "watch"),
                             "detail": f"peak confidence {c['max_conf']:.2f}"})

    recommendations = [{"text": RECOMMENDATIONS.get(c["name"], "Clinician review recommended."),
                        "rationale": f"{c['display']} seen in {c['frame_count']} sampled frames"}
                       for c in conditions]

    findings_text = (f"{verdict} (video — {n_dental}/{len(frames)} sampled frames were dental): " +
                     (", ".join(f"{c['display']} (seen in {c['frame_count']} frames, "
                                f"peak {int(c['max_conf']*100)}%)" for c in conditions)
                      if conditions else "no lesions"))

    action_specs = []
    if n_dental > 0 and not conditions:
        action_specs = [("Record as healthy screen", "add_findings")]
    elif conditions:
        action_specs = [("Add findings to record", "add_findings"),
                        ("Flag for specialist review", "flag_specialist")]
    actions = []
    for label, atype in action_specs:
        payload = {"verdict": verdict}
        if patient_id and atype == "add_findings":
            payload = {"patient_id": patient_id, "content": findings_text}
        aid = db.save_action("imaging", subject, label, atype, payload)
        actions.append({"id": str(aid), "label": label, "type": atype, "status": "proposed"})

    summary = (f"Video screened: {n_dental} of {len(frames)} sampled frames were dental. " +
              (f"{len(conditions)} condition(s) flagged across the sweep."
               if conditions else "No lesions detected in any sampled frame."))

    # one representative (highest-confidence) frame per finding, so the UI
    # has something to show similar to a photo upload's thumbnail
    representative = {}
    for c in conditions:
        img = frames[c["best_frame_idx"]]
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=80)
        representative[c["name"]] = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    return {
        "agent": "imaging", "subject": "scan", "source": "model", "mode": "video",
        "verdict": verdict, "gated": n_dental > 0,
        "frames_sampled": len(frames), "dental_frames": n_dental,
        "summary": summary, "insights": insights,
        "recommendations": recommendations, "actions": actions,
        "conditions": [{"name": c["name"], "display": c["display"],
                        "frame_count": c["frame_count"], "max_conf": round(c["max_conf"], 3),
                        "representative_image": representative.get(c["name"])} for c in conditions],
    }
