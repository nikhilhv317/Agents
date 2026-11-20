"""
Per-patient memory (roadmap item 5) — semantic search over a patient's history.

Agents used to see the last few visits, plus older ones sharing a keyword. Here
every visit, saved note (imaging, SOAP, brief, Copilot) and finalized report is
embedded with the local `nomic-embed-text` model (Ollama) and searched by
meaning: "swollen gum" finds "localised abscess", "loose tooth" finds "mobility".

The index is lazy: search() embeds whatever the patient has that isn't indexed
yet (keyed by a hash of the text), so new records and old databases both work
without write hooks or a migration. If the embedding model is unavailable,
search falls back to keyword overlap, so callers always get an answer.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re

import db

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
_STOP = {"the", "a", "an", "and", "or", "for", "of", "to", "with", "on", "in", "at", "is", "was",
         "patient", "review", "follow-up", "check-up", "noted", "today"}


def _schema():
    conn = db.connect()
    conn.execute("""CREATE TABLE IF NOT EXISTS memory_index (
        id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL, kind TEXT NOT NULL,
        ref TEXT NOT NULL, date TEXT, text TEXT NOT NULL, text_hash TEXT NOT NULL,
        model TEXT NOT NULL, vector TEXT NOT NULL,
        UNIQUE(patient_id, text_hash, model))""")
    conn.commit(); conn.close()


def patient_items(patient_id: int) -> list[dict]:
    """Everything on the chart worth remembering, as {kind, ref, date, text}."""
    p = db.get_patient(int(patient_id)) or {}
    items = [{"kind": "visit", "ref": f"visit:{v['date']}", "date": v["date"],
              "text": f"{v['date']} visit: {v['complaint']}. Diagnosis: {v['diagnosis']}. "
                      f"Treatment: {v['treatment']}. {v['notes'] or ''}".strip()}
             for v in p.get("visits", [])]
    items += [{"kind": f"note:{n['kind']}", "ref": f"note:{n['kind']}:{n['created_at']}",
               "date": (n["created_at"] or "")[:10], "text": f"{n['kind']} note: {n['content']}"}
              for n in db.get_notes(int(patient_id))]
    items += [{"kind": "report", "ref": f"report:{r['id']}", "date": (r["finalized_at"] or "")[:10],
               "text": f"Report: {r['ai_summary']}"}
              for r in db.list_reports("finalized") if r["patient_id"] == int(patient_id) and r["ai_summary"]]
    return items


def _embed(texts: list[str]) -> list[list[float]]:
    import llm
    client = llm._client()
    resp = client.embed(model=EMBED_MODEL, input=texts, keep_alive="30m")
    return resp["embeddings"]


def _cos(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _ensure_indexed(patient_id: int, items: list[dict]) -> dict[str, list[float]]:
    """text_hash -> vector for every item, embedding only what's missing."""
    _schema()
    conn = db.connect()
    have = {r["text_hash"]: json.loads(r["vector"]) for r in conn.execute(
        "SELECT text_hash, vector FROM memory_index WHERE patient_id=? AND model=?",
        (patient_id, EMBED_MODEL))}
    missing = [it for it in items if _hash(it["text"]) not in have]
    if missing:
        # nomic-embed-text wants a task prefix; documents and queries differ
        vecs = _embed(["search_document: " + it["text"] for it in missing])
        for it, v in zip(missing, vecs):
            h = _hash(it["text"])
            conn.execute("INSERT OR IGNORE INTO memory_index (patient_id,kind,ref,date,text,text_hash,model,vector) "
                         "VALUES (?,?,?,?,?,?,?,?)",
                         (patient_id, it["kind"], it["ref"], it["date"], it["text"], h, EMBED_MODEL, json.dumps(v)))
            have[h] = v
        conn.commit()
    conn.close()
    return have


def _keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9#]+", (text or "").lower()) if w not in _STOP and len(w) > 2}


def search(patient_id: int, query: str, k: int = 5) -> list[dict]:
    """The k history items most relevant to `query`, best first, each with a score
    and how it was ranked ("embedding" or "keyword")."""
    items = patient_items(patient_id)
    if not items or not (query or "").strip():
        return []
    try:
        vectors = _ensure_indexed(int(patient_id), items)
        qv = _embed(["search_query: " + query])[0]
        scored = [(round(_cos(qv, vectors[_hash(it["text"])]), 3), it) for it in items]
        method = "embedding"
    except Exception:   # Ollama down / model not pulled -> still useful, just cruder
        qk = _keywords(query)
        scored = [(len(qk & _keywords(it["text"])), it) for it in items]
        method = "keyword"
    scored.sort(key=lambda s: s[0], reverse=True)
    return [{**it, "score": s, "method": method} for s, it in scored[:k]]


def relevant_visits(patient: dict, query: str, max_visits: int = 6, recent: int = 2) -> list[dict]:
    """Visits to put in a prompt: all of them when there are few, otherwise the
    `recent` latest plus the most relevant older ones — in date order, so source
    numbering ([V1], [V2] ...) stays chronological."""
    visits = patient.get("visits", [])
    if len(visits) <= max_visits:
        return visits
    keep = {v["date"] for v in visits[-recent:]}
    for hit in search(patient["id"], query, k=max_visits * 2):
        if hit["kind"] == "visit" and len(keep) < max_visits:
            keep.add(hit["date"])
    return [v for v in visits if v["date"] in keep]
