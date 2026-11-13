"""
LLM layer  (local, via Ollama).

The ONLY place that talks to a model. Inference runs on a local Ollama server,
so patient data never leaves the machine — which is the point for a clinical
system. No API key, no cloud call.

`complete_json()` asks the model for JSON and parses it. When a `schema` is
given it is passed to Ollama's `format` parameter, which constrains generation
at the token level so the reply is guaranteed to match the shape — the thing
small local models otherwise get wrong. If Ollama isn't running or the model
isn't pulled, it raises LLMUnavailable so the caller can fall back to a
deterministic demo result, and the whole app still runs with nothing installed.

--- Model routing ---------------------------------------------------------------
Every call site names the TASK it's running (report_summary, soap_note, ...).
TASK_MODELS maps each task to the model tier that should serve it. This is real
routing infrastructure, not a single hardcoded model: moving one task to a
stronger model later (e.g. a SOAP note that needs more reasoning) is a one-line
change to TASK_MODELS, with zero changes to the calling agent.

v1 deployment (this AWS instance): every task is pinned to the SAME small
model — one model resident in Ollama, lower/predictable memory footprint,
one thing to pull and tune. The routing table exists so that changes when
there's a reason to, not by accident (see git history: this app once had
qwen3:4b, qwen3:8b, and qwen3:1.7b all referenced in different files at once —
routing through one table is what stops that from happening again).

Setup:
    # install Ollama from https://ollama.com, then:
    ollama pull qwen3:1.7b
    ollama serve            # usually already running as a service
"""
import os
import json

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")  # tiny + fast; qwen3 family so think=False works
HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# task name -> model tag. Every entry points at DEFAULT_MODEL for v1; bump an
# individual task here (after pulling that tag) without touching agent code.
TASK_MODELS: dict[str, str] = {
    "report_summary":       DEFAULT_MODEL,   # report_agent.py — 3-4 line summary
    "patient_brief":        DEFAULT_MODEL,   # patient_intelligence.py — clinical brief
    "soap_note":            DEFAULT_MODEL,   # clinical_assistant.py — SOAP note
    "scheduling_reminder":  DEFAULT_MODEL,   # scheduling_agent.py — SMS reminder text
    # copilot.py — multi-step tool calling. Same 1.7B model, but run with
    # thinking ON (see copilot._model): with thinking off it describes tool
    # calls in prose instead of making them. Measured on this CPU box, a
    # two-round question takes ~35s. The local "qwen3:4b" tag is the 2507
    # thinking-only build: it can't turn thinking off, Ollama leaks the
    # reasoning into the answer, and it ran ~4-5 min per question. If you want
    # a bigger model here, use a non-thinking build such as qwen3:4b-instruct-2507.
    "copilot":              os.getenv("COPILOT_MODEL", DEFAULT_MODEL),
}


def resolve_model(task: str | None) -> str:
    """The one place a task name becomes an actual model tag."""
    if task and task in TASK_MODELS:
        return TASK_MODELS[task]
    return DEFAULT_MODEL


class LLMUnavailable(Exception):
    pass


def _client():
    try:
        import ollama
    except ImportError as e:
        raise LLMUnavailable("ollama package not installed (pip install ollama)") from e
    return ollama.Client(host=HOST)


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    return t.strip()


def complete_json(system: str, user: str, schema: dict | None = None,
                  task: str | None = None) -> dict:
    """
    Run one local completion and parse a JSON object out of it.

    schema: a JSON Schema dict (e.g. SomeModel.model_json_schema()). When given,
    Ollama constrains the output to match it. When None, plain JSON mode is used.
    task: which TASK_MODELS entry routes this call (see module docstring). Omit
    to use DEFAULT_MODEL directly.
    """
    model = resolve_model(task)
    client = _client()
    fmt = schema if schema else "json"
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        try:
            resp = client.chat(
                model=model, messages=msgs, format=fmt,
                think=False,            # skip Qwen3's slow "thinking" pass — big speedup
                keep_alive="30m",       # keep the model loaded between calls
                options={"temperature": 0},
            )
        except TypeError:
            # older ollama client without think= support
            resp = client.chat(model=model, messages=msgs, format=fmt,
                               keep_alive="30m", options={"temperature": 0})
        content = resp["message"]["content"]
        return json.loads(_strip_fences(content))
    except LLMUnavailable:
        raise
    except Exception as e:  # connection refused, model missing, parse error -> demo
        raise LLMUnavailable(str(e)) from e
