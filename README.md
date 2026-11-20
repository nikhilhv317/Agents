# DPMS Agentic AI — prototype

Two agents, wired end to end, running fully local (Ollama). Patient data never
leaves the machine.

- **Patient Intelligence** (AI Summary screen) — reads a patient's record + visits
  and drafts a clinical brief. Local LLM writes it; clinician approves.
- **Smart Scheduling** (Appointments screen) — every morning scores today's
  schedule for no-show risk with a logistic-regression model, drafts reminders
  with the local LLM, and waits for approval before any go out.

## Stack

```
HTML/JS (dentaVIS UI) → HTTP/JSON → FastAPI → Agents → Ollama (local)
                                       │         │
                                       │         └─ scikit-learn (no-show model)
                                     SQLite         LangGraph (scheduling flow)
```

Division of labour in the scheduling agent — the important idea:
- **scikit-learn LogisticRegression** produces the risk NUMBER (reproducible, explainable).
- **qwen3:4b via Ollama** produces only the WORDS (reason line, reminder, summary).
- **LangGraph** sequences it and PAUSES at approval via `interrupt()`, checkpointed
  to SQLite so a paused run survives a restart and resumes on approve.

## One-time setup

Install Ollama (https://ollama.com), then:

```bat
ollama pull qwen3:4b
setup.bat
```

`setup.bat` installs packages, generates synthetic history, and trains the model.

## Run

```bat
run.bat
```

Opens http://127.0.0.1:8000. Or manually: `uvicorn app:app`.

- **AI Summary**: pick a patient → Generate brief.
- **Appointments**: opens today's scored schedule; tick the reminders to send → Approve.

A green **Local model** tag means the LLM answered; **Demo (offline)** means Ollama
wasn't reachable and a deterministic fallback ran (the app never hard-fails).

## Files

| Path | Role |
|------|------|
| `app.py` | FastAPI — serves UI + both agents' APIs + morning job |
| `db.py` | SQLite schema, seed, appointment/log helpers |
| `contracts.py` | Shared agent contract (Insights→Recs→Actions) |
| `llm.py` | Ollama client for Patient Intelligence (+ demo fallback) |
| `agents/base.py` | BaseAgent lifecycle |
| `agents/patient_intelligence.py` | Patient Intelligence agent |
| `agents/scheduling_agent.py` | Smart Scheduling agent (LangGraph graph + runner) |
| `agents/noshow_features.py` | One feature definition — shared train ↔ serve |
| `agents/copilot.py` | Clinic Copilot — tool-calling agent (model picks tools; writes are proposals only) |
| `agents/__init__.py` | Registry + action handlers (Patient Intelligence) |
| `ml/generate_data.py` | Synthetic appointment history + today's schedule |
| `ml/train.py` | Trains the model → `models/noshow.pkl` |
| `models/noshow.pkl` | Trained logistic-regression pipeline (regenerate locally) |
| `static/index.html` | dentaVIS UI (AI Summary + Appointments) |
| `setup.bat` / `run.bat` | Windows one-time setup / daily launch |

## Model notes

- Target AUC ~0.73 — deliberately realistic, not a fake 0.95. No-show is partly
  random; a flagged patient is right ~half the time, which is why the only action
  on a prediction is a soft reminder, never an auto-cancel.
- Always `python ml\train.py` on your own machine so `noshow.pkl` matches your
  installed scikit-learn version.
- The scheduling model uses behavioural features only (prior attendance, lead time,
  type, confirmation, hour/day) — no demographics, by design.

## Visit Pipeline (event-driven, durable, human-in-the-loop)

`agents/visit_pipeline.py` chains the agents for one visit and pauses for a
clinician at each decision (**Visit Pipeline** page):

```
imaging finding approved ─▶ gather imaging ─▶ Clinical agent drafts SOAP ─▶ ⏸ review
   ─▶ Report agent drafts report ─▶ ⏸ review ─▶ finalize + add visit to history
```

- **Event trigger:** approving an imaging finding on the Scan page starts it. A
  second finding approved while the SOAP note awaits review triggers a redraft.
- **Pauses:** LangGraph `interrupt()`; the clinician approves (optionally edited),
  redrafts or rejects, passed back via `Command(resume=...)`. Who decided is logged.
- **Durable:** state is checkpointed to `pipeline_checkpoints.db` (next to
  `dpms.db`). Paused runs survive restarts; a run that was mid-step when the
  server stopped continues from its last checkpoint on startup.
- **Safety check:** every draft is compared with the record before review —
  each recorded allergy must be mentioned and not negated, each condition
  mentioned. (Testing caught qwen3:1.7b turning a history note "Latex-free
  gloves" into "patient is latex-free" for a latex-allergic patient.)
- **Grounding check** (`agents/grounding.py`): every clinical finding in a draft
  must trace to a source. The SOAP Objective may only use *today's* complaint
  and imaging; Subjective/Assessment may also use visit history; the report may
  also use the clinician-approved SOAP note. It flags invented exams
  ("examination reveals…" with no exam entered), unsourced findings, findings
  copied from a past visit as if current, and unsourced tooth numbers / mm
  measurements. Hedged mentions ("possible infection") aren't flagged. It's a
  vocabulary-based check: it narrows what the reviewer must verify, it doesn't
  replace the reviewer. The SOAP part lives in the Clinical Assistant
  (`clinical_assistant.ground_soap`), so `/api/clinical/draft` returns the same
  `unsupported` list (plus a "Grounding check" insight) on every draft; an
  examination may only be described when the clinician typed the findings.
  The report part lives in the Report agent (`report_agent.ground_report`), so
  the **Create Report** page shows the same warning before you finalize. Its
  sources are the record, imaging and clinician findings; visit history is not
  in its prompt, so a finding found only there is flagged as from a past visit.

## Clinic Copilot (tool-calling agent)

The **Copilot** page is a chat where the *model* decides which steps to take,
unlike the other agents whose steps are fixed in code. It loops agent → tools →
agent (LangGraph, max 6 rounds) over:

- **Read tools** (run freely): `find_patients`, `get_patient_record`,
  `get_patient_notes`, `get_imaging_findings`, `get_schedule`.
- **Write tools** (proposals only): `propose_followup`, `propose_chart_note`,
  `propose_patient_message`. They appear as **Approve** buttons; nothing is
  written or sent until a human approves, and the log records who approved.

Guards: a patient_id is only accepted after a lookup returned it (small models
guess ids), an answer that claims an action with nothing behind it gets a
visible warning, and the **grounding check** (`agents/grounding.py`, same as
the visit pipeline) flags clinical findings in the answer that appear in no
tool result or chat message — e.g. it caught "active infection" claimed from a
year-old abscess record. Negations ("no caries on record") are allowed. Each reply shows its tool calls and the model's reasoning.

Model: `qwen3:1.7b` with thinking on (set `COPILOT_MODEL` to change). With
thinking off it describes tool calls instead of making them. The local
`qwen3:4b` tag is the thinking-only 2507 build and leaks its reasoning into
answers — use a non-thinking build such as `qwen3:4b-instruct-2507` if you want
a bigger model. On CPU expect roughly 40–100 s per question.

## Adding the next agent (e.g. Imaging)

Same shape: a module in `agents/` returning the shared contract, plus its screen.
Imaging needs a vision-capable local model (e.g. a Qwen-VL or llama3.2-vision tag)
since it reads an image rather than a record.

## Strategy

Where the project stands against the market, and a prioritised roadmap: [docs/STRATEGY.md](docs/STRATEGY.md).
