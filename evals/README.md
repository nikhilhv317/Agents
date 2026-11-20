# Evaluations

Roadmap item 1 in [docs/STRATEGY.md](../docs/STRATEGY.md): a fixed set of cases
with expected outcomes, so changes to the agents, prompts, model or checks are
measured rather than eyeballed.

```bash
python evals/run.py                     # offline checker suite (CI runs this)
python evals/run.py --suite live        # drafting + Copilot on the local Ollama model
python evals/run.py --suite all --json evals/results/run.json
python evals/run.py --case latex        # only cases whose id contains "latex"
```

Every run uses a fresh temporary database with the seeded 12-patient roster, so
it never reads or changes `dpms.db`.

## Suites

| Suite | Cases | Needs | Measures |
|---|---|---|---|
| `checks` | 62 | nothing (runs in CI) | Whether the safety, grounding, citation and action-claim checks flag exactly what they should, on hand-labelled drafts. Any regression fails the run. |
| `drafting` | 16 | Ollama | SOAP and Report agents: **grounding failure rate** (drafts with invented clinical content), **safety violation rate** (a recorded allergy or condition missing, negated or denied) and **citation failure rate** (missing, wrong or unknown source tags, counted separately so the grounding rate stays comparable). |
| `copilot` | 14 | Ollama | Did it call the right tools, avoid forbidden ones, propose the right actions and state the right facts? Includes prompt-injection and semantic history-search cases. |

Live suites report rates. To make them gates, add thresholds:
`--max-grounding-failure 0.3 --max-safety-violation 0.1 --min-copilot-pass 0.9`.
`--model TAG` runs the SOAP and report agents on another Ollama model, to compare models.

Cases live in `cases/*.yaml`, with the format described at the top of each file.
Expectations were written from reading each draft clinically, not copied from the
checker's output.

### Known gaps

A `checks` case marked `known_gap: true` records something the checks *should*
do but don't yet. It's reported as `gap` and doesn't fail CI. If a change makes
one pass, it shows as `FIXED`, so you can remove the label and keep it as a
regression test.

The first baseline had 10 known gaps: timing, vocabulary, hedge scope, advice
wording, paraphrased conditions, "no allergic reactions", prescribing the
allergen, and two action-claim phrasings. **All 10 are now fixed** and kept as
regression cases. There are none open.

## How the numbers moved

`qwen3:1.7b` via Ollama, 100% on this laptop's GPU. Each drafting row is one run of
16 cases; runs vary by a case or two even at temperature 0.

| Stage | Invented content | Safety violations | Citation failures | Copilot | Checks / notes |
|---|---|---|---|---|---|
| Baseline (flagged) | 50% (8/16) | 31% (5/16) | — | 11/12 | 34 pass, 10 gaps |
| Baseline (manual read) | ~31% | ~19% | — | | |
| Checks fixed: safety paraphrase, prescribing, Copilot retry | 44% | 19% | — | 12/12 | 53 pass, 1 gap |
| Citations + structured sources + objective rule | 12–19% | 25% → 6% after restating allergies from the record | 75% | 12/12 | 62 pass, 0 gaps |
| qwen2.5:3b instead (comparison) | 25% | 0% | 94% | — | |
| Per-patient memory, hardening | 19% | 6% | 75% | 13/13 | 62 pass |
| LoRA v1 (synthetic data, plans ignore the record) | 0% | 0% | 0% | — | relevance misses 4/8 (base: 1/8) |
| **LoRA v2** (plans carry allergy/condition precautions), now the default | **0%** | **0%** | **0%** | 13/13 (Copilot stays on base) | relevance misses **0/8** |
**Clinical relevance** (added with the first LoRA): 8 drafting cases list terms the Assessment or Plan
must contain, for example latex-free precautions, pregnancy, extraction plus bleeding precautions. LoRA
v1 scored perfectly on grounding but missed 4/8: its synthetic training plans ignored the patient's
record, and the grounding checks can't see that. v2's training data adds those precautions.

**Be careful reading v2's perfect scores.** The relevance terms were written before v2's data, but v2's
data was written to fix that failure category, so the metric isn't fully independent. Two things back it
up. First, a held-out spot check with allergy/condition combinations no eval case uses (lidocaine +
aspirin allergy on warfarin; chlorhexidine allergy, pregnancy and asthma together) got correct
precautions from v2 and missing ones from the base model. Second, three visit-pipeline runs end to end
produced clean drafts.

**Known weaknesses of v2:**
- Plans are partly templated from 11 synthetic scenarios. A pregnant patient with swollen gums got both
  "periapical radiograph" and "defer elective radiographs".
- It was trained only on synthetic cases.
- It runs through transformers (`hf:` tags in `llm.py`), not Ollama: Ollama 0.40 on Windows can't import
  safetensors.
- It's used only when `ml/lora/out/merged` exists locally (3.4 GB, not in git). Otherwise the base model
  is used, with the base model's numbers above.

## First baseline in detail (21 Nov 2026)

`qwen3:1.7b` via Ollama, running 100% on this laptop's GPU.

| Measure | Result |
|---|---|
| Checks | 34 pass, 0 fail, 10 known gaps. Flag precision 0.89, recall 0.89. |
| Drafting: grounding failure rate | **50%** (8/16), 3.4 s per draft |
| Drafting: safety violation rate | **31%** (5/16) |
| Copilot | **11/12** pass. Tool selection 0.92, actions 1.0, 5.2 s per question. Prompt injection resisted. |

What a manual read of those drafts showed:

- **Grounding:** 5 of the 16 drafts invent an examination with findings ("small
  abscess", "tenderness", "evidence of infection"). This happens when no exam
  findings were entered, which is the failure the grounding check was written for.
  The other flags are unrecorded negatives ("no redness noted") or wording like
  "plaque control". So the rate of drafts with a genuinely invented finding is
  about **31%**.
- **Safety:** 3 of the 5 are real. One SOAP note says "Patient is latex-free" for a
  latex-allergic patient, and two 3–4 line reports leave out a penicillin allergy.
  The other 2 are the pregnancy paraphrase false alarm, so the real rate is
  about **19%**.
- **Copilot:** the failure is `pregnancy-lookup`. The model answered "no information
  available" without calling any tool.

Live results vary by a case or two between runs even at temperature 0. Leave some
margin when setting thresholds.

## Bug found while building this

`safety-condition-with-parentheses`: a recorded condition ending in ")", such as
"Pregnant (2nd trimester)", could never match, so every draft for that patient got
a false "condition not mentioned" warning. It's fixed in
`agents/safety.py` (`_mentions`) and kept as a regression case.
