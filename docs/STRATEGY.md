# dentaVIS: where it stands, and where to take it

*Assessment as of November 2026 (commit `d6db51d`, which added the agent registry). Market notes come from general knowledge up to mid-2026; check them before relying on them.*

**In short:** the engineering patterns are ahead of where a prototype usually is. Specifically: human approval before any action, durable workflows, and checks on what the model writes. The product is behind the market. There is no voice input, no connections to other clinic systems, no evaluation data, and no clinical validation, and the models are small and slow. Being fully local is the strongest strategic card, and it isn't being used as a selling point yet.

## What's already strong (keep and promote)

| Capability | Why it matters to the market |
|---|---|
| **Fully local (Ollama, SQLite)**, so patient data never leaves the clinic | Most dental AI products run in the cloud. Privacy laws (India's DPDP Act, HIPAA in the US) and clinics with poor internet make on-premises software a real point of difference. |
| **The AI only proposes; a human approves**, with the approver's name logged | This is now the standard expectation for clinical AI. Regulators and buyers ask about it first. |
| **Durable visit pipeline** (LangGraph checkpoints, pause/resume, triggered by events) | Agents built into the workflow, not chatbots, are where agentic software is heading. Few dental tools chain scan → note → report like this. |
| **Safety and grounding checks** on every draft, in all four drafting agents | Invented findings are the main barrier to AI-written clinical notes. A check you can point to is a selling point. |
| **Tool-calling Copilot** with guarded writes and checked patient IDs | This is the current agent design: the model picks its tools, and code enforces the limits. |
| **Agent registry** (`GET /api/agents`, `POST /api/agents/{key}/run`) | One list of all 8 agents, with parameter checks and admin-only controls. It's the base for exposing them to outside tools (see MCP below). |
| **Explainable no-show model** (logistic regression, no demographic data) | This is the right call. Not every problem needs an LLM. |

## Gaps versus the market

### Competitors

Dental AI spending has gathered in four areas:

- **X-ray detection:** Overjet, Pearl and VideaHealth have FDA clearances for it. Competing head-on with them is a losing fight unless you have regulatory clearance.
- **AI scribes:** voice-to-clinical-note tools are the fastest-adopted category across healthcare.
- **Billing and insurance claims automation:** clinics feel the return on investment directly here.
- **Front-desk voice agents:** answering calls, booking, and sending recall and reminder messages.

dentaVIS touches all four lightly but leads in none.

### Technical gaps

- **Model quality and speed:** qwen3:1.7b on a CPU takes 30–100 seconds per answer and makes mistakes, as testing showed. A clinic won't wait that long.
- **No evaluations:** there's no fixed test set and no measured grounding-failure or safety-violation rate. Without numbers, you can't show any clinic or regulator that the system is safe.
- **Grounding matches words:** it can't catch claims about *when* something happened. "Abscess" written as current passes if the word appears anywhere in the record.
- **No connections to other systems:** no FHIR or HL7 for health records, no DICOM for X-rays, no link to existing practice-management software or insurers. Clinics won't re-enter their data.
- **Tests aren't in the repo,** and there's no CI to run them on each change.
- **Production readiness:** SQLite, a single admin role and no multi-clinic setup. There's no monitoring of agent runs and no encryption of stored data.
- **Prompt injection:** text from the public booking page and patient notes goes into the model's prompts unfiltered.

## Methods worth adopting, in priority order

### Do now (biggest gain for the effort)

1. **Evaluation set:** 50–100 real-style cases with expected outputs. Measure the grounding-failure rate, safety violations and whether the Copilot picks the right tool. Run it in CI on every change. Everything else depends on this.
2. **Citations:** make the model attach a source ID to each sentence (for example `imaging#3` or `visit 2025-07-01`), then check the claim against that source with a small local entailment model. This fixes the "when did it happen" gap and gives reviewers one-click verification.
3. **A faster, better model:** a GPU, or a quantised 7–8B instruct model through llama.cpp or vLLM. Send simple tasks to the 1.7B model and harder ones to the bigger one.

### Next (product differentiation)

4. **Voice-to-SOAP:** run Whisper locally to turn a recording into a draft SOAP note, with the existing grounding check. This joins the hottest category while keeping data on-site, which is a clear pitch.
5. **Memory per patient:** search over each patient's past notes (for example with `nomic-embed-text`), so the agents see full history instead of the last few visits.
6. **MCP server:** expose the agent registry and the Copilot's tools through MCP, the standard way AI tools plug into each other. Other agents and assistants could then use dentaVIS without custom integration.
7. **Tracing:** use OpenTelemetry or Langfuse to record each agent step, tool call and approval. This also strengthens the audit trail for compliance.

### Later (to sell it)

8. **Interoperability:** FHIR or HL7 export, DICOM import, and a connector to one existing practice-management system.
9. **Fine-tuning:** a LoRA adapter trained on de-identified dental notes, which helps the small models most.
10. **Regulatory path for the vision model:** a clinical validation study, then CDSCO (India) or FDA 510(k) clearance, before claiming it detects anything.
11. **Hardening:** Postgres, role-based access, encryption of stored data, separate data per clinic, and filtering of untrusted text before it reaches the model.

## Suggested strategy

Don't sell it as "AI that reads X-rays." Sell it as **a private AI assistant for the whole visit, where nothing leaves the clinic**: voice → grounded SOAP → report → recall, with every AI claim cited and approved by a clinician. Start with privacy-sensitive or low-connectivity clinics, such as Indian private practices and dental chains. That's where cloud competitors are weakest and the local-first design is strongest.

The first step that unlocks the rest is the evaluation set (item 1). Without measured accuracy, nothing else can be shown to a clinic or regulator.
