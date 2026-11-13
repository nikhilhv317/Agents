"""
Grounding check — does every clinical finding in a draft come from a source?

Found in testing: given only the complaint "sensitivity in upper right molar"
and no exam findings, qwen3:1.7b wrote "Clinical examination reveals localized
tenderness and mild swelling". Nobody examined anything; the model invented it,
and it then flowed into the report. The safety check (allergies/conditions)
can't see this, so this module checks the other direction: for each FINDING
the draft asserts, is it in the material the model was actually given?

Deterministic on purpose. An LLM judge on a 1.7B model would be slow and as
unreliable as the drafter it's checking. The trade-off: it only knows the
finding vocabulary below, so it catches the common inventions, not all of them
— it narrows what the reviewer must look for, it doesn't replace the reviewer.

Rules:
  - A finding term (tenderness, swelling, caries, ...) in the draft must also
    appear in the sources (complaint, clinician findings, imaging, record,
    visit history — and, for a report, the SOAP note the clinician approved).
  - Hedged mentions are not claims: "possible infection", "rule out abscess",
    "monitor for swelling" are skipped.
  - Tooth/site numbers and mm measurements must appear in the sources.
  - "Examination reveals ..." with no clinician exam findings is flagged: the
    draft is claiming an exam that wasn't recorded.
"""
from __future__ import annotations

import re

# canonical finding -> regex matching its word forms
FINDINGS = {
    "tenderness": r"tender(?:ness)?",
    "swelling": r"swell(?:ing|ed)?|swollen|o?edema(?:tous)?",
    "bleeding": r"bleed(?:ing|s)?|bled|ha?emorrhag\w*",
    "pus / suppuration": r"pus|purulent|suppurat\w*",
    "mobility": r"mobil(?:e|ity)|loose(?:ness)?",
    "pain": r"pain(?:ful)?|(?:tooth)?ache|aching|sore(?:ness)?",
    "sensitivity": r"sensitiv(?:e|ity)",
    "caries": r"caries|carious|cavit(?:y|ies)|decay(?:ed)?",
    "abscess": r"abscess(?:es|ed)?",
    "fracture": r"fractur\w*|crack(?:ed|s)?|chipped",
    "gingival recession": r"recession|receding",
    "inflammation": r"inflam\w*",
    "gingivitis": r"gingivitis",
    "periodontitis": r"periodont\w*",
    "plaque": r"plaque",
    "calculus": r"calculus|tartar",
    "redness": r"redness|erythema\w*",
    "percussion": r"percussion",
    "radiolucency": r"radiolucen\w*",
    "bone loss": r"bone loss",
    "pocketing": r"pocket(?:s|ing)?",
    "ulcer": r"ulcer\w*",
    "lesion": r"lesions?",
    "sinus tract / fistula": r"fistula|sinus tract",
    "discoloration": r"discolou?r\w*",
    "infection": r"infect(?:ed|ion|ions)",
    "missing teeth": r"edentul\w*|hypodontia|missing tee?th",
}
_FINDING_RE = {k: re.compile(rf"\b(?:{v})\b", re.IGNORECASE) for k, v in FINDINGS.items()}

# words that turn a mention into a question/plan rather than a finding
_HEDGE = re.compile(r"\b(possible|possibly|probable|suspect\w*|query|r/o|rule out|consider|"
                    r"risk of|monitor(?:ing)? for|assess(?:ing)? for|evaluat\w* for|check for|"
                    r"watch for|to (?:assess|evaluate|exclude|rule out)|if|unless|may|might|"
                    r"could|likely|differential)\b|\?", re.IGNORECASE)
_EXAM_CLAIM = re.compile(r"\b(?:clinical )?exam(?:ination)?\s+(?:reveals?|revealed|shows?|showed|"
                         r"demonstrates?|demonstrated|notes?|noted|finds?|found)\b|"
                         r"\bon (?:clinical )?exam(?:ination)?\b", re.IGNORECASE)
_TOOTH = re.compile(r"(?:#|\btooth\s*|\bsite\s*|\bteeth\s*)(\d{1,2})\b", re.IGNORECASE)
_MM = re.compile(r"\b(\d+(?:\.\d+)?)\s?mm\b", re.IGNORECASE)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.;!?])\s+|\n+", text or "") if s.strip()]


_NEGATION = re.compile(r"\b(no|not|without|none|absent|denies|denied|free of|neither|nor|never)\b",
                       re.IGNORECASE)


def _clause_before(sentence: str, start: int) -> str:
    # the same clause, up to the previous comma/colon
    return re.split(r"[,:]", sentence[:start])[-1]


def _hedged(sentence: str, start: int) -> bool:
    return bool(_HEDGE.search(_clause_before(sentence, start)))


def _negated(sentence: str, start: int) -> bool:
    return bool(_NEGATION.search(_clause_before(sentence, start)))


def check(draft: str, sources: str, exam_findings_given: bool = False,
          history: str = "", allow_negated: bool = False) -> list[dict]:
    """Return unsupported claims as [{"claim", "detail", "snippet"}], deduplicated.

    history: past-visit text that is NOT a valid source for this draft (e.g. for
    the Objective section, which must describe today). A finding found only
    there is still flagged, with a clearer message — we watched the model take
    "Swollen gum" from a past visit and write "examination reveals swelling".

    allow_negated: skip "no caries", "without swelling". Right for the Copilot,
    whose "no caries on record" is a statement about the records; wrong for a
    SOAP note, where "no swelling" claims an examination that never happened."""
    src = sources or ""
    out, seen = [], set()

    def why(word: str, rx) -> str:
        if history and rx.search(history):
            return f"'{word}' appears only in a PAST visit, but is stated as a current finding."
        return f"'{word}' is not in the complaint, findings, imaging or history."

    def add(claim, detail, snippet):
        if claim not in seen:
            seen.add(claim)
            out.append({"claim": claim, "detail": detail, "snippet": snippet[:160]})

    for sent in _sentences(draft):
        if not exam_findings_given and _EXAM_CLAIM.search(sent):
            add("examination claimed",
                "The draft describes an examination, but no exam findings were entered.", sent)
        for name, rx in _FINDING_RE.items():
            for m in rx.finditer(sent):
                if (_hedged(sent, m.start()) or rx.search(src)
                        or (allow_negated and _negated(sent, m.start()))):
                    continue
                add(name, why(m.group(0), rx), sent)
                break
        for m in _TOOTH.finditer(sent):
            if not re.search(rf"(?<!\d){m.group(1)}(?!\d)", src):
                add(f"tooth/site {m.group(1)}", f"'{m.group(0).strip()}' does not appear in any source.", sent)
        for m in _MM.finditer(sent):
            if not re.search(rf"(?<![\d.]){re.escape(m.group(1))}\s?mm", src, re.IGNORECASE):
                add(f"{m.group(1)} mm", f"Measurement '{m.group(0)}' does not appear in any source.", sent)
    return out


def mentions_exam(text: str) -> bool:
    """True if a source itself records an examination — then a draft may report one."""
    return bool(_EXAM_CLAIM.search(text or "") or re.search(r"\bexam(?:ination)? findings?\b", text or "", re.I))


def record_sources(patient: dict) -> str:
    return f"Allergies: {patient.get('allergies') or ''}\nConditions: {patient.get('conditions') or ''}"


def history_sources(patient: dict) -> str:
    return "\n".join(f"{v['date']}: {v['complaint']} {v['diagnosis']} {v['treatment']} {v['notes']}"
                     for v in patient.get("visits", []))


def patient_sources(patient: dict) -> str:
    """Everything on the record a draft may legitimately draw from."""
    return record_sources(patient) + "\n" + history_sources(patient)
