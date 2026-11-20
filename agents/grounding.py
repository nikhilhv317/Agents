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
from datetime import date

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
    "pericoronitis": r"pericoronitis",
    "pulpitis": r"pulpitis",
    "impaction": r"impact(?:ed|ion)",
    "necrosis": r"necro(?:sis|tic)",
    "attrition / wear": r"attrition|abrasion|erosion",
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


# "and/but" + a verb starts a new clause: in "may have caries and has visible
# swelling" the "may" must not cover the swelling (eval case grounding-hedge-leaks).
_CLAUSE_BREAK = re.compile(r"[,:]|\b(?:and|but|with)\s+(?=(?:has|have|is|are|was|were|shows?|showed|"
                           r"presents?|presented|reveals?|revealed|demonstrates?)\b)", re.IGNORECASE)
# a finding word used as advice, not as an observation: "plaque control", "caries prevention"
_ADVICE_AFTER = re.compile(r"\s+(?:control|prevention|management|instructions?|education|reduction|"
                           r"risk|precautions?)\b", re.IGNORECASE)


def _clause_before(sentence: str, start: int) -> str:
    # the same clause, up to the previous comma/colon or "and"+verb
    return _CLAUSE_BREAK.split(sentence[:start])[-1]


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

    for sent in _sentences(strip_tags(draft)):
        if not exam_findings_given and _EXAM_CLAIM.search(sent):
            add("examination claimed",
                "The draft describes an examination, but no exam findings were entered.", sent)
        for name, rx in _FINDING_RE.items():
            for m in rx.finditer(sent):
                if (_hedged(sent, m.start()) or _ADVICE_AFTER.match(sent, m.end())
                        or (allow_negated and _negated(sent, m.start()))):
                    continue
                if rx.search(src):
                    # in the sources — but stated as CURRENT when every mention is a dated past record?
                    if _CURRENT.search(sent) and _only_in_past_records(rx, src):
                        add(name, f"'{m.group(0)}' is stated as current, but the sources only have it "
                                  "in a past, dated record.", sent)
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


# ---- timing: a past finding stated as current ----------------------------------
# Word matching alone passed "Deepa has an active abscess right now" because
# "abscess" IS in her record — from a 2025 visit. If the sentence asserts the
# present and every mention in the sources sits next to a past date, flag it.
_CURRENT = re.compile(r"\b(currently|current|active|right now|at present|presently|ongoing|still|today)\b",
                      re.IGNORECASE)
_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def _only_in_past_records(rx, src: str) -> bool:
    today = date.today().isoformat()
    hits = list(rx.finditer(src))
    for m in hits:
        line_start = src.rfind("\n", 0, m.start()) + 1
        dates = _DATE.findall(src[max(line_start, m.start() - 160):m.start()])
        if not dates or "-".join(dates[-1]) >= today:
            return False          # an undated (record/today) or current mention exists
    return bool(hits)


# ---- citations ------------------------------------------------------------------
# Drafts cite their sources inline: "Tender to percussion on #46 [F]." Sources are
# a dict of id -> text: R record, C complaint, F findings, I imaging, V1..Vn past
# visits (dated). Checking a claim against the source it CITES (not the whole pile)
# catches a finding borrowed from the wrong place, and `allowed` stops the Objective
# section citing a past visit as today's finding.
_TAG = re.compile(r"\[\s*([A-Z]\d*(?:\s*[,;/]\s*[A-Z]\d*)*)\s*\]")


def strip_tags(text: str) -> str:
    return re.sub(r"\s+([.;,])", r"\1", _TAG.sub("", text or ""))


def cited_ids(sentence: str) -> list[str]:
    return [i.strip() for g in _TAG.findall(sentence) for i in re.split(r"[,;/]", g) if i.strip()]


def has_citations(text: str) -> bool:
    return bool(cited_ids(text or ""))


def no_citations_issue(text: str) -> list[dict]:
    """Note-level: a model draft that cites nothing at all can't be verified."""
    if (text or "").strip() and not has_citations(text):
        return [{"claim": "no citations", "snippet": text[:160],
                 "detail": "The draft doesn't cite any source, so its findings can't be verified."}]
    return []


def check_citations(draft: str, sources: dict, allowed: set | None = None,
                    require: bool = True) -> list[dict]:
    """Per-sentence citation check for one piece of text (a section or a summary).
    require: flag findings with no tag. The note-level "no citations at all" check
    is no_citations_issue(), run once on the whole draft by the caller."""
    out, seen = [], set()

    def add(claim, detail, snippet):
        if claim not in seen:
            seen.add(claim)
            out.append({"claim": claim, "detail": detail, "snippet": snippet[:160]})

    sents = _sentences(draft)
    for sent in sents:
        ids, bare = cited_ids(sent), strip_tags(sent)
        terms = [(n, m) for n, rx in _FINDING_RE.items() for m in [rx.search(bare)]
                 if m and not _hedged(bare, m.start()) and not _negated(bare, m.start())
                 and not _ADVICE_AFTER.match(bare, m.end())]
        for i in ids:
            if i not in sources:
                add(f"unknown source [{i}]", f"[{i}] is not one of the sources given.", sent)
        if not terms:
            continue
        if not ids:
            if require:
                add(f"{terms[0][0]} (uncited)", "A finding with no source tag.", sent)
            continue
        if allowed is not None:
            for i in ids:
                if i in sources and i not in allowed:
                    add(f"{terms[0][0]} (cites [{i}])",
                        f"This section must describe today, but it cites [{i}] ({sources[i][:60]}).", sent)
        cited = "\n".join(sources[i] for i in ids if i in sources)   # disallowed cites are flagged above
        for name, _ in terms:
            if not _FINDING_RE[name].search(cited):
                add(f"{name} (not in cited source)",
                    f"'{name}' does not appear in the cited source(s) {', '.join('[' + i + ']' for i in ids)}.", sent)
    return out


def soap_sources(patient: dict, complaint: str, findings: str, findings_from_exam: bool) -> dict:
    # C and F are always present (the model cites [F] even when nothing was entered;
    # "none entered" turns that into a checkable "not in cited source" instead of an
    # unknown tag).
    src = {"R": record_sources(patient).replace("\n", "; "),
           "C": f"Today's chief complaint: {(complaint or '').strip() or 'none entered'}"}
    if (findings or "").strip():
        src["F"] = (f"Today's findings ({'entered by the clinician' if findings_from_exam else 'from imaging, not an examination'}): "
                    f"{findings}")
    else:
        src["F"] = "Today's examination findings: none entered"
    for n, v in enumerate(patient.get("visits", []), 1):
        src[f"V{n}"] = f"Past visit {v['date']}: {v['complaint']}; {v['diagnosis']}; {v['treatment']}; {v['notes']}"
    return src


def report_sources(patient: dict, imaging: str, clinical: str) -> dict:
    src = {"R": record_sources(patient).replace("\n", "; ")}
    if (imaging or "").strip():
        src["I"] = f"Imaging today: {imaging}"
    if (clinical or "").strip():
        src["F"] = f"Clinician findings today: {clinical}"
    return src


def format_sources(sources: dict) -> str:
    return "\n".join(f"[{k}] {v}" for k, v in sources.items())


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
