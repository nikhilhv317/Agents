"""
Safety check — every draft against the patient RECORD, deterministically.

Allergies and conditions come from the chart, never from the model's memory.
Used by the visit pipeline (before each review pause), by the Clinical Assistant
and Report agents on their own drafts, and by evals/.
"""
from __future__ import annotations

import re
from typing import Optional

# Deterministic check of every draft against the RECORD before a human sees it.
# Found in testing: qwen3:1.7b turned a recorded "Latex" allergy into
# "latex-free" — the opposite — in both the SOAP note and the report. The
# review pause only helps if the reviewer notices, so the pipeline points at it.
# "allerg(y|ies)\b", not "allerg": "no allergic reactions during treatment" is
# not a claim that the patient has no allergies (seen in the live eval baseline).
_NO_ALLERGY = re.compile(r"\b(no (known |recorded )?(drug )?allerg(y|ies)\b|nkda|nka\b|allerg(y|ies)[^.]{0,20}\bnone\b)",
                         re.IGNORECASE)

# Recorded allergy -> what a draft must not prescribe or use. A mention check
# alone passed "Prescribed ibuprofen 400 mg" for an ibuprofen-allergic patient.
_NSAIDS = ["ibuprofen", "diclofenac", "naproxen", "aspirin", "ketorolac", "aceclofenac",
           "mefenamic acid", "ketoprofen", "nimesulide"]
_PENICILLINS = ["penicillin", "amoxicillin", "amoxycillin", "ampicillin", "co-amoxiclav", "amoxiclav",
                "augmentin", "flucloxacillin", "cloxacillin", "phenoxymethylpenicillin", "piperacillin"]
_ALLERGY_CLASSES = {"penicillin": _PENICILLINS, "penicillins": _PENICILLINS, "amoxicillin": _PENICILLINS,
                    "ibuprofen": _NSAIDS, "nsaid": _NSAIDS, "nsaids": _NSAIDS, "aspirin": _NSAIDS,
                    "diclofenac": _NSAIDS, "latex": ["latex gloves", "latex glove", "latex dam"]}
_PRESCRIBE = re.compile(r"\b(prescrib\w*|rx|administer\w*|give|given|giving|start(?:ed)? on|dispens\w*|"
                        r"take|taking|course of|commenc\w*|use|used|using|apply|applied)\b", re.IGNORECASE)
_AVOID = re.compile(r"\b(avoid\w*|not|no|never|non|contraindicat\w*|allerg\w*|instead of|rather than|"
                    r"alternative to|withh[oe]ld\w*|substitut\w*|free)\b", re.IGNORECASE)


def _split_terms(text: str) -> list[str]:
    return [t.strip() for t in re.split(r"[,;/]|\band\b", text or "") if len(t.strip()) >= 3]


def _prescribed(text: str, allergy: str) -> Optional[str]:
    """The allergen (or a drug in its class) being prescribed or used, if any.
    Judged on the clause leading up to the drug, so "Despite the penicillin
    allergy, prescribed amoxicillin" is caught but "do not prescribe ibuprofen"
    and "paracetamol instead of ibuprofen" are not."""
    for sent in re.split(r"(?<=[.;!?])\s+|\n+", text):
        for drug in _ALLERGY_CLASSES.get(allergy.lower(), [allergy.lower()]):
            for m in re.finditer(rf"(?<!\w){re.escape(drug)}(?!\w)", sent, re.IGNORECASE):
                before = re.split(r"[,:]", sent[:m.start()])[-1]
                after = sent[m.end():m.end() + 40]
                if _AVOID.search(before):
                    continue
                if _PRESCRIBE.search(before) or re.match(
                        r"(?:\s+\S+){0,3}?\s+(prescribed|given|administered|dispensed|started)\b", after, re.I):
                    return drug
    return None


def safety_check(patient: dict, draft_text: str) -> list[str]:
    warnings, text = [], draft_text or ""
    allergies = _split_terms(patient.get("allergies"))
    if allergies and _NO_ALLERGY.search(text):
        warnings.append(f"Draft says no allergies, but the record lists: {patient['allergies']}.")
    for a in allergies:
        drug = _prescribed(text, a)
        if drug:
            warnings.append(f"Draft prescribes '{drug}', but the record lists an allergy to '{a}'.")
        t = re.escape(a)
        # "use latex-free gloves" is CORRECT for a latex allergy — only flag the
        # patient being described as free of / negative for the allergen. (The
        # model's "latex-free" came from a history note saying "Latex-free gloves".)
        negated = re.search(rf"\b{t}[- ]?(free|negative)\b(?!\s+(gloves?|products?|materials?|equipment|"
                            rf"environment|instruments?|room|setting|supplies|alternatives?))"
                            rf"|\b(no|not allergic to|denies|without) {t}\b(?!\s+(gloves?|products?|exposure))",
                            text, re.IGNORECASE)
        if negated:
            warnings.append(f"Recorded allergy '{a}' appears NEGATED in the draft: \"{negated.group(0)}\".")
        elif not _mentions(text, a):
            warnings.append(f"Recorded allergy '{a}' is not mentioned in the draft.")
    for c in _split_terms(patient.get("conditions")):
        if not _mentions_condition(text, c):
            warnings.append(f"Recorded condition '{c}' is not mentioned in the draft.")
    return warnings


def _mentions(text: str, term: str) -> bool:
    # (?<!\w)/(?!\w), not \b: a term ending in ")" like "Pregnant (2nd trimester)"
    # never matched \b...\b, so it was always reported missing (found by evals/).
    return bool(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, re.IGNORECASE))


_GENERIC_WORDS = {"type", "chronic", "history", "known", "severe", "moderate", "condition"}


def _mentions_condition(text: str, condition: str) -> bool:
    """Exact text, or every key word by its 6-letter stem, so "diabetic" counts for
    "Type 2 diabetes" and "pregnant in her second trimester" for "Pregnant (2nd
    trimester)". Exact-phrase-only flagged correct paraphrases (eval baseline)."""
    if _mentions(text, condition):
        return True
    core = re.sub(r"\([^)]*\)", " ", condition)
    stems = [w.lower()[:6] for w in re.findall(r"[A-Za-z]{5,}", core) if w.lower() not in _GENERIC_WORDS]
    return bool(stems) and all(re.search(rf"\b{re.escape(s)}", text, re.IGNORECASE) for s in stems)


def missing_record_items(patient: dict, text: str) -> tuple[list[str], list[str]]:
    """Recorded allergies / conditions the text doesn't mention at all."""
    allergies = [a for a in _split_terms(patient.get("allergies")) if not _mentions(text, a)]
    conditions = [c for c in _split_terms(patient.get("conditions")) if not _mentions_condition(text, c)]
    return allergies, conditions


def record_sentence(patient: dict, text: str) -> str:
    """A sentence restating, from the chart, whatever the draft left out — so a
    safety-critical fact never depends on the model remembering it. Negations are
    NOT repaired here: a draft that contradicts the record stays flagged for review."""
    allergies, conditions = missing_record_items(patient, text)
    parts = []
    if allergies:
        parts.append(f"Recorded allergies: {', '.join(allergies)}.")
    if conditions:
        parts.append(f"Recorded conditions: {', '.join(conditions)}.")
    return (" ".join(parts)[:-1] + " [R].") if parts else ""
