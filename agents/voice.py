"""
Voice-to-SOAP (roadmap item 4) — local speech-to-text, then the normal SOAP agent.

A clinician dictates ("Patient complains of pain lower right molar. On exam,
tender to percussion on 46, no swelling."). faster-whisper transcribes it on this
machine — audio never leaves the clinic — and the transcript goes to the
Clinical Assistant as the clinician's own dictated findings, so the note is
drafted, grounded, safety-checked and cited exactly like a typed one.

Model: Whisper "small" (WHISPER_MODEL to change), on the GPU in float16 when
CTranslate2 can use it, otherwise on the CPU in int8. Loaded once, kept warm.
"""
from __future__ import annotations

import os
import re

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
# Biases Whisper toward dental terms: without it "caries" came out as "carries".
DENTAL_PROMPT = ("Dental clinical dictation. Caries, gingivitis, periodontitis, pulpitis, pericoronitis, "
                 "calculus, plaque, abscess, percussion, mobility, pocketing, tooth #46, #38, mm.")
_model = None
_device = None


class VoiceUnavailable(Exception):
    pass


def _load(force_cpu: bool = False):
    global _model, _device
    if _model is not None and not (force_cpu and _device != "cpu"):
        return _model
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise VoiceUnavailable("install faster-whisper") from e
    errors = []
    wanted = os.getenv("WHISPER_DEVICE", "cpu" if force_cpu else "auto")
    options = [("cpu", "int8")] if wanted == "cpu" else [("cuda", "float16"), ("cpu", "int8")]
    for device, compute in options:
        try:
            _model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute)
            _device = device
            return _model
        except Exception as e:   # no CUDA runtime for CTranslate2 -> try the CPU
            errors.append(f"{device}: {e}")
    raise VoiceUnavailable("could not load Whisper: " + "; ".join(errors))


def _decode(path: str, rate: int = 16000):
    """Any audio/video file (wav, webm/opus from the browser, m4a ...) -> mono float32
    at 16 kHz. Done here rather than by faster-whisper, whose decoder passes an
    argument newer PyAV (>=15) no longer accepts."""
    import av
    import numpy as np
    chunks = []
    with av.open(path) as container:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
        for frame in container.decode(audio=0):
            chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(frame)]
        chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(None)]
    if not chunks:
        raise ValueError("the recording contains no audio")
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def transcribe(audio_path: str) -> dict:
    """Transcribe one recording. Returns {text, language, seconds, device}."""
    audio = _decode(audio_path)

    def run(model):
        segments, info = model.transcribe(audio, language="en", beam_size=5, vad_filter=True,
                                          initial_prompt=DENTAL_PROMPT)
        return " ".join(s.text.strip() for s in segments).strip(), info

    try:
        text, info = run(_load())
    except RuntimeError as e:
        # CTranslate2's GPU build needs CUDA 12 cuBLAS/cuDNN DLLs; without them the
        # model LOADS on cuda but fails on first use. Fall back to the CPU for good.
        if _device != "cuda" or not any(k in str(e).lower() for k in ("cublas", "cudnn", "cuda")):
            raise
        text, info = run(_load(force_cpu=True))
    return {"text": fix_dental_terms(text), "language": info.language,
            "seconds": round(info.duration, 1), "device": _device}


# Homophones Whisper gets wrong in dental dictation, corrected only in a dental
# context (a surface/severity word before, or a tooth number after) so the plain
# verb "carries" elsewhere is left alone.
_CARIES = re.compile(r"\b(?:(?<=deep )|(?<=early )|(?<=advanced )|(?<=distal )|(?<=mesial )|(?<=occlusal )|"
                     r"(?<=buccal )|(?<=lingual )|(?<=recurrent )|(?<=interproximal )|(?<=root )|(?<=active ))"
                     r"carries\b|\bcarries(?=\s+(?:on|at|in|of)?\s*(?:tooth\s*)?#?\d)", re.IGNORECASE)


def fix_dental_terms(text: str) -> str:
    return _CARIES.sub("caries", text or "")


# Dictation usually starts with what the patient says, then the exam. Split there
# so the complaint and the clinician's findings go to the right SOAP sources.
_EXAM_START = re.compile(r"\b(on (?:clinical )?exam(?:ination)?|exam(?:ination)? (?:shows|reveals|finds)|"
                         r"findings?:|objectively|clinically)\b", re.IGNORECASE)


def split_dictation(text: str) -> tuple[str, str]:
    """(complaint, findings). Without an exam marker everything is treated as findings."""
    m = _EXAM_START.search(text or "")
    if not m:
        return "", (text or "").strip()
    return text[:m.start()].strip(" .,;"), text[m.start():].strip()


def voice_soap(audio_path: str, patient_id: int) -> dict:
    """Transcribe a dictation and draft the SOAP note from it."""
    from agents import clinical_assistant
    t = transcribe(audio_path)
    if not t["text"]:
        raise ValueError("no speech was recognised in the recording")
    complaint, findings = split_dictation(t["text"])
    result = clinical_assistant.draft(patient_id, complaint, findings, findings_from_exam=True)
    result["transcript"] = t
    result["complaint_from_dictation"] = complaint
    result["findings_from_dictation"] = findings
    return result
