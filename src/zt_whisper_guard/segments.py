"""Segment-level gate for faster-whisper / openai-whisper output, plus decoding settings that reduce loops."""
from __future__ import annotations

from typing import Any, Iterable

from .filters import clean_text, dedupe_consecutive, is_hallucination

NO_SPEECH_THRESHOLD = 0.50
# -0.75 dropped genuine sentences on quiet, noisy phone audio with the "small" model; -1.0 lets real speech
# through and leaves hallucinations to the text filters.
AVG_LOGPROB_MIN = -1.0

# faster-whisper ``WhisperModel.transcribe`` keyword arguments that held up on telephone audio.
# condition_on_previous_text=False is the single most effective switch against runaway loops.
RECOMMENDED_TRANSCRIBE_KWARGS: dict[str, Any] = {
    "beam_size": 3,
    "vad_filter": True,
    "vad_parameters": {"min_silence_duration_ms": 350, "speech_pad_ms": 80, "threshold": 0.45},
    "temperature": 0.0,
    "condition_on_previous_text": False,
    "compression_ratio_threshold": 2.0,
    "log_prob_threshold": AVG_LOGPROB_MIN,
    "no_speech_threshold": NO_SPEECH_THRESHOLD,
}


def _get(seg: Any, name: str, default: float) -> float:
    if isinstance(seg, dict):
        return float(seg.get(name, default))
    return float(getattr(seg, name, default))


def filter_segments(
    segments: Iterable[Any],
    no_speech_threshold: float = NO_SPEECH_THRESHOLD,
    avg_logprob_min: float = AVG_LOGPROB_MIN,
    max_repeat: int = 2,
) -> str:
    """Join the trustworthy segments into one text; return "" if the result still looks invented.

    Works with faster-whisper ``Segment`` objects and with openai-whisper segment dicts.
    """
    parts: list[str] = []
    for seg in segments:
        if _get(seg, "no_speech_prob", 0.0) > no_speech_threshold:
            continue
        if _get(seg, "avg_logprob", 0.0) < avg_logprob_min:
            continue
        text = seg["text"] if isinstance(seg, dict) else seg.text
        part = clean_text(text)
        if part and not is_hallucination(part):
            parts.append(part)
    text = " ".join(dedupe_consecutive(parts, max_repeat)).strip()
    return "" if is_hallucination(text) else text
