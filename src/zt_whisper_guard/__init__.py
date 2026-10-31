"""zt-whisper-guard: stop Whisper from inventing text on noisy, real-world audio."""
from .filters import (
    HALLUCINATION_PHRASES,
    clean_text,
    dedupe_consecutive,
    is_gibberish,
    is_hallucination,
    is_repetitive,
)
from .segments import RECOMMENDED_TRANSCRIBE_KWARGS, filter_segments

__all__ = [
    "HALLUCINATION_PHRASES", "RECOMMENDED_TRANSCRIBE_KWARGS", "UtteranceBuffer", "clean_text",
    "dedupe_consecutive", "filter_segments", "is_gibberish", "is_hallucination", "is_repetitive",
]
__version__ = "0.1.0"


def __getattr__(name: str):  # numpy is needed only by the live buffer
    if name == "UtteranceBuffer":
        from .buffer import UtteranceBuffer
        return UtteranceBuffer
    raise AttributeError(name)
