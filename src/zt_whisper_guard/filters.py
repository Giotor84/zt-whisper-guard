"""Text filters for Whisper output: phrase loops, gibberish, subtitle boilerplate, cross-segment repeats.

Each rule was added after a real failure on noisy phone audio (see README); thresholds are the ones that held
up there. All functions are pure and dependency-free.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Optional

# Phrases Whisper invents from its training data (YouTube videos, subtitle credits).
HALLUCINATION_PHRASES: tuple[str, ...] = (
    "thank you for watching",
    "thanks for watching",
    "thanks for listening",
    "please subscribe",
    "like and subscribe",
    "see you in the next video",
    "see you next time",
    "stay tuned",
    "subtitles by",
    "subtitle by",
    "sottotitol",
    "translated by",
    "amara.org",
    "copyright",
    "all rights reserved",
)

_JUNK_RE = re.compile(
    r"^\.{2,}$"
    r"|^[\s\W]+$"
    r"|♪+|\U0001f3b5+"
    r"|\[\s*(?:BLANK_AUDIO|SILENCE|RUMORE|MUSICA?|NOISE)\s*\]"
    r"|\(\s*(?:Musica|Music|Applausi|Applause|Rumore|Noise|Silenzio)\s*\)"
    r"|\bsottotitol[io]\b|\bsubtitl|\bunder[- ]?titl"
    r"|\b(?:www\.|http)",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[a-zA-Z0-9àèéìòùáéíóúâêîôû']+")
_ALLOWED_UPPER = ("OK", "NO", "YES", "ID")


def is_repetitive(text: str) -> bool:
    """True for a single word or a 2-5 word phrase looping through the text.

    Covers the most common real failure ("did you say that" repeated 40 times), not only single-word spam,
    and digit spam such as "3, 4, 3, 4, 3, 4" (invented phone numbers or amounts).
    """
    words = _WORD_RE.findall(text.lower())
    if len(words) < 4:
        return False
    _, count = Counter(words).most_common(1)[0]
    if count / len(words) > 0.45:
        return True
    for n in (2, 3, 4, 5):
        if len(words) < n * 3:
            continue
        grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
        _, gcount = Counter(grams).most_common(1)[0]
        if gcount >= 3 and (gcount * n) / len(words) > 0.4:
            return True
    return False


def is_gibberish(text: str) -> bool:
    """True when most words have no vowels or are random all-caps tokens (e.g. "BAMEES DRUCKY").

    Text with no letters at all (a dictated number "3, 4, 2, 0, 5") is NOT gibberish: digits are legitimate.
    """
    words = re.findall(r"[A-Za-z']+", text)
    if not words:
        return False
    weird = 0
    for w in words:
        core = w.strip("'")
        if len(core) <= 2:
            continue
        if core.isupper() and core not in _ALLOWED_UPPER:
            weird += 1
        elif len(core) > 3 and not re.search(r"[aeiouyAEIOUYàèéìòù]", core):
            weird += 1
    return weird >= max(2, len(words) // 2)


def is_hallucination(text: str, phrases: Iterable[str] = HALLUCINATION_PHRASES) -> bool:
    """True for empty text, junk markers, known invented phrases, loops or gibberish."""
    if not text or _JUNK_RE.search(text):
        return True
    tl = text.lower().strip()
    if any(p in tl for p in phrases):
        return True
    return is_repetitive(text) or is_gibberish(text)


def dedupe_consecutive(parts: list[str], max_repeat: int = 2) -> list[str]:
    """Keep at most ``max_repeat`` identical consecutive segments.

    Whisper sometimes repeats the same short sentence across separate segments ("Perfect." five times):
    each segment looks legitimate, so only a cross-segment check catches it.
    """
    out: list[str] = []
    prev: Optional[str] = None
    run = 0
    for p in parts:
        norm = re.sub(r"\s+", " ", p.strip().lower())
        if norm and norm == prev:
            run += 1
            if run <= max_repeat:
                out.append(p)
        else:
            run = 1
            prev = norm
            out.append(p)
    return out


def clean_text(text: str) -> str:
    """Remove music notes, [bracketed] tags and (noise) markers; normalise spaces."""
    text = re.sub(r"♪+|\U0001f3b5+", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(
        r"\(\s*(?:Musica|Music|Applausi|Applause|Rumore|Noise|Silenzio)\s*\)",
        "", text, flags=re.IGNORECASE,
    )
    return re.sub(r"\s{2,}", " ", text).strip()
