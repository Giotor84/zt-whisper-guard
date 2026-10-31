"""Energy-based utterance buffer for live audio (mono float32), with a hard length cap and buffer wiping."""
from __future__ import annotations

from typing import Optional

import numpy as np


def rms(data: np.ndarray) -> float:
    if data.size == 0:
        return 0.0
    arr = np.asarray(data, dtype=np.float32)
    return float(np.sqrt(np.mean(arr * arr)))


class UtteranceBuffer:
    """Collects audio chunks and releases one utterance when enough silence follows the speech
    or when the hard cap is reached.

    The cap is checked on EVERY chunk, not only during silence: with continuous speech (pauses shorter than
    ``silence_chunks``) the buffer otherwise grows without limit (a 49 s utterance against a 30 s cap was
    observed) and long contexts make Whisper loop. 12 s works well for phone calls.

    Writeable chunks are zero-filled before the buffer is cleared so raw audio does not linger in memory,
    while read-only caller buffers (e.g. ``np.frombuffer(pcm_bytes, ...)``) are handled safely without ValueError.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        energy_threshold: float = 0.001,
        silence_chunks: int = 3,
        min_speech_chunks: int = 4,
        max_utterance_secs: float = 12.0,
    ) -> None:
        self.energy_threshold = energy_threshold
        self.silence_chunks = silence_chunks
        self.min_speech_chunks = min_speech_chunks
        self.max_samples = int(sample_rate * max_utterance_secs)
        self._chunks: list[np.ndarray] = []
        self._speech_chunks = 0
        self._silence = 0

    def add_chunk(self, chunk: np.ndarray) -> Optional[np.ndarray]:
        """Add a chunk; return the utterance to transcribe when it is ready, else None."""
        if chunk.size == 0:
            return None
        if rms(chunk) > self.energy_threshold:
            self._chunks.append(chunk)
            self._speech_chunks += 1
            self._silence = 0
        else:
            self._silence += 1
            if self._chunks:
                self._chunks.append(chunk)
        if not self._chunks:
            return None

        samples = sum(c.shape[0] for c in self._chunks)
        ready = (
            (self._silence >= self.silence_chunks and self._speech_chunks >= self.min_speech_chunks)
            or samples >= self.max_samples
        )
        if not ready:
            return None

        audio = np.concatenate(self._chunks).astype(np.float32)
        speech_ok = self._speech_chunks >= self.min_speech_chunks or samples >= self.max_samples
        for c in self._chunks:
            if c.flags.writeable:
                c.fill(0)
        self._chunks = []
        self._speech_chunks = 0
        self._silence = 0
        if not speech_ok or rms(audio) < self.energy_threshold * 0.5:
            return None
        return audio
