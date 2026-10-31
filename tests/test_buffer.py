import pytest
import numpy as np

from zt_whisper_guard import UtteranceBuffer

CHUNK = 4800  # 0.3 s at 16 kHz


def _speech():
    return np.full(CHUNK, 0.1, dtype=np.float32)


def _silence():
    return np.zeros(CHUNK, dtype=np.float32)


def test_flush_after_silence():
    buf = UtteranceBuffer()
    out = None
    for _ in range(4):
        assert buf.add_chunk(_speech()) is None
    for _ in range(3):
        out = buf.add_chunk(_silence())
    assert out is not None and out.shape[0] == 7 * CHUNK


def test_hard_cap_during_continuous_speech():
    buf = UtteranceBuffer(max_utterance_secs=1.2)  # 4 chunks
    outs = [buf.add_chunk(_speech()) for _ in range(4)]
    assert outs[:3] == [None, None, None] and outs[3].shape[0] == 4 * CHUNK


def test_released_chunks_are_wiped():
    buf = UtteranceBuffer(max_utterance_secs=0.6)
    a, b = _speech(), _speech()
    buf.add_chunk(a)
    out = buf.add_chunk(b)
    assert out is not None and float(np.abs(out).sum()) > 0
    assert not a.any() and not b.any()


def test_silence_only_returns_nothing():
    buf = UtteranceBuffer()
    assert all(buf.add_chunk(_silence()) is None for _ in range(10))


def test_readonly_frombuffer_chunk_does_not_crash():
    pcm = (np.ones(1600, dtype=np.float32) * 0.05).tobytes()
    ro_chunk = np.frombuffer(pcm, dtype=np.float32)
    assert not ro_chunk.flags.writeable
    buf = UtteranceBuffer(sample_rate=16_000, silence_chunks=1, min_speech_chunks=1)
    assert buf.add_chunk(ro_chunk) is None
    out = buf.add_chunk(np.zeros(1600, dtype=np.float32))
    assert out is not None and out.shape[0] == 3200
    assert float(ro_chunk[0]) == pytest.approx(0.05)


def test_single_noise_spike_plus_silence_does_not_satisfy_min_speech_chunks():
    buf = UtteranceBuffer(sample_rate=16_000, silence_chunks=3, min_speech_chunks=4, max_utterance_secs=12.0)
    assert buf.add_chunk(np.ones(1600, dtype=np.float32) * 0.05) is None
    for _ in range(4):
        assert buf.add_chunk(np.zeros(1600, dtype=np.float32)) is None


def test_empty_chunk_rms_is_zero():
    from zt_whisper_guard.buffer import rms
    assert rms(np.array([], dtype=np.float32)) == 0.0
