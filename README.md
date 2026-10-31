# zt-whisper-guard

[![tests](https://github.com/Giotor84/zt-whisper-guard/actions/workflows/tests.yml/badge.svg)](https://github.com/Giotor84/zt-whisper-guard/actions/workflows/tests.yml) [![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE) ![python](https://img.shields.io/badge/python-3.10%2B-blue)

**In italiano:** filtri senza dipendenze contro i testi inventati da Whisper su audio telefonico reale (frasi in loop, cifre ripetute, formule da sottotitoli).

**Stop Whisper from inventing text on noisy, real-world audio.**

Whisper is excellent on clean speech and creative on everything else: on phone calls it loops a phrase forty
times, credits non-existent subtitlers, thanks you for watching, and turns line noise into "BAMEES DRUCKY".
zt-whisper-guard is a small, dependency-free set of filters that came out of a real-time call translator running on
telephone audio. Every rule exists because of a failure seen in production.

```bash
pip install "zt-whisper-guard @ git+https://github.com/Giotor84/zt-whisper-guard"          # filters only
pip install "zt-whisper-guard[live] @ git+https://github.com/Giotor84/zt-whisper-guard"    # + numpy utterance buffer
```

## Use with faster-whisper

```python
from faster_whisper import WhisperModel
from zt_whisper_guard import RECOMMENDED_TRANSCRIBE_KWARGS, filter_segments

model = WhisperModel("small", device="cpu", compute_type="int8")
segments, info = model.transcribe(audio, language="en", **RECOMMENDED_TRANSCRIBE_KWARGS)
text = filter_segments(segments)      # "" when what is left still looks invented
```

`filter_segments` also accepts openai-whisper segment dicts. The single functions are available too:
`is_hallucination`, `is_repetitive`, `is_gibberish`, `dedupe_consecutive`, `clean_text`.

## What it catches (and why each rule exists)

| failure seen | rule |
|---|---|
| a 2–6 word **phrase loop** ("did you say that" × 40) that single-word checks miss | n-gram repetition (2–5 words) covering > 40% of the text |
| **digit spam** "3, 4, 3, 4, 3, 4…" (invented phone numbers, amounts) | digits are tokens too |
| the **same short sentence in separate segments** ("Perfect." × 5) — each segment looks fine | cross-segment dedupe, at most 2 in a row |
| subtitle/YouTube boilerplate, `[BLANK_AUDIO]`, `(Music)`, URLs | phrase list + junk regex (English and Italian markers) |
| phonetic noise ("BAMEES DRUCKY") | vowel-less or random all-caps words |
| **false positive**: a dictated number "3, 4, 2, 0, 5" flagged as gibberish | text without letters is never gibberish |
| genuine sentences dropped on quiet, noisy audio with `avg_logprob < -0.75` | gate at -1.0; text filters handle the rest |
| utterances growing to 49 s with continuous speech (and looping) | `UtteranceBuffer` checks the 12 s cap on every chunk |

Decoding settings in `RECOMMENDED_TRANSCRIBE_KWARGS`: VAD on, `condition_on_previous_text=False` (the most
effective switch against loops), temperature 0, compression-ratio threshold 2.0.
Model choice matters too: `base` hallucinated too much on phone audio; `small` int8 on CPU was the floor.

## Live audio

```python
from zt_whisper_guard import UtteranceBuffer

buf = UtteranceBuffer(sample_rate=16_000, max_utterance_secs=12)
for chunk in mic_chunks():            # mono float32, e.g. 0.3 s each
    utterance = buf.add_chunk(chunk)
    if utterance is not None:
        transcribe(utterance)
```

Released chunks are zero-filled, so raw audio does not linger in memory after transcription.

## Known limits

- Rules tuned on phone audio in English and Italian: other languages may need their own phrase lists.
- A genuine but repetitive utterance ("yes, yes, yes, yes") can be dropped: the trade-off is deliberate.
- It cleans transcripts; it does not make recognition more accurate.

## Related projects

- [zt-call-translator](https://github.com/Giotor84/zt-call-translator) — real-time call translation that never leaves the PC
- All projects: [github.com/Giotor84](https://github.com/Giotor84)

## License

MIT — see [LICENSE](LICENSE).
