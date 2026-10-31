from zt_whisper_guard import (
    clean_text,
    dedupe_consecutive,
    filter_segments,
    is_gibberish,
    is_hallucination,
    is_repetitive,
)


def test_phrase_loop_is_repetitive():
    assert is_repetitive("did you say that " * 10)
    assert not is_repetitive("I can pay the first instalment next Monday, then the rest in March.")


def test_digit_spam_is_repetitive():
    assert is_repetitive("3, 4, 3, 4, 3, 4, 3, 4, 3, 4")


def test_dictated_number_is_not_gibberish():
    assert not is_gibberish("3, 4, 2, 0, 5, 9, 6, 5, 2, 6")
    assert not is_hallucination("3 4 2 0 5 9 6 5 2 6 is my number")


def test_gibberish_detected():
    assert is_gibberish("BAMEES DRUCKY XRTPL")


def test_subtitle_boilerplate():
    assert is_hallucination("Thanks for watching!")
    assert is_hallucination("Sottotitoli a cura di QTSS")
    assert is_hallucination("[BLANK_AUDIO]")
    assert is_hallucination("...")


def test_normal_sentence_passes():
    assert not is_hallucination("Good morning, I am calling about the payment plan we discussed.")


def test_dedupe_consecutive_keeps_two():
    assert dedupe_consecutive(["Perfect."] * 5 + ["Thank you."]) == ["Perfect.", "Perfect.", "Thank you."]


def test_clean_text():
    assert clean_text("♪ hello [music] (Applause)  world ") == "hello world"


class _Seg:
    def __init__(self, text, no_speech_prob=0.0, avg_logprob=-0.2):
        self.text, self.no_speech_prob, self.avg_logprob = text, no_speech_prob, avg_logprob


def test_filter_segments_objects_and_dicts():
    segs = [
        _Seg("Hello, this is Anna."),
        _Seg("Thanks for watching!"),
        _Seg("background hum", no_speech_prob=0.9),
        {"text": "I will pay on Friday.", "no_speech_prob": 0.1, "avg_logprob": -0.4},
        {"text": "mumble", "no_speech_prob": 0.1, "avg_logprob": -1.6},
    ]
    assert filter_segments(segs) == "Hello, this is Anna. I will pay on Friday."
