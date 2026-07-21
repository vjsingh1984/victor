"""Unit tests for the intra-turn streaming repetition detector (P5).

A single degenerate generation ("Let me check the state. " looped hundreds of
times) previously ran until max_tokens: victor only detected repetition BETWEEN
turns. The detector watches the running text of ONE generation.
"""

from __future__ import annotations

from victor.agent.streaming.repetition_guard import IntraTurnRepetitionDetector


def make_detector(**overrides) -> IntraTurnRepetitionDetector:
    defaults = {
        "window_chars": 4000,
        "min_segment_chars": 24,
        "max_repeats": 6,
        "check_every_chars": 512,
    }
    defaults.update(overrides)
    return IntraTurnRepetitionDetector(**defaults)


def feed_all(detector: IntraTurnRepetitionDetector, text: str, chunk_size: int = 37):
    """Feed text in odd-sized chunks (mirrors arbitrary provider chunking)."""
    for i in range(0, len(text), chunk_size):
        repeated = detector.feed(text[i : i + chunk_size])
        if repeated is not None:
            return repeated
    return None


class TestTrigger:
    def test_looping_sentence_triggers(self):
        # The live failure shape: alternating short sentences repeated forever.
        loop = "Let me check the state. Let me check the remote tracking state. "
        repeated = feed_all(make_detector(), loop * 40)
        assert repeated is not None
        assert "let me check the" in repeated.lower()

    def test_single_repeated_line_triggers(self):
        line = "Checking the remote branch tracking status now...\n"
        assert feed_all(make_detector(), line * 30) is not None

    def test_varied_prose_does_not_trigger(self):
        text = " ".join(
            f"Sentence number {i} discusses a different topic entirely, item {i * 7}."
            for i in range(120)
        )
        assert feed_all(make_detector(), text) is None

    def test_repeated_short_code_lines_do_not_trigger(self):
        # Closing braces / short boilerplate lines repeat legitimately in code.
        code = ("    }\n" * 50) + ("    return x\n" * 10) + ("})\n" * 40)
        assert feed_all(make_detector(), code) is None

    def test_below_repeat_threshold_does_not_trigger(self):
        sentence = "This exact sentence appears a handful of times in the output. "
        varied = " ".join(f"Filler sentence {i} with distinct content here." for i in range(60))
        assert feed_all(make_detector(), sentence * 4 + varied) is None

    def test_long_period_block_loop_triggers(self):
        # A multi-sentence paragraph (~300 chars) looped — caught by the block rule
        # even though each inner sentence stays under max_repeats per window.
        para = (
            "First we examine the configuration files for the project setup. "
            "Then we validate every dependency against the recorded lockfile. "
            "Next we compile the sources and collect all emitted diagnostics. "
            "Finally we execute the verification suite and gather the results. "
        )
        assert feed_all(make_detector(), para * 12) is not None


class TestWindowing:
    def test_repeats_far_apart_outside_window_do_not_trigger(self):
        sentence = "A rare marker sentence that shows up occasionally in output. "
        filler = " ".join(f"Unique padding sentence number {i} for spacing." for i in range(30))
        text = (sentence + filler + " ") * 8  # repeats spaced ~1.5k chars apart, window 4k
        detector = make_detector(window_chars=1000)
        assert feed_all(detector, text) is None


class TestTruncation:
    def test_truncation_point_keeps_first_occurrence(self):
        detector = make_detector()
        prefix = "Here is a legitimate answer about the topic at hand. "
        loop = "Let me check the remote tracking state of the branch. "
        text = prefix + loop * 20
        repeated = feed_all(detector, text)
        assert repeated is not None
        cut = detector.truncation_point(text)
        truncated = text[:cut]
        assert prefix.strip() in truncated
        # keeps at most a couple of instances of the loop, not twenty
        assert truncated.lower().count("remote tracking state") <= 2

    def test_truncation_point_defaults_to_full_length_when_not_found(self):
        detector = make_detector()
        assert detector.truncation_point("short unrelated text") == len("short unrelated text")
