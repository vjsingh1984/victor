"""Intra-turn streaming repetition guard (Tool Reliability P5).

Detects a DEGENERATE GENERATION — the model looping the same sentence/paragraph
within one response — so the stream loop can stop it instead of burning tokens
until the provider's max_tokens cap. Victor's existing repetition detection
(TurnEvaluationController / SpinDetector) compares BETWEEN turns and cannot see
an intra-turn loop.

Design notes:
- Fed incrementally from the single stream choke point
  (``chat_stream_helpers._stream_provider_response_inner``); checks run only
  every ``check_every_chars`` of new content, over a bounded tail window —
  O(window) per check, no cost on healthy short responses.
- Two complementary rules:
  1. segment rule — any normalized sentence/line of at least
     ``min_segment_chars`` occurring ``max_repeats``+ times in the window
     (catches short-period loops like "Let me check the state.");
  2. block rule — the trailing 256-char block occurring 3+ times in the
     window (catches long-period multi-sentence loops whose individual
     sentences stay under ``max_repeats``).
- False-positive guards: the segment length floor excludes legitimately
  repetitive short lines (code braces, list bullets), and ``max_repeats``
  is high enough that stylistic echoes don't trip it.
- The dormant ``StreamingConfidenceMonitor`` (streaming/confidence_monitor.py)
  is prior art for mid-stream early stop but is injected-and-never-called;
  this detector is self-contained at the choke point instead.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

_SEGMENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WS = re.compile(r"\s+")

_BLOCK_CHARS = 256
_BLOCK_REPEATS = 3


def _normalize(segment: str) -> str:
    return _WS.sub(" ", segment).strip().lower()


class IntraTurnRepetitionDetector:
    """Watches the running text of one generation for degenerate repetition."""

    def __init__(
        self,
        window_chars: int = 4000,
        min_segment_chars: int = 24,
        max_repeats: int = 6,
        check_every_chars: int = 512,
    ) -> None:
        self.window_chars = window_chars
        self.min_segment_chars = min_segment_chars
        self.max_repeats = max_repeats
        self.check_every_chars = check_every_chars
        self._tail = ""
        self._since_last_check = 0
        self._repeated_segment: Optional[str] = None

    @property
    def repeated_segment(self) -> Optional[str]:
        """The segment that triggered detection, if any."""
        return self._repeated_segment

    def feed(self, text: str) -> Optional[str]:
        """Accumulate streamed text; returns the repeated segment on trigger.

        Returns None while the stream looks healthy. Once triggered, keeps
        returning the same segment (idempotent).
        """
        if self._repeated_segment is not None:
            return self._repeated_segment
        if not text:
            return None
        self._tail = (self._tail + text)[-self.window_chars :]
        self._since_last_check += len(text)
        if self._since_last_check < self.check_every_chars:
            return None
        self._since_last_check = 0
        return self._check()

    def _check(self) -> Optional[str]:
        # Rule 1: repeated normalized segment (sentence/line) within the window.
        counts: Counter[str] = Counter()
        originals: dict[str, str] = {}
        for raw in _SEGMENT_SPLIT.split(self._tail):
            if len(raw) < self.min_segment_chars:
                continue
            key = _normalize(raw)
            if len(key) < self.min_segment_chars:
                continue
            counts[key] += 1
            originals.setdefault(key, raw.strip())
        if counts:
            key, count = counts.most_common(1)[0]
            if count >= self.max_repeats:
                self._repeated_segment = originals[key]
                return self._repeated_segment

        # Rule 2: the trailing block recurring — long-period loops.
        if len(self._tail) >= _BLOCK_CHARS * _BLOCK_REPEATS:
            block = _normalize(self._tail[-_BLOCK_CHARS:])
            if len(block) >= self.min_segment_chars:
                if _normalize(self._tail).count(block) >= _BLOCK_REPEATS:
                    self._repeated_segment = self._tail[-_BLOCK_CHARS:].strip()
                    return self._repeated_segment
        return None

    def truncation_point(self, full_content: str) -> int:
        """Index to cut ``full_content`` at, keeping the first loop instance.

        Finds the second occurrence of the triggering segment and cuts there;
        falls back to the full length when the segment cannot be located.
        """
        if not self._repeated_segment:
            return len(full_content)
        needle = self._repeated_segment
        first = full_content.find(needle)
        if first < 0:
            return len(full_content)
        second = full_content.find(needle, first + len(needle))
        if second < 0:
            return len(full_content)
        return second
