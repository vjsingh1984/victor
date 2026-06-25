# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Prompt candidate hygiene checks for optimization strategies."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, List, Optional


@dataclass
class PromptHygieneReport:
    """Validation report for an optimized prompt candidate."""

    accepted: bool
    growth_chars: int
    seed_similarity: float
    repeated_trigrams: int
    unsupported_additions: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)


def evaluate_prompt_candidate(
    seed_text: str,
    candidate_text: str,
    *,
    allowed_additions: Iterable[str] = (),
    max_growth_chars: int = 240,
    min_seed_similarity: float = 0.6,
    max_repeated_trigrams: int = 0,
) -> PromptHygieneReport:
    """Validate an optimized prompt against basic safety and quality constraints."""
    growth_chars = len(candidate_text) - len(seed_text)
    seed_similarity = _seed_similarity(seed_text, candidate_text)
    repeated_trigrams = _repeated_trigram_count(candidate_text)
    unsupported_additions = _find_unsupported_additions(
        seed_text,
        candidate_text,
        allowed_additions,
    )

    violations = []
    if growth_chars > max_growth_chars:
        violations.append("growth_exceeded")
    if seed_similarity < min_seed_similarity:
        violations.append("seed_similarity_too_low")
    if repeated_trigrams > max_repeated_trigrams:
        violations.append("repeated_trigrams")
    if unsupported_additions:
        violations.append("unsupported_additions")

    return PromptHygieneReport(
        accepted=not violations,
        growth_chars=growth_chars,
        seed_similarity=seed_similarity,
        repeated_trigrams=repeated_trigrams,
        unsupported_additions=unsupported_additions,
        violations=violations,
    )


def _seed_similarity(seed_text: str, candidate_text: str) -> float:
    seed_tokens = set(_tokens(seed_text))
    candidate_tokens = set(_tokens(candidate_text))
    if not seed_tokens or not candidate_tokens:
        return 0.0
    return len(seed_tokens & candidate_tokens) / len(seed_tokens)


def _repeated_trigram_count(text: str) -> int:
    tokens = _tokens(text)
    if len(tokens) < 3:
        return 0
    trigrams = [tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
    counts = Counter(trigrams)
    return sum(count - 1 for count in counts.values() if count > 1)


def _find_unsupported_additions(
    seed_text: str,
    candidate_text: str,
    allowed_additions: Iterable[str],
) -> List[str]:
    seed_lines = {_normalize_line(line) for line in seed_text.splitlines() if line.strip()}
    allowed_lines = [_normalize_line(line) for line in allowed_additions if str(line).strip()]
    unsupported = []

    for raw_line in candidate_text.splitlines():
        line = _normalize_line(raw_line)
        if not line or line in seed_lines:
            continue
        if any(line.startswith(allowed) or allowed.startswith(line) for allowed in allowed_lines):
            continue
        unsupported.append(raw_line.strip())
    return unsupported


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip().lower())


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_()]+", text.lower())


# ── Sanitization (transform) ───────────────────────────────────────────────
# evaluate_prompt_candidate() above is a *validation* gate (report-only). The
# two helpers below are the matching *transform* used by GEPA/PrefPO mutation
# paths to clean a candidate before it is stored. They were referenced by
# gepa_service.mutate / prompt_optimizer but never defined, which surfaced as
# ImportError on those code paths.


def boundary_aware_truncate(text: str, limit: int) -> tuple[str, bool]:
    """Truncate ``text`` to ``limit`` chars without splitting a token.

    When truncation is required the cut lands on the last whitespace at or
    before ``limit`` so words/sentences are never severed mid-token. If no
    whitespace is found, a hard cut at ``limit`` is used as a last resort.

    Returns ``(truncated_text, was_truncated)``.
    """
    if limit <= 0 or len(text) <= limit:
        return text, False
    cut = text.rfind(" ", 0, limit)
    if cut <= 0:
        cut = limit
    return text[:cut].rstrip(), True


def sanitize_prompt_candidate(
    result: str,
    limit: int = 0,
    seed_text: str = "",
) -> str:
    """Clean a mutated prompt before storage.

    Applies, in order:
      1. code-fence stripping — drop ```` ``` ```` delimiters, keep inner text
      2. consecutive-line dedupe — collapse immediately-repeated lines
      3. boundary-aware truncation to ``limit`` chars (when ``limit > 0``)

    ``seed_text`` is accepted for API symmetry with ``evaluate_prompt_candidate``
    and for future similarity-preserving strategies; it does not alter the
    transform today.
    """
    text = _strip_code_fences(result)
    text = _dedupe_consecutive_lines(text)
    if limit > 0 and len(text) > limit:
        text, _ = boundary_aware_truncate(text, limit)
    return text


def _strip_code_fences(text: str) -> str:
    """Remove ```` ``` ```` fence delimiters, preserving the fenced content."""
    out: List[str] = []
    for line in text.splitlines():
        if line.strip().startswith("```"):
            continue
        out.append(line)
    return "\n".join(out)


def _dedupe_consecutive_lines(text: str) -> str:
    """Collapse runs of immediately-repeated identical lines to a single line."""
    out: List[str] = []
    prev: Optional[str] = None
    for line in text.splitlines():
        if line != prev:
            out.append(line)
        prev = line
    return "\n".join(out)
