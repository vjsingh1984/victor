# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Prompt candidate hygiene checks for optimization strategies."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, List


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
