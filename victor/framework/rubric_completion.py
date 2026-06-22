# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rubric-based completion evaluation (EVR-3, ADR-009, FEP-0008 Phase A).

Replaces a single scalar completion score with **task-adaptive rubric dimensions**, each scored with
a confidence, and gated by a **DimensionAwareFilter** that requires *every engaged* dimension to
clear its threshold before COMPLETE (AdaRubric 2603.21362). This fixes the two failure modes the
FEP-0007 cutover exposed: one strong axis masking a failed one (premature stop), and a finished
answer being under-scored overall (restatement).

This module is the machinery, offline-testable and additive — it is **not** yet wired into
``AgenticLoop._evaluate`` (that cutover is EVR-3b, behind ``settings.evaluation.completion_strategy``,
gated by the parity/characterization batteries). The judge and the rubric generator are injected
Protocols with deterministic defaults, so nothing here requires an LLM call.

**Production wiring note:** the injected :class:`RubricJudge` should be an LLM judge wrapped in
``OrderSwapEnsembleJudge`` and only trusted when it clears ``JudgeReliabilityGate`` (EVR-2 / ADR-011);
otherwise fall back to :class:`HeuristicRubricJudge`. Rubrics are cached per task family (>95% cost
cut, AdaRubric).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Sequence

from victor.core.utils import clamp


@dataclass(frozen=True)
class RubricDimension:
    """One orthogonal completion axis: a weight, a pass threshold, and grading guidance."""

    name: str
    weight: float = 0.25
    threshold: float = 0.6
    description: str = ""


@dataclass(frozen=True)
class Rubric:
    """A task-family rubric: a set of weighted, thresholded dimensions."""

    task_family: str
    dimensions: tuple[RubricDimension, ...]

    def validate(self) -> "Rubric":
        """Validate dimension names are distinct and weights are sane; return self.

        Weights are normalized to sum to 1.0 by the caller if needed; here we only reject empties and
        duplicate names (AdaRubric requires orthogonal, non-overlapping dimensions).
        """
        if not self.dimensions:
            raise ValueError("rubric must have at least one dimension")
        names = [d.name for d in self.dimensions]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate rubric dimension names: {names}")
        return self

    def normalized(self) -> "Rubric":
        """Return a copy with weights normalized to sum to 1.0 (no-op if already ~1.0)."""
        total = sum(d.weight for d in self.dimensions)
        if total <= 0:
            raise ValueError("rubric weights must be positive")
        if math.isclose(total, 1.0, abs_tol=1e-6):
            return self
        dims = tuple(
            RubricDimension(d.name, d.weight / total, d.threshold, d.description)
            for d in self.dimensions
        )
        return Rubric(self.task_family, dims)


@dataclass(frozen=True)
class RubricDimensionScore:
    """A judge's score for one rubric dimension. ``score``/``confidence`` are in [0, 1]."""

    name: str
    score: float
    confidence: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RubricCompletionResult:
    """Outcome of a rubric completion evaluation."""

    complete: bool
    aggregate: float
    scores: tuple[RubricDimensionScore, ...]
    failed_dimensions: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "aggregate": round(self.aggregate, 4),
            "failed_dimensions": list(self.failed_dimensions),
            "reason": self.reason,
            "scores": [s.to_dict() for s in self.scores],
        }


def confidence_weighted_mean(scores: Sequence[RubricDimensionScore]) -> float:
    """Confidence-weighted mean of dimension scores (un-engaged axes contribute less)."""
    total_w = sum(s.confidence for s in scores)
    if total_w <= 0:
        return 0.0
    return sum(s.score * s.confidence for s in scores) / total_w


class RubricJudge(Protocol):
    """Scores a single rubric dimension for given content. The LLM-judge seam (wrap + gate it)."""

    def score(
        self, dimension: RubricDimension, content: str, context: Mapping[str, Any]
    ) -> RubricDimensionScore:
        """Return the dimension's :class:`RubricDimensionScore` for ``content``."""
        ...


class RubricGenerator(Protocol):
    """Produces a :class:`Rubric` for a task family. The task-adaptive (LLM) generator seam."""

    def generate(self, task_family: str, context: Mapping[str, Any]) -> Rubric:
        """Return a validated rubric for ``task_family``."""
        ...


# General default rubric (deterministic, non-LLM). A task-adaptive LLM generator supersedes this in
# production; these four orthogonal axes are a sensible baseline for agent task completion.
_DEFAULT_DIMENSIONS = (
    RubricDimension("correctness", 0.40, 0.6, "Is the result/answer correct for the request?"),
    RubricDimension("tool_grounding", 0.25, 0.5, "Is the answer grounded in tool/file evidence?"),
    RubricDimension(
        "completeness", 0.25, 0.6, "Does it address the full request, nothing missing?"
    ),
    RubricDimension("recovery", 0.10, 0.4, "Were tool failures recovered from?"),
)


@dataclass
class DefaultRubricGenerator:
    """Returns a fixed, sensible rubric (the non-LLM baseline)."""

    dimensions: tuple[RubricDimension, ...] = _DEFAULT_DIMENSIONS

    def generate(self, task_family: str, context: Mapping[str, Any]) -> Rubric:
        return Rubric(task_family, self.dimensions).validate().normalized()


@dataclass
class HeuristicRubricJudge:
    """Deterministic, no-LLM fallback judge.

    A coarse content-shape heuristic — substantial, structured content scores higher — used only as
    the fallback when no reliable LLM judge is available (EVR-2 gate). Low/moderate confidence by
    design so the DimensionAwareFilter does not over-trust it.
    """

    def score(
        self, dimension: RubricDimension, content: str, context: Mapping[str, Any]
    ) -> RubricDimensionScore:
        text = (content or "").strip()
        length = len(text)
        has_structure = any(m in text for m in ("\n-", "\n*", "\n#", "1.", "```"))
        base = clamp(length / 400.0, 0.0, 1.0)
        if has_structure:
            base = clamp(base + 0.15, 0.0, 1.0)
        return RubricDimensionScore(
            dimension.name, base, 0.3, f"heuristic: len={length}, structured={has_structure}"
        )


@dataclass
class RubricCache:
    """Per-task-family rubric cache (AdaRubric caches rubrics per family for ~free reuse)."""

    _store: dict[str, Rubric] = field(default_factory=dict)

    def get_or_generate(
        self, task_family: str, generator: RubricGenerator, context: Mapping[str, Any]
    ) -> Rubric:
        cached = self._store.get(task_family)
        if cached is not None:
            return cached
        rubric = generator.generate(task_family, context).validate().normalized()
        self._store[task_family] = rubric
        return rubric

    def clear(self) -> None:
        self._store.clear()


@dataclass
class DimensionAwareFilter:
    """COMPLETE only when every *engaged* dimension clears its threshold (AdaRubric).

    A dimension is "engaged" when its score confidence is at or above ``confidence_floor``; un-engaged
    dimensions (e.g. recovery on a no-error task) are not gated. This is the structural fix that stops
    one strong axis from masking a failed one — and stops an under-scored axis the trajectory never
    engaged from blocking completion.
    """

    confidence_floor: float = 0.25

    def failed(
        self, rubric: Rubric, scores: Sequence[RubricDimensionScore]
    ) -> tuple[RubricDimensionScore, ...]:
        thresholds = {d.name: d.threshold for d in rubric.dimensions}
        return tuple(
            s
            for s in scores
            if s.confidence >= self.confidence_floor and s.score < thresholds.get(s.name, 0.6)
        )


class RubricCompletionEvaluator:
    """Generate → score → DimensionAwareFilter, producing a :class:`RubricCompletionResult`.

    Default-constructed it is fully deterministic (default generator + heuristic judge), so it runs
    with no LLM. Inject an LLM judge (wrapped per the module note) and a task-adaptive generator for
    the real thing.
    """

    def __init__(
        self,
        *,
        judge: Optional[RubricJudge] = None,
        generator: Optional[RubricGenerator] = None,
        dimension_filter: Optional[DimensionAwareFilter] = None,
        cache: Optional[RubricCache] = None,
    ) -> None:
        self._judge: RubricJudge = judge or HeuristicRubricJudge()
        self._generator: RubricGenerator = generator or DefaultRubricGenerator()
        self._filter = dimension_filter or DimensionAwareFilter()
        self._cache = cache or RubricCache()

    def evaluate(
        self,
        *,
        task_family: str,
        content: str,
        context: Optional[Mapping[str, Any]] = None,
    ) -> RubricCompletionResult:
        ctx: Mapping[str, Any] = context or {}
        rubric = self._cache.get_or_generate(task_family, self._generator, ctx)
        scores = tuple(self._judge.score(d, content, ctx) for d in rubric.dimensions)
        failed = self._filter.failed(rubric, scores)
        aggregate = confidence_weighted_mean(scores)
        complete = not failed
        if complete:
            reason = f"all engaged dimensions cleared their thresholds (aggregate={aggregate:.2f})"
        else:
            names = ", ".join(f"{f.name}={f.score:.2f}" for f in failed)
            reason = f"dimensions below threshold: {names}"
        return RubricCompletionResult(
            complete=complete,
            aggregate=aggregate,
            scores=scores,
            failed_dimensions=tuple(f.name for f in failed),
            reason=reason,
        )
