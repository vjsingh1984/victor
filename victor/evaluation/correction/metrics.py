# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Correction metrics and observability.

This module provides metrics collection and reporting for the
self-correction system. It tracks validation results, correction
attempts, and success rates.

Design Pattern: Observer Pattern (collects events)
Enterprise Integration Pattern: Wire Tap (observability)
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .types import Language, CodeValidationResult

logger = logging.getLogger(__name__)


@dataclass
class CorrectionAttempt:
    """Record of a single correction attempt."""

    task_id: str
    language: Language
    iteration: int
    timestamp: datetime
    duration_ms: float
    validation_before: CodeValidationResult
    validation_after: Optional[CodeValidationResult]
    test_passed_before: Optional[int]
    test_passed_after: Optional[int]
    test_total: Optional[int]
    success: bool
    auto_fixed: bool = False


@dataclass
class CorrectionMetrics:
    """Aggregate metrics for correction sessions.

    Tracks:
    - Total attempts and success rates
    - Per-language statistics
    - Iteration distribution
    - Auto-fix effectiveness
    """

    total_validations: int = 0
    total_corrections: int = 0
    successful_corrections: int = 0
    auto_fixes_applied: int = 0

    # Per-language stats
    language_validations: dict[Language, int] = field(default_factory=dict)
    language_corrections: dict[Language, int] = field(default_factory=dict)
    language_successes: dict[Language, int] = field(default_factory=dict)

    # Iteration distribution
    corrections_by_iteration: dict[int, int] = field(default_factory=dict)
    successes_by_iteration: dict[int, int] = field(default_factory=dict)

    # Error type tracking
    syntax_errors: int = 0
    import_errors: int = 0
    test_failures: int = 0

    # Timing
    total_correction_time_ms: float = 0.0
    min_correction_time_ms: float = float("inf")
    max_correction_time_ms: float = 0.0

    # Detailed records (optional)
    attempts: list[CorrectionAttempt] = field(default_factory=list)

    def record_validation(
        self,
        language: Language,
        result: CodeValidationResult,
    ) -> None:
        """Record a validation event."""
        self.total_validations += 1
        self.language_validations[language] = self.language_validations.get(language, 0) + 1

        if not result.syntax_valid:
            self.syntax_errors += 1
        if not result.imports_valid:
            self.import_errors += 1

    def record_correction(
        self,
        attempt: CorrectionAttempt,
    ) -> None:
        """Record a correction attempt."""
        self.total_corrections += 1
        self.attempts.append(attempt)

        # Language stats
        lang = attempt.language
        self.language_corrections[lang] = self.language_corrections.get(lang, 0) + 1

        # Iteration stats
        self.corrections_by_iteration[attempt.iteration] = (
            self.corrections_by_iteration.get(attempt.iteration, 0) + 1
        )

        # Success tracking
        if attempt.success:
            self.successful_corrections += 1
            self.language_successes[lang] = self.language_successes.get(lang, 0) + 1
            self.successes_by_iteration[attempt.iteration] = (
                self.successes_by_iteration.get(attempt.iteration, 0) + 1
            )

        # Auto-fix tracking
        if attempt.auto_fixed:
            self.auto_fixes_applied += 1

        # Timing
        self.total_correction_time_ms += attempt.duration_ms
        self.min_correction_time_ms = min(self.min_correction_time_ms, attempt.duration_ms)
        self.max_correction_time_ms = max(self.max_correction_time_ms, attempt.duration_ms)

        # Test failure tracking
        if (
            attempt.test_passed_before is not None
            and attempt.test_total is not None
            and attempt.test_passed_before < attempt.test_total
        ):
            self.test_failures += 1

    @property
    def correction_success_rate(self) -> float:
        """Overall correction success rate."""
        if self.total_corrections == 0:
            return 0.0
        return self.successful_corrections / self.total_corrections

    @property
    def auto_fix_rate(self) -> float:
        """Rate of corrections that used auto-fix."""
        if self.total_corrections == 0:
            return 0.0
        return self.auto_fixes_applied / self.total_corrections

    @property
    def avg_correction_time_ms(self) -> float:
        """Average correction time in milliseconds."""
        if self.total_corrections == 0:
            return 0.0
        return self.total_correction_time_ms / self.total_corrections

    def success_rate_by_language(self, language: Language) -> float:
        """Get success rate for a specific language."""
        corrections = self.language_corrections.get(language, 0)
        if corrections == 0:
            return 0.0
        successes = self.language_successes.get(language, 0)
        return successes / corrections

    def success_rate_by_iteration(self, iteration: int) -> float:
        """Get success rate for a specific iteration."""
        corrections = self.corrections_by_iteration.get(iteration, 0)
        if corrections == 0:
            return 0.0
        successes = self.successes_by_iteration.get(iteration, 0)
        return successes / corrections

    def to_dict(self) -> dict:
        """Export metrics as dictionary for JSON serialization."""
        return {
            "summary": {
                "total_validations": self.total_validations,
                "total_corrections": self.total_corrections,
                "successful_corrections": self.successful_corrections,
                "correction_success_rate": round(self.correction_success_rate, 4),
                "auto_fixes_applied": self.auto_fixes_applied,
                "auto_fix_rate": round(self.auto_fix_rate, 4),
            },
            "errors": {
                "syntax_errors": self.syntax_errors,
                "import_errors": self.import_errors,
                "test_failures": self.test_failures,
            },
            "timing": {
                "total_correction_time_ms": round(self.total_correction_time_ms, 2),
                "avg_correction_time_ms": round(self.avg_correction_time_ms, 2),
                "min_correction_time_ms": (
                    round(self.min_correction_time_ms, 2)
                    if self.min_correction_time_ms != float("inf")
                    else 0
                ),
                "max_correction_time_ms": round(self.max_correction_time_ms, 2),
            },
            "by_language": {
                lang.name: {
                    "validations": self.language_validations.get(lang, 0),
                    "corrections": self.language_corrections.get(lang, 0),
                    "successes": self.language_successes.get(lang, 0),
                    "success_rate": round(self.success_rate_by_language(lang), 4),
                }
                for lang in set(self.language_validations.keys())
                | set(self.language_corrections.keys())
            },
            "by_iteration": {
                str(i): {
                    "corrections": self.corrections_by_iteration.get(i, 0),
                    "successes": self.successes_by_iteration.get(i, 0),
                    "success_rate": round(self.success_rate_by_iteration(i), 4),
                }
                for i in sorted(self.corrections_by_iteration.keys())
            },
        }

    def report(self) -> str:
        """Generate a human-readable report."""
        lines = [
            "=" * 50,
            "       CORRECTION METRICS REPORT",
            "=" * 50,
            "",
            "SUMMARY",
            "-" * 30,
            f"  Total Validations:    {self.total_validations}",
            f"  Total Corrections:    {self.total_corrections}",
            f"  Successful:           {self.successful_corrections}",
            f"  Success Rate:         {self.correction_success_rate:.1%}",
            f"  Auto-fixes Applied:   {self.auto_fixes_applied}",
            "",
            "ERROR TYPES",
            "-" * 30,
            f"  Syntax Errors:        {self.syntax_errors}",
            f"  Import Errors:        {self.import_errors}",
            f"  Test Failures:        {self.test_failures}",
            "",
            "TIMING",
            "-" * 30,
            f"  Avg Correction Time:  {self.avg_correction_time_ms:.1f}ms",
            (
                f"  Min Correction Time:  {self.min_correction_time_ms:.1f}ms"
                if self.min_correction_time_ms != float("inf")
                else "  Min Correction Time:  N/A"
            ),
            f"  Max Correction Time:  {self.max_correction_time_ms:.1f}ms",
            "",
        ]

        # Per-language breakdown
        if self.language_corrections:
            lines.extend(
                [
                    "BY LANGUAGE",
                    "-" * 30,
                ]
            )
            for lang in sorted(self.language_corrections.keys(), key=lambda x: x.name):
                rate = self.success_rate_by_language(lang)
                corr = self.language_corrections[lang]
                succ = self.language_successes.get(lang, 0)
                lines.append(f"  {lang.name:12} {succ}/{corr} ({rate:.0%})")
            lines.append("")

        # Per-iteration breakdown
        if self.corrections_by_iteration:
            lines.extend(
                [
                    "BY ITERATION",
                    "-" * 30,
                ]
            )
            for i in sorted(self.corrections_by_iteration.keys()):
                rate = self.success_rate_by_iteration(i)
                corr = self.corrections_by_iteration[i]
                succ = self.successes_by_iteration.get(i, 0)
                lines.append(f"  Iteration {i}: {succ}/{corr} ({rate:.0%})")
            lines.append("")

        lines.append("=" * 50)
        return "\n".join(lines)


class CorrectionMetricsCollector:
    """Collector for correction metrics.

    Provides a convenient interface for tracking correction events
    and generating reports.

    Usage:
        collector = CorrectionMetricsCollector()

        # Track validation
        collector.record_validation(Language.PYTHON, result)

        # Track correction with timing
        with collector.track_correction(task_id, Language.PYTHON, 1) as tracker:
            # ... perform correction ...
            tracker.set_result(success=True, validation_after=result)

        # Generate report
        print(collector.metrics.report())
    """

    def __init__(self, keep_attempts: bool = True):
        """Initialize collector.

        Args:
            keep_attempts: Whether to store individual attempt records
        """
        self.metrics = CorrectionMetrics()
        self._keep_attempts = keep_attempts

    def record_validation(
        self,
        language: Language,
        result: CodeValidationResult,
    ) -> None:
        """Record a validation event."""
        self.metrics.record_validation(language, result)

    def track_correction(
        self,
        task_id: str,
        language: Language,
        iteration: int,
        validation_before: CodeValidationResult,
        test_passed_before: Optional[int] = None,
        test_total: Optional[int] = None,
    ) -> "CorrectionTracker":
        """Create a context manager for tracking a correction attempt."""
        return CorrectionTracker(
            collector=self,
            task_id=task_id,
            language=language,
            iteration=iteration,
            validation_before=validation_before,
            test_passed_before=test_passed_before,
            test_total=test_total,
        )

    def _record_attempt(self, attempt: CorrectionAttempt) -> None:
        """Record a completed correction attempt."""
        # Record the correction (updates aggregate stats)
        self.metrics.record_correction(attempt)

        if not self._keep_attempts:
            # Clear attempts list after recording to save memory
            # (aggregate stats are already updated)
            self.metrics.attempts = []


class CorrectionTracker:
    """Context manager for tracking a single correction attempt."""

    def __init__(
        self,
        collector: CorrectionMetricsCollector,
        task_id: str,
        language: Language,
        iteration: int,
        validation_before: CodeValidationResult,
        test_passed_before: Optional[int] = None,
        test_total: Optional[int] = None,
    ):
        self._collector = collector
        self._task_id = task_id
        self._language = language
        self._iteration = iteration
        self._validation_before = validation_before
        self._test_passed_before = test_passed_before
        self._test_total = test_total
        self._start_time: float = 0
        self._validation_after: Optional[CodeValidationResult] = None
        self._test_passed_after: Optional[int] = None
        self._success: bool = False
        self._auto_fixed: bool = False

    def __enter__(self) -> "CorrectionTracker":
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        duration_ms = (time.perf_counter() - self._start_time) * 1000

        attempt = CorrectionAttempt(
            task_id=self._task_id,
            language=self._language,
            iteration=self._iteration,
            timestamp=datetime.now(),
            duration_ms=duration_ms,
            validation_before=self._validation_before,
            validation_after=self._validation_after,
            test_passed_before=self._test_passed_before,
            test_passed_after=self._test_passed_after,
            test_total=self._test_total,
            success=self._success,
            auto_fixed=self._auto_fixed,
        )

        self._collector._record_attempt(attempt)

    def set_result(
        self,
        success: bool,
        validation_after: Optional[CodeValidationResult] = None,
        test_passed_after: Optional[int] = None,
        auto_fixed: bool = False,
    ) -> None:
        """Set the result of the correction attempt."""
        self._success = success
        self._validation_after = validation_after
        self._test_passed_after = test_passed_after
        self._auto_fixed = auto_fixed


# Global metrics collector (optional singleton pattern)
_global_collector: Optional[CorrectionMetricsCollector] = None


def get_metrics_collector() -> CorrectionMetricsCollector:
    """Get the global metrics collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = CorrectionMetricsCollector()
    return _global_collector


def reset_metrics() -> None:
    """Reset global metrics collector."""
    global _global_collector
    _global_collector = None
