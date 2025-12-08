# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for correction metrics module."""

from datetime import datetime

from victor.evaluation.correction import (
    Language,
    ValidationResult,
)
from victor.evaluation.correction.metrics import (
    CorrectionAttempt,
    CorrectionMetrics,
    CorrectionMetricsCollector,
    get_metrics_collector,
    reset_metrics,
)


class TestCorrectionMetrics:
    """Tests for CorrectionMetrics dataclass."""

    def test_initial_state(self):
        """Test initial metrics state."""
        metrics = CorrectionMetrics()

        assert metrics.total_validations == 0
        assert metrics.total_corrections == 0
        assert metrics.successful_corrections == 0
        assert metrics.auto_fixes_applied == 0
        assert metrics.syntax_errors == 0
        assert metrics.import_errors == 0
        assert metrics.test_failures == 0
        assert metrics.correction_success_rate == 0.0
        assert metrics.auto_fix_rate == 0.0
        assert metrics.avg_correction_time_ms == 0.0

    def test_record_validation_python(self):
        """Test recording a Python validation."""
        metrics = CorrectionMetrics()

        result = ValidationResult(
            valid=True,
            language=Language.PYTHON,
            syntax_valid=True,
            imports_valid=True,
            errors=(),
            warnings=(),
        )

        metrics.record_validation(Language.PYTHON, result)

        assert metrics.total_validations == 1
        assert metrics.language_validations[Language.PYTHON] == 1
        assert metrics.syntax_errors == 0
        assert metrics.import_errors == 0

    def test_record_validation_with_errors(self):
        """Test recording validation with syntax and import errors."""
        metrics = CorrectionMetrics()

        result = ValidationResult(
            valid=False,
            language=Language.PYTHON,
            syntax_valid=False,
            imports_valid=False,
            errors=("SyntaxError: invalid syntax",),
            warnings=(),
            missing_imports=("numpy", "pandas"),
        )

        metrics.record_validation(Language.PYTHON, result)

        assert metrics.total_validations == 1
        assert metrics.syntax_errors == 1
        assert metrics.import_errors == 1

    def test_record_correction_success(self):
        """Test recording a successful correction."""
        metrics = CorrectionMetrics()

        validation_before = ValidationResult(
            valid=False,
            language=Language.PYTHON,
            syntax_valid=False,
            imports_valid=True,
            errors=("SyntaxError",),
            warnings=(),
        )
        validation_after = ValidationResult(
            valid=True,
            language=Language.PYTHON,
            syntax_valid=True,
            imports_valid=True,
            errors=(),
            warnings=(),
        )

        attempt = CorrectionAttempt(
            task_id="test-001",
            language=Language.PYTHON,
            iteration=1,
            timestamp=datetime.now(),
            duration_ms=150.0,
            validation_before=validation_before,
            validation_after=validation_after,
            test_passed_before=None,
            test_passed_after=None,
            test_total=None,
            success=True,
            auto_fixed=True,
        )

        metrics.record_correction(attempt)

        assert metrics.total_corrections == 1
        assert metrics.successful_corrections == 1
        assert metrics.auto_fixes_applied == 1
        assert metrics.correction_success_rate == 1.0
        assert metrics.auto_fix_rate == 1.0
        assert metrics.language_corrections[Language.PYTHON] == 1
        assert metrics.language_successes[Language.PYTHON] == 1
        assert metrics.corrections_by_iteration[1] == 1
        assert metrics.successes_by_iteration[1] == 1

    def test_record_correction_failure(self):
        """Test recording a failed correction."""
        metrics = CorrectionMetrics()

        validation = ValidationResult(
            valid=False,
            language=Language.PYTHON,
            syntax_valid=False,
            imports_valid=True,
            errors=("SyntaxError",),
            warnings=(),
        )

        attempt = CorrectionAttempt(
            task_id="test-002",
            language=Language.PYTHON,
            iteration=1,
            timestamp=datetime.now(),
            duration_ms=200.0,
            validation_before=validation,
            validation_after=None,
            test_passed_before=0,
            test_passed_after=0,
            test_total=5,
            success=False,
            auto_fixed=False,
        )

        metrics.record_correction(attempt)

        assert metrics.total_corrections == 1
        assert metrics.successful_corrections == 0
        assert metrics.correction_success_rate == 0.0
        assert metrics.test_failures == 1

    def test_success_rate_by_language(self):
        """Test calculating success rate by language."""
        metrics = CorrectionMetrics()

        # Add successful Python correction
        metrics.language_corrections[Language.PYTHON] = 10
        metrics.language_successes[Language.PYTHON] = 8

        # Add successful JavaScript correction
        metrics.language_corrections[Language.JAVASCRIPT] = 5
        metrics.language_successes[Language.JAVASCRIPT] = 5

        assert metrics.success_rate_by_language(Language.PYTHON) == 0.8
        assert metrics.success_rate_by_language(Language.JAVASCRIPT) == 1.0
        assert metrics.success_rate_by_language(Language.RUST) == 0.0

    def test_success_rate_by_iteration(self):
        """Test calculating success rate by iteration."""
        metrics = CorrectionMetrics()

        # Iteration 1: 10 corrections, 6 successes
        metrics.corrections_by_iteration[1] = 10
        metrics.successes_by_iteration[1] = 6

        # Iteration 2: 5 corrections, 4 successes
        metrics.corrections_by_iteration[2] = 5
        metrics.successes_by_iteration[2] = 4

        assert metrics.success_rate_by_iteration(1) == 0.6
        assert metrics.success_rate_by_iteration(2) == 0.8
        assert metrics.success_rate_by_iteration(3) == 0.0

    def test_timing_metrics(self):
        """Test timing metrics calculation."""
        metrics = CorrectionMetrics()

        validation = ValidationResult(
            valid=True,
            language=Language.PYTHON,
            syntax_valid=True,
            imports_valid=True,
            errors=(),
            warnings=(),
        )

        # Add two attempts with different durations
        attempt1 = CorrectionAttempt(
            task_id="test-001",
            language=Language.PYTHON,
            iteration=1,
            timestamp=datetime.now(),
            duration_ms=100.0,
            validation_before=validation,
            validation_after=validation,
            test_passed_before=None,
            test_passed_after=None,
            test_total=None,
            success=True,
        )

        attempt2 = CorrectionAttempt(
            task_id="test-002",
            language=Language.PYTHON,
            iteration=1,
            timestamp=datetime.now(),
            duration_ms=300.0,
            validation_before=validation,
            validation_after=validation,
            test_passed_before=None,
            test_passed_after=None,
            test_total=None,
            success=True,
        )

        metrics.record_correction(attempt1)
        metrics.record_correction(attempt2)

        assert metrics.total_correction_time_ms == 400.0
        assert metrics.min_correction_time_ms == 100.0
        assert metrics.max_correction_time_ms == 300.0
        assert metrics.avg_correction_time_ms == 200.0

    def test_to_dict(self):
        """Test exporting metrics to dictionary."""
        metrics = CorrectionMetrics()
        metrics.total_validations = 10
        metrics.total_corrections = 5
        metrics.successful_corrections = 4

        result = metrics.to_dict()

        assert "summary" in result
        assert result["summary"]["total_validations"] == 10
        assert result["summary"]["total_corrections"] == 5
        assert result["summary"]["successful_corrections"] == 4
        assert "errors" in result
        assert "timing" in result
        assert "by_language" in result
        assert "by_iteration" in result

    def test_report_generation(self):
        """Test human-readable report generation."""
        metrics = CorrectionMetrics()
        metrics.total_validations = 100
        metrics.total_corrections = 50
        metrics.successful_corrections = 40
        metrics.syntax_errors = 10
        metrics.import_errors = 5

        report = metrics.report()

        assert "CORRECTION METRICS REPORT" in report
        assert "Total Validations:    100" in report
        assert "Total Corrections:    50" in report
        assert "Successful:           40" in report
        assert "Success Rate:         80.0%" in report


class TestCorrectionMetricsCollector:
    """Tests for CorrectionMetricsCollector."""

    def test_record_validation(self):
        """Test recording validation through collector."""
        collector = CorrectionMetricsCollector()

        result = ValidationResult(
            valid=True,
            language=Language.PYTHON,
            syntax_valid=True,
            imports_valid=True,
            errors=(),
            warnings=(),
        )

        collector.record_validation(Language.PYTHON, result)

        assert collector.metrics.total_validations == 1

    def test_track_correction_context_manager(self):
        """Test tracking correction with context manager."""
        collector = CorrectionMetricsCollector()

        validation_before = ValidationResult(
            valid=False,
            language=Language.PYTHON,
            syntax_valid=False,
            imports_valid=True,
            errors=("SyntaxError",),
            warnings=(),
        )
        validation_after = ValidationResult(
            valid=True,
            language=Language.PYTHON,
            syntax_valid=True,
            imports_valid=True,
            errors=(),
            warnings=(),
        )

        with collector.track_correction(
            task_id="test-001",
            language=Language.PYTHON,
            iteration=1,
            validation_before=validation_before,
        ) as tracker:
            # Simulate correction work
            tracker.set_result(
                success=True,
                validation_after=validation_after,
                auto_fixed=True,
            )

        assert collector.metrics.total_corrections == 1
        assert collector.metrics.successful_corrections == 1
        assert collector.metrics.auto_fixes_applied == 1

    def test_memory_efficient_mode(self):
        """Test collector in memory-efficient mode (keep_attempts=False)."""
        collector = CorrectionMetricsCollector(keep_attempts=False)

        validation = ValidationResult(
            valid=True,
            language=Language.PYTHON,
            syntax_valid=True,
            imports_valid=True,
            errors=(),
            warnings=(),
        )

        with collector.track_correction(
            task_id="test-001",
            language=Language.PYTHON,
            iteration=1,
            validation_before=validation,
        ) as tracker:
            tracker.set_result(success=True)

        # Metrics should be tracked but attempts list should be empty
        assert collector.metrics.total_corrections == 1
        assert len(collector.metrics.attempts) == 0


class TestCorrectionTracker:
    """Tests for CorrectionTracker context manager."""

    def test_duration_tracking(self):
        """Test that duration is tracked correctly."""
        collector = CorrectionMetricsCollector()

        validation = ValidationResult(
            valid=True,
            language=Language.PYTHON,
            syntax_valid=True,
            imports_valid=True,
            errors=(),
            warnings=(),
        )

        import time

        with collector.track_correction(
            task_id="test-001",
            language=Language.PYTHON,
            iteration=1,
            validation_before=validation,
        ) as tracker:
            time.sleep(0.01)  # 10ms
            tracker.set_result(success=True)

        # Duration should be at least 10ms
        assert collector.metrics.min_correction_time_ms >= 10.0

    def test_test_metrics_tracking(self):
        """Test tracking of test pass/fail metrics."""
        collector = CorrectionMetricsCollector()

        validation = ValidationResult(
            valid=True,
            language=Language.PYTHON,
            syntax_valid=True,
            imports_valid=True,
            errors=(),
            warnings=(),
        )

        with collector.track_correction(
            task_id="test-001",
            language=Language.PYTHON,
            iteration=1,
            validation_before=validation,
            test_passed_before=3,
            test_total=10,
        ) as tracker:
            tracker.set_result(
                success=True,
                test_passed_after=8,
            )

        assert collector.metrics.total_corrections == 1
        assert collector.metrics.test_failures == 1  # 3/10 < 10


class TestGlobalCollector:
    """Tests for global collector functions."""

    def test_get_metrics_collector_singleton(self):
        """Test that get_metrics_collector returns same instance."""
        reset_metrics()

        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()

        assert collector1 is collector2

    def test_reset_metrics(self):
        """Test that reset_metrics creates new instance."""
        collector1 = get_metrics_collector()
        collector1.metrics.total_validations = 100

        reset_metrics()

        collector2 = get_metrics_collector()
        assert collector2.metrics.total_validations == 0
        assert collector1 is not collector2


class TestMultiLanguageMetrics:
    """Tests for metrics across multiple languages."""

    def test_multi_language_validation(self):
        """Test validation metrics across multiple languages."""
        metrics = CorrectionMetrics()

        python_result = ValidationResult(
            valid=True,
            language=Language.PYTHON,
            syntax_valid=True,
            imports_valid=True,
            errors=(),
            warnings=(),
        )

        js_result = ValidationResult(
            valid=False,
            language=Language.JAVASCRIPT,
            syntax_valid=False,
            imports_valid=True,
            errors=("SyntaxError",),
            warnings=(),
        )

        rust_result = ValidationResult(
            valid=True,
            language=Language.RUST,
            syntax_valid=True,
            imports_valid=True,
            errors=(),
            warnings=(),
        )

        metrics.record_validation(Language.PYTHON, python_result)
        metrics.record_validation(Language.PYTHON, python_result)
        metrics.record_validation(Language.JAVASCRIPT, js_result)
        metrics.record_validation(Language.RUST, rust_result)

        assert metrics.total_validations == 4
        assert metrics.language_validations[Language.PYTHON] == 2
        assert metrics.language_validations[Language.JAVASCRIPT] == 1
        assert metrics.language_validations[Language.RUST] == 1
        assert metrics.syntax_errors == 1  # Only JS had syntax error
