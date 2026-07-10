# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Unit tests for FEP-0014 Phase 1 canonical validation/metrics contracts."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from victor.core import (
    MetricsCollectorProtocol,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    severity_rank,
)
from victor.core.validation import severity_rank as severity_rank_direct

# =============================================================================
# ValidationSeverity (superset incl. CRITICAL)
# =============================================================================


def test_severity_members_are_superset() -> None:
    """The canonical enum carries the {INFO, WARNING, ERROR, CRITICAL} superset."""
    names = {m.name for m in ValidationSeverity}
    assert names == {"INFO", "WARNING", "ERROR", "CRITICAL"}


def test_existing_severity_values_unchanged() -> None:
    """Pre-existing members keep their string values (backward compatible)."""
    assert ValidationSeverity.ERROR.value == "error"
    assert ValidationSeverity.WARNING.value == "warning"
    assert ValidationSeverity.INFO.value == "info"
    assert ValidationSeverity.CRITICAL.value == "critical"


def test_severity_is_str_enum() -> None:
    """Members are still plain strings (str Enum contract preserved)."""
    assert ValidationSeverity.ERROR == "error"
    assert isinstance(ValidationSeverity.CRITICAL, str)


def test_severity_rank_ordering_including_critical() -> None:
    """Logical ordering is INFO < WARNING < ERROR < CRITICAL."""
    order = [
        ValidationSeverity.INFO,
        ValidationSeverity.WARNING,
        ValidationSeverity.ERROR,
        ValidationSeverity.CRITICAL,
    ]
    ranks = [severity_rank(s) for s in order]
    assert ranks == sorted(ranks)
    assert ranks == [0, 1, 2, 3]
    assert severity_rank(ValidationSeverity.CRITICAL) > severity_rank(ValidationSeverity.ERROR)


def test_severity_rank_reexported_from_package() -> None:
    """severity_rank is importable from both victor.core and victor.core.validation."""
    assert severity_rank is severity_rank_direct


# =============================================================================
# ValidationResult (canonical frozen value object)
# =============================================================================


def _issue(severity: ValidationSeverity, message: str) -> ValidationIssue:
    return ValidationIssue(path="p", message=message, severity=severity)


def test_result_is_frozen() -> None:
    """The canonical result is an immutable value object."""
    result = ValidationResult(is_valid=True)
    try:
        result.is_valid = False  # type: ignore[misc]
    except Exception as exc:  # dataclasses.FrozenInstanceError
        assert "FrozenInstance" in type(exc).__name__ or "frozen" in str(exc).lower()
    else:  # pragma: no cover - must not happen
        raise AssertionError("ValidationResult should be frozen")


def test_result_defaults_to_empty_issues() -> None:
    result = ValidationResult(is_valid=True)
    assert result.issues == []
    assert result.errors == []
    assert result.error_message == ""


def test_errors_include_error_and_critical_only() -> None:
    """errors property returns ERROR+CRITICAL, excludes INFO/WARNING."""
    issues = [
        _issue(ValidationSeverity.INFO, "info msg"),
        _issue(ValidationSeverity.WARNING, "warn msg"),
        _issue(ValidationSeverity.ERROR, "error msg"),
        _issue(ValidationSeverity.CRITICAL, "critical msg"),
    ]
    result = ValidationResult(is_valid=False, issues=issues)
    error_msgs = [i.message for i in result.errors]
    assert error_msgs == ["error msg", "critical msg"]


def test_errors_uses_logical_not_string_ordering() -> None:
    """Guard: CRITICAL ('critical' < 'error' alphabetically) still counts as error."""
    result = ValidationResult(is_valid=False, issues=[_issue(ValidationSeverity.CRITICAL, "boom")])
    # A naive str-Enum ``>= ERROR`` would drop CRITICAL ("critical" < "error"); the
    # logical severity_rank comparison must keep it.
    assert len(result.errors) == 1


def test_error_message_joins_error_level_messages() -> None:
    issues = [
        _issue(ValidationSeverity.WARNING, "ignored"),
        _issue(ValidationSeverity.ERROR, "first"),
        _issue(ValidationSeverity.CRITICAL, "second"),
    ]
    result = ValidationResult(is_valid=False, issues=issues)
    assert result.error_message == "first\nsecond"


# =============================================================================
# MetricsCollectorProtocol
# =============================================================================


class _ConformingCollector:
    """Minimal object exposing the canonical collector surface."""

    def __init__(self) -> None:
        self._store: Dict[str, float] = {}

    def record_metric(self, name: str, value: float, **tags: str) -> None:
        self._store[name] = value

    def get_snapshot(self) -> Mapping[str, Any]:
        return dict(self._store)


class _AnotherConformingCollector:
    """A second, differently-shaped conforming collector."""

    def record_metric(self, name: str, value: float, **tags: str) -> None:  # noqa: D401
        return None

    def get_snapshot(self) -> Mapping[str, Any]:
        return {}


class _MissingSnapshot:
    def record_metric(self, name: str, value: float, **tags: str) -> None:
        return None


def test_protocol_is_runtime_checkable() -> None:
    assert isinstance(_ConformingCollector(), MetricsCollectorProtocol)


def test_protocol_matches_at_least_two_collectors() -> None:
    """isinstance holds for >=2 distinct conforming collector implementations."""
    collectors = [_ConformingCollector(), _AnotherConformingCollector()]
    assert all(isinstance(c, MetricsCollectorProtocol) for c in collectors)
    assert len(collectors) >= 2


def test_protocol_rejects_partial_surface() -> None:
    """An object missing get_snapshot does not satisfy the protocol."""
    assert not isinstance(_MissingSnapshot(), MetricsCollectorProtocol)


def test_named_collectors_not_yet_conforming_phase2_surface() -> None:
    """FEP-0014 finding: the five named MetricsCollectors do not yet conform.

    Their present public surfaces are disjoint (no shared public method); Phase 2
    adapts/renames them onto MetricsCollectorProtocol. This test documents the
    Phase-1 baseline so a future accidental partial-conformance is noticed.
    """
    from victor.framework.observability.metrics import (
        MetricsCollector as FrameworkObsCollector,
    )

    collector = FrameworkObsCollector()
    # Has get_snapshot but its record method is named ``record`` (not record_metric),
    # so it does not structurally satisfy the canonical protocol yet.
    assert hasattr(collector, "get_snapshot")
    assert not hasattr(collector, "record_metric")
    assert not isinstance(collector, MetricsCollectorProtocol)
