from __future__ import annotations

# Copyright 2026 Vijaykumar Singh <singhvijay@users.noreply.github.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Canonical evaluation-level service factories and exports."""

from pathlib import Path
from typing import Optional
from typing import Any, Mapping, Protocol, runtime_checkable

from victor.evaluation.protocol import BenchmarkType, EvaluationResult
from victor.evaluation.validated_session_truth_emitters import ValidatedSessionTruthEmitterRegistry
from victor.evaluation.validated_session_truth_service import (
    ValidatedSessionTruthService,
    create_default_validated_session_truth_service,
)


@runtime_checkable
class ValidatedSessionTruthServiceProtocol(Protocol):
    """Contract for evaluation runtimes that persist validated session truth."""

    def persist_evaluation_result(
        self,
        result: EvaluationResult,
        *,
        results_dir: Path,
        source_result_path: Path,
        summary: Optional[Mapping[str, Any]] = None,
        refresh_when_empty: bool = True,
    ) -> list[Path]:
        """Persist per-task validated session-truth artifacts from an evaluation result."""
        ...

    def persist_validation_result(
        self,
        *,
        benchmark: BenchmarkType,
        results_dir: Path,
        task_id: str,
        validation_result: Any,
        score: Any = None,
        source_result_path: Optional[Path] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        refresh_when_empty: bool = False,
    ) -> Optional[Path]:
        """Persist validated session-truth artifacts from validation output."""
        ...


def create_validated_session_truth_service(
    emitters: Optional[ValidatedSessionTruthEmitterRegistry] = None,
) -> ValidatedSessionTruthServiceProtocol:
    """Return the canonical validated session-truth service for evaluation flows."""
    return create_default_validated_session_truth_service(emitters)


def resolve_validated_session_truth_service(
    *,
    service: Optional[ValidatedSessionTruthServiceProtocol] = None,
    emitters: Optional[ValidatedSessionTruthEmitterRegistry] = None,
) -> ValidatedSessionTruthServiceProtocol:
    """Resolve the canonical validated session-truth service for evaluation runtimes."""
    return service or create_validated_session_truth_service(emitters)


__all__ = [
    "ValidatedSessionTruthServiceProtocol",
    "ValidatedSessionTruthService",
    "create_validated_session_truth_service",
    "resolve_validated_session_truth_service",
]
