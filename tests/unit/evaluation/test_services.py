# Copyright 2026 Vijaykumar Singh <singhvijay@users.noreply.github.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for canonical evaluation service exports."""

from victor.evaluation.services import (
    ValidatedSessionTruthServiceProtocol,
    ValidatedSessionTruthService as exported_service,
    create_validated_session_truth_service,
)
from victor.evaluation.validated_session_truth_emitters import (
    ValidatedSessionTruthEmitterRegistry,
)
from victor.evaluation.validated_session_truth_service import (
    ValidatedSessionTruthService as concrete_service,
)


def test_evaluation_services_reexport_validated_session_truth_service():
    """Evaluation services should expose the canonical validated-session service."""
    assert exported_service is concrete_service


def test_create_validated_session_truth_service_uses_supplied_registry():
    registry = ValidatedSessionTruthEmitterRegistry()

    service = create_validated_session_truth_service(registry)

    assert isinstance(service, concrete_service)
    assert service._emitters is registry


def test_validated_session_truth_service_protocol_accepts_duck_typed_service():
    class StubService:
        def persist_evaluation_result(self, result, **kwargs):
            return []

        def persist_validation_result(self, **kwargs):
            return None

    assert isinstance(StubService(), ValidatedSessionTruthServiceProtocol)
