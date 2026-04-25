from __future__ import annotations

# Copyright 2026 Vijaykumar Singh <singhvijay@users.noreply.github.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Canonical evaluation-level service factories and exports."""

from typing import Optional

from victor.evaluation.validated_session_truth_emitters import (
    ValidatedSessionTruthEmitterRegistry,
)
from victor.evaluation.validated_session_truth_service import (
    ValidatedSessionTruthService,
    create_default_validated_session_truth_service,
)


def create_validated_session_truth_service(
    emitters: Optional[ValidatedSessionTruthEmitterRegistry] = None,
) -> ValidatedSessionTruthService:
    """Return the canonical validated session-truth service for evaluation flows."""
    return create_default_validated_session_truth_service(emitters)


__all__ = [
    "ValidatedSessionTruthService",
    "create_validated_session_truth_service",
]
