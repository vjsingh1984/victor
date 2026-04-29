# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for removed compatibility shims.

Note: tool_compat, recovery_compat, prompt_compat, and state_compat were
removed as part of the service-first architecture migration. This test file
verifies that the removed modules are no longer importable.
"""

from __future__ import annotations

import pytest

# Removed compatibility shims - verify they cannot be imported
def test_prompt_compat_module_removed():
    """Verify that prompt_compat module has been removed."""
    with pytest.raises(ImportError, match="prompt_compat"):
        from victor.agent.services import prompt_compat  # noqa: F401


def test_state_compat_module_removed():
    """Verify that state_compat module has been removed."""
    with pytest.raises(ImportError, match="state_compat"):
        from victor.agent.services import state_compat  # noqa: F401


def test_recovery_compat_module_removed():
    """Verify that recovery_compat module has been removed."""
    with pytest.raises(ImportError, match="recovery_compat"):
        from victor.agent.services import recovery_compat  # noqa: F401


# Verify canonical imports still work
def test_canonical_prompt_runtime_imports():
    """Verify canonical prompt runtime imports work."""
    from victor.agent.services.prompt_runtime import (
        PromptRuntimeAdapter,
        PromptRuntimeConfig,
        PromptRuntimeContext,
    )

    assert PromptRuntimeAdapter is not None
    assert PromptRuntimeConfig is not None
    assert PromptRuntimeContext is not None


def test_canonical_state_runtime_imports():
    """Verify canonical state runtime imports work."""
    from victor.agent.services.state_runtime import StateRuntimeAdapter

    assert StateRuntimeAdapter is not None


def test_canonical_recovery_service_imports():
    """Verify canonical recovery service imports work."""
    from victor.agent.services.recovery_service import (
        RecoveryService,
        StreamingRecoveryContext,
    )

    assert RecoveryService is not None
    assert StreamingRecoveryContext is not None


# Verify deprecated names are no longer exported from services/__init__.py
def test_prompt_coordinator_not_exported_from_services():
    """Verify PromptCoordinator is no longer exported from services."""
    from victor.agent import services

    assert "PromptCoordinator" not in dir(services)
    assert "create_prompt_coordinator" not in dir(services)


def test_state_coordinator_not_exported_from_services():
    """Verify StateCoordinator is no longer exported from services."""
    from victor.agent import services

    assert "StateCoordinator" not in dir(services)
    assert "create_state_coordinator" not in dir(services)


# Verify deprecated names are no longer exported from coordinators/__init__.py
def test_state_coordinator_not_exported_from_coordinators():
    """Verify StateCoordinator is no longer exported from coordinators."""
    from victor.agent import coordinators

    # StateCoordinator should not be importable
    with pytest.raises(AttributeError, match="has no attribute"):
        coordinators.StateCoordinator()

    with pytest.raises(AttributeError, match="has no attribute"):
        coordinators.StateCoordinatorConfig()

    with pytest.raises(AttributeError, match="has no attribute"):
        coordinators.create_state_coordinator()
