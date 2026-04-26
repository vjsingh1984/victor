# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Deprecation coverage for public compatibility shims."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock

import pytest

from victor.agent.services.prompt_compat import (
    PromptCoordinator,
    create_prompt_coordinator,
)
from victor.agent.services.recovery_compat import StreamingRecoveryCoordinator
from victor.agent.services.session_compat import (
    SessionCoordinator,
    create_session_coordinator,
)
from victor.agent.services.state_compat import (
    StateCoordinator,
    create_state_coordinator,
)
from victor.agent.services.tool_compat import ToolCoordinator


def test_tool_coordinator_warns_with_tool_service_target():
    with pytest.warns(DeprecationWarning, match="ToolService"):
        ToolCoordinator(
            tool_pipeline=MagicMock(),
            tool_registry=MagicMock(),
        )


def test_tool_coordinator_can_suppress_init_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ToolCoordinator(
            tool_pipeline=MagicMock(),
            tool_registry=MagicMock(),
            warn_on_init=False,
        )

    assert not caught


def test_session_coordinator_warns_with_session_service_target():
    with pytest.warns(DeprecationWarning, match="SessionService"):
        SessionCoordinator(session_state_manager=MagicMock())


def test_create_session_coordinator_warns_with_session_service_target():
    with pytest.warns(DeprecationWarning, match="SessionService"):
        create_session_coordinator(session_state_manager=MagicMock())


def test_create_session_coordinator_can_suppress_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        create_session_coordinator(
            session_state_manager=MagicMock(),
            warn_on_init=False,
        )

    assert not caught


def test_prompt_coordinator_warns_with_unified_prompt_pipeline_target():
    with pytest.warns(DeprecationWarning, match="UnifiedPromptPipeline"):
        PromptCoordinator()


def test_create_prompt_coordinator_warns_with_unified_prompt_pipeline_target():
    with pytest.warns(DeprecationWarning, match="UnifiedPromptPipeline"):
        create_prompt_coordinator()


def test_create_prompt_coordinator_can_suppress_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        create_prompt_coordinator(warn_on_init=False)

    assert not caught


def test_state_coordinator_warns_with_live_and_persisted_targets():
    with pytest.warns(DeprecationWarning, match="ConversationController"):
        StateCoordinator(conversation_controller=MagicMock())


def test_create_state_coordinator_warns_with_state_targets():
    with pytest.warns(DeprecationWarning, match="StateService"):
        create_state_coordinator(conversation_controller=MagicMock())


def test_create_state_coordinator_can_suppress_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        create_state_coordinator(
            conversation_controller=MagicMock(),
            warn_on_init=False,
        )

    assert not caught


def test_streaming_recovery_coordinator_warns_with_recovery_service_target():
    with pytest.warns(DeprecationWarning, match="RecoveryService"):
        StreamingRecoveryCoordinator(
            recovery_handler=None,
            recovery_integration=None,
            streaming_handler=MagicMock(),
            context_compactor=None,
            unified_tracker=MagicMock(),
            settings=MagicMock(),
        )


def test_streaming_recovery_coordinator_can_suppress_init_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        StreamingRecoveryCoordinator(
            recovery_handler=None,
            recovery_integration=None,
            streaming_handler=MagicMock(),
            context_compactor=None,
            unified_tracker=MagicMock(),
            settings=MagicMock(),
            warn_on_init=False,
        )

    assert not caught
