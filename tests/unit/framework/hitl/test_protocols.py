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

"""Tests for HITL protocol definitions."""

import pytest

from victor.framework.hitl.protocols import (
    BaseHITLGate,
    BaseHITLResponse,
    FallbackBehavior,
    FallbackStrategy,
)


# =============================================================================
# FallbackBehavior Tests
# =============================================================================


class TestFallbackBehavior:
    """Tests for FallbackBehavior enum."""

    def test_abort_value(self):
        """FallbackBehavior.ABORT should have correct value."""
        assert FallbackBehavior.ABORT.value == "abort"

    def test_continue_value(self):
        """FallbackBehavior.CONTINUE should have correct value."""
        assert FallbackBehavior.CONTINUE.value == "continue"

    def test_skip_value(self):
        """FallbackBehavior.SKIP should have correct value."""
        assert FallbackBehavior.SKIP.value == "skip"

    def test_retry_value(self):
        """FallbackBehavior.RETRY should have correct value."""
        assert FallbackBehavior.RETRY.value == "retry"


# =============================================================================
# FallbackStrategy Tests
# =============================================================================


class TestFallbackStrategy:
    """Tests for FallbackStrategy."""

    def test_default_strategy_is_abort(self):
        """Default strategy should be abort."""
        strategy = FallbackStrategy()

        assert strategy.behavior == FallbackBehavior.ABORT
        assert strategy.default_value is None

    def test_abort_factory(self):
        """abort() factory should create abort strategy."""
        strategy = FallbackStrategy.abort()

        assert strategy.behavior == FallbackBehavior.ABORT

    def test_continue_with_default_factory(self):
        """continue_with_default() factory should create continue strategy."""
        strategy = FallbackStrategy.continue_with_default("default_value")

        assert strategy.behavior == FallbackBehavior.CONTINUE
        assert strategy.default_value == "default_value"

    def test_skip_factory(self):
        """skip() factory should create skip strategy."""
        strategy = FallbackStrategy.skip()

        assert strategy.behavior == FallbackBehavior.SKIP

    def test_retry_factory(self):
        """retry() factory should create retry strategy."""
        strategy = FallbackStrategy.retry(max_retries=5, delay=2.0)

        assert strategy.behavior == FallbackBehavior.RETRY
        assert strategy.max_retries == 5
        assert strategy.retry_delay == 2.0

    def test_retry_factory_defaults(self):
        """retry() factory should use defaults."""
        strategy = FallbackStrategy.retry()

        assert strategy.max_retries == 3
        assert strategy.retry_delay == 5.0


# =============================================================================
# BaseHITLResponse Tests
# =============================================================================


class TestBaseHITLResponse:
    """Tests for BaseHITLResponse."""

    def test_response_initialization(self):
        """BaseHITLResponse should initialize correctly."""
        response = BaseHITLResponse(
            gate_id="test_gate",
            approved=True,
            value="test_value",
            reason="Test reason",
            responder="test_user",
        )

        assert response.gate_id == "test_gate"
        assert response.approved is True
        assert response.value == "test_value"
        assert response.reason == "Test reason"
        assert response.responder == "test_user"
        assert response.created_at > 0
        assert response.metadata == {}

    def test_is_approved_property(self):
        """is_approved should return approved status."""
        approved_response = BaseHITLResponse(gate_id="test", approved=True)
        rejected_response = BaseHITLResponse(gate_id="test", approved=False)

        assert approved_response.is_approved is True
        assert rejected_response.is_approved is False

    def test_is_rejected_property(self):
        """is_rejected should return opposite of approved."""
        approved_response = BaseHITLResponse(gate_id="test", approved=True)
        rejected_response = BaseHITLResponse(gate_id="test", approved=False)

        assert approved_response.is_rejected is False
        assert rejected_response.is_rejected is True

    def test_is_timeout_property_default(self):
        """is_timeout should return False by default."""
        response = BaseHITLResponse(gate_id="test", approved=True)

        assert response.is_timeout is False

    def test_is_timeout_property_when_set(self):
        """is_timeout should return True when metadata indicates."""
        response = BaseHITLResponse(
            gate_id="test",
            approved=False,
            metadata={"timed_out": True},
        )

        assert response.is_timeout is True

    def test_is_skipped_property_default(self):
        """is_skipped should return False by default."""
        response = BaseHITLResponse(gate_id="test", approved=True)

        assert response.is_skipped is False

    def test_is_skipped_property_when_set(self):
        """is_skipped should return True when metadata indicates."""
        response = BaseHITLResponse(
            gate_id="test",
            approved=False,
            metadata={"skipped": True},
        )

        assert response.is_skipped is True

    def test_to_dict(self):
        """to_dict should serialize to dictionary."""
        response = BaseHITLResponse(
            gate_id="test_gate",
            approved=True,
            value="test_value",
            reason="Test reason",
            responder="test_user",
            metadata={"key": "value"},
        )

        data = response.to_dict()

        assert data["gate_id"] == "test_gate"
        assert data["approved"] is True
        assert data["value"] == "test_value"
        assert data["reason"] == "Test reason"
        assert data["responder"] == "test_user"
        assert data["metadata"] == {"key": "value"}
        assert "created_at" in data


# =============================================================================
# BaseHITLGate Tests
# =============================================================================


class TestBaseHITLGate:
    """Tests for BaseHITLGate."""

    def test_gate_initialization(self):
        """BaseHITLGate should initialize correctly."""
        gate = BaseHITLGate(
            _gate_id="test_gate",
            gate_type="test_type",
            title="Test Title",
            prompt="Test prompt",
        )

        assert gate.gate_id == "test_gate"
        assert gate.gate_type == "test_type"
        assert gate.title == "Test Title"
        assert gate.prompt == "Test prompt"
        assert gate.timeout_seconds == 300.0
        assert gate.fallback_strategy.behavior == FallbackBehavior.ABORT
        assert gate.is_required is True
        assert gate.context == {}
        assert gate.validator is None

    def test_gate_with_custom_timeout(self):
        """Gate should accept custom timeout."""
        gate = BaseHITLGate(
            _gate_id="test",
            gate_type="test",
            title="Test",
            prompt="Test",
            timeout_seconds=60.0,
        )

        assert gate.timeout_seconds == 60.0

    def test_gate_with_context(self):
        """Gate should accept context."""
        gate = BaseHITLGate(
            _gate_id="test",
            gate_type="test",
            title="Test",
            prompt="Test",
            context={"key": "value"},
        )

        assert gate.context == {"key": "value"}

    def test_gate_with_custom_fallback(self):
        """Gate should accept custom fallback strategy."""
        strategy = FallbackStrategy.skip()
        gate = BaseHITLGate(
            _gate_id="test",
            gate_type="test",
            title="Test",
            prompt="Test",
            fallback_strategy=strategy,
        )

        assert gate.fallback_strategy.behavior == FallbackBehavior.SKIP

    def test_gate_optional(self):
        """Gate can be marked as not required."""
        gate = BaseHITLGate(
            _gate_id="test",
            gate_type="test",
            title="Test",
            prompt="Test",
            required=False,
        )

        assert gate.is_required is False

    def test_with_context_creates_new_gate(self):
        """with_context should create new gate with merged context."""
        gate = BaseHITLGate(
            _gate_id="test",
            gate_type="test",
            title="Test",
            prompt="Test",
            context={"a": 1},
        )

        new_gate = gate.with_context({"b": 2})

        # The returned gate should be a BaseHITLGate
        assert isinstance(new_gate, BaseHITLGate)
        assert new_gate.context == {"a": 1, "b": 2}
        # Original should be unchanged
        assert gate.context == {"a": 1}

    def test_with_timeout_creates_new_gate(self):
        """with_timeout should create new gate with custom timeout."""
        gate = BaseHITLGate(
            _gate_id="test",
            gate_type="test",
            title="Test",
            prompt="Test",
        )

        new_gate = gate.with_timeout(120)

        assert new_gate.timeout_seconds == 120
        assert gate.timeout_seconds == 300.0

    def test_with_fallback_creates_new_gate(self):
        """with_fallback should create new gate with custom fallback."""
        gate = BaseHITLGate(
            _gate_id="test",
            gate_type="test",
            title="Test",
            prompt="Test",
        )

        strategy = FallbackStrategy.continue_with_default("default")
        new_gate = gate.with_fallback(strategy)

        assert new_gate.fallback_strategy == strategy
        assert gate.fallback_strategy.behavior == FallbackBehavior.ABORT

    def test_render_prompt_without_variables(self):
        """_render_prompt should return prompt as-is with no variables."""
        gate = BaseHITLGate(
            _gate_id="test",
            gate_type="test",
            title="Test",
            prompt="Simple prompt with no variables",
        )

        rendered = gate._render_prompt()

        assert rendered == "Simple prompt with no variables"

    def test_render_prompt_with_context_variables(self):
        """_render_prompt should substitute variables from context."""
        gate = BaseHITLGate(
            _gate_id="test",
            gate_type="test",
            title="Test",
            prompt="Hello $name, welcome to $place",
            context={"name": "Alice", "place": "Wonderland"},
        )

        rendered = gate._render_prompt()

        assert rendered == "Hello Alice, welcome to Wonderland"

    def test_render_prompt_with_additional_context(self):
        """_render_prompt should merge additional context."""
        gate = BaseHITLGate(
            _gate_id="test",
            gate_type="test",
            title="Test",
            prompt="Action: $action on $target",
            context={"action": "deploy"},
        )

        rendered = gate._render_prompt({"target": "production"})

        assert rendered == "Action: deploy on production"

    def test_render_prompt_safe_substitution(self):
        """_render_prompt should use safe_substitution."""
        gate = BaseHITLGate(
            _gate_id="test",
            gate_type="test",
            title="Test",
            prompt="Hello $name, $unknown_var",
            context={"name": "Alice"},
        )

        rendered = gate._render_prompt()

        # Safe substitution leaves unknown variables
        assert "Hello Alice" in rendered
        assert "$unknown_var" in rendered
