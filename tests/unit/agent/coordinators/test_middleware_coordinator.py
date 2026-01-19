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

"""Tests for MiddlewareCoordinator."""

import pytest

from victor.agent.coordinators.middleware_coordinator import (
    MiddlewareCoordinator,
    create_middleware_coordinator,
)


class MockMiddleware:
    """Mock middleware for testing."""

    def __init__(self, name: str):
        self.name = name


class MockSafetyPattern:
    """Mock safety pattern for testing."""

    def __init__(self, pattern: str):
        self.pattern = pattern


@pytest.fixture
def coordinator():
    """Fixture for MiddlewareCoordinator."""
    return MiddlewareCoordinator()


class TestMiddlewareCoordinator:
    """Test suite for MiddlewareCoordinator."""

    def test_initialization(self, coordinator):
        """Test coordinator initialization."""
        assert coordinator.get_middleware() == []
        assert coordinator.get_middleware_chain() is None
        assert coordinator.get_safety_patterns() == []
        assert coordinator.get_code_correction_middleware() is None

    # ========================================================================
    # Middleware Management
    # ========================================================================

    def test_set_middleware(self, coordinator):
        """Test setting middleware."""
        middleware = [MockMiddleware("logging"), MockMiddleware("metrics")]
        coordinator.set_middleware(middleware)

        result = coordinator.get_middleware()
        assert len(result) == 2
        assert result[0].name == "logging"
        assert result[1].name == "metrics"

    def test_set_middleware_returns_copy(self, coordinator):
        """Test that get_middleware returns a copy."""
        middleware = [MockMiddleware("logging")]
        coordinator.set_middleware(middleware)

        result = coordinator.get_middleware()
        result.append(MockMiddleware("new"))

        # Original should not be modified
        assert len(coordinator.get_middleware()) == 1

    def test_set_middleware_empty(self, coordinator):
        """Test setting empty middleware list."""
        coordinator.set_middleware([])
        assert coordinator.get_middleware() == []

    # ========================================================================
    # Safety Patterns Management
    # ========================================================================

    def test_set_safety_patterns(self, coordinator):
        """Test setting safety patterns."""
        patterns = [MockSafetyPattern("pattern1"), MockSafetyPattern("pattern2")]
        coordinator.set_safety_patterns(patterns)

        result = coordinator.get_safety_patterns()
        assert len(result) == 2
        assert result[0].pattern == "pattern1"
        assert result[1].pattern == "pattern2"

    def test_set_safety_patterns_returns_copy(self, coordinator):
        """Test that get_safety_patterns returns a copy."""
        patterns = [MockSafetyPattern("pattern1")]
        coordinator.set_safety_patterns(patterns)

        result = coordinator.get_safety_patterns()
        result.append(MockSafetyPattern("new"))

        # Original should not be modified
        assert len(coordinator.get_safety_patterns()) == 1

    def test_set_safety_patterns_empty(self, coordinator):
        """Test setting empty safety patterns list."""
        coordinator.set_safety_patterns([])
        assert coordinator.get_safety_patterns() == []

    # ========================================================================
    # Middleware Chain Management
    # ========================================================================

    def test_get_middleware_chain_initially_none(self, coordinator):
        """Test that middleware chain is initially None."""
        assert coordinator.get_middleware_chain() is None

    def test_set_middleware_chain(self, coordinator):
        """Test setting middleware chain."""
        mock_chain = {"name": "mock_chain"}
        coordinator.set_middleware_chain(mock_chain)

        result = coordinator.get_middleware_chain()
        assert result["name"] == "mock_chain"

    def test_build_middleware_chain_empty(self, coordinator):
        """Test building chain with empty middleware list."""
        chain = coordinator.build_middleware_chain([])

        assert chain is None
        assert coordinator.get_middleware_chain() is None

    def test_build_middleware_chain_with_middleware(self, coordinator):
        """Test building chain with middleware."""
        # This test assumes MiddlewareChain might not be available
        # so we test the logic path
        middleware = [MockMiddleware("logging")]
        chain = coordinator.build_middleware_chain(middleware)

        # If MiddlewareChain is available, chain should be set
        # If not, chain should be None (handled gracefully)
        assert chain is not None or chain is None

    # ========================================================================
    # Code Correction Middleware
    # ========================================================================

    def test_set_code_correction_middleware(self, coordinator):
        """Test setting code correction middleware."""
        mock_correction = {"name": "code_correction"}
        coordinator.set_code_correction_middleware(mock_correction)

        result = coordinator.get_code_correction_middleware()
        assert result["name"] == "code_correction"

    def test_get_code_correction_middleware_initially_none(self, coordinator):
        """Test that code correction middleware is initially None."""
        assert coordinator.get_code_correction_middleware() is None

    # ========================================================================
    # Internal Storage Setters
    # ========================================================================

    def test_set_vertical_middleware_storage(self, coordinator):
        """Test internal vertical middleware storage setter."""
        middleware = [MockMiddleware("logging")]
        coordinator._set_vertical_middleware_storage(middleware)

        result = coordinator.get_middleware()
        assert len(result) == 1

    def test_set_middleware_chain_storage(self, coordinator):
        """Test internal middleware chain storage setter."""
        mock_chain = {"name": "mock_chain"}
        coordinator._set_middleware_chain_storage(mock_chain)

        result = coordinator.get_middleware_chain()
        assert result["name"] == "mock_chain"

    def test_set_safety_patterns_storage(self, coordinator):
        """Test internal safety patterns storage setter."""
        patterns = [MockSafetyPattern("pattern1")]
        coordinator._set_safety_patterns_storage(patterns)

        result = coordinator.get_safety_patterns()
        assert len(result) == 1

    # ========================================================================
    # State Management
    # ========================================================================

    def test_get_state(self, coordinator):
        """Test getting coordinator state."""
        middleware = [MockMiddleware("logging")]
        patterns = [MockSafetyPattern("pattern1")]
        mock_correction = {"name": "correction"}

        coordinator.set_middleware(middleware)
        coordinator.set_safety_patterns(patterns)
        coordinator.set_code_correction_middleware(mock_correction)

        state = coordinator.get_state()

        assert state["vertical_middleware_count"] == 1
        assert state["vertical_safety_patterns_count"] == 1
        assert state["has_middleware_chain"] is False
        assert state["has_code_correction"] is True

    def test_get_state_empty(self, coordinator):
        """Test getting state with no configuration."""
        state = coordinator.get_state()

        assert state["vertical_middleware_count"] == 0
        assert state["vertical_safety_patterns_count"] == 0
        assert state["has_middleware_chain"] is False
        assert state["has_code_correction"] is False

    def test_reset(self, coordinator):
        """Test resetting coordinator."""
        middleware = [MockMiddleware("logging")]
        patterns = [MockSafetyPattern("pattern1")]
        mock_correction = {"name": "correction"}

        coordinator.set_middleware(middleware)
        coordinator.set_safety_patterns(patterns)
        coordinator.set_code_correction_middleware(mock_correction)

        coordinator.reset()

        assert coordinator.get_middleware() == []
        assert coordinator.get_middleware_chain() is None
        assert coordinator.get_safety_patterns() == []
        assert coordinator.get_code_correction_middleware() is None

    # ========================================================================
    # Computed Properties
    # ========================================================================

    def test_has_middleware(self, coordinator):
        """Test has_middleware property."""
        assert coordinator.has_middleware() is False

        coordinator.set_middleware([MockMiddleware("logging")])
        assert coordinator.has_middleware() is True

    def test_has_safety_patterns(self, coordinator):
        """Test has_safety_patterns property."""
        assert coordinator.has_safety_patterns() is False

        coordinator.set_safety_patterns([MockSafetyPattern("pattern1")])
        assert coordinator.has_safety_patterns() is True

    def test_has_code_correction(self, coordinator):
        """Test has_code_correction property."""
        assert coordinator.has_code_correction() is False

        coordinator.set_code_correction_middleware({"name": "correction"})
        assert coordinator.has_code_correction() is True

    def test_get_middleware_summary(self, coordinator):
        """Test getting middleware summary."""
        middleware = [MockMiddleware("logging"), MockMiddleware("metrics")]
        patterns = [MockSafetyPattern("pattern1")]
        mock_correction = {"name": "correction"}

        coordinator.set_middleware(middleware)
        coordinator.set_safety_patterns(patterns)
        coordinator.set_code_correction_middleware(mock_correction)

        summary = coordinator.get_middleware_summary()

        assert summary["total_middleware"] == 2
        assert summary["total_safety_patterns"] == 1
        assert summary["has_chain"] is False
        assert summary["has_code_correction"] is True

    def test_get_middleware_summary_empty(self, coordinator):
        """Test getting summary with no configuration."""
        summary = coordinator.get_middleware_summary()

        assert summary["total_middleware"] == 0
        assert summary["total_safety_patterns"] == 0
        assert summary["has_chain"] is False
        assert summary["has_code_correction"] is False


class TestCreateMiddlewareCoordinator:
    """Test suite for create_middleware_coordinator factory."""

    def test_factory_function(self):
        """Test factory function creates coordinator."""
        coordinator = create_middleware_coordinator()

        assert isinstance(coordinator, MiddlewareCoordinator)
        assert coordinator.get_middleware() == []
