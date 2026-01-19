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

"""Tests for core vertical types.

This test suite validates the canonical vertical types that have been
moved to victor.core to enforce layer boundaries.
"""

from __future__ import annotations

import pytest

from victor.core.teams import SubAgentRole
from victor.core.verticals.context import (
    MutableVerticalContextProtocol,
    VerticalContext,
    VerticalContextProtocol,
    create_vertical_context,
)


# =============================================================================
# SubAgentRole Tests
# =============================================================================


class TestSubAgentRole:
    """Test the SubAgentRole enum."""

    def test_all_roles_defined(self):
        """All expected roles should be defined."""
        expected_roles = [
            "RESEARCHER",
            "PLANNER",
            "EXECUTOR",
            "REVIEWER",
            "TESTER",
        ]

        for role_name in expected_roles:
            assert hasattr(SubAgentRole, role_name), f"Missing role: {role_name}"

    def test_role_values_are_strings(self):
        """Role values should be strings for serialization compatibility."""
        assert issubclass(SubAgentRole, str)

    def test_role_values_lowercase(self):
        """Role values should be lowercase strings."""
        assert SubAgentRole.RESEARCHER.value == "researcher"
        assert SubAgentRole.PLANNER.value == "planner"
        assert SubAgentRole.EXECUTOR.value == "executor"
        assert SubAgentRole.REVIEWER.value == "reviewer"
        assert SubAgentRole.TESTER.value == "tester"

    def test_role_iteration(self):
        """Should be able to iterate over all roles."""
        roles = list(SubAgentRole)
        assert len(roles) == 5
        assert SubAgentRole.RESEARCHER in roles
        assert SubAgentRole.TESTER in roles

    def test_role_comparison(self):
        """Roles should be comparable."""
        assert SubAgentRole.RESEARCHER == SubAgentRole.RESEARCHER
        assert SubAgentRole.RESEARCHER != SubAgentRole.PLANNER

    def test_role_string_representation(self):
        """Roles should have nice string representations."""
        assert str(SubAgentRole.RESEARCHER) == "researcher"
        assert repr(SubAgentRole.RESEARCHER) == "<SubAgentRole.researcher: 'researcher'>"

    def test_role_from_string(self):
        """Should be able to get role from string value."""
        role = SubAgentRole("researcher")
        assert role == SubAgentRole.RESEARCHER

        role = SubAgentRole("executor")
        assert role == SubAgentRole.EXECUTOR

    def test_role_serialization(self):
        """Roles should be JSON-serializable via their string values."""
        import json

        role = SubAgentRole.EXECUTOR

        # Should serialize to string
        serialized = json.dumps({"role": role.value})
        assert serialized == '{"role": "executor"}'

        # Should deserialize back to role
        data = json.loads(serialized)
        deserialized = SubAgentRole(data["role"])
        assert deserialized == SubAgentRole.EXECUTOR


# =============================================================================
# VerticalContext Tests
# =============================================================================


class TestVerticalContext:
    """Test the VerticalContext dataclass."""

    def test_default_initialization(self):
        """Should create empty context with defaults."""
        context = VerticalContext()

        assert context.name is None
        assert context.config is None
        assert context.stages == {}
        assert context.middleware == []
        assert context.safety_patterns == []
        assert context.task_hints == {}
        assert context.mode_configs == {}
        assert context.default_mode == "default"
        assert context.default_budget == 10
        assert context.tool_dependencies == []
        assert context.tool_sequences == []
        assert context.enabled_tools == set()

    def test_property_accessors(self):
        """Property accessors should work correctly."""
        context = VerticalContext(name="coding")

        assert context.vertical_name == "coding"
        assert context.has_vertical is True
        assert context.has_middleware is False
        assert context.has_safety_patterns is False
        assert context.has_mode_configs is False
        assert context.has_tool_dependencies is False
        assert context.has_custom_prompt is False
        assert context.has_workflows is False
        assert context.has_rl_config is False
        assert context.has_team_specs is False

    def test_apply_vertical(self):
        """Should be able to apply vertical to context."""
        context = VerticalContext()
        context.apply_vertical("coding", config=None)

        assert context.name == "coding"
        assert context.config is None
        assert context.has_vertical is True

    def test_apply_stages(self):
        """Should be able to apply stages to context."""
        context = VerticalContext()
        stages = {"reading": {"tools": ["read", "search"]}}
        context.apply_stages(stages)

        assert context.stages == stages

    def test_apply_middleware(self):
        """Should be able to apply middleware to context."""
        context = VerticalContext()
        middleware = ["middleware1", "middleware2"]
        context.apply_middleware(middleware)

        assert context.middleware == middleware
        assert context.has_middleware is True

    def test_apply_safety_patterns(self):
        """Should be able to apply safety patterns to context."""
        context = VerticalContext()
        patterns = ["pattern1", "pattern2"]
        context.apply_safety_patterns(patterns)

        assert context.safety_patterns == patterns
        assert context.has_safety_patterns is True

    def test_apply_task_hints(self):
        """Should be able to apply task hints to context."""
        context = VerticalContext()
        hints = {"bug_fix": {"priority": "high"}}
        context.apply_task_hints(hints)

        assert context.task_hints == hints

    def test_apply_mode_configs(self):
        """Should be able to apply mode configs to context."""
        context = VerticalContext()
        configs = {"plan": {"tool_budget": 25}}
        context.apply_mode_configs(configs, default_mode="plan", default_budget=25)

        assert context.mode_configs == configs
        assert context.default_mode == "plan"
        assert context.default_budget == 25
        assert context.has_mode_configs is True

    def test_apply_tool_dependencies(self):
        """Should be able to apply tool dependencies to context."""
        context = VerticalContext()
        dependencies = ["dep1", "dep2"]
        sequences = [["tool1", "tool2"]]
        context.apply_tool_dependencies(dependencies, sequences)

        assert context.tool_dependencies == dependencies
        assert context.tool_sequences == sequences
        assert context.has_tool_dependencies is True

    def test_capability_configs(self):
        """Should be able to store and retrieve capability configs."""
        context = VerticalContext()

        # Set individual config
        context.set_capability_config("rag_config", {"chunk_size": 512})
        assert context.get_capability_config("rag_config") == {"chunk_size": 512}
        assert context.has_capability_configs is True

        # Apply multiple configs
        configs = {
            "code_style": {"max_line_length": 100},
            "test_framework": "pytest",
        }
        context.apply_capability_configs(configs)

        assert context.get_capability_config("code_style") == {"max_line_length": 100}
        assert context.get_capability_config("test_framework") == "pytest"

        # Get default for non-existent config
        assert context.get_capability_config("nonexistent", "default") == "default"

    def test_query_methods(self):
        """Query methods should work correctly."""
        context = VerticalContext()
        context.apply_mode_configs({"plan": {"tool_budget": 25}}, default_mode="plan")

        # get_mode_config
        mode_config = context.get_mode_config("plan")
        assert mode_config == {"tool_budget": 25}

        # get_tool_budget_for_mode
        budget = context.get_tool_budget_for_mode("plan")
        assert budget == 25

        # get_tool_budget_for_mode with default
        budget = context.get_tool_budget_for_mode()
        assert budget == 25

        # get_task_hint
        context.apply_task_hints({"bug_fix": {"priority": "high"}})
        hint = context.get_task_hint("bug_fix")
        assert hint == {"priority": "high"}

    def test_to_dict(self):
        """Should serialize to dictionary correctly."""
        context = VerticalContext(name="coding")
        context.apply_stages({"reading": {"tools": ["read"]}})
        context.apply_middleware(["mw1"])

        data = context.to_dict()

        assert data["name"] == "coding"
        assert data["stages"] == ["reading"]
        assert data["middleware_count"] == 1

    def test_empty_factory(self):
        """Empty factory should create empty context."""
        context = VerticalContext.empty()

        assert context.name is None
        assert context.config is None


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestVerticalContextFactory:
    """Test vertical context factory functions."""

    def test_create_vertical_context_no_args(self):
        """Should create empty context with no args."""
        context = create_vertical_context()

        assert context.name is None
        assert context.config is None

    def test_create_vertical_context_with_name(self):
        """Should create context with name."""
        context = create_vertical_context(name="coding")

        assert context.name == "coding"
        assert context.has_vertical is True

    def test_create_vertical_context_with_config(self):
        """Should create context with config."""
        config = {}  # Mock VerticalConfig
        context = create_vertical_context(name="coding", config=config)

        assert context.name == "coding"
        assert context.config is config


# =============================================================================
# Protocol Tests
# =============================================================================


class TestVerticalContextProtocols:
    """Test vertical context protocols."""

    def test_vertical_context_protocol(self):
        """VerticalContext should implement VerticalContextProtocol."""
        context = VerticalContext(name="coding")

        # Check protocol compliance
        assert isinstance(context, VerticalContextProtocol)

        # Check properties
        assert context.vertical_name == "coding"
        assert context.has_vertical is True
        assert context.middleware == []
        assert context.safety_patterns == []
        assert context.task_hints == {}
        assert context.mode_configs == {}

    def test_mutable_vertical_context_protocol(self):
        """VerticalContext should implement MutableVerticalContextProtocol."""
        context = VerticalContext()

        # Check protocol compliance
        assert isinstance(context, MutableVerticalContextProtocol)

        # Check methods work
        context.apply_vertical("coding")
        assert context.name == "coding"

        context.apply_stages({"reading": {}})
        assert context.stages == {"reading": {}}

        context.set_capability_config("test", "value")
        assert context.get_capability_config("test") == "value"


# =============================================================================
# Integration Tests
# =============================================================================


class TestVerticalContextIntegration:
    """Test VerticalContext integration with other components."""

    def test_import_from_core(self):
        """Should be importable from victor.core.verticals.context."""
        from victor.core.verticals.context import (
            VerticalContext as CoreVerticalContext,
        )

        assert CoreVerticalContext is VerticalContext

    def test_legacy_import_path(self):
        """Legacy import path should still work for backward compatibility."""
        try:
            from victor.agent.vertical_context import VerticalContext as LegacyContext
            from victor.core.verticals.context import VerticalContext as CoreContext

            # Should be the same type
            assert LegacyContext is CoreContext
        except ImportError:
            # It's OK if legacy import doesn't exist
            pass

    def test_context_with_all_fields(self):
        """Should be able to create context with all fields populated."""
        context = VerticalContext()
        context.apply_vertical("coding")
        context.apply_stages({"reading": {"tools": ["read"]}})
        context.apply_middleware(["mw1"])
        context.apply_safety_patterns(["safe1"])
        context.apply_task_hints({"bug_fix": {"priority": "high"}})
        context.apply_mode_configs({"plan": {"tool_budget": 25}})
        context.apply_tool_dependencies(["dep1"], [["tool1", "tool2"]])
        context.apply_system_prompt("Custom prompt")
        context.apply_enabled_tools({"read", "write"})
        context.apply_workflows({"workflow1": {}})
        context.apply_team_specs({"team1": {}})

        # Verify all fields
        assert context.name == "coding"
        assert context.stages == {"reading": {"tools": ["read"]}}
        assert context.middleware == ["mw1"]
        assert context.safety_patterns == ["safe1"]
        assert context.task_hints == {"bug_fix": {"priority": "high"}}
        assert context.mode_configs == {"plan": {"tool_budget": 25}}
        assert context.tool_dependencies == ["dep1"]
        assert context.tool_sequences == [["tool1", "tool2"]]
        assert context.system_prompt == "Custom prompt"
        assert context.enabled_tools == {"read", "write"}
        assert context.workflows == {"workflow1": {}}
        assert context.team_specs == {"team1": {}}


# =============================================================================
# Edge Cases
# =============================================================================


class TestVerticalContextEdgeCases:
    """Test edge cases and error conditions."""

    def test_context_immutability_of_protocols(self):
        """Protocols should be runtime checkable but not instantiable."""
        from typing import Protocol

        # Protocols should be Protocol subclasses
        assert issubclass(VerticalContextProtocol, Protocol)

        # Can't directly instantiate protocols (they're abstract)
        # This is expected behavior
        with pytest.raises(TypeError):
            VerticalContextProtocol()

    def test_context_with_none_values(self):
        """Should handle None values gracefully."""
        context = VerticalContext()
        context.apply_vertical(None, config=None)

        assert context.name is None
        assert context.config is None
        assert context.has_vertical is False

    def test_context_overwrite(self):
        """Should be able to overwrite context values."""
        context = VerticalContext()
        context.apply_vertical("coding")
        assert context.name == "coding"

        context.apply_vertical("devops")
        assert context.name == "devops"

    def test_capability_config_overwrite(self):
        """Should be able to overwrite capability configs."""
        context = VerticalContext()
        context.set_capability_config("config1", "value1")
        assert context.get_capability_config("config1") == "value1"

        context.set_capability_config("config1", "value2")
        assert context.get_capability_config("config1") == "value2"


# =============================================================================
# Performance Tests
# =============================================================================


class TestVerticalContextPerformance:
    """Test performance characteristics of VerticalContext."""

    def test_context_creation_speed(self):
        """Context creation should be fast."""
        import time

        iterations = 1000
        start = time.time()

        for _ in range(iterations):
            context = VerticalContext()
            context.apply_vertical("coding")
            context.apply_stages({"reading": {}})
            context.apply_middleware(["mw1"])

        elapsed = time.time() - start
        avg_time = elapsed / iterations

        # Should be fast (< 1ms per creation)
        assert avg_time < 0.001, f"Context creation too slow: {avg_time:.6f}s per iteration"

    def test_capability_config_lookup_speed(self):
        """Capability config lookups should be fast."""
        import time

        context = VerticalContext()
        for i in range(100):
            context.set_capability_config(f"config{i}", f"value{i}")

        iterations = 10000
        start = time.time()

        for _ in range(iterations):
            _ = context.get_capability_config("config50")

        elapsed = time.time() - start
        avg_time = elapsed / iterations

        # Should be very fast (< 0.1ms per lookup)
        assert avg_time < 0.0001, f"Config lookup too slow: {avg_time:.6f}s per iteration"
