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

"""Tests for lazy runtime initialization in AgentOrchestrator.

These tests validate that P4 decomposition properly defers runtime
component initialization until first access, ensuring faster startup.
"""

import pytest

from victor.framework import Agent


class TestRuntimeLazyInitialization:
    """Tests for lazy initialization of runtime components."""

    @pytest.mark.asyncio
    async def test_coordination_runtime_components_are_lazy(self):
        """Test that coordination runtime components are NOT initialized during Agent.create()."""
        agent = await Agent.create(
            provider="ollama",
            model="qwen3-coder:30b",
            enable_observability=False,
        )

        try:
            orchestrator = agent.get_orchestrator()

            # Access coordination runtime
            coordination_runtime = getattr(orchestrator, "_coordination_runtime", None)
            assert coordination_runtime is not None, "coordination_runtime should exist"

            # Check that components are NOT initialized (lazy)
            recovery_coordinator = getattr(coordination_runtime, "recovery_coordinator", None)
            chunk_generator = getattr(coordination_runtime, "chunk_generator", None)
            tool_planner = getattr(coordination_runtime, "tool_planner", None)
            task_coordinator = getattr(coordination_runtime, "task_coordinator", None)

            # All should be LazyRuntimeProxy instances
            for component_name, component in [
                ("recovery_coordinator", recovery_coordinator),
                ("chunk_generator", chunk_generator),
                ("tool_planner", tool_planner),
                ("task_coordinator", task_coordinator),
            ]:
                assert component is not None, f"{component_name} should exist"
                # Check initialized property - should be False
                initialized = getattr(component, "initialized", False)
                assert (
                    not initialized
                ), f"{component_name} should NOT be initialized after Agent.create()"

        finally:
            await agent.close()

    @pytest.mark.asyncio
    async def test_interaction_runtime_components_are_lazy(self):
        """Test that interaction runtime components are NOT initialized during Agent.create()."""
        agent = await Agent.create(
            provider="ollama",
            model="qwen3-coder:30b",
            enable_observability=False,
        )

        try:
            orchestrator = agent.get_orchestrator()

            # Access interaction runtime
            interaction_runtime = getattr(orchestrator, "_interaction_runtime", None)
            assert interaction_runtime is not None, "interaction_runtime should exist"

            # Check that components are NOT initialized (lazy)
            chat_coordinator = getattr(interaction_runtime, "chat_coordinator", None)
            tool_coordinator = getattr(interaction_runtime, "tool_coordinator", None)
            session_coordinator = getattr(interaction_runtime, "session_coordinator", None)

            # All should be LazyRuntimeProxy instances
            for component_name, component in [
                ("chat_coordinator", chat_coordinator),
                ("tool_coordinator", tool_coordinator),
                ("session_coordinator", session_coordinator),
            ]:
                assert component is not None, f"{component_name} should exist"
                # Check initialized property - should be False
                initialized = getattr(component, "initialized", False)
                assert (
                    not initialized
                ), f"{component_name} should NOT be initialized after Agent.create()"

        finally:
            await agent.close()

    @pytest.mark.asyncio
    async def test_resilience_runtime_components_are_lazy(self):
        """Test that resilience runtime components are NOT initialized during Agent.create()."""
        agent = await Agent.create(
            provider="ollama",
            model="qwen3-coder:30b",
            enable_observability=False,
        )

        try:
            orchestrator = agent.get_orchestrator()
            resilience_runtime = getattr(orchestrator, "_resilience_runtime", None)
            assert resilience_runtime is not None, "resilience_runtime should exist"

            recovery_handler = getattr(resilience_runtime, "recovery_handler", None)
            recovery_integration = getattr(resilience_runtime, "recovery_integration", None)

            for component_name, component in [
                ("recovery_handler", recovery_handler),
                ("recovery_integration", recovery_integration),
            ]:
                assert component is not None, f"{component_name} should exist"
                initialized = getattr(component, "initialized", False)
                assert (
                    not initialized
                ), f"{component_name} should NOT be initialized after Agent.create()"

        finally:
            await agent.close()

    @pytest.mark.asyncio
    async def test_runtime_components_initialize_on_first_access(self):
        """Test that runtime components DO initialize when first accessed via get_instance()."""
        agent = await Agent.create(
            provider="ollama",
            model="qwen3-coder:30b",
            enable_observability=False,
        )

        try:
            orchestrator = agent.get_orchestrator()

            # Access coordination runtime
            coordination_runtime = getattr(orchestrator, "_coordination_runtime", None)
            assert coordination_runtime is not None

            # Get the LazyRuntimeProxy for recovery_coordinator
            recovery_coordinator = getattr(coordination_runtime, "recovery_coordinator", None)
            assert recovery_coordinator is not None

            # Verify it's NOT initialized initially
            assert (
                not recovery_coordinator.initialized
            ), "recovery_coordinator should NOT be initialized initially"

            # Call get_instance() to trigger lazy initialization
            _ = recovery_coordinator.get_instance()

            # After calling get_instance(), the component should be initialized
            assert (
                recovery_coordinator.initialized
            ), "recovery_coordinator should be initialized after get_instance() call"

        finally:
            await agent.close()

    @pytest.mark.asyncio
    async def test_provider_runtime_components_are_lazy(self):
        """Test that provider runtime components are NOT initialized during Agent.create()."""
        agent = await Agent.create(
            provider="ollama",
            model="qwen3-coder:30b",
            enable_observability=False,
        )

        try:
            orchestrator = agent.get_orchestrator()

            # Access provider runtime
            provider_runtime = getattr(orchestrator, "_provider_runtime", None)
            assert provider_runtime is not None, "provider_runtime should exist"

            # Check that components are NOT initialized (lazy)
            provider_coordinator = getattr(provider_runtime, "provider_coordinator", None)
            provider_switch_coordinator = getattr(
                provider_runtime, "provider_switch_coordinator", None
            )

            for component_name, component in [
                ("provider_coordinator", provider_coordinator),
                ("provider_switch_coordinator", provider_switch_coordinator),
            ]:
                assert component is not None, f"{component_name} should exist"
                initialized = getattr(component, "initialized", False)
                assert (
                    not initialized
                ), f"{component_name} should NOT be initialized after Agent.create()"

        finally:
            await agent.close()

    @pytest.mark.asyncio
    async def test_workflow_runtime_components_are_lazy(self):
        """Test that workflow runtime components are NOT initialized during Agent.create()."""
        agent = await Agent.create(
            provider="ollama",
            model="qwen3-coder:30b",
            enable_observability=False,
        )

        try:
            orchestrator = agent.get_orchestrator()

            # Access workflow runtime
            workflow_runtime = getattr(orchestrator, "_workflow_runtime", None)
            assert workflow_runtime is not None, "workflow_runtime should exist"

            # Check that components are NOT initialized (lazy)
            workflow_registry = getattr(workflow_runtime, "workflow_registry", None)

            assert workflow_registry is not None, "workflow_registry should exist"
            initialized = getattr(workflow_registry, "initialized", False)
            assert (
                not initialized
            ), "workflow_registry should NOT be initialized after Agent.create()"

        finally:
            await agent.close()

    @pytest.mark.asyncio
    async def test_metrics_runtime_components_are_lazy(self):
        """Test that metrics runtime components are NOT initialized during Agent.create()."""
        agent = await Agent.create(
            provider="ollama",
            model="qwen3-coder:30b",
            enable_observability=False,
        )

        try:
            orchestrator = agent.get_orchestrator()

            # Access metrics runtime
            metrics_runtime = getattr(orchestrator, "_metrics_runtime", None)
            assert metrics_runtime is not None, "metrics_runtime should exist"

            # Check that components are NOT initialized (lazy)
            metrics_collector = getattr(metrics_runtime, "metrics_collector", None)

            assert metrics_collector is not None, "metrics_collector should exist"
            initialized = getattr(metrics_collector, "initialized", False)
            assert (
                not initialized
            ), "metrics_collector should NOT be initialized after Agent.create()"

        finally:
            await agent.close()

    @pytest.mark.asyncio
    async def test_all_runtime_runtimes_exist(self):
        """Test that all runtime boundaries exist and have the expected structure."""
        agent = await Agent.create(
            provider="ollama",
            model="qwen3-coder:30b",
            enable_observability=False,
        )

        try:
            orchestrator = agent.get_orchestrator()

            # All runtimes should exist
            assert hasattr(orchestrator, "_provider_runtime"), "provider_runtime missing"
            assert hasattr(orchestrator, "_memory_runtime"), "memory_runtime missing"
            assert hasattr(orchestrator, "_metrics_runtime"), "metrics_runtime missing"
            assert hasattr(orchestrator, "_workflow_runtime"), "workflow_runtime missing"
            assert hasattr(orchestrator, "_coordination_runtime"), "coordination_runtime missing"
            assert hasattr(orchestrator, "_interaction_runtime"), "interaction_runtime missing"
            assert hasattr(orchestrator, "_resilience_runtime"), "resilience_runtime missing"

        finally:
            await agent.close()
