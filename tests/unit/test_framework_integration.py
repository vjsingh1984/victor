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

"""Integration tests for the Victor Framework with Verticals and Observability.

Tests the complete integration of:
1. Framework API (Agent.create, vertical support)
2. Verticals (CodingAssistant, custom verticals)
3. Observability (EventBus, ObservabilityIntegration)
4. State management (ConversationStateMachine, StateHookManager)

Design Patterns Tested:
- Factory Method: Agent.create() with vertical parameter
- Template Method: VerticalBase configuration assembly
- Observer Pattern: EventBus subscriptions
- Mediator Pattern: ObservabilityIntegration wiring
"""

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.framework.tools import ToolSet
from victor.observability import (
    EventBus,
    EventCategory,
    ObservabilityIntegration,
    StateHookManager,
    TransitionHistory,
    VictorEvent,
)
from victor.core.verticals.base import (
    StageDefinition,
    VerticalBase,
    VerticalConfig,
    VerticalRegistry,
)


class TestVerticalBase:
    """Test VerticalBase abstract class and configuration."""

    def test_vertical_config_creation(self):
        """Test VerticalConfig can be created with all fields."""
        tools = ToolSet.from_tools(["read", "write"])
        config = VerticalConfig(
            tools=tools,
            system_prompt="You are an expert.",
            stages={
                "INITIAL": StageDefinition(
                    name="INITIAL",
                    description="Starting stage",
                    keywords=["start", "begin"],
                )
            },
            provider_hints={"preferred_providers": ["anthropic"]},
            evaluation_criteria=["accuracy", "efficiency"],
            metadata={"version": "1.0"},
        )

        assert config.tools == tools
        assert config.system_prompt == "You are an expert."
        assert "INITIAL" in config.stages
        assert config.provider_hints["preferred_providers"] == ["anthropic"]

    def test_vertical_config_to_agent_kwargs(self):
        """Test VerticalConfig.to_agent_kwargs() returns correct dict."""
        tools = ToolSet.from_tools(["read", "write"])
        config = VerticalConfig(
            tools=tools,
            system_prompt="Test prompt",
        )

        kwargs = config.to_agent_kwargs()
        assert "tools" in kwargs
        assert kwargs["tools"] == tools

    def test_stage_definition_to_dict(self):
        """Test StageDefinition serialization."""
        stage = StageDefinition(
            name="PLANNING",
            description="Planning the approach",
            tools={"read", "search"},
            keywords=["plan", "strategy"],
            next_stages={"EXECUTION"},
            min_confidence=0.7,
        )

        d = stage.to_dict()
        assert d["name"] == "PLANNING"
        assert d["description"] == "Planning the approach"
        assert set(d["tools"]) == {"read", "search"}
        assert d["min_confidence"] == 0.7


class TestCustomVertical:
    """Tests for custom vertical implementations."""

    def test_custom_vertical_get_config(self):
        """Test creating a custom vertical and getting its config."""

        class TestVertical(VerticalBase):
            name = "test_vertical"
            description = "Test vertical for unit tests"
            version = "0.1.0"

            @classmethod
            def get_tools(cls) -> List[str]:
                return ["read", "write", "shell"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "You are a test assistant."

            @classmethod
            def get_evaluation_criteria(cls) -> List[str]:
                return ["test_coverage", "correctness"]

        config = TestVertical.get_config()

        assert config.system_prompt == "You are a test assistant."
        assert config.metadata["vertical_name"] == "test_vertical"
        assert config.metadata["vertical_version"] == "0.1.0"
        assert "test_coverage" in config.evaluation_criteria

    def test_custom_vertical_tool_set(self):
        """Test getting ToolSet from vertical."""

        class ToolTestVertical(VerticalBase):
            name = "tool_test"
            description = "Tool test vertical"

            @classmethod
            def get_tools(cls) -> List[str]:
                return ["read", "write"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Test."

        tool_set = ToolTestVertical.get_tool_set()
        assert isinstance(tool_set, ToolSet)

    def test_custom_vertical_customize_config_hook(self):
        """Test the customize_config hook."""

        class CustomHookVertical(VerticalBase):
            name = "custom_hook"
            description = "Tests customize_config hook"

            @classmethod
            def get_tools(cls) -> List[str]:
                return ["read"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Base prompt."

            @classmethod
            def customize_config(cls, config: VerticalConfig) -> VerticalConfig:
                # Add custom metadata
                config.metadata["customized"] = True
                config.metadata["custom_value"] = 42
                return config

        config = CustomHookVertical.get_config()
        assert config.metadata.get("customized") is True
        assert config.metadata.get("custom_value") == 42


class TestVerticalRegistry:
    """Tests for VerticalRegistry pattern."""

    @pytest.fixture(autouse=True)
    def clear_registry(self):
        """Clear registry before each test (without built-ins for isolated testing)."""
        VerticalRegistry.clear(reregister_builtins=False)
        yield
        # Re-register built-ins on teardown to avoid polluting other tests
        VerticalRegistry.clear(reregister_builtins=True)

    def test_register_and_get_vertical(self):
        """Test registering and retrieving a vertical."""

        class RegisteredVertical(VerticalBase):
            name = "registered"
            description = "Registered vertical"

            @classmethod
            def get_tools(cls):
                return ["read"]

            @classmethod
            def get_system_prompt(cls):
                return "Test."

        VerticalRegistry.register(RegisteredVertical)

        retrieved = VerticalRegistry.get("registered")
        assert retrieved is RegisteredVertical

    def test_register_vertical_without_name_raises(self):
        """Test registering vertical without name raises ValueError."""

        class NoNameVertical(VerticalBase):
            name = ""
            description = "No name"

            @classmethod
            def get_tools(cls):
                return []

            @classmethod
            def get_system_prompt(cls):
                return ""

        with pytest.raises(ValueError, match="has no name defined"):
            VerticalRegistry.register(NoNameVertical)

    def test_list_all_verticals(self):
        """Test listing all registered verticals."""

        class V1(VerticalBase):
            name = "v1"
            description = "V1"

            @classmethod
            def get_tools(cls):
                return []

            @classmethod
            def get_system_prompt(cls):
                return ""

        class V2(VerticalBase):
            name = "v2"
            description = "V2"

            @classmethod
            def get_tools(cls):
                return []

            @classmethod
            def get_system_prompt(cls):
                return ""

        VerticalRegistry.register(V1)
        VerticalRegistry.register(V2)

        all_verticals = VerticalRegistry.list_all()
        names = VerticalRegistry.list_names()

        assert len(all_verticals) == 2
        assert "v1" in names
        assert "v2" in names

    def test_unregister_vertical(self):
        """Test unregistering a vertical."""

        class TempVertical(VerticalBase):
            name = "temp"
            description = "Temporary"

            @classmethod
            def get_tools(cls):
                return []

            @classmethod
            def get_system_prompt(cls):
                return ""

        VerticalRegistry.register(TempVertical)
        assert VerticalRegistry.get("temp") is not None

        VerticalRegistry.unregister("temp")
        assert VerticalRegistry.get("temp") is None


class TestObservabilityIntegrationWithVerticals:
    """Tests for observability integration with vertical-based agents."""

    @pytest.fixture(autouse=True)
    def reset_event_bus(self):
        """Reset EventBus singleton before each test."""
        EventBus.reset_instance()
        yield
        EventBus.reset_instance()

    def test_observability_integration_properties(self):
        """Test ObservabilityIntegration has expected properties."""
        integration = ObservabilityIntegration(session_id="test-session")

        assert integration.event_bus is not None
        assert integration.state_hook_manager is None  # Before wiring
        assert integration.state_transition_history is None  # Before wiring

    def test_observability_integration_event_emission(self):
        """Test ObservabilityIntegration emits events correctly."""
        integration = ObservabilityIntegration()
        received_events: List[VictorEvent] = []

        integration.event_bus.subscribe(EventCategory.TOOL, lambda e: received_events.append(e))

        # Emit tool events
        integration.on_tool_start("read", {"path": "/test"}, "tool-1")
        integration.on_tool_end("read", {"content": "..."}, success=True, tool_id="tool-1")

        assert len(received_events) == 2
        assert received_events[0].name == "read.start"
        assert received_events[1].name == "read.end"

    def test_observability_with_state_machine_wiring(self):
        """Test ObservabilityIntegration wires state machine correctly."""
        integration = ObservabilityIntegration()

        # Create a mock state machine with set_hooks
        mock_state_machine = MagicMock()
        mock_state_machine.set_hooks = MagicMock()

        integration.wire_state_machine(mock_state_machine)

        # State hook manager should be created
        assert integration.state_hook_manager is not None
        assert integration.state_transition_history is not None

        # set_hooks should have been called
        mock_state_machine.set_hooks.assert_called_once()

    def test_observability_metrics_without_transitions(self):
        """Test get_state_transition_metrics with no transitions."""
        integration = ObservabilityIntegration()
        mock_state_machine = MagicMock()
        mock_state_machine.set_hooks = MagicMock()

        integration.wire_state_machine(mock_state_machine)

        metrics = integration.get_state_transition_metrics()
        assert metrics["total_transitions"] == 0
        assert metrics["unique_stages_visited"] == 0
        assert metrics["has_cycles"] is False


class TestFrameworkInternalHelpers:
    """Tests for framework internal helper functions."""

    def test_setup_observability_integration(self):
        """Test setup_observability_integration function."""
        from victor.framework._internal import setup_observability_integration

        # Create a mock orchestrator
        mock_orchestrator = MagicMock()
        mock_orchestrator.conversation_state = MagicMock()
        mock_orchestrator.conversation_state.set_hooks = MagicMock()

        EventBus.reset_instance()

        integration = setup_observability_integration(mock_orchestrator, session_id="test-123")

        assert integration is not None
        assert isinstance(integration, ObservabilityIntegration)
        assert hasattr(mock_orchestrator, "observability")
        assert mock_orchestrator.observability is integration

        EventBus.reset_instance()

    def test_apply_system_prompt(self):
        """Test apply_system_prompt function uses public methods only (DIP compliance)."""
        from victor.framework._internal import apply_system_prompt
        from victor.framework.protocols import CapabilityRegistryProtocol

        # Create mock that implements capability registry protocol
        mock_orchestrator = MagicMock(spec=CapabilityRegistryProtocol)
        mock_orchestrator.has_capability.return_value = True
        mock_orchestrator.invoke_capability = MagicMock(return_value=True)

        apply_system_prompt(mock_orchestrator, "Custom prompt text")

        # Should invoke via capability registry
        mock_orchestrator.invoke_capability.assert_called_once()
        call_args = mock_orchestrator.invoke_capability.call_args
        assert call_args[0][0] == "custom_prompt"
        assert call_args[0][1] == "Custom prompt text"


class TestAgentVerticalIntegration:
    """Tests for Agent.create() with vertical parameter."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset global state before each test."""
        EventBus.reset_instance()
        VerticalRegistry.clear(reregister_builtins=False)
        yield
        EventBus.reset_instance()
        # Re-register built-ins on teardown to avoid polluting other tests
        VerticalRegistry.clear(reregister_builtins=True)

    def test_vertical_config_extraction(self):
        """Test that vertical config is properly extracted."""

        class TestVertical(VerticalBase):
            name = "test"
            description = "Test vertical"

            @classmethod
            def get_tools(cls) -> List[str]:
                return ["read", "write"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "You are a test assistant specialized in testing."

        # Get config (use_cache=False to avoid stale cached config)
        config = TestVertical.get_config(use_cache=False)

        assert config.system_prompt == "You are a test assistant specialized in testing."
        assert config.tools is not None
        assert config.metadata["vertical_name"] == "test"

    def test_vertical_provider_hints(self):
        """Test vertical provider hints are accessible."""

        class HintVertical(VerticalBase):
            name = "hint_vertical"
            description = "Vertical with provider hints"

            @classmethod
            def get_tools(cls) -> List[str]:
                return ["read"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "Test."

            @classmethod
            def get_provider_hints(cls) -> Dict[str, Any]:
                return {
                    "preferred_providers": ["openai", "anthropic"],
                    "min_context_window": 50000,
                    "requires_tool_calling": True,
                }

        config = HintVertical.get_config()
        hints = config.provider_hints

        assert "openai" in hints["preferred_providers"]
        assert hints["min_context_window"] == 50000
        assert hints["requires_tool_calling"] is True


class TestStateHookManagerIntegration:
    """Tests for StateHookManager with observability."""

    @pytest.fixture(autouse=True)
    def reset_event_bus(self):
        """Reset EventBus singleton."""
        EventBus.reset_instance()
        yield
        EventBus.reset_instance()

    def test_history_aware_hooks_fire_correctly(self):
        """Test history-aware hooks receive full context."""
        manager = StateHookManager(enable_history=True)
        fired_callbacks: List[Dict[str, Any]] = []

        @manager.on_transition_with_history
        def capture(old: str, new: str, ctx: Dict, history: TransitionHistory) -> None:
            fired_callbacks.append(
                {
                    "old": old,
                    "new": new,
                    "sequence_len": len(history.get_stage_sequence()),
                    "has_cycle": history.has_cycle(),
                }
            )

        # Transition sequence: A→B, B→C, C→B (cycle - B visited twice as new_stage)
        manager.fire_transition("A", "B", {})
        manager.fire_transition("B", "C", {})
        manager.fire_transition("C", "B", {})  # Cycle - transitioning TO B again

        assert len(fired_callbacks) == 3
        assert fired_callbacks[0]["old"] == "A"
        assert fired_callbacks[0]["new"] == "B"
        # has_cycle() checks if same new_stage appears twice
        assert fired_callbacks[2]["has_cycle"] is True

    def test_transition_history_metrics(self):
        """Test TransitionHistory provides correct metrics."""
        manager = StateHookManager(enable_history=True)

        manager.fire_transition("INITIAL", "PLANNING", {})
        manager.fire_transition("PLANNING", "READING", {})
        manager.fire_transition("READING", "PLANNING", {})  # Back to planning

        history = manager.history
        assert len(history) == 3
        assert history.current_stage == "PLANNING"
        assert history.get_stage_visit_count("PLANNING") == 2
        assert history.has_cycle() is True


class TestEndToEndFrameworkFlow:
    """End-to-end tests demonstrating complete integration."""

    @pytest.fixture(autouse=True)
    def reset_state(self):
        """Reset all global state."""
        EventBus.reset_instance()
        VerticalRegistry.clear(reregister_builtins=False)
        yield
        EventBus.reset_instance()
        # Re-register built-ins on teardown to avoid polluting other tests
        VerticalRegistry.clear(reregister_builtins=True)

    def test_complete_vertical_observability_flow(self):
        """Test complete flow from vertical config to event emission."""

        # 1. Define a vertical
        class AnalysisVertical(VerticalBase):
            name = "analysis"
            description = "Code analysis vertical"

            @classmethod
            def get_tools(cls) -> List[str]:
                return ["read", "search", "code_review"]

            @classmethod
            def get_system_prompt(cls) -> str:
                return "You are a code analysis expert."

            @classmethod
            def get_stages(cls) -> Dict[str, StageDefinition]:
                return {
                    "SCANNING": StageDefinition(
                        name="SCANNING",
                        description="Scanning codebase",
                        tools={"read", "search"},
                        keywords=["scan", "find"],
                        next_stages={"ANALYZING"},
                    ),
                    "ANALYZING": StageDefinition(
                        name="ANALYZING",
                        description="Analyzing code",
                        tools={"code_review"},
                        keywords=["analyze", "review"],
                        next_stages={"REPORTING"},
                    ),
                    "REPORTING": StageDefinition(
                        name="REPORTING",
                        description="Generating report",
                        keywords=["report", "summarize"],
                        next_stages=set(),
                    ),
                }

        # 2. Register vertical
        VerticalRegistry.register(AnalysisVertical)

        # 3. Get config
        config = AnalysisVertical.get_config()
        assert "SCANNING" in config.stages
        assert "ANALYZING" in config.stages

        # 4. Set up observability
        integration = ObservabilityIntegration(session_id="analysis-001")
        all_events: List[VictorEvent] = []
        integration.event_bus.subscribe_all(lambda e: all_events.append(e))

        # 5. Wire a mock state machine
        mock_sm = MagicMock()
        mock_sm.set_hooks = MagicMock()
        integration.wire_state_machine(mock_sm)

        # 6. Simulate the workflow
        hook_manager = integration.state_hook_manager
        assert hook_manager is not None

        # Start session
        integration.on_session_start({"vertical": "analysis", "model": "claude-3"})

        # State transitions through stages
        hook_manager.fire_transition("INITIAL", "SCANNING", {"confidence": 0.9})
        integration.on_tool_start("read", {"path": "/src"}, "tool-1")
        integration.on_tool_end("read", {"files": 10}, success=True, tool_id="tool-1")

        hook_manager.fire_transition("SCANNING", "ANALYZING", {"confidence": 0.85})
        integration.on_tool_start("code_review", {"file": "main.py"}, "tool-2")
        integration.on_tool_end("code_review", {"issues": 3}, success=True, tool_id="tool-2")

        hook_manager.fire_transition("ANALYZING", "REPORTING", {"confidence": 0.95})

        # End session
        integration.on_session_end(tool_calls=2, duration_seconds=30.0, success=True)

        # 7. Verify events
        event_names = [e.name for e in all_events]
        assert "session.start" in event_names
        assert "read.start" in event_names
        assert "read.end" in event_names
        assert "session.end" in event_names

        # 8. Verify metrics
        metrics = integration.get_state_transition_metrics()
        assert metrics["total_transitions"] == 3
        assert "SCANNING" in metrics["stage_sequence"]
        assert "ANALYZING" in metrics["stage_sequence"]
        assert "REPORTING" in metrics["stage_sequence"]

    def test_complete_flow_with_cqrs_bridge(self):
        """Test complete flow including CQRS event bridging.

        This test verifies the full integration chain:
        1. Vertical configuration
        2. ObservabilityIntegration with CQRS bridge enabled
        3. State transitions via StateHookManager
        4. Events flow to both EventBus and CQRS EventDispatcher
        5. Metrics and history tracking
        """
        from victor.observability import (
            CQRSEventAdapter,
            UnifiedEventBridge,
            create_unified_bridge,
        )
        from victor.core.event_sourcing import EventDispatcher, Event as CQRSEvent

        # 1. Set up the event systems
        event_bus = EventBus.get_instance()
        event_dispatcher = EventDispatcher()

        # Track events on both systems
        bus_events: List[VictorEvent] = []
        cqrs_events: List[CQRSEvent] = []

        event_bus.subscribe_all(lambda e: bus_events.append(e))
        event_dispatcher.subscribe_all(lambda e: cqrs_events.append(e))

        # 2. Create integration with CQRS bridge
        integration = ObservabilityIntegration(
            event_bus=event_bus,
            session_id="e2e-cqrs-test",
            enable_cqrs_bridge=True,
        )

        # Verify CQRS bridge is enabled
        assert integration.cqrs_bridge is not None
        assert isinstance(integration.cqrs_bridge, UnifiedEventBridge)

        # 3. Wire a mock state machine
        mock_sm = MagicMock()
        mock_sm.set_hooks = MagicMock()
        integration.wire_state_machine(mock_sm)

        hook_manager = integration.state_hook_manager
        assert hook_manager is not None

        # 4. Simulate workflow with state transitions
        integration.on_session_start(
            {
                "model": "claude-3",
                "vertical": "coding",
            }
        )

        # First transition
        hook_manager.fire_transition("INITIAL", "PLANNING", {"confidence": 0.9})

        # Tool usage
        integration.on_tool_start("read", {"path": "/src/main.py"}, "tool-1")
        integration.on_tool_end(
            "read",
            {"content": "def main(): pass"},
            success=True,
            tool_id="tool-1",
        )

        # Second transition
        hook_manager.fire_transition("PLANNING", "EXECUTION", {"confidence": 0.85})

        integration.on_tool_start("write", {"path": "/src/main.py"}, "tool-2")
        integration.on_tool_end(
            "write",
            {"written": True},
            success=True,
            tool_id="tool-2",
        )

        # Final transition
        hook_manager.fire_transition("EXECUTION", "COMPLETION", {"confidence": 0.95})

        integration.on_session_end(
            tool_calls=2,
            duration_seconds=45.0,
            success=True,
        )

        # 5. Verify events on EventBus
        bus_event_names = [e.name for e in bus_events]
        assert "session.start" in bus_event_names
        assert "read.start" in bus_event_names
        assert "read.end" in bus_event_names
        assert "write.start" in bus_event_names
        assert "write.end" in bus_event_names
        assert "session.end" in bus_event_names

        # State change events should be present
        state_events = [e for e in bus_events if e.category == EventCategory.STATE]
        assert len(state_events) >= 3  # 3 transitions

        # 6. Verify state transition metrics
        metrics = integration.get_state_transition_metrics()
        assert metrics["total_transitions"] == 3
        assert metrics["has_cycles"] is False
        assert "INITIAL" in metrics["stage_sequence"]
        assert "PLANNING" in metrics["stage_sequence"]
        assert "EXECUTION" in metrics["stage_sequence"]
        assert "COMPLETION" in metrics["stage_sequence"]
        assert metrics["current_stage"] == "COMPLETION"

        # 7. Verify stage visit counts
        assert metrics["stage_visit_counts"]["PLANNING"] == 1
        assert metrics["stage_visit_counts"]["EXECUTION"] == 1
        assert metrics["stage_visit_counts"]["COMPLETION"] == 1

        # 8. Clean up
        integration.disable_cqrs_bridge()
        assert integration.cqrs_bridge is None

    def test_cqrs_bridge_event_conversion(self):
        """Test that events are properly converted when bridging."""
        from victor.observability import CQRSEventAdapter, AdapterConfig
        from victor.core.event_sourcing import EventDispatcher

        # Create adapter with custom config
        event_bus = EventBus.get_instance()
        event_dispatcher = EventDispatcher()

        converted_events: List[Any] = []
        event_dispatcher.subscribe_all(lambda e: converted_events.append(e))

        config = AdapterConfig(
            enable_observability_to_cqrs=True,
            enable_cqrs_to_observability=False,  # One-way for this test
        )

        adapter = CQRSEventAdapter(
            event_bus=event_bus,
            event_dispatcher=event_dispatcher,
            config=config,
        )
        adapter.start()

        # Emit a tool event on EventBus
        from victor.observability import VictorEvent, EventCategory

        tool_event = VictorEvent(
            category=EventCategory.TOOL,
            name="read.start",
            data={
                "tool_name": "read",
                "arguments": {"path": "/test.py"},
            },
        )
        event_bus.publish(tool_event)

        # Emit a state event
        state_event = VictorEvent(
            category=EventCategory.STATE,
            name="stage_transition",
            data={
                "old_stage": "INITIAL",
                "new_stage": "PLANNING",
                "confidence": 0.9,
            },
        )
        event_bus.publish(state_event)

        # Check adapter metrics
        metrics = adapter.get_metrics()
        assert metrics["events_bridged_to_cqrs"] >= 2

        adapter.stop()
