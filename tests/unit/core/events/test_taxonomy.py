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

"""Tests for the unified event taxonomy.

This module tests the UnifiedEventType enum and mapping functions
that consolidate event types across the Victor codebase.
"""

import warnings

from victor.core.events.taxonomy import (
    UnifiedEventType,
    map_workflow_event,
    map_event_category,
    map_framework_event,
    map_tool_event,
    map_agent_event,
    map_system_event,
    get_all_categories,
    get_events_by_category,
    is_valid_event_type,
    emit_deprecation_warning,
)


class TestUnifiedEventType:
    """Tests for the UnifiedEventType enum."""

    def test_event_type_values_use_dot_notation(self):
        """All event types should use hierarchical dot notation."""
        for event in UnifiedEventType:
            if event not in (UnifiedEventType.CUSTOM, UnifiedEventType.UNKNOWN):
                assert "." in event.value, f"{event.name} should use dot notation"

    def test_event_type_category_extraction(self):
        """Category property should extract first segment correctly."""
        assert UnifiedEventType.WORKFLOW_START.category == "workflow"
        assert UnifiedEventType.TOOL_CALL.category == "tool"
        assert UnifiedEventType.AGENT_THINKING.category == "agent"
        assert UnifiedEventType.SYSTEM_HEALTH.category == "system"
        assert UnifiedEventType.FRAMEWORK_CONTENT.category == "framework"

    def test_is_workflow_event(self):
        """is_workflow_event should return True for workflow events."""
        assert UnifiedEventType.WORKFLOW_START.is_workflow_event is True
        assert UnifiedEventType.WORKFLOW_NODE_START.is_workflow_event is True
        assert UnifiedEventType.WORKFLOW_ERROR.is_workflow_event is True
        assert UnifiedEventType.TOOL_CALL.is_workflow_event is False
        assert UnifiedEventType.AGENT_THINKING.is_workflow_event is False

    def test_is_tool_event(self):
        """is_tool_event should return True for tool events."""
        assert UnifiedEventType.TOOL_CALL.is_tool_event is True
        assert UnifiedEventType.TOOL_RESULT.is_tool_event is True
        assert UnifiedEventType.TOOL_ERROR.is_tool_event is True
        assert UnifiedEventType.WORKFLOW_START.is_tool_event is False

    def test_is_agent_event(self):
        """is_agent_event should return True for agent events."""
        assert UnifiedEventType.AGENT_THINKING.is_agent_event is True
        assert UnifiedEventType.AGENT_RESPONSE.is_agent_event is True
        assert UnifiedEventType.AGENT_CONTENT.is_agent_event is True
        assert UnifiedEventType.WORKFLOW_START.is_agent_event is False

    def test_is_system_event(self):
        """is_system_event should return True for system events."""
        assert UnifiedEventType.SYSTEM_HEALTH.is_system_event is True
        assert UnifiedEventType.SYSTEM_METRICS.is_system_event is True
        assert UnifiedEventType.SYSTEM_ERROR.is_system_event is True
        assert UnifiedEventType.WORKFLOW_START.is_system_event is False

    def test_is_error_event(self):
        """is_error_event should return True for error events."""
        assert UnifiedEventType.WORKFLOW_ERROR.is_error_event is True
        assert UnifiedEventType.TOOL_ERROR.is_error_event is True
        assert UnifiedEventType.SYSTEM_ERROR.is_error_event is True
        assert UnifiedEventType.WORKFLOW_NODE_ERROR.is_error_event is True
        assert UnifiedEventType.WORKFLOW_START.is_error_event is False

    def test_from_string_valid(self):
        """from_string should parse valid event type strings."""
        assert UnifiedEventType.from_string("workflow.start") == UnifiedEventType.WORKFLOW_START
        assert UnifiedEventType.from_string("tool.call") == UnifiedEventType.TOOL_CALL
        assert UnifiedEventType.from_string("agent.thinking") == UnifiedEventType.AGENT_THINKING

    def test_from_string_invalid(self):
        """from_string should return UNKNOWN for invalid strings."""
        assert UnifiedEventType.from_string("invalid.event") == UnifiedEventType.UNKNOWN
        assert UnifiedEventType.from_string("not_an_event") == UnifiedEventType.UNKNOWN
        assert UnifiedEventType.from_string("") == UnifiedEventType.UNKNOWN

    def test_string_serialization(self):
        """Event types should serialize to their value string."""
        event = UnifiedEventType.WORKFLOW_NODE_START
        assert str(event.value) == "workflow.node.start"
        assert event.value == "workflow.node.start"


class TestWorkflowEventMapping:
    """Tests for mapping WorkflowEventType to UnifiedEventType."""

    def test_map_workflow_start(self):
        """WORKFLOW_START should map correctly."""
        from victor.workflows.streaming import WorkflowEventType

        result = map_workflow_event(WorkflowEventType.WORKFLOW_START)
        assert result == UnifiedEventType.WORKFLOW_START

    def test_map_workflow_complete(self):
        """WORKFLOW_COMPLETE should map correctly."""
        from victor.workflows.streaming import WorkflowEventType

        result = map_workflow_event(WorkflowEventType.WORKFLOW_COMPLETE)
        assert result == UnifiedEventType.WORKFLOW_COMPLETE

    def test_map_workflow_error(self):
        """WORKFLOW_ERROR should map correctly."""
        from victor.workflows.streaming import WorkflowEventType

        result = map_workflow_event(WorkflowEventType.WORKFLOW_ERROR)
        assert result == UnifiedEventType.WORKFLOW_ERROR

    def test_map_node_start(self):
        """NODE_START should map to WORKFLOW_NODE_START."""
        from victor.workflows.streaming import WorkflowEventType

        result = map_workflow_event(WorkflowEventType.NODE_START)
        assert result == UnifiedEventType.WORKFLOW_NODE_START

    def test_map_node_complete(self):
        """NODE_COMPLETE should map to WORKFLOW_NODE_COMPLETE."""
        from victor.workflows.streaming import WorkflowEventType

        result = map_workflow_event(WorkflowEventType.NODE_COMPLETE)
        assert result == UnifiedEventType.WORKFLOW_NODE_COMPLETE

    def test_map_node_error(self):
        """NODE_ERROR should map to WORKFLOW_NODE_ERROR."""
        from victor.workflows.streaming import WorkflowEventType

        result = map_workflow_event(WorkflowEventType.NODE_ERROR)
        assert result == UnifiedEventType.WORKFLOW_NODE_ERROR

    def test_map_agent_content(self):
        """AGENT_CONTENT should map correctly."""
        from victor.workflows.streaming import WorkflowEventType

        result = map_workflow_event(WorkflowEventType.AGENT_CONTENT)
        assert result == UnifiedEventType.AGENT_CONTENT

    def test_map_agent_tool_call(self):
        """AGENT_TOOL_CALL should map correctly."""
        from victor.workflows.streaming import WorkflowEventType

        result = map_workflow_event(WorkflowEventType.AGENT_TOOL_CALL)
        assert result == UnifiedEventType.AGENT_TOOL_CALL

    def test_map_progress_update(self):
        """PROGRESS_UPDATE should map to WORKFLOW_PROGRESS."""
        from victor.workflows.streaming import WorkflowEventType

        result = map_workflow_event(WorkflowEventType.PROGRESS_UPDATE)
        assert result == UnifiedEventType.WORKFLOW_PROGRESS

    def test_map_checkpoint_saved(self):
        """CHECKPOINT_SAVED should map to WORKFLOW_CHECKPOINT."""
        from victor.workflows.streaming import WorkflowEventType

        result = map_workflow_event(WorkflowEventType.CHECKPOINT_SAVED)
        assert result == UnifiedEventType.WORKFLOW_CHECKPOINT


class TestEventCategoryMapping:
    """Tests for mapping EventCategory to UnifiedEventType."""

    def test_map_tool_category(self):
        """TOOL topic prefix should map to TOOL_CALL."""
        # EventCategory removed - use topic prefixes

        result = map_event_category("tool")
        assert result == UnifiedEventType.TOOL_CALL

    def test_map_state_category(self):
        """STATE topic prefix should map to STATE_TRANSITION."""
        # EventCategory removed - use topic prefixes

        result = map_event_category("state")
        assert result == UnifiedEventType.STATE_TRANSITION

    def test_map_model_category(self):
        """MODEL topic prefix should map to MODEL_REQUEST."""
        # EventCategory removed - use topic prefixes

        result = map_event_category("model")
        assert result == UnifiedEventType.MODEL_REQUEST

    def test_map_error_category(self):
        """ERROR topic prefix should map to SYSTEM_ERROR."""
        # EventCategory removed - use topic prefixes

        result = map_event_category("error")
        assert result == UnifiedEventType.SYSTEM_ERROR

    def test_map_metric_category(self):
        """METRIC topic prefix should map to SYSTEM_METRICS."""
        # EventCategory removed - use topic prefixes

        result = map_event_category("metric")
        assert result == UnifiedEventType.SYSTEM_METRICS

    def test_map_lifecycle_category(self):
        """LIFECYCLE topic prefix should map to SYSTEM_LIFECYCLE."""
        # EventCategory removed - use topic prefixes

        result = map_event_category("lifecycle")
        assert result == UnifiedEventType.SYSTEM_LIFECYCLE

    def test_map_custom_category(self):
        """CUSTOM topic prefix should map to CUSTOM."""
        # EventCategory removed - use topic prefixes

        result = map_event_category("custom")
        assert result == UnifiedEventType.CUSTOM


class TestStringMappingFunctions:
    """Tests for string-based mapping functions."""

    def test_map_framework_event(self):
        """Framework events should map correctly."""
        assert map_framework_event("content") == UnifiedEventType.FRAMEWORK_CONTENT
        assert map_framework_event("thinking") == UnifiedEventType.FRAMEWORK_THINKING
        assert map_framework_event("chunk") == UnifiedEventType.FRAMEWORK_CHUNK
        assert map_framework_event("invalid") == UnifiedEventType.UNKNOWN

    def test_map_framework_event_case_insensitive(self):
        """Framework event mapping should be case insensitive."""
        assert map_framework_event("CONTENT") == UnifiedEventType.FRAMEWORK_CONTENT
        assert map_framework_event("Thinking") == UnifiedEventType.FRAMEWORK_THINKING

    def test_map_tool_event(self):
        """Tool events should map correctly."""
        assert map_tool_event("call") == UnifiedEventType.TOOL_CALL
        assert map_tool_event("result") == UnifiedEventType.TOOL_RESULT
        assert map_tool_event("error") == UnifiedEventType.TOOL_ERROR
        assert map_tool_event("start") == UnifiedEventType.TOOL_START
        assert map_tool_event("end") == UnifiedEventType.TOOL_END
        assert map_tool_event("invalid") == UnifiedEventType.UNKNOWN

    def test_map_agent_event(self):
        """Agent events should map correctly."""
        assert map_agent_event("thinking") == UnifiedEventType.AGENT_THINKING
        assert map_agent_event("response") == UnifiedEventType.AGENT_RESPONSE
        assert map_agent_event("content") == UnifiedEventType.AGENT_CONTENT
        assert map_agent_event("tool_call") == UnifiedEventType.AGENT_TOOL_CALL
        assert map_agent_event("tool_result") == UnifiedEventType.AGENT_TOOL_RESULT
        assert map_agent_event("invalid") == UnifiedEventType.UNKNOWN

    def test_map_system_event(self):
        """System events should map correctly."""
        assert map_system_event("health") == UnifiedEventType.SYSTEM_HEALTH
        assert map_system_event("metrics") == UnifiedEventType.SYSTEM_METRICS
        assert map_system_event("error") == UnifiedEventType.SYSTEM_ERROR
        assert map_system_event("warning") == UnifiedEventType.SYSTEM_WARNING
        assert map_system_event("lifecycle") == UnifiedEventType.SYSTEM_LIFECYCLE
        assert map_system_event("invalid") == UnifiedEventType.UNKNOWN


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_all_categories(self):
        """Should return all unique categories."""
        categories = get_all_categories()
        assert "workflow" in categories
        assert "tool" in categories
        assert "agent" in categories
        assert "system" in categories
        assert "framework" in categories
        # Categories should be sorted
        assert categories == sorted(categories)

    def test_get_events_by_category(self):
        """Should return all events for a category."""
        workflow_events = get_events_by_category("workflow")
        assert len(workflow_events) > 0
        assert all(e.category == "workflow" for e in workflow_events)
        assert UnifiedEventType.WORKFLOW_START in workflow_events
        assert UnifiedEventType.WORKFLOW_NODE_START in workflow_events

    def test_get_events_by_category_tool(self):
        """Should return all tool events."""
        tool_events = get_events_by_category("tool")
        assert len(tool_events) > 0
        assert all(e.category == "tool" for e in tool_events)
        assert UnifiedEventType.TOOL_CALL in tool_events
        assert UnifiedEventType.TOOL_ERROR in tool_events

    def test_get_events_by_category_empty(self):
        """Should return empty list for non-existent category."""
        events = get_events_by_category("nonexistent")
        assert events == []

    def test_is_valid_event_type(self):
        """Should validate event type strings."""
        assert is_valid_event_type("workflow.start") is True
        assert is_valid_event_type("tool.call") is True
        assert is_valid_event_type("agent.thinking") is True
        assert is_valid_event_type("invalid.event") is False
        assert is_valid_event_type("") is False


class TestDeprecationWarning:
    """Tests for deprecation warning helper."""

    def test_emit_deprecation_warning(self):
        """Should emit a deprecation warning with correct message."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            emit_deprecation_warning(
                "workflow_start",
                UnifiedEventType.WORKFLOW_START,
                "victor.workflows.streaming",
            )
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "workflow_start" in str(w[0].message)
            assert "WORKFLOW_START" in str(w[0].message)
            assert "victor.workflows.streaming" in str(w[0].message)


class TestPackageImports:
    """Tests that package imports work correctly."""

    def test_import_from_package(self):
        """Should be able to import from package init."""
        from victor.core.events import (
            UnifiedEventType,
            map_workflow_event,
            map_event_category,
            get_all_categories,
        )

        # Verify imports work
        assert UnifiedEventType.WORKFLOW_START is not None
        assert callable(map_workflow_event)
        assert callable(map_event_category)
        assert callable(get_all_categories)

    def test_all_exports_available(self):
        """All __all__ exports should be available."""
        from victor.core.events import taxonomy

        for name in taxonomy.__all__:
            assert hasattr(taxonomy, name), f"{name} not found in taxonomy module"
