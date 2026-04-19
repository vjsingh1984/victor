"""Unit tests for Pydantic state models.

Tests WorkflowExecutionContextModel and WorkflowStateModel for validation,
serialization, and adapter functionality.
"""

import pytest

from victor.workflows.models import WorkflowExecutionContextModel, WorkflowStateModel
from victor.workflows.models.adapters import StateAdapter, WorkflowExecutionContextAdapter


class TestWorkflowExecutionContextModel:
    """Test WorkflowExecutionContextModel Pydantic model."""

    def test_default_values(self):
        """Model should create with sensible defaults."""
        model = WorkflowExecutionContextModel()
        assert model.data == {}
        assert model.messages == []
        assert model.workflow_id != ""  # Auto-generated UUID
        assert model.workflow_name == ""
        assert model.current_node == ""
        assert model.node_results == {}
        assert model.error is None
        assert model.iteration == 0
        assert model.visited_nodes == []
        assert model.parallel_results == {}
        assert not model.hitl_pending
        assert model.hitl_response is None
        assert model.as_of_date is None
        assert model.lookback_periods is None
        assert model.include_end_date is True
        assert not model.is_complete
        assert not model.success

    def test_custom_values(self):
        """Model should accept custom values."""
        model = WorkflowExecutionContextModel(
            workflow_name="test_workflow",
            current_node="node_1",
            data={"key": "value"},
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert model.workflow_name == "test_workflow"
        assert model.current_node == "node_1"
        assert model.data["key"] == "value"
        assert len(model.messages) == 1

    def test_workflow_id_validation(self):
        """workflow_id should accept valid identifiers."""
        # Valid identifiers
        WorkflowExecutionContextModel(workflow_id="abc123")
        WorkflowExecutionContextModel(workflow_id="test-workflow_123")

        # Empty string should fail
        with pytest.raises(ValueError, match="workflow_id cannot be empty"):
            WorkflowExecutionContextModel(workflow_id="")

        # Invalid characters should fail
        with pytest.raises(ValueError, match="must be alphanumeric"):
            WorkflowExecutionContextModel(workflow_id="test workflow!")

    def test_iteration_validation(self):
        """iteration should be non-negative."""
        # Valid values
        WorkflowExecutionContextModel(iteration=0)
        WorkflowExecutionContextModel(iteration=5)

        # Negative should fail with Pydantic v2 built-in error
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            WorkflowExecutionContextModel(iteration=-1)

    def test_lookback_periods_validation(self):
        """lookback_periods should be non-negative if provided."""
        # Valid values
        WorkflowExecutionContextModel(lookback_periods=0)
        WorkflowExecutionContextModel(lookback_periods=10)

        # Negative should fail with Pydantic v2 built-in error
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            WorkflowExecutionContextModel(lookback_periods=-5)

    def test_visited_nodes_uniqueness(self):
        """visited_nodes should not allow duplicates."""
        # Valid list with unique values
        model = WorkflowExecutionContextModel(visited_nodes=["node1", "node2"])
        assert model.visited_nodes == ["node1", "node2"]

        # Duplicates should fail
        with pytest.raises(ValueError, match="must be unique"):
            WorkflowExecutionContextModel(visited_nodes=["node1", "node1"])

    def test_to_dict(self):
        """to_dict should convert to TypedDict-compatible format."""
        model = WorkflowExecutionContextModel(
            workflow_id="test_workflow",
            workflow_name="My Workflow",
            current_node="node_1",
            data={"key": "value"},
            iteration=5,
        )
        result = model.to_dict()

        # Should have underscore prefixes for TypedDict compatibility
        assert result["_workflow_id"] == "test_workflow"
        assert result["_workflow_name"] == "My Workflow"
        assert result["_current_node"] == "node_1"
        assert result["data"]["key"] == "value"
        assert result["_iteration"] == 5

    def test_from_dict(self):
        """from_dict should create model from TypedDict-compatible dict."""
        data = {
            "_workflow_id": "test_workflow",
            "_workflow_name": "My Workflow",
            "_current_node": "node_1",
            "data": {"key": "value"},
            "_iteration": 5,
        }
        model = WorkflowExecutionContextModel.from_dict(data)

        assert model.workflow_id == "test_workflow"
        assert model.workflow_name == "My Workflow"
        assert model.current_node == "node_1"
        assert model.data["key"] == "value"
        assert model.iteration == 5

    def test_roundtrip_conversion(self):
        """to_dict and from_dict should be inverses."""
        original = WorkflowExecutionContextModel(
            workflow_name="test",
            current_node="node_1",
            data={"key": "value"},
        )

        # Convert to dict and back
        dict_form = original.to_dict()
        restored = WorkflowExecutionContextModel.from_dict(dict_form)

        assert restored.workflow_name == original.workflow_name
        assert restored.current_node == original.current_node
        assert restored.data == original.data

    def test_add_node_result(self):
        """add_node_result should add results to context."""
        model = WorkflowExecutionContextModel()
        model.add_node_result("node1", {"output": "success"})

        assert "node1" in model.node_results
        assert model.node_results["node1"]["output"] == "success"

    def test_visit_node(self):
        """visit_node should track visited nodes."""
        model = WorkflowExecutionContextModel()
        model.visit_node("node1")
        model.visit_node("node2")

        assert model.visited_nodes == ["node1", "node2"]

    def test_visit_node_deduplication(self):
        """visit_node should not add duplicates."""
        model = WorkflowExecutionContextModel()
        model.visit_node("node1")
        model.visit_node("node1")  # Visit again

        assert model.visited_nodes == ["node1"]  # No duplicate

    def test_increment_iteration(self):
        """increment_iteration should increase counter."""
        model = WorkflowExecutionContextModel(iteration=0)
        model.increment_iteration()
        assert model.iteration == 1

        model.increment_iteration()
        assert model.iteration == 2

    def test_mark_complete(self):
        """mark_complete should set completion status."""
        model = WorkflowExecutionContextModel()
        assert not model.is_complete
        assert not model.success

        model.mark_complete(success=True)
        assert model.is_complete
        assert model.success


class TestWorkflowStateModel:
    """Test WorkflowStateModel Pydantic model."""

    def test_default_values(self):
        """Model should create with sensible defaults."""
        model = WorkflowStateModel()
        assert model.workflow_id != ""  # Auto-generated UUID
        assert model.workflow_name == ""
        assert model.current_node == ""
        assert model.node_results == {}
        assert model.error is None
        assert model.iteration == 0
        assert model.parallel_results == {}
        assert not model.hitl_pending
        assert model.hitl_response is None

    def test_custom_values(self):
        """Model should accept custom values."""
        model = WorkflowStateModel(
            workflow_name="test_workflow",
            current_node="node_1",
            node_results={"node1": "result"},
        )
        assert model.workflow_name == "test_workflow"
        assert model.current_node == "node_1"
        assert model.node_results["node1"] == "result"

    def test_to_dict(self):
        """to_dict should convert to TypedDict-compatible format."""
        model = WorkflowStateModel(
            workflow_id="test_workflow",
            workflow_name="My Workflow",
            current_node="node_1",
        )
        result = model.to_dict()

        # Should have underscore prefixes
        assert result["_workflow_id"] == "test_workflow"
        assert result["_workflow_name"] == "My Workflow"
        assert result["_current_node"] == "node_1"

    def test_from_dict(self):
        """from_dict should create model from TypedDict-compatible dict."""
        data = {
            "_workflow_id": "test_workflow",
            "_workflow_name": "My Workflow",
            "_current_node": "node_1",
        }
        model = WorkflowStateModel.from_dict(data)

        assert model.workflow_id == "test_workflow"
        assert model.workflow_name == "My Workflow"
        assert model.current_node == "node_1"


class TestStateAdapter:
    """Test generic StateAdapter utilities."""

    def test_to_dict(self):
        """StateAdapter.to_dict should convert Pydantic models to dict."""
        model = WorkflowExecutionContextModel(workflow_name="test")
        result = StateAdapter.to_dict(model)

        assert isinstance(result, dict)
        assert result["_workflow_name"] == "test"

    def test_from_dict(self):
        """StateAdapter.from_dict should create Pydantic model from dict."""
        data = {"_workflow_id": "test", "_workflow_name": "workflow"}
        model = StateAdapter.from_dict(data, WorkflowExecutionContextModel)

        assert isinstance(model, WorkflowExecutionContextModel)
        assert model.workflow_id == "test"

    def test_roundtrip_conversion(self):
        """to_dict and from_dict should be inverses."""
        original = WorkflowExecutionContextModel(workflow_name="test")
        dict_form = StateAdapter.to_dict(original)
        restored = StateAdapter.from_dict(dict_form, WorkflowExecutionContextModel)

        assert restored.workflow_name == original.workflow_name
        assert restored.data == original.data


class TestWorkflowExecutionContextAdapter:
    """Test specialized WorkflowExecutionContextAdapter utilities."""

    def test_create_initial(self):
        """create_initial should create context with defaults."""
        model = WorkflowExecutionContextAdapter.create_initial(
            workflow_name="test_workflow",
            current_node="node_1",
        )

        assert model.workflow_name == "test_workflow"
        assert model.current_node == "node_1"
        assert model.data == {}
        assert model.messages == []

    def test_create_initial_with_custom_data(self):
        """create_initial should merge initial data."""
        model = WorkflowExecutionContextAdapter.create_initial(
            workflow_name="test",
            initial_data={"key": "value"},
        )

        assert model.data["key"] == "value"

    def test_update_from_legacy(self):
        """update_from_legacy should update model from legacy dict."""
        original = WorkflowExecutionContextModel(
            workflow_name="original",
            current_node="node_1",
            iteration=0,
        )

        legacy_update = {
            "_workflow_name": "updated",
            "_current_node": "node_2",
            "_iteration": 5,
            "data": {"new_key": "new_value"},
        }

        updated = WorkflowExecutionContextAdapter.update_from_legacy(original, legacy_update)

        assert updated.workflow_name == "updated"
        assert updated.current_node == "node_2"
        assert updated.iteration == 5
        assert updated.data["new_key"] == "new_value"

        # Original should be unchanged (immutable pattern)
        assert original.workflow_name == "original"
        assert original.iteration == 0

    def test_update_from_legacy_preserves_unmentioned_fields(self):
        """update_from_legacy should preserve fields not in legacy update."""
        original = WorkflowExecutionContextModel(
            workflow_name="test",
            hitl_pending=True,
        )

        legacy_update = {
            "_iteration": 10,
        }

        updated = WorkflowExecutionContextAdapter.update_from_legacy(original, legacy_update)

        assert updated.iteration == 10
        assert updated.workflow_name == "test"  # Preserved
        assert updated.hitl_pending is True  # Preserved
