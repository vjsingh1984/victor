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

"""Tests for WorkflowRegistry and TeamSpecRegistry hash-based idempotence.

Tests that registries skip validation when definitions haven't changed,
improving startup performance through idempotent registration.
"""

import pytest
from unittest.mock import Mock

from victor.workflows.registry import WorkflowRegistry
from victor.workflows.definition import WorkflowDefinition, AgentNode
from victor.framework.team_registry import TeamSpecRegistry


class MockWorkflow:
    """Mock workflow for testing."""

    def __init__(self, name: str, description: str = "Test workflow"):
        self.name = name
        self.description = description
        self.nodes = {}
        self.metadata = {}
        self.validate_count = 0

    def validate(self):
        """Track validation calls."""
        self.validate_count += 1
        return []  # No errors

    def get_agent_count(self):
        """Return agent count for registry."""
        return 0

    def get_total_budget(self):
        """Return total budget for registry."""
        return 0

    def to_hash(self):
        """Return hash for idempotence testing."""
        import hashlib

        return hashlib.sha256(f"{self.name}:{self.description}".encode()).hexdigest()


class TestWorkflowRegistryIdempotence:
    """Tests for hash-based idempotence in WorkflowRegistry."""

    def test_workflow_has_to_hash_method(self, reset_singletons):
        """WorkflowDefinition should have a to_hash() method.

        Phase 4 implementation: to_hash() computes a deterministic hash
        from the workflow definition for change detection.
        """
        # Create a simple workflow
        workflow = WorkflowDefinition(
            name="test_workflow",
            description="Test workflow",
            nodes={
                "node1": AgentNode(
                    id="node1", name="Test Node", role="test_agent", goal="Do something"
                )
            },
            start_node="node1",
        )

        # Should have to_hash method
        assert hasattr(workflow, "to_hash"), "WorkflowDefinition should have to_hash() method"

        # Hash should be string
        hash_value = workflow.to_hash()
        assert isinstance(hash_value, str), "to_hash() should return string"
        assert len(hash_value) == 64, "SHA256 hash should be 64 characters (hex)"

    def test_workflow_to_hash_is_deterministic(self, reset_singletons):
        """to_hash() should return same hash for same workflow.

        Phase 4: Hash is deterministic based on workflow structure.
        """
        workflow = WorkflowDefinition(
            name="test_workflow",
            description="Test workflow",
            nodes={
                "node1": AgentNode(
                    id="node1", name="Test Node", role="test_agent", goal="Do something"
                )
            },
            start_node="node1",
        )

        hash1 = workflow.to_hash()
        hash2 = workflow.to_hash()

        assert hash1 == hash2, "to_hash() should be deterministic"

    def test_workflow_to_hash_detects_changes(self, reset_singletons):
        """to_hash() should detect changes to workflow definition.

        Phase 4: Different workflows should have different hashes.
        """
        workflow1 = WorkflowDefinition(
            name="test_workflow",
            description="Test workflow",
            nodes={
                "node1": AgentNode(
                    id="node1", name="Test Node", role="test_agent", goal="Do something"
                )
            },
            start_node="node1",
        )

        workflow2 = WorkflowDefinition(
            name="test_workflow",
            description="Modified workflow",  # Changed!
            nodes={
                "node1": AgentNode(
                    id="node1",
                    name="Test Node",
                    role="test_agent",
                    goal="Do something else",  # Changed!
                )
            },
            start_node="node1",
        )

        hash1 = workflow1.to_hash()
        hash2 = workflow2.to_hash()

        assert hash1 != hash2, "Different workflows should have different hashes"

    def test_workflow_registry_skips_validation_when_unchanged(self, reset_singletons):
        """Registry should skip validation when workflow hash matches.

        Phase 4 implementation: Hash-based idempotence means:
        1. First registration: validates and stores hash
        2. Second registration with same workflow: skips validation (fast!)
        """
        registry = WorkflowRegistry()

        # Create mock workflow that tracks validation calls
        workflow = MockWorkflow("test_workflow", "Test workflow")

        # First registration - should validate
        registry.register(workflow)
        assert workflow.validate_count == 1, "First registration should validate"

        # Second registration with same workflow - should skip validation
        # (Note: need replace=True to re-register same name)
        registry.register(workflow, replace=True)
        # With idempotence, this should NOT call validate() again
        # But current implementation doesn't have this yet

        # This test will fail initially, then pass after implementation
        assert (
            workflow.validate_count == 1
        ), "Second registration with same workflow should skip validation"

    def test_workflow_registry_validates_when_changed(self, reset_singletons):
        """Registry should validate when workflow definition changes.

        Phase 4: Hash change triggers re-validation.
        """
        registry = WorkflowRegistry()

        workflow1 = MockWorkflow("test_workflow", "Version 1")
        registry.register(workflow1)
        first_count = workflow1.validate_count

        # Register modified workflow
        workflow2 = MockWorkflow("test_workflow", "Version 2")
        registry.register(workflow2, replace=True)

        # Should validate again because workflow changed
        assert workflow2.validate_count == 1, "Modified workflow should trigger validation"


class TestTeamSpecRegistryIdempotence:
    """Tests for hash-based idempotence in TeamSpecRegistry."""

    def test_team_spec_registry_skips_validation_when_unchanged(self, reset_singletons):
        """TeamSpecRegistry should skip processing when team specs unchanged.

        Phase 4 implementation: register_from_vertical should skip
        reprocessing when team specs haven't changed.
        """
        registry = TeamSpecRegistry()

        # Create mock team spec
        team_spec = Mock()
        team_spec.members = []

        # First registration
        team_specs = {"team1": team_spec}
        count1 = registry.register_from_vertical("test_vertical", team_specs)
        assert count1 == 1

        # Second registration with same specs - should skip processing
        # (replace=True to force re-registration attempt)
        count2 = registry.register_from_vertical("test_vertical", team_specs, replace=True)
        # With idempotence, should still return count but skip actual processing
        assert count2 == 1  # Still returns count even if skipped processing

    def test_team_spec_registry_processes_when_changed(self, reset_singletons):
        """TeamSpecRegistry should process when team specs change.

        Phase 4: Change detection triggers reprocessing.
        """
        registry = TeamSpecRegistry()

        # First registration
        team_specs1 = {"team1": Mock(spec="v1")}
        count1 = registry.register_from_vertical("test_vertical", team_specs1, replace=True)
        assert count1 == 1

        # Second registration with different specs
        team_specs2 = {"team2": Mock(spec="v2")}
        count2 = registry.register_from_vertical("test_vertical", team_specs2, replace=True)
        assert count2 == 1

        # Verify both teams are registered
        assert "test_vertical:team1" in registry._teams
        assert "test_vertical:team2" in registry._teams


class TestIdempotencePerformance:
    """Tests for performance improvements from idempotence."""

    def test_idempotent_registration_is_faster(self, reset_singletons):
        """Skipping validation should significantly improve performance.

        Phase 4: Demonstrate performance benefit of hash-based idempotence.
        """
        import time

        registry = WorkflowRegistry()

        # Create workflow that simulates expensive validation
        class SlowWorkflow:
            def __init__(self, name):
                self.name = name
                self.nodes = {}
                self.description = "Slow workflow"
                self.metadata = {}
                self.validate_count = 0

            def validate(self):
                # Simulate expensive validation
                self.validate_count += 1
                time.sleep(0.01)  # 10ms delay
                return []

            def get_agent_count(self):
                return 0

            def get_total_budget(self):
                return 0

            def to_hash(self):
                import hashlib

                return hashlib.sha256(f"{self.name}:{self.description}".encode()).hexdigest()

        workflow = SlowWorkflow("slow_workflow")

        # First registration - should be slow
        start = time.time()
        registry.register(workflow)
        first_duration = time.time() - start

        # Second registration - should be fast (skip validation)
        start = time.time()
        registry.register(workflow, replace=True)
        second_duration = time.time() - start

        # First should have validated, second should skip
        assert workflow.validate_count == 1, "Second registration should skip validation"

        # Second should be faster (no validation delay)
        assert (
            second_duration < first_duration / 2
        ), f"Idempotent registration should be faster: {second_duration:.4f}s vs {first_duration:.4f}s"
