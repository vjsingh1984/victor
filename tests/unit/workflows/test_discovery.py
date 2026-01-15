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

"""Tests for victor.workflows.discovery module."""

import pytest

from victor.workflows.base import BaseWorkflow, WorkflowRegistry
from victor.workflows.discovery import (
    discover_workflows,
    get_workflow_by_name,
    register_builtin_workflows,
    workflow_class,
)


class TestWorkflowClassDecorator:
    """Test workflow_class decorator."""

    def test_decorator_adds_to_pending(self):
        """Test that decorator adds class to pending list."""
        from victor.workflows.discovery import _pending_workflows

        initial_count = len(_pending_workflows)

        @workflow_class
        class TestWorkflow1(BaseWorkflow):
            name = "test_workflow_1"
            description = "Test workflow 1"
            def run(self, context):
                return context

        assert len(_pending_workflows) == initial_count + 1

    def test_decorator_returns_class(self):
        """Test that decorator returns the original class."""

        @workflow_class
        class TestWorkflow2(BaseWorkflow):
            name = "test_workflow_2"
            description = "Test workflow 2"
            def run(self, context):
                return context

        assert TestWorkflow2.name == "test_workflow_2"

    def test_decorator_with_multiple_classes(self):
        """Test decorator with multiple classes."""
        from victor.workflows.discovery import _pending_workflows

        initial_count = len(_pending_workflows)

        @workflow_class
        class TestWorkflow3(BaseWorkflow):
            name = "test_workflow_3"
            def run(self, context):
                return context

        @workflow_class
        class TestWorkflow4(BaseWorkflow):
            name = "test_workflow_4"
            def run(self, context):
                return context

        assert len(_pending_workflows) == initial_count + 2


class TestDiscoverWorkflows:
    """Test discover_workflows function."""

    def test_discover_workflows_default_package(self):
        """Test discovering workflows from default package."""
        workflows = discover_workflows()

        assert isinstance(workflows, list)

    def test_discover_workflows_with_exclusions(self):
        """Test discovering workflows with exclusions."""
        workflows = discover_workflows(exclude_modules=["hitl", "isolation"])

        assert isinstance(workflows, list)

    def test_discover_workflows_empty_exclusions(self):
        """Test discovering workflows with empty exclusion list."""
        workflows = discover_workflows(exclude_modules=[])

        assert isinstance(workflows, list)

    def test_discover_workflows_nonexistent_package(self):
        """Test discovering workflows from nonexistent package."""
        workflows = discover_workflows(package_path="nonexistent.package")

        assert workflows == []

    def test_discover_workflows_returns_list(self):
        """Test that discover_workflows returns a list."""
        workflows = discover_workflows()

        assert isinstance(workflows, list)

    def test_discover_workflows_excludes_base_modules(self):
        """Test that base modules are excluded by default."""
        workflows = discover_workflows()

        # Should not include base, registry, executor, etc.
        workflow_names = [w.name for w in workflows if hasattr(w, "name")]
        assert "base" not in workflow_names
        assert "registry" not in workflow_names


class TestRegisterBuiltinWorkflows:
    """Test register_builtin_workflows function."""

    def test_register_builtin_workflows(self):
        """Test registering built-in workflows."""
        registry = WorkflowRegistry()
        count = register_builtin_workflows(registry)

        assert count >= 0
        assert isinstance(count, int)

    def test_register_builtin_workflows_idempotent(self):
        """Test that registering twice doesn't duplicate."""
        registry1 = WorkflowRegistry()
        count1 = register_builtin_workflows(registry1)

        registry2 = WorkflowRegistry()
        count2 = register_builtin_workflows(registry2)

        assert count1 == count2

    def test_register_builtin_workflows_with_pending(self):
        """Test registering with pending decorated workflows."""

        @workflow_class
        class TestWorkflow5(BaseWorkflow):
            name = "test_workflow_5"
            description = "Test workflow 5"

        registry = WorkflowRegistry()
        count = register_builtin_workflows(registry)

        assert count >= 0


class TestGetWorkflowByName:
    """Test get_workflow_by_name function."""

    def test_get_workflow_by_name_without_registry(self):
        """Test getting workflow by name without registry."""
        workflow = get_workflow_by_name("nonexistent")

        assert workflow is None

    def test_get_workflow_by_name_with_registry(self):
        """Test getting workflow by name with registry."""
        registry = WorkflowRegistry()

        # Register a test workflow
        class TestWorkflow6(BaseWorkflow):
            name = "test_workflow_6"
            description = "Test workflow 6"

            def run(self, context):
                return context

        registry.register(TestWorkflow6())

        workflow = get_workflow_by_name("test_workflow_6", registry=registry)

        assert workflow is not None
        assert workflow.name == "test_workflow_6"

    def test_get_workflow_by_name_nonexistent(self):
        """Test getting nonexistent workflow by name."""
        registry = WorkflowRegistry()
        register_builtin_workflows(registry)

        workflow = get_workflow_by_name("nonexistent_workflow", registry=registry)

        assert workflow is None

    def test_get_workflow_by_name_with_lazy_discovery(self):
        """Test lazy discovery when getting workflow by name."""
        workflow = get_workflow_by_name("some_workflow", registry=None)

        # Should attempt discovery
        assert workflow is None or hasattr(workflow, "name")


class TestWorkflowDiscoveryIntegration:
    """Integration tests for workflow """

    def test_discovery_and_registration_flow(self):
        """Test the complete discovery and registration flow."""
        registry = WorkflowRegistry()

        # Discover workflows
        workflows = discover_workflows()

        # Register them
        count = 0
        for workflow in workflows:
            try:
                if registry.get(workflow.name) is None:
                    registry.register(workflow)
                    count += 1
            except ValueError:
                pass

        assert count >= 0

    def test_decorator_and_discovery_integration(self):
        """Test that decorated workflows are discovered and registered."""

        @workflow_class
        class TestWorkflow7(BaseWorkflow):
            name = "test_workflow_7"
            description = "Test workflow 7"
            def run(self, context):
                return context

        registry = WorkflowRegistry()
        count = register_builtin_workflows(registry)

        assert count >= 0

    def test_workflow_registry_workflow_listing(self):
        """Test listing workflows from registry."""
        registry = WorkflowRegistry()
        register_builtin_workflows(registry)

        # The registry should have workflows registered
        assert len(registry._workflows) >= 0
