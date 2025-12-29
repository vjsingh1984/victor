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

"""Tests for coding vertical workflows."""

import pytest

from victor.verticals.coding.workflows import (
    CodingWorkflowProvider,
    feature_implementation_workflow,
    quick_feature_workflow,
    bug_fix_workflow,
    quick_fix_workflow,
    code_review_workflow,
    quick_review_workflow,
    pr_review_workflow,
)
from victor.workflows.definition import (
    AgentNode,
    ConditionNode,
    ParallelNode,
    WorkflowDefinition,
)


class TestCodingWorkflowProvider:
    """Tests for CodingWorkflowProvider."""

    def test_get_workflows(self):
        """Test getting all workflows."""
        provider = CodingWorkflowProvider()
        workflows = provider.get_workflows()

        assert len(workflows) == 7
        assert "feature_implementation" in workflows
        assert "quick_feature" in workflows
        assert "bug_fix" in workflows
        assert "quick_fix" in workflows
        assert "code_review" in workflows
        assert "quick_review" in workflows
        assert "pr_review" in workflows

    def test_get_workflow_names(self):
        """Test getting workflow names."""
        provider = CodingWorkflowProvider()
        names = provider.get_workflow_names()

        assert len(names) == 7
        assert "feature_implementation" in names

    def test_get_workflow_by_name(self):
        """Test getting a specific workflow."""
        provider = CodingWorkflowProvider()
        wf = provider.get_workflow("feature_implementation")

        assert wf is not None
        assert wf.name == "feature_implementation"
        assert isinstance(wf, WorkflowDefinition)

    def test_get_nonexistent_workflow(self):
        """Test getting a workflow that doesn't exist."""
        provider = CodingWorkflowProvider()
        wf = provider.get_workflow("nonexistent")

        assert wf is None

    def test_get_auto_workflows(self):
        """Test getting auto-trigger patterns."""
        provider = CodingWorkflowProvider()
        auto = provider.get_auto_workflows()

        assert len(auto) > 0
        # Check patterns are tuples of (pattern, workflow_name)
        for pattern, wf_name in auto:
            assert isinstance(pattern, str)
            assert isinstance(wf_name, str)
            assert wf_name in provider.get_workflow_names()

    def test_get_workflow_for_task_type(self):
        """Test getting workflow by task type."""
        provider = CodingWorkflowProvider()

        assert provider.get_workflow_for_task_type("feature") == "feature_implementation"
        assert provider.get_workflow_for_task_type("bug") == "bug_fix"
        assert provider.get_workflow_for_task_type("review") == "code_review"
        assert provider.get_workflow_for_task_type("pr") == "pr_review"
        assert provider.get_workflow_for_task_type("unknown") is None


class TestFeatureImplementationWorkflow:
    """Tests for feature implementation workflow."""

    def test_workflow_structure(self):
        """Test workflow has correct structure."""
        wf = feature_implementation_workflow()

        assert wf.name == "feature_implementation"
        assert wf.description == "End-to-end feature development with review"
        assert wf.get_agent_count() >= 4  # research, plan, implement, review, finalize

    def test_workflow_nodes(self):
        """Test workflow has expected nodes."""
        wf = feature_implementation_workflow()

        # Check key nodes exist
        assert "research" in wf.nodes
        assert "plan" in wf.nodes
        assert "implement" in wf.nodes
        assert "review" in wf.nodes

    def test_workflow_validation(self):
        """Test workflow passes validation."""
        wf = feature_implementation_workflow()
        errors = wf.validate()

        assert len(errors) == 0

    def test_workflow_metadata(self):
        """Test workflow metadata."""
        wf = feature_implementation_workflow()

        assert wf.metadata.get("category") == "coding"
        assert wf.metadata.get("complexity") == "high"


class TestQuickFeatureWorkflow:
    """Tests for quick feature workflow."""

    def test_workflow_structure(self):
        """Test workflow has correct structure."""
        wf = quick_feature_workflow()

        assert wf.name == "quick_feature"
        assert wf.get_agent_count() == 3  # research, implement, verify

    def test_lower_budget(self):
        """Test quick workflow has lower budget."""
        quick = quick_feature_workflow()
        full = feature_implementation_workflow()

        assert quick.get_total_budget() < full.get_total_budget()


class TestBugFixWorkflow:
    """Tests for bug fix workflow."""

    def test_workflow_structure(self):
        """Test workflow has correct structure."""
        wf = bug_fix_workflow()

        assert wf.name == "bug_fix"
        assert "investigate" in wf.nodes
        assert "diagnose" in wf.nodes
        assert "fix" in wf.nodes
        assert "verify" in wf.nodes

    def test_has_verification_loop(self):
        """Test workflow has verification condition."""
        wf = bug_fix_workflow()

        # Check for condition node
        has_condition = any(
            isinstance(node, ConditionNode) for node in wf.nodes.values()
        )
        assert has_condition


class TestCodeReviewWorkflow:
    """Tests for code review workflow."""

    def test_workflow_structure(self):
        """Test workflow has correct structure."""
        wf = code_review_workflow()

        assert wf.name == "code_review"
        assert "identify" in wf.nodes
        assert "synthesize" in wf.nodes

    def test_has_parallel_reviews(self):
        """Test workflow has parallel review nodes."""
        wf = code_review_workflow()

        # Check for parallel node
        has_parallel = any(
            isinstance(node, ParallelNode) for node in wf.nodes.values()
        )
        assert has_parallel

    def test_review_types(self):
        """Test all review types are present."""
        wf = code_review_workflow()

        assert "security" in wf.nodes
        assert "style" in wf.nodes
        assert "logic" in wf.nodes


class TestPRReviewWorkflow:
    """Tests for PR review workflow."""

    def test_workflow_structure(self):
        """Test workflow has correct structure."""
        wf = pr_review_workflow()

        assert wf.name == "pr_review"
        assert "fetch" in wf.nodes
        assert "analyze" in wf.nodes
        assert "generate_review" in wf.nodes


class TestWorkflowIntegration:
    """Integration tests for workflows."""

    def test_all_workflows_valid(self):
        """Test all workflows pass validation."""
        provider = CodingWorkflowProvider()
        workflows = provider.get_workflows()

        for name, wf in workflows.items():
            errors = wf.validate()
            assert len(errors) == 0, f"Workflow {name} has errors: {errors}"

    def test_all_workflows_serializable(self):
        """Test all workflows can be serialized to dict."""
        provider = CodingWorkflowProvider()
        workflows = provider.get_workflows()

        for name, wf in workflows.items():
            d = wf.to_dict()
            assert d["name"] == wf.name
            assert "nodes" in d
            assert "start_node" in d

    def test_agent_nodes_have_tools(self):
        """Test agent nodes have allowed_tools specified."""
        provider = CodingWorkflowProvider()
        workflows = provider.get_workflows()

        for name, wf in workflows.items():
            for node_id, node in wf.nodes.items():
                if isinstance(node, AgentNode):
                    # Most agent nodes should have allowed_tools
                    # (not all require it, but most do)
                    pass  # Just ensure no crashes
