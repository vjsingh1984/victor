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

"""Tests for RAG workflow provider."""

import pytest

from victor.core.verticals.protocols import WorkflowProviderProtocol
from victor.workflows.definition import WorkflowDefinition


class TestRAGWorkflowProvider:
    """Tests for RAGWorkflowProvider."""

    def test_implements_protocol(self):
        """RAGWorkflowProvider should implement WorkflowProviderProtocol."""
        from victor.rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        assert isinstance(provider, WorkflowProviderProtocol)

    def test_get_workflows_returns_dict(self):
        """get_workflows should return a dict of workflow definitions."""
        from victor.rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        workflows = provider.get_workflows()

        assert isinstance(workflows, dict)
        assert len(workflows) >= 3  # ingest, query, maintenance

    def test_has_ingest_workflow(self):
        """Provider should have an ingest workflow."""
        from victor.rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        workflows = provider.get_workflows()

        assert (
            "ingest" in workflows
            or "document_ingest" in workflows
            or "ingest_workflow" in workflows
        )
        # Get the ingest workflow (check different possible names)
        ingest_key = next(k for k in workflows if "ingest" in k.lower())
        workflow = workflows[ingest_key]
        assert isinstance(workflow, WorkflowDefinition)

    def test_has_query_workflow(self):
        """Provider should have a query workflow."""
        from victor.rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        workflows = provider.get_workflows()

        # Check for query workflow
        query_key = next((k for k in workflows if "query" in k.lower()), None)
        assert query_key is not None, "Should have a query workflow"
        workflow = workflows[query_key]
        assert isinstance(workflow, WorkflowDefinition)

    def test_has_maintenance_workflow(self):
        """Provider should have a maintenance workflow."""
        from victor.rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        workflows = provider.get_workflows()

        # Check for maintenance workflow
        maint_key = next(
            (k for k in workflows if "maintenance" in k.lower() or "optimize" in k.lower()), None
        )
        assert maint_key is not None, "Should have a maintenance workflow"
        workflow = workflows[maint_key]
        assert isinstance(workflow, WorkflowDefinition)

    def test_get_workflow_by_name(self):
        """get_workflow should return a specific workflow by name."""
        from victor.rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        workflows = provider.get_workflows()
        first_name = next(iter(workflows.keys()))

        workflow = provider.get_workflow(first_name)
        assert workflow is not None
        assert isinstance(workflow, WorkflowDefinition)

    def test_get_workflow_names(self):
        """get_workflow_names should return list of workflow names."""
        from victor.rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        names = provider.get_workflow_names()

        assert isinstance(names, list)
        assert len(names) >= 3

    def test_get_auto_workflows(self):
        """get_auto_workflows should return pattern-workflow tuples."""
        from victor.rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        auto_workflows = provider.get_auto_workflows()

        assert isinstance(auto_workflows, list)
        # Should have at least some auto-trigger patterns
        assert len(auto_workflows) >= 1
        # Each entry should be (pattern, workflow_name) tuple
        for pattern, workflow_name in auto_workflows:
            assert isinstance(pattern, str)
            assert isinstance(workflow_name, str)

    def test_workflow_validation(self):
        """All workflows should pass validation."""
        from victor.rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        workflows = provider.get_workflows()

        for name, workflow in workflows.items():
            errors = workflow.validate()
            assert not errors, f"Workflow '{name}' has validation errors: {errors}"

    def test_workflow_has_agents(self):
        """Each workflow should have at least one agent node (except utility workflows)."""
        from victor.rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        workflows = provider.get_workflows()

        # Some workflows are utility/compute-only (no LLM agents)
        utility_workflows = {"incremental_update", "maintenance"}

        for name, workflow in workflows.items():
            if name in utility_workflows:
                # Utility workflows may not have agent nodes
                continue
            agent_count = workflow.get_agent_count()
            assert agent_count >= 1, f"Workflow '{name}' should have at least one agent"

    def test_repr(self):
        """Provider should have a useful repr."""
        from victor.rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        repr_str = repr(provider)

        assert "RAGWorkflowProvider" in repr_str


class TestIngestWorkflow:
    """Tests for the document ingest workflow."""

    def test_ingest_workflow_structure(self):
        """Ingest workflow should have proper structure."""
        from victor.rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        workflow = provider.get_workflow("document_ingest")
        assert workflow is not None
        assert isinstance(workflow, WorkflowDefinition)
        assert "ingest" in workflow.name.lower()

    def test_ingest_has_parse_step(self):
        """Ingest workflow should have a parsing/processing step."""
        from victor.rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        workflow = provider.get_workflow("document_ingest")
        assert workflow is not None
        nodes = workflow.nodes

        # Should have a node related to parsing/processing
        has_parse = any(
            "parse" in node_id.lower()
            or "process" in node_id.lower()
            or "extract" in node_id.lower()
            or "discover" in node_id.lower()
            for node_id in nodes
        )
        assert has_parse, "Ingest workflow should have a parse/process step"


class TestQueryWorkflow:
    """Tests for the query workflow."""

    def test_query_workflow_structure(self):
        """Query workflow should have proper structure."""
        from victor.rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        workflow = provider.get_workflow("rag_query")
        assert workflow is not None
        assert isinstance(workflow, WorkflowDefinition)
        assert "query" in workflow.name.lower()

    def test_query_has_search_step(self):
        """Query workflow should have a search/retrieve step."""
        from victor.rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        workflow = provider.get_workflow("rag_query")
        assert workflow is not None
        nodes = workflow.nodes

        # Should have a node related to search/retrieve
        has_search = any(
            "search" in node_id.lower() or "retrieve" in node_id.lower() for node_id in nodes
        )
        assert has_search, "Query workflow should have a search/retrieve step"

    def test_query_has_synthesis_step(self):
        """Query workflow should have a synthesis/answer step."""
        from victor.rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        workflow = provider.get_workflow("rag_query")
        assert workflow is not None
        nodes = workflow.nodes

        # Should have a node related to synthesis
        has_synthesis = any(
            "synth" in node_id.lower()
            or "answer" in node_id.lower()
            or "generate" in node_id.lower()
            for node_id in nodes
        )
        assert has_synthesis, "Query workflow should have a synthesis step"


class TestMaintenanceWorkflow:
    """Tests for the maintenance workflow."""

    def test_maintenance_workflow_structure(self):
        """Maintenance workflow should have proper structure."""
        from victor.rag.workflows import RAGWorkflowProvider

        provider = RAGWorkflowProvider()
        workflow = provider.get_workflow("maintenance")
        assert workflow is not None
        assert isinstance(workflow, WorkflowDefinition)
        assert "maintenance" in workflow.name.lower() or "optimize" in workflow.name.lower()
