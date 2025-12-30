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

"""Integration tests for vertical workflow provider streaming support.

Tests verify that all vertical workflow providers properly implement:
- create_executor() - Returns WorkflowExecutor
- create_streaming_executor() - Returns StreamingWorkflowExecutor
- astream() - Async generator yielding WorkflowStreamChunk events

Verticals tested:
- CodingWorkflowProvider
- DevOpsWorkflowProvider
- ResearchWorkflowProvider
- DataAnalysisWorkflowProvider
"""

import pytest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from victor.workflows.executor import WorkflowExecutor
from victor.workflows.streaming_executor import (
    StreamingWorkflowExecutor,
    WorkflowEventType,
    WorkflowStreamChunk,
)


# ============ Test Fixtures ============


def create_mock_orchestrator():
    """Create a mock orchestrator for testing without real LLM calls."""
    mock_orchestrator = MagicMock()
    mock_orchestrator.settings = MagicMock()
    mock_orchestrator.settings.tool_budget = 15
    return mock_orchestrator


@pytest.fixture
def mock_orchestrator():
    """Fixture providing a mock orchestrator."""
    return create_mock_orchestrator()


@pytest.fixture
def coding_provider():
    """Fixture providing CodingWorkflowProvider."""
    from victor.coding.workflows.provider import CodingWorkflowProvider

    return CodingWorkflowProvider()


@pytest.fixture
def devops_provider():
    """Fixture providing DevOpsWorkflowProvider."""
    from victor.devops.workflows import DevOpsWorkflowProvider

    return DevOpsWorkflowProvider()


@pytest.fixture
def research_provider():
    """Fixture providing ResearchWorkflowProvider."""
    from victor.research.workflows import ResearchWorkflowProvider

    return ResearchWorkflowProvider()


@pytest.fixture
def dataanalysis_provider():
    """Fixture providing DataAnalysisWorkflowProvider."""
    from victor.dataanalysis.workflows import DataAnalysisWorkflowProvider

    return DataAnalysisWorkflowProvider()


# ============ CodingWorkflowProvider Tests ============


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.workflows
class TestCodingWorkflowProviderStreaming:
    """Tests for CodingWorkflowProvider streaming support."""

    def test_create_executor_returns_workflow_executor(self, mock_orchestrator, coding_provider):
        """Test create_executor returns a WorkflowExecutor instance."""
        executor = coding_provider.create_executor(mock_orchestrator)
        assert isinstance(executor, WorkflowExecutor)

    def test_create_streaming_executor_returns_streaming_executor(
        self, mock_orchestrator, coding_provider
    ):
        """Test create_streaming_executor returns StreamingWorkflowExecutor."""
        executor = coding_provider.create_streaming_executor(mock_orchestrator)
        assert isinstance(executor, StreamingWorkflowExecutor)

    async def test_astream_raises_for_unknown_workflow(self, mock_orchestrator, coding_provider):
        """Test astream raises ValueError for unknown workflow name."""
        with pytest.raises(ValueError, match="Unknown workflow: nonexistent"):
            async for _ in coding_provider.astream("nonexistent", mock_orchestrator, {}):
                pass

    def test_provider_has_workflows(self, coding_provider):
        """Test provider has expected workflows available."""
        workflows = coding_provider.get_workflow_names()
        assert len(workflows) > 0
        # Should have common coding workflows
        assert "code_review" in workflows or "feature_implementation" in workflows

    def test_get_workflow_returns_valid_definition(self, coding_provider):
        """Test get_workflow returns valid workflow definitions."""
        workflow_names = coding_provider.get_workflow_names()
        for name in workflow_names:
            workflow = coding_provider.get_workflow(name)
            assert workflow is not None
            assert hasattr(workflow, "name")
            assert hasattr(workflow, "nodes")


# ============ DevOpsWorkflowProvider Tests ============


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.workflows
class TestDevOpsWorkflowProviderStreaming:
    """Tests for DevOpsWorkflowProvider streaming support."""

    def test_create_executor_returns_workflow_executor(self, mock_orchestrator, devops_provider):
        """Test create_executor returns a WorkflowExecutor instance."""
        executor = devops_provider.create_executor(mock_orchestrator)
        assert isinstance(executor, WorkflowExecutor)

    def test_create_streaming_executor_returns_streaming_executor(
        self, mock_orchestrator, devops_provider
    ):
        """Test create_streaming_executor returns StreamingWorkflowExecutor."""
        executor = devops_provider.create_streaming_executor(mock_orchestrator)
        assert isinstance(executor, StreamingWorkflowExecutor)

    async def test_astream_raises_for_unknown_workflow(self, mock_orchestrator, devops_provider):
        """Test astream raises ValueError for unknown workflow name."""
        with pytest.raises(ValueError, match="Unknown workflow: nonexistent"):
            async for _ in devops_provider.astream("nonexistent", mock_orchestrator, {}):
                pass

    def test_provider_has_workflows(self, devops_provider):
        """Test provider has expected workflows available."""
        workflows = devops_provider.get_workflow_names()
        assert len(workflows) > 0
        # Should have common devops workflows
        assert "deploy_infrastructure" in workflows or "cicd_pipeline" in workflows

    def test_get_workflow_returns_valid_definition(self, devops_provider):
        """Test get_workflow returns valid workflow definitions."""
        workflow_names = devops_provider.get_workflow_names()
        for name in workflow_names:
            workflow = devops_provider.get_workflow(name)
            assert workflow is not None
            assert hasattr(workflow, "name")
            assert hasattr(workflow, "nodes")


# ============ ResearchWorkflowProvider Tests ============


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.workflows
class TestResearchWorkflowProviderStreaming:
    """Tests for ResearchWorkflowProvider streaming support."""

    def test_create_executor_returns_workflow_executor(self, mock_orchestrator, research_provider):
        """Test create_executor returns a WorkflowExecutor instance."""
        executor = research_provider.create_executor(mock_orchestrator)
        assert isinstance(executor, WorkflowExecutor)

    def test_create_streaming_executor_returns_streaming_executor(
        self, mock_orchestrator, research_provider
    ):
        """Test create_streaming_executor returns StreamingWorkflowExecutor."""
        executor = research_provider.create_streaming_executor(mock_orchestrator)
        assert isinstance(executor, StreamingWorkflowExecutor)

    async def test_astream_raises_for_unknown_workflow(self, mock_orchestrator, research_provider):
        """Test astream raises ValueError for unknown workflow name."""
        with pytest.raises(ValueError, match="Unknown workflow: nonexistent"):
            async for _ in research_provider.astream("nonexistent", mock_orchestrator, {}):
                pass

    def test_provider_has_workflows(self, research_provider):
        """Test provider has expected workflows available."""
        workflows = research_provider.get_workflow_names()
        assert len(workflows) > 0
        # Should have common research workflows
        assert "deep_research" in workflows or "fact_check" in workflows

    def test_get_workflow_returns_valid_definition(self, research_provider):
        """Test get_workflow returns valid workflow definitions."""
        workflow_names = research_provider.get_workflow_names()
        for name in workflow_names:
            workflow = research_provider.get_workflow(name)
            assert workflow is not None
            assert hasattr(workflow, "name")
            assert hasattr(workflow, "nodes")


# ============ DataAnalysisWorkflowProvider Tests ============


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.workflows
class TestDataAnalysisWorkflowProviderStreaming:
    """Tests for DataAnalysisWorkflowProvider streaming support."""

    def test_create_executor_returns_workflow_executor(
        self, mock_orchestrator, dataanalysis_provider
    ):
        """Test create_executor returns a WorkflowExecutor instance."""
        executor = dataanalysis_provider.create_executor(mock_orchestrator)
        assert isinstance(executor, WorkflowExecutor)

    def test_create_streaming_executor_returns_streaming_executor(
        self, mock_orchestrator, dataanalysis_provider
    ):
        """Test create_streaming_executor returns StreamingWorkflowExecutor."""
        executor = dataanalysis_provider.create_streaming_executor(mock_orchestrator)
        assert isinstance(executor, StreamingWorkflowExecutor)

    async def test_astream_raises_for_unknown_workflow(
        self, mock_orchestrator, dataanalysis_provider
    ):
        """Test astream raises ValueError for unknown workflow name."""
        with pytest.raises(ValueError, match="Unknown workflow: nonexistent"):
            async for _ in dataanalysis_provider.astream("nonexistent", mock_orchestrator, {}):
                pass

    def test_provider_has_workflows(self, dataanalysis_provider):
        """Test provider has expected workflows available."""
        workflows = dataanalysis_provider.get_workflow_names()
        assert len(workflows) > 0
        # Should have common data analysis workflows
        assert "eda_workflow" in workflows or "ml_pipeline" in workflows

    def test_get_workflow_returns_valid_definition(self, dataanalysis_provider):
        """Test get_workflow returns valid workflow definitions."""
        workflow_names = dataanalysis_provider.get_workflow_names()
        for name in workflow_names:
            workflow = dataanalysis_provider.get_workflow(name)
            assert workflow is not None
            assert hasattr(workflow, "name")
            assert hasattr(workflow, "nodes")


# ============ Cross-Vertical Tests ============


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.workflows
class TestCrossVerticalStreamingConsistency:
    """Tests for consistent streaming API across all verticals."""

    @pytest.fixture
    def all_providers(
        self,
        coding_provider,
        devops_provider,
        research_provider,
        dataanalysis_provider,
    ):
        """Fixture providing all workflow providers."""
        return [
            ("coding", coding_provider),
            ("devops", devops_provider),
            ("research", research_provider),
            ("dataanalysis", dataanalysis_provider),
        ]

    def test_all_providers_have_create_executor(self, mock_orchestrator, all_providers):
        """Test all providers implement create_executor."""
        for name, provider in all_providers:
            assert hasattr(provider, "create_executor"), f"{name} provider missing create_executor"
            executor = provider.create_executor(mock_orchestrator)
            assert isinstance(executor, WorkflowExecutor), f"{name} create_executor wrong type"

    def test_all_providers_have_create_streaming_executor(self, mock_orchestrator, all_providers):
        """Test all providers implement create_streaming_executor."""
        for name, provider in all_providers:
            assert hasattr(
                provider, "create_streaming_executor"
            ), f"{name} provider missing create_streaming_executor"
            executor = provider.create_streaming_executor(mock_orchestrator)
            assert isinstance(
                executor, StreamingWorkflowExecutor
            ), f"{name} create_streaming_executor wrong type"

    def test_all_providers_have_astream(self, all_providers):
        """Test all providers implement astream method."""
        for name, provider in all_providers:
            assert hasattr(provider, "astream"), f"{name} provider missing astream"
            # Verify it's callable
            assert callable(provider.astream), f"{name} astream not callable"

    def test_all_providers_have_consistent_api(self, all_providers):
        """Test all providers have consistent API surface."""
        expected_methods = [
            "get_workflows",
            "get_workflow",
            "get_workflow_names",
            "create_executor",
            "create_streaming_executor",
            "astream",
        ]
        for name, provider in all_providers:
            for method in expected_methods:
                assert hasattr(provider, method), f"{name} provider missing {method}"
