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

"""Integration tests for vertical workflow provider canonical API.

Tests verify that all vertical workflow providers properly implement:
- compile_workflow() - Returns CompiledGraph (CANONICAL)
- invoke() - Executes workflow with caching (CANONICAL)
- stream() - Streams workflow execution (CANONICAL)
- run_compiled_workflow() - Convenience wrapper for invoke
- stream_compiled_workflow() - Convenience wrapper for stream

Verticals tested:
- CodingWorkflowProvider
- DevOpsWorkflowProvider
- ResearchWorkflowProvider
- DataAnalysisWorkflowProvider
"""

import pytest
from typing import Any


# ============ Test Fixtures ============


@pytest.fixture
def coding_provider() -> Any:
    """Fixture providing CodingWorkflowProvider."""
    from victor.coding.workflows.provider import CodingWorkflowProvider

    return CodingWorkflowProvider()


@pytest.fixture
def devops_provider() -> Any:
    """Fixture providing DevOpsWorkflowProvider."""
    from victor.devops.workflows import DevOpsWorkflowProvider

    return DevOpsWorkflowProvider()


@pytest.fixture
def research_provider() -> Any:
    """Fixture providing ResearchWorkflowProvider."""
    from victor.research.workflows import ResearchWorkflowProvider

    return ResearchWorkflowProvider()


@pytest.fixture
def dataanalysis_provider() -> Any:
    """Fixture providing DataAnalysisWorkflowProvider."""
    from victor.dataanalysis.workflows import DataAnalysisWorkflowProvider

    return DataAnalysisWorkflowProvider()


# ============ Per-Vertical Workflow Tests ============


@pytest.mark.integration
@pytest.mark.workflows
class TestCodingWorkflowProvider:
    """Tests for CodingWorkflowProvider canonical API."""

    def test_provider_has_workflows(self, coding_provider: Any) -> None:
        """Test provider has expected workflows available."""
        workflows = coding_provider.get_workflow_names()
        assert len(workflows) > 0
        # Should have common coding workflows
        assert "code_review" in workflows or "feature_implementation" in workflows

    def test_get_workflow_returns_valid_definition(self, coding_provider: Any) -> None:
        """Test get_workflow returns valid workflow definitions."""
        workflow_names = coding_provider.get_workflow_names()
        for name in workflow_names:
            workflow = coding_provider.get_workflow(name)
            assert workflow is not None
            assert hasattr(workflow, "name")
            assert hasattr(workflow, "nodes")


@pytest.mark.integration
@pytest.mark.workflows
class TestDevOpsWorkflowProvider:
    """Tests for DevOpsWorkflowProvider canonical API."""

    def test_provider_has_workflows(self, devops_provider: Any) -> None:
        """Test provider has expected workflows available."""
        workflows = devops_provider.get_workflow_names()
        assert len(workflows) > 0
        # Should have common devops workflows (deploy, cicd, container_*)
        assert "deploy" in workflows or "cicd" in workflows

    def test_get_workflow_returns_valid_definition(self, devops_provider: Any) -> None:
        """Test get_workflow returns valid workflow definitions."""
        workflow_names = devops_provider.get_workflow_names()
        for name in workflow_names:
            workflow = devops_provider.get_workflow(name)
            assert workflow is not None
            assert hasattr(workflow, "name")
            assert hasattr(workflow, "nodes")


@pytest.mark.integration
@pytest.mark.workflows
class TestResearchWorkflowProvider:
    """Tests for ResearchWorkflowProvider canonical API."""

    def test_provider_has_workflows(self, research_provider: Any) -> None:
        """Test provider has expected workflows available."""
        workflows = research_provider.get_workflow_names()
        assert len(workflows) > 0
        # Should have common research workflows
        assert "deep_research" in workflows or "fact_check" in workflows

    def test_get_workflow_returns_valid_definition(self, research_provider: Any) -> None:
        """Test get_workflow returns valid workflow definitions."""
        workflow_names = research_provider.get_workflow_names()
        for name in workflow_names:
            workflow = research_provider.get_workflow(name)
            assert workflow is not None
            assert hasattr(workflow, "name")
            assert hasattr(workflow, "nodes")


@pytest.mark.integration
@pytest.mark.workflows
class TestDataAnalysisWorkflowProvider:
    """Tests for DataAnalysisWorkflowProvider canonical API."""

    def test_provider_has_workflows(self, dataanalysis_provider: Any) -> None:
        """Test provider has expected workflows available."""
        workflows = dataanalysis_provider.get_workflow_names()
        assert len(workflows) > 0
        # Should have common data analysis workflows
        assert "eda_workflow" in workflows or "ml_pipeline" in workflows

    def test_get_workflow_returns_valid_definition(self, dataanalysis_provider: Any) -> None:
        """Test get_workflow returns valid workflow definitions."""
        workflow_names = dataanalysis_provider.get_workflow_names()
        for name in workflow_names:
            workflow = dataanalysis_provider.get_workflow(name)
            assert workflow is not None
            assert hasattr(workflow, "name")
            assert hasattr(workflow, "nodes")


# ============ Canonical API Tests (UnifiedWorkflowCompiler) ============


@pytest.mark.integration
@pytest.mark.workflows
class TestCanonicalWorkflowAPI:
    """Tests for canonical UnifiedWorkflowCompiler API across all verticals.

    Tests verify that all vertical workflow providers properly implement:
    - compile_workflow() - Returns CompiledGraph with caching
    - invoke() - Executes workflow with caching
    - stream() - Streams workflow execution with caching
    - run_compiled_workflow() - Convenience wrapper for invoke
    - stream_compiled_workflow() - Convenience wrapper for stream

    This is the NEW canonical API that replaces deprecated create_executor/astream.
    """

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

    def test_all_providers_have_compile_workflow(self, all_providers: Any) -> None:
        """Test all providers implement compile_workflow (canonical API)."""
        for name, provider in all_providers:
            assert hasattr(
                provider, "compile_workflow"
            ), f"{name} provider missing compile_workflow"
            # Verify it's callable
            assert callable(provider.compile_workflow), f"{name} compile_workflow not callable"

    def test_compile_workflow_returns_compiled_graph(self, all_providers: Any) -> None:
        """Test compile_workflow returns a CompiledGraph instance."""
        for name, provider in all_providers:
            workflows = provider.get_workflow_names()
            if workflows:
                workflow_name = workflows[0]
                compiled = provider.compile_workflow(workflow_name)
                # CompiledGraph should have invoke and stream methods
                assert hasattr(compiled, "invoke"), f"{name} compiled graph missing invoke"
                assert hasattr(compiled, "stream"), f"{name} compiled graph missing stream"
                assert callable(compiled.invoke), f"{name} invoke not callable"
                assert callable(compiled.stream), f"{name} stream not callable"

    @pytest.mark.asyncio
    async def test_run_compiled_workflow_raises_for_unknown_workflow(self, all_providers) -> None:
        """Test run_compiled_workflow raises ValueError for unknown workflow."""
        for name, provider in all_providers:
            with pytest.raises(ValueError, match="Workflow not found: nonexistent"):
                await provider.run_compiled_workflow("nonexistent", {})

    @pytest.mark.asyncio
    async def test_stream_compiled_workflow_raises_for_unknown_workflow(
        self, all_providers
    ) -> None:
        """Test stream_compiled_workflow raises ValueError for unknown workflow."""
        for name, provider in all_providers:
            try:
                async for _ in provider.stream_compiled_workflow("nonexistent", {}):
                    pass
                pytest.fail(f"{name} provider should have raised ValueError")
            except ValueError as e:
                assert "Workflow not found: nonexistent" in str(e)

    def test_all_providers_have_canonical_convenience_methods(self, all_providers: Any) -> None:
        """Test all providers have canonical convenience methods."""
        canonical_methods = [
            "compile_workflow",
            "run_compiled_workflow",
            "stream_compiled_workflow",
        ]
        for name, provider in all_providers:
            for method in canonical_methods:
                assert hasattr(
                    provider, method
                ), f"{name} provider missing canonical method {method}"

    def test_canonical_api_consistency(self, all_providers: Any) -> None:
        """Test all providers have consistent canonical API surface."""
        expected_canonical_methods = [
            "get_workflows",
            "get_workflow",
            "get_workflow_names",
            "compile_workflow",
            "run_compiled_workflow",
            "stream_compiled_workflow",
        ]
        for name, provider in all_providers:
            for method in expected_canonical_methods:
                assert hasattr(
                    provider, method
                ), f"{name} provider missing canonical method {method}"
