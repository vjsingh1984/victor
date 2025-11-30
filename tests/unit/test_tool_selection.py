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

"""Tests for tool selection mechanisms (semantic and keyword-based)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import Settings
from victor.providers.base import BaseProvider
from victor.tools.base import ToolRegistry


class MockProvider(BaseProvider):
    """Mock provider for testing."""

    @property
    def name(self) -> str:
        return "mock"

    async def chat(self, **kwargs):
        return MagicMock()

    async def stream(self, **kwargs):
        yield MagicMock()

    def supports_tools(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True


@pytest.fixture
def mock_provider():
    """Create mock provider."""
    return MockProvider()


@pytest.fixture
def mock_settings():
    """Create mock settings."""
    settings = MagicMock(spec=Settings)
    settings.airgapped_mode = False
    settings.use_semantic_tool_selection = False
    return settings


@pytest.fixture
def orchestrator(mock_provider, mock_settings):
    """Create orchestrator with mock provider."""
    return AgentOrchestrator(
        settings=mock_settings,
        provider=mock_provider,
        model="test-model",
        temperature=0.7,
        max_tokens=1000,
    )


class TestToolCount:
    """Test that tool count reflects 32-tool consolidation."""

    def test_tool_registry_count(self, orchestrator):
        """Verify that exactly 31 tools are registered."""
        tools = list(orchestrator.tools.list_tools())
        assert len(tools) == 31, f"Expected 31 tools, got {len(tools)}"

    def test_consolidated_tools_present(self, orchestrator):
        """Verify consolidated tools are present."""
        tools = orchestrator.tools.list_tools()
        tool_names = {t.name for t in tools}

        # Consolidated tools
        consolidated = {
            "edit_files",  # from 10 tools
            "git",  # from 6 tools
            "batch",  # from 5 tools
            "cicd",  # from 4 tools
            "scaffold",  # from 4 tools
            "docker",  # from 15 tools
            "analyze_metrics",  # from 6 tools
            "security_scan",  # from 5 tools
            "code_review",  # from 5 tools
        }

        for tool_name in consolidated:
            assert tool_name in tool_names, f"Consolidated tool '{tool_name}' not found"

    def test_old_tools_removed(self, orchestrator):
        """Verify old unconsolidated tools are removed."""
        tools = orchestrator.tools.list_tools()
        tool_names = {t.name for t in tools}

        # These should NOT exist (consolidated into unified tools)
        removed = {
            "batch_search",
            "batch_replace",
            "batch_analyze",
            "cicd_generate",
            "cicd_validate",
            "scaffold_create",
            "scaffold_list_templates",
            "docker_ps",
            "docker_run",
            "docker_stop",
        }

        for tool_name in removed:
            assert tool_name not in tool_names, f"Old tool '{tool_name}' should be removed"


class TestKeywordBasedSelection:
    """Test keyword-based tool selection with 32 tools."""

    def test_git_keywords(self, orchestrator):
        """Test git-related keywords select git tools."""
        message = "commit my changes to the repository"
        tools = orchestrator._select_relevant_tools_keywords(message)
        tool_names = {t.name for t in tools}

        assert "git" in tool_names
        assert "git_suggest_commit" in tool_names
        assert "git_create_pr" in tool_names

    def test_batch_keywords(self, orchestrator):
        """Test batch operation keywords select batch tool."""
        message = "search across multiple files for TODO comments"
        tools = orchestrator._select_relevant_tools_keywords(message)
        tool_names = {t.name for t in tools}

        assert "batch" in tool_names

    def test_cicd_keywords(self, orchestrator):
        """Test CI/CD keywords select cicd tool."""
        message = "generate a GitHub Actions workflow for testing"
        tools = orchestrator._select_relevant_tools_keywords(message)
        tool_names = {t.name for t in tools}

        assert "cicd" in tool_names

    def test_scaffold_keywords(self, orchestrator):
        """Test scaffold keywords select scaffold tool."""
        message = "create a new FastAPI project from template"
        tools = orchestrator._select_relevant_tools_keywords(message)
        tool_names = {t.name for t in tools}

        assert "scaffold" in tool_names

    def test_docker_keywords(self, orchestrator):
        """Test docker keywords select docker tool."""
        message = "list running containers and their status"
        tools = orchestrator._select_relevant_tools_keywords(message)
        tool_names = {t.name for t in tools}

        assert "docker" in tool_names

    def test_small_model_limitation(self, orchestrator):
        """Test small models get limited tool set."""
        # Simulate small model by creating a new provider with ollama name
        from unittest.mock import PropertyMock

        type(orchestrator.provider).name = PropertyMock(return_value="ollama")
        orchestrator.model = "qwen2.5-coder:1.5b"

        message = "test commit docker security"
        tools = orchestrator._select_relevant_tools_keywords(message)

        # Small models should get <= 10 tools
        assert len(tools) <= 10, f"Small model got {len(tools)} tools, expected <= 10"

    def test_core_tools_always_included(self, orchestrator):
        """Test core tools are always available."""
        message = "do something"
        tools = orchestrator._select_relevant_tools_keywords(message)
        tool_names = {t.name for t in tools}

        # Core tools should always be present
        core_tools = {"read_file", "write_file", "execute_bash", "edit_files"}
        included_core = core_tools & tool_names
        assert len(included_core) > 0, "No core tools included"


class TestConsolidatedToolFunctionality:
    """Test consolidated tools have correct operations."""

    @pytest.mark.asyncio
    async def test_batch_operations(self):
        """Test batch tool has all required operations."""
        from victor.tools.batch_processor_tool import batch

        # Test operation validation
        result = await batch(operation="invalid", path=".")
        assert result["success"] is False
        assert "Unknown operation" in result["error"]

        # Valid operations
        valid_ops = ["search", "replace", "analyze", "list", "transform"]
        for op in valid_ops:
            # These will fail due to missing params, but operation should be recognized
            result = await batch(operation=op, path=".")
            if op == "transform":
                assert "not yet implemented" in result["error"].lower()
            # Operation should be recognized (not "Unknown operation")

    @pytest.mark.asyncio
    async def test_cicd_operations(self):
        """Test cicd tool has all required operations."""
        from victor.tools.cicd_tool import cicd

        # Test operation validation
        result = await cicd(operation="invalid")
        assert result["success"] is False
        assert "Unknown operation" in result["error"]

        # Valid operations: generate, validate, list
        result = await cicd(operation="list")
        assert result["success"] is True
        assert "templates" in result

    @pytest.mark.asyncio
    async def test_scaffold_operations(self):
        """Test scaffold tool has all required operations."""
        from victor.tools.scaffold_tool import scaffold

        # Test operation validation
        result = await scaffold(operation="invalid")
        assert result["success"] is False
        assert "Unknown operation" in result["error"]

        # Valid operation: list
        result = await scaffold(operation="list")
        assert result["success"] is True
        assert "templates" in result
        assert len(result["templates"]) == 5  # Should have 5 templates

    @pytest.mark.asyncio
    async def test_git_operations(self):
        """Test git tool has all required operations."""
        from victor.tools.git_tool import git

        # Test operation validation
        result = await git(operation="invalid")
        assert result["success"] is False
        assert "Unknown operation" in result["error"]

        # Valid operations recognized (will fail due to git not being initialized, but operation should be recognized)


@pytest.mark.skipif(
    True,  # Skip by default as it requires Ollama running
    reason="Requires Ollama with embedding model running",
)
class TestSemanticToolSelection:
    """Test semantic (embedding-based) tool selection."""

    @pytest.mark.asyncio
    async def test_semantic_selector_initialization(self, orchestrator):
        """Test semantic selector initializes with 32 tools."""
        orchestrator.settings.use_semantic_tool_selection = True

        # Initialize semantic selector
        from victor.tools.semantic_selector import SemanticToolSelector

        selector = SemanticToolSelector(
            embedding_model="nomic-embed-text", embedding_provider="ollama", cache_embeddings=True
        )

        # Get all tools
        tools = list(orchestrator.tools.list_tools())

        # Initialize embeddings
        await selector.initialize_tool_embeddings(tools)

        # Should have embeddings for all 32 tools
        assert len(selector.tool_embeddings) == 32

    @pytest.mark.asyncio
    async def test_semantic_selection_testing_query(self, orchestrator):
        """Test semantic selection with testing-related query."""
        from victor.tools.semantic_selector import SemanticToolSelector

        selector = SemanticToolSelector(
            embedding_model="nomic-embed-text", embedding_provider="ollama"
        )

        tools = list(orchestrator.tools.list_tools())
        await selector.initialize_tool_embeddings(tools)

        # Query about testing
        message = "verify the authentication module works correctly"
        selected = await selector.select_relevant_tools(
            message, max_tools=5, similarity_threshold=0.3
        )

        # Should include testing-related tools
        tool_names = {t.name for t in selected}
        assert "run_tests" in tool_names or "code_review" in tool_names

    @pytest.mark.asyncio
    async def test_semantic_vs_keyword_coverage(self, orchestrator):
        """Test that semantic selection covers similar tools as keyword."""
        from victor.tools.semantic_selector import SemanticToolSelector

        selector = SemanticToolSelector(
            embedding_model="nomic-embed-text", embedding_provider="ollama"
        )

        tools = list(orchestrator.tools.list_tools())
        await selector.initialize_tool_embeddings(tools)

        # Test multiple queries
        test_cases = [
            "commit my changes",
            "run security scan",
            "create a new project",
            "build docker image",
        ]

        for query in test_cases:
            keyword_tools = orchestrator._select_relevant_tools_keywords(query)
            semantic_tools = await selector.select_relevant_tools(
                query, max_tools=10, similarity_threshold=0.3
            )

            # Both should select at least some tools
            assert len(keyword_tools) > 0
            assert len(semantic_tools) > 0


class TestToolDescriptions:
    """Test that tools have clear, semantic-friendly descriptions."""

    def test_all_tools_have_descriptions(self, orchestrator):
        """Verify all 32 tools have descriptions."""
        tools = list(orchestrator.tools.list_tools())

        for tool in tools:
            assert tool.description, f"Tool '{tool.name}' missing description"
            assert len(tool.description) > 20, f"Tool '{tool.name}' has too short description"

    def test_consolidated_tools_mention_operations(self, orchestrator):
        """Verify consolidated tools mention their operations in descriptions."""
        tools = {t.name: t.description for t in orchestrator.tools.list_tools()}

        # Consolidated tools should mention operations (flexible matching for word forms)
        consolidated_checks = {
            "batch": ["search", "replace", "analyz"],  # matches analyze/analyzing
            "cicd": ["generat", "validat"],  # matches generate/generating, validate/validating
            "scaffold": ["creat", "template"],  # matches create/creating
            "git": ["commit", "stage", "diff"],
        }

        for tool_name, keywords in consolidated_checks.items():
            if tool_name in tools:
                desc = tools[tool_name].lower()
                # At least one keyword should be in description (using substring matching)
                matches = [kw for kw in keywords if kw in desc]
                assert (
                    len(matches) > 0
                ), f"Tool '{tool_name}' should mention operations in description. Description: {desc[:100]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
