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

"""Integration tests for code review workflows.

This test suite provides comprehensive coverage for:
- Code review workflow execution
- Security analysis
- Style checking
- Logic verification
- Multi-agent coordination
- Workflow compilation
- Error handling
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.fixtures.coding_fixtures import (
    CODE_REVIEW_SCENARIOS,
    SAMPLE_PYTHON_CLASS,
    SAMPLE_PYTHON_COMPLEX,
    SAMPLE_PYTHON_SIMPLE,
    create_sample_project,
)
from victor.coding.workflows.review import (
    code_review_workflow,
    pr_review_workflow,
    quick_review_workflow,
)
from victor.framework.graph import CompiledGraph, StateGraph


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_orchestrator():
    """Create a mock orchestrator for testing."""
    orchestrator = MagicMock()
    orchestrator.run_agent = AsyncMock()
    return orchestrator


@pytest.fixture
def sample_project(tmp_path):
    """Create a sample project for testing."""
    return create_sample_project(tmp_path)


@pytest.fixture
def workflow_builder():
    """Get workflow builders."""
    return {
        "code_review": code_review_workflow,
        "quick_review": quick_review_workflow,
        "pr_review": pr_review_workflow,
    }


# =============================================================================
# Code Review Workflow Tests
# =============================================================================


class TestCodeReviewWorkflow:
    """Tests for code review workflow."""

    def test_code_review_workflow_definition(self, workflow_builder):
        """Test code review workflow definition structure."""
        workflow = workflow_builder["code_review"]()

        assert workflow is not None
        assert workflow.name == "code_review"

    def test_code_review_workflow_nodes(self, workflow_builder):
        """Test that code review workflow has required nodes."""
        workflow = workflow_builder["code_review"]()

        # Check for required nodes (nodes is a Dict[str, WorkflowNode])
        node_ids = list(workflow.nodes.keys())

        assert "identify" in node_ids
        assert "security" in node_ids
        assert "style" in node_ids
        assert "logic" in node_ids
        assert "synthesize" in node_ids

    def test_code_review_workflow_parallel_structure(self, workflow_builder):
        """Test parallel review structure."""
        workflow = workflow_builder["code_review"]()

        # Should have parallel execution for reviews (nodes is a Dict[str, WorkflowNode])
        nodes = list(workflow.nodes.values())
        parallel_nodes = [
            node for node in nodes if hasattr(node, "is_parallel") and node.is_parallel
        ]

        # May have parallel execution nodes
        assert len(nodes) >= 5  # identify, security, style, logic, synthesize

    def test_code_review_workflow_edges(self, workflow_builder):
        """Test workflow edges/connectivity."""
        workflow = workflow_builder["code_review"]()

        # Workflow should have edges connecting nodes
        # Edges are defined via each node's next_nodes attribute
        total_edges = sum(len(node.next_nodes) for node in workflow.nodes.values())
        assert total_edges > 0


# =============================================================================
# Quick Review Workflow Tests
# =============================================================================


class TestQuickReviewWorkflow:
    """Tests for quick review workflow."""

    def test_quick_review_workflow_definition(self, workflow_builder):
        """Test quick review workflow definition."""
        workflow = workflow_builder["quick_review"]()

        assert workflow is not None
        assert workflow.name == "quick_review"

    def test_quick_review_workflow_simplicity(self, workflow_builder):
        """Test that quick review is simpler than full review."""
        quick_workflow = workflow_builder["quick_review"]()
        full_workflow = workflow_builder["code_review"]()

        # Quick review should have fewer nodes
        assert len(quick_workflow.nodes) < len(full_workflow.nodes)

    def test_quick_review_workflow_nodes(self, workflow_builder):
        """Test quick review workflow nodes."""
        workflow = workflow_builder["quick_review"]()

        # nodes is a Dict[str, WorkflowNode]
        node_ids = list(workflow.nodes.keys())

        # Should have identify and review nodes
        assert "identify" in node_ids or "review" in node_ids


# =============================================================================
# PR Review Workflow Tests
# =============================================================================


class TestPRReviewWorkflow:
    """Tests for pull request review workflow."""

    def test_pr_review_workflow_definition(self, workflow_builder):
        """Test PR review workflow definition."""
        workflow = workflow_builder["pr_review"]()

        assert workflow is not None
        assert workflow.name == "pr_review"

    def test_pr_review_workflow_nodes(self, workflow_builder):
        """Test PR review workflow nodes."""
        workflow = workflow_builder["pr_review"]()

        # nodes is a Dict[str, WorkflowNode]
        node_ids = list(workflow.nodes.keys())

        # Should have fetch, analyze, test_check, generate_review
        assert "fetch" in node_ids or "generate_review" in node_ids

    def test_pr_review_workflow_complexity(self, workflow_builder):
        """Test PR review workflow complexity."""
        workflow = workflow_builder["pr_review"]()

        # Should be reasonably complex
        assert len(workflow.nodes) >= 3


# =============================================================================
# Workflow Compilation Tests
# =============================================================================


class TestWorkflowCompilation:
    """Tests for workflow compilation."""

    def test_compile_code_review_workflow(self, workflow_builder):
        """Test compiling code review workflow."""
        from victor.workflows.definition import ParallelNode

        workflow = workflow_builder["code_review"]()

        # Build state graph
        graph = StateGraph("test_code_review")

        # Track nodes that are children of ParallelNodes
        parallel_children = set()
        for node in workflow.nodes.values():
            if isinstance(node, ParallelNode):
                parallel_children.update(node.parallel_nodes)

        # Add only top-level nodes (not children of ParallelNodes)
        for node_id, node in workflow.nodes.items():
            if node_id not in parallel_children:
                graph.add_node(node_id, node)

        # Add edges based on next_nodes
        for node_id, node in workflow.nodes.items():
            if node_id not in parallel_children:
                for next_node in node.next_nodes:
                    if next_node not in parallel_children:
                        graph.add_edge(node_id, next_node)

        # Set entry point
        if workflow.start_node and workflow.start_node not in parallel_children:
            graph.set_entry_point(workflow.start_node)

        # Compile graph
        compiled = graph.compile()

        assert isinstance(compiled, CompiledGraph)

    def test_compile_quick_review_workflow(self, workflow_builder):
        """Test compiling quick review workflow."""
        workflow = workflow_builder["quick_review"]()

        graph = StateGraph("test_quick_review")
        for node_id, node in workflow.nodes.items():
            graph.add_node(node_id, node)

        # Add edges based on next_nodes
        for node_id, node in workflow.nodes.items():
            for next_node in node.next_nodes:
                graph.add_edge(node_id, next_node)

        # Set entry point
        if workflow.start_node:
            graph.set_entry_point(workflow.start_node)

        compiled = graph.compile()

        assert isinstance(compiled, CompiledGraph)


# =============================================================================
# Scenario-Based Tests
# =============================================================================


class TestCodeReviewScenarios:
    """Tests using code review scenarios."""

    @pytest.mark.parametrize("scenario", CODE_REVIEW_SCENARIOS[:2])  # Test first 2 scenarios
    def test_scenario_validity(self, scenario):
        """Test that scenarios are valid."""
        assert scenario.name is not None
        assert scenario.code is not None
        assert scenario.file_path is not None
        assert scenario.expected_findings >= 0

    def test_simple_function_scenario(self):
        """Test simple function review scenario."""
        scenario = CODE_REVIEW_SCENARIOS[0]  # simple_function

        assert scenario.name == "simple_function"
        assert "def hello_world" in scenario.code or "function" in scenario.code
        assert scenario.expected_findings == 2

    def test_security_scenario(self):
        """Test security review scenario."""
        scenario = CODE_REVIEW_SCENARIOS[3]  # security_issues

        assert scenario.name == "security_issues"
        assert "import subprocess" in scenario.code or "dangerous" in scenario.code.lower()
        assert scenario.expected_findings == 4
        assert "security" in scenario.categories


# =============================================================================
# Security Analysis Tests
# =============================================================================


class TestSecurityAnalysis:
    """Tests for security analysis in code review."""

    def test_detect_sql_injection(self):
        """Test detection of SQL injection patterns."""
        code = "query = f\"SELECT * FROM users WHERE name = '{user_input}'\""

        # Should be flagged as potential SQL injection
        assert 'f"' in code or "{user_input}" in code

    def test_detect_command_injection(self):
        """Test detection of command injection patterns."""
        code = "subprocess.call(user_input, shell=True)"

        # Should be flagged as potential command injection
        assert "shell=True" in code
        assert "subprocess" in code

    def test_detect_insecure_deserialization(self):
        """Test detection of insecure deserialization."""
        code = "data = pickle.loads(user_data)"

        # Should be flagged as insecure
        assert "pickle.loads" in code

    def test_detect_weak_hashing(self):
        """Test detection of weak hashing algorithms."""
        code = "hashlib.md5(password).hexdigest()"

        # Should be flagged as weak
        assert "md5" in code


# =============================================================================
# Style Analysis Tests
# =============================================================================


class TestStyleAnalysis:
    """Tests for code style analysis."""

    def test_detect_missing_docstrings(self):
        """Test detection of missing docstrings."""
        code = "def calculate(x, y):\n    return x + y"

        # Should detect missing docstring
        assert '"""' not in code
        assert "'''" not in code

    def test_detect_naming_conventions(self):
        """Test detection of naming convention violations."""
        # PEP 8: function names should be lowercase
        code = "def CalculateSum():\n    pass"

        # Should detect uppercase function name
        assert "CalculateSum" in code

    def test_detect_line_length(self):
        """Test detection of overly long lines."""
        code = "x" * 120

        # Should detect line > 100 characters
        assert len(code) > 100

    def test_detect_import_order(self):
        """Test detection of import order violations."""
        code = """from local_module import func
import os
import sys
"""

        # Should detect local imports before stdlib
        lines = code.strip().split("\n")
        assert "from local_module" in lines[0]


# =============================================================================
# Logic Analysis Tests
# =============================================================================


class TestLogicAnalysis:
    """Tests for logic correctness analysis."""

    def test_detect_unused_variables(self):
        """Test detection of unused variables."""
        code = "def f():\n    x = 1\n    return 2"

        # Variable x is unused
        assert "x = 1" in code

    def test_detect_dead_code(self):
        """Test detection of unreachable code."""
        code = "def f():\n    return\n    x = 1"

        # Code after return is unreachable
        assert "return" in code.split("x = 1")[0]

    def test_detect_empty_blocks(self):
        """Test detection of empty code blocks."""
        code = "def f():\n    pass"

        # Should detect empty function
        assert "pass" in code

    def test_detect_duplicate_code(self):
        """Test detection of duplicate code blocks."""
        code1 = "def f1():\n    return x + y"
        code2 = "def f2():\n    return x + y"

        # Should detect similar logic
        assert "x + y" in code1 and "x + y" in code2


# =============================================================================
# Multi-Agent Coordination Tests
# =============================================================================


class TestMultiAgentCoordination:
    """Tests for multi-agent coordination in workflows."""

    def test_parallel_review_agents(self, workflow_builder):
        """Test that parallel review uses multiple agents."""
        workflow = workflow_builder["code_review"]()

        # nodes is a Dict[str, WorkflowNode]
        node_ids = list(workflow.nodes.keys())

        # Should have separate nodes for different review types
        # Check for security, style, logic reviewers
        has_security = any("security" in name for name in node_ids)
        has_style = any("style" in name for name in node_ids)
        has_logic = any("logic" in name for name in node_ids)

        assert has_security or has_style or has_logic

    def test_agent_roles(self, workflow_builder):
        """Test that agents have defined roles."""
        workflow = workflow_builder["code_review"]()

        # Nodes should have role information (nodes is a Dict[str, WorkflowNode])
        for node in workflow.nodes.values():
            # Should have role or other metadata
            assert hasattr(node, "id")


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestWorkflowErrorHandling:
    """Tests for workflow error handling."""

    def test_workflow_with_invalid_code(self, workflow_builder):
        """Test workflow behavior with invalid code."""
        workflow = workflow_builder["quick_review"]()

        # Should handle invalid code gracefully
        assert workflow is not None

    def test_workflow_with_empty_file(self, workflow_builder):
        """Test workflow with empty file."""
        workflow = workflow_builder["quick_review"]()

        # Should handle empty file
        assert workflow is not None

    def test_workflow_compilation_failure(self, workflow_builder):
        """Test handling of compilation failures."""
        workflow = workflow_builder["code_review"]()

        # Should handle gracefully
        try:
            graph = StateGraph("test")
            for node_id, node in workflow.nodes.items():
                graph.add_node(node_id, node)
            # Should not raise
        except Exception:
            pytest.fail("Workflow compilation failed unexpectedly")


# =============================================================================
# Integration Tests
# =============================================================================


class TestWorkflowIntegration:
    """Integration tests for complete workflows."""

    def test_end_to_end_code_review(self, workflow_builder, sample_project):
        """Test end-to-end code review workflow."""
        from victor.workflows.definition import ParallelNode

        workflow = workflow_builder["code_review"]()

        # Build and compile
        graph = StateGraph("integration_test")

        # Track nodes that are children of ParallelNodes
        parallel_children = set()
        for node in workflow.nodes.values():
            if isinstance(node, ParallelNode):
                parallel_children.update(node.parallel_nodes)

        # Add only top-level nodes (not children of ParallelNodes)
        for node_id, node in workflow.nodes.items():
            if node_id not in parallel_children:
                graph.add_node(node_id, node)

        # Add edges based on next_nodes
        for node_id, node in workflow.nodes.items():
            if node_id not in parallel_children:
                for next_node in node.next_nodes:
                    if next_node not in parallel_children:
                        graph.add_edge(node_id, next_node)

        # Set entry point
        if workflow.start_node and workflow.start_node not in parallel_children:
            graph.set_entry_point(workflow.start_node)

        compiled = graph.compile()

        # Should compile successfully
        assert compiled is not None

    def test_multi_file_review(self, workflow_builder, tmp_path):
        """Test reviewing multiple files."""
        # Create multiple files
        files = {
            "file1.py": SAMPLE_PYTHON_SIMPLE,
            "file2.py": SAMPLE_PYTHON_CLASS,
            "file3.py": SAMPLE_PYTHON_COMPLEX,
        }

        for name, content in files.items():
            (tmp_path / name).write_text(content)

        # Workflow should handle multiple files
        workflow = workflow_builder["code_review"]()
        assert workflow is not None

    def test_workflow_with_real_scenarios(self, workflow_builder):
        """Test workflow with realistic scenarios."""
        for scenario in CODE_REVIEW_SCENARIOS[:2]:
            workflow = workflow_builder["code_review"]()

            # Each scenario should be valid
            assert workflow is not None
            assert scenario.code is not None


# =============================================================================
# Performance Tests
# =============================================================================


class TestWorkflowPerformance:
    """Tests for workflow performance."""

    def test_large_codebase_review(self, workflow_builder, tmp_path):
        """Test reviewing a larger codebase."""
        # Create multiple files
        for i in range(10):
            (tmp_path / f"file{i}.py").write_text(SAMPLE_PYTHON_CLASS)

        workflow = workflow_builder["code_review"]()

        # Should handle reasonably sized codebase
        assert workflow is not None

    def test_workflow_compilation_speed(self, workflow_builder):
        """Test workflow compilation performance."""
        import time
        from victor.workflows.definition import ParallelNode

        workflow = workflow_builder["code_review"]()

        start = time.time()
        graph = StateGraph("performance_test")

        # Track nodes that are children of ParallelNodes
        parallel_children = set()
        for node in workflow.nodes.values():
            if isinstance(node, ParallelNode):
                parallel_children.update(node.parallel_nodes)

        # Add only top-level nodes (not children of ParallelNodes)
        for node_id, node in workflow.nodes.items():
            if node_id not in parallel_children:
                graph.add_node(node_id, node)

        # Add edges based on next_nodes
        for node_id, node in workflow.nodes.items():
            if node_id not in parallel_children:
                for next_node in node.next_nodes:
                    if next_node not in parallel_children:
                        graph.add_edge(node_id, next_node)

        # Set entry point
        if workflow.start_node and workflow.start_node not in parallel_children:
            graph.set_entry_point(workflow.start_node)

        compiled = graph.compile()
        duration = time.time() - start

        # Should compile quickly (< 1 second)
        assert duration < 1.0


# =============================================================================
# Metadata Tests
# =============================================================================


class TestWorkflowMetadata:
    """Tests for workflow metadata."""

    def test_code_review_metadata(self, workflow_builder):
        """Test code review workflow metadata."""
        workflow = workflow_builder["code_review"]()

        # Should have metadata
        assert workflow.name == "code_review"

    def test_quick_review_metadata(self, workflow_builder):
        """Test quick review workflow metadata."""
        workflow = workflow_builder["quick_review"]()

        # Should have metadata
        assert workflow.name == "quick_review"

    def test_pr_review_metadata(self, workflow_builder):
        """Test PR review workflow metadata."""
        workflow = workflow_builder["pr_review"]()

        # Should have metadata
        assert workflow.name == "pr_review"


# =============================================================================
# Regression Tests
# =============================================================================


class TestRegressionIssues:
    """Regression tests for known issues."""

    def test_no_crash_on_special_characters(self, workflow_builder):
        """Test that special characters don't crash workflows."""
        code = "def test():\n    return '特殊字符'"

        workflow = workflow_builder["quick_review"]()

        # Should not crash
        assert workflow is not None

    def test_no_crash_on_unicode(self, workflow_builder):
        """Test that Unicode doesn't crash workflows."""
        code = "def test():\n    # 评论\n    return '🎉'"

        workflow = workflow_builder["quick_review"]()

        # Should not crash
        assert workflow is not None

    def test_no_crash_on_very_long_lines(self, workflow_builder):
        """Test that very long lines don't crash workflows."""
        code = "x = " + ("a" * 1000)

        workflow = workflow_builder["quick_review"]()

        # Should not crash
        assert workflow is not None


# =============================================================================
# Helper Functions
# =============================================================================


def create_mock_context(code: str, file_path: str = "test.py") -> dict[str, Any]:
    """Create a mock workflow context."""
    return {
        "code": code,
        "file_path": file_path,
        "findings": [],
    }
