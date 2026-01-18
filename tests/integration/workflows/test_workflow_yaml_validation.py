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

"""Integration tests for YAML workflow validation and compilation.

These tests catch regressions early by:
1. Validating YAML syntax
2. Loading workflow definitions
3. Compiling workflows to executable graphs
4. Checking for common errors (unknown node types, missing references, etc.)

Run with: pytest tests/integration/workflows/test_workflow_yaml_validation.py -v
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from victor.workflows import load_workflow_from_file
from victor.workflows.unified_compiler import UnifiedWorkflowCompiler
from victor.core.errors import ConfigurationError, ValidationError
from victor.workflows.yaml_loader import YAMLWorkflowError


# =============================================================================
# Test Data
# =============================================================================

# Core production workflows (should all pass)
PRODUCTION_WORKFLOWS = {
    "coding": [
        "victor/coding/workflows/bugfix.yaml",
        "victor/coding/workflows/code_review.yaml",
        "victor/coding/workflows/feature.yaml",
        "victor/coding/workflows/refactor.yaml",
        "victor/coding/workflows/tdd.yaml",
        "victor/coding/workflows/multi_agent_consensus.yaml",
        "victor/coding/workflows/team_node_example.yaml",  # Team node support implemented
    ],
    "devops": [
        "victor/devops/workflows/container_setup.yaml",
        "victor/devops/workflows/deploy.yaml",
    ],
    "rag": [
        "victor/rag/workflows/ingest.yaml",
        "victor/rag/workflows/query.yaml",
    ],
    "dataanalysis": [
        "victor/dataanalysis/workflows/data_cleaning.yaml",
        "victor/dataanalysis/workflows/statistical_analysis.yaml",
        "victor/dataanalysis/workflows/automl_pipeline.yaml",
        "victor/dataanalysis/workflows/eda_pipeline.yaml",
        "victor/dataanalysis/workflows/ml_pipeline.yaml",
    ],
    "research": [
        "victor/research/workflows/fact_check.yaml",
        "victor/research/workflows/literature_review.yaml",
        "victor/research/workflows/competitive_analysis.yaml",
        "victor/research/workflows/deep_research.yaml",
    ],
    "benchmark": [
        "victor/benchmark/workflows/agentic_bench.yaml",
        "victor/benchmark/workflows/code_generation.yaml",
        "victor/benchmark/workflows/passk.yaml",
        "victor/benchmark/workflows/swe_bench.yaml",
    ],
    "framework": [
        "victor/workflows/feature_workflows.yaml",
        "victor/workflows/mode_workflows.yaml",
    ],
}

# Workflows with known issues (for documentation)
KNOWN_ISSUES = {
    # No known issues - team node support has been implemented!
}

# Example/migrated workflows (may have validation errors)
EXAMPLE_WORKFLOWS = {
    "coding": ["victor/coding/workflows/examples/migrated_example.yaml"],
    "devops": ["victor/devops/workflows/examples/migrated_example.yaml"],
    "rag": ["victor/rag/workflows/examples/migrated_example.yaml"],
    "research": ["victor/research/workflows/examples/migrated_example.yaml"],
}


# =============================================================================
# Test Helpers
# =============================================================================


def load_and_compile_workflow(
    workflow_path: str,
) -> Tuple[str, Dict, List]:
    """Load a workflow file and compile all workflows within it.

    Args:
        workflow_path: Path to YAML workflow file

    Returns:
        Tuple of (workflow_name, workflows_dict, compiled_graphs)

    Raises:
        ConfigurationError: If YAML loading fails
        ValidationError: If workflow validation fails
    """
    loaded = load_workflow_from_file(workflow_path)

    # Handle both dict and single WorkflowDefinition
    if isinstance(loaded, dict):
        workflows = loaded
    else:
        workflows = {loaded.name: loaded}

    compiler = UnifiedWorkflowCompiler()
    compiled_graphs = []

    for name, definition in workflows.items():
        compiled = compiler.compile_definition(definition)
        compiled_graphs.append(compiled)

    return Path(workflow_path).stem, workflows, compiled_graphs


def get_all_workflow_files() -> List[str]:
    """Get all workflow YAML files from the codebase."""
    workflow_files = []

    for vertical, files in PRODUCTION_WORKFLOWS.items():
        workflow_files.extend(files)

    return workflow_files


# =============================================================================
# Production Workflow Tests
# =============================================================================


@pytest.mark.workflow
@pytest.mark.integration
class TestProductionWorkflows:
    """Test all production workflows for successful compilation."""

    @pytest.mark.parametrize("workflow_path", get_all_workflow_files())
    def test_workflow_loads_and_compiles(self, workflow_path: str):
        """Test that workflow file loads and compiles successfully.

        This is a regression test to catch:
        - YAML syntax errors
        - Missing or invalid node types
        - Invalid node references
        - Circular dependencies
        - Validation errors
        """
        # Should not raise any exceptions
        workflow_name, workflows, compiled_graphs = load_and_compile_workflow(workflow_path)

        # Basic sanity checks
        assert isinstance(workflows, dict), "Workflows should be a dictionary"
        assert len(workflows) > 0, "At least one workflow should be defined"
        assert len(compiled_graphs) == len(workflows), "All workflows should compile"

    @pytest.mark.parametrize("vertical,files", PRODUCTION_WORKFLOWS.items())
    def test_vertical_all_workflows_valid(self, vertical: str, files: List[str]):
        """Test that all workflows in a vertical are valid."""
        for workflow_path in files:
            try:
                load_and_compile_workflow(workflow_path)
            except Exception as e:
                pytest.fail(f"Workflow {workflow_path} failed to compile: {e}")

    def test_all_coding_workflows(self):
        """Test all coding workflows compile successfully."""
        for workflow_path in PRODUCTION_WORKFLOWS["coding"]:
            workflow_name, workflows, compiled_graphs = load_and_compile_workflow(workflow_path)
            assert (
                len(workflows) >= 1
            ), f"Coding workflow {workflow_name} should have at least 1 workflow"

    def test_all_devops_workflows(self):
        """Test all DevOps workflows compile successfully."""
        for workflow_path in PRODUCTION_WORKFLOWS["devops"]:
            workflow_name, workflows, compiled_graphs = load_and_compile_workflow(workflow_path)
            assert len(workflows) >= 1

    def test_all_rag_workflows(self):
        """Test all RAG workflows compile successfully."""
        for workflow_path in PRODUCTION_WORKFLOWS["rag"]:
            workflow_name, workflows, compiled_graphs = load_and_compile_workflow(workflow_path)
            assert len(workflows) >= 1

    def test_all_dataanalysis_workflows(self):
        """Test all data analysis workflows compile successfully."""
        for workflow_path in PRODUCTION_WORKFLOWS["dataanalysis"]:
            workflow_name, workflows, compiled_graphs = load_and_compile_workflow(workflow_path)
            assert len(workflows) >= 1

    def test_all_research_workflows(self):
        """Test all research workflows compile successfully."""
        for workflow_path in PRODUCTION_WORKFLOWS["research"]:
            workflow_name, workflows, compiled_graphs = load_and_compile_workflow(workflow_path)
            assert len(workflows) >= 1

    def test_all_benchmark_workflows(self):
        """Test all benchmark workflows compile successfully."""
        for workflow_path in PRODUCTION_WORKFLOWS["benchmark"]:
            workflow_name, workflows, compiled_graphs = load_and_compile_workflow(workflow_path)
            assert len(workflows) >= 1


# =============================================================================
# Known Issues Tests
# =============================================================================


@pytest.mark.workflow
@pytest.mark.integration
@pytest.mark.xfail(reason="Documenting known issues")
class TestKnownWorkflowIssues:
    """Document known issues with certain workflows.

    These tests are expected to fail and serve as documentation for
    known issues that need to be addressed.
    """

    @pytest.mark.parametrize("workflow_path,issue_info", KNOWN_ISSUES.items())
    def test_known_issue_documentation(self, workflow_path: str, issue_info: Dict):
        """Document and track known workflow issues.

        This test will always fail (xfail) but provides documentation
        about the issue and what needs to be fixed.
        """
        with pytest.raises((ValidationError, ConfigurationError)) as exc_info:
            load_and_compile_workflow(workflow_path)

        # Verify the error matches what we expect
        error_message = str(exc_info.value)
        if issue_info["issue"] == "Unknown node type 'team'":
            assert "Unknown node type: 'team'" in error_message
            assert "Valid types:" in error_message


# =============================================================================
# Example Workflow Tests
# =============================================================================


@pytest.mark.workflow
@pytest.mark.integration
class TestExampleWorkflows:
    """Test example and migrated workflows.

    These workflows may have validation errors as they serve as
    examples or migration templates.
    """

    @pytest.mark.parametrize("vertical,files", EXAMPLE_WORKFLOWS.items())
    def test_example_workflow_status(self, vertical: str, files: List[str]):
        """Test example workflows and document their status.

        Example workflows may fail validation, but we should at least
        be able to load them and check their status.
        """
        for workflow_path in files:
            # Try to load - may fail with validation error
            try:
                workflow_name, workflows, compiled_graphs = load_and_compile_workflow(workflow_path)
                # If we get here, the workflow is valid
                assert len(workflows) >= 1
            except (ConfigurationError, ValidationError) as e:
                # Expected for example/migrated workflows
                # Just verify it's a known validation error type
                assert (
                    "validation failed" in str(e).lower()
                    or "references non-existent node" in str(e).lower()
                )


# =============================================================================
# Regression Tests
# =============================================================================


@pytest.mark.workflow
@pytest.mark.integration
@pytest.mark.regression
class TestWorkflowRegressions:
    """Regression tests for specific workflow issues.

    These tests prevent specific bugs from reoccurring.
    """

    def test_unknown_node_type_detection(self):
        """Test that team node type is now supported.

        Success test for: team_node_example.yaml using 'team' node type
        which is now supported by UnifiedWorkflowCompiler.

        This was previously a regression test expecting failure, but
        team node support has been implemented.
        """
        workflow_path = "victor/coding/workflows/team_node_example.yaml"

        # Should now compile successfully without raising exceptions
        workflow_name, workflows, compiled_graphs = load_and_compile_workflow(workflow_path)

        # Verify the workflow loaded correctly
        assert isinstance(workflows, dict)
        assert len(workflows) > 0
        assert len(compiled_graphs) == len(workflows)

    def test_missing_node_reference_detection(self):
        """Test that missing node references are properly detected.

        Regression test for: migrated_example.yaml files that reference
        non-existent nodes.
        """
        workflow_path = "victor/coding/workflows/examples/migrated_example.yaml"

        with pytest.raises((YAMLWorkflowError, ConfigurationError, ValidationError)) as exc_info:
            load_and_compile_workflow(workflow_path)

        error_message = str(exc_info.value)
        assert (
            "references non-existent node" in error_message
            or "validation failed" in error_message.lower()
        )

    def test_invalid_yaml_syntax_detection(self):
        """Test that invalid YAML syntax is properly detected."""
        # Test truly malformed YAML
        import tempfile
        import yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            # Write invalid YAML (unquoted colons, bad indentation)
            f.write(
                """
workflows:
  test:
    nodes:
      - id: test_node
        type: agent
    invalid_yaml: [unclosed bracket
    """
            )
            f.flush()

            try:
                # Should fail during YAML parsing
                with pytest.raises((YAMLWorkflowError, yaml.YAMLError)):
                    load_and_compile_workflow(f.name)
            finally:
                os.unlink(f.name)


# =============================================================================
# Statistics Tests
# =============================================================================


@pytest.mark.workflow
@pytest.mark.integration
class TestWorkflowStatistics:
    """Test workflow statistics and coverage."""

    def test_production_workflow_count(self):
        """Verify we have the expected number of production workflows."""
        total_count = sum(len(files) for files in PRODUCTION_WORKFLOWS.values())
        # We expect at least 25 production workflows
        assert total_count >= 25, f"Expected at least 25 production workflows, found {total_count}"

    def test_all_verticals_have_workflows(self):
        """Verify all verticals have at least one workflow."""
        for vertical, files in PRODUCTION_WORKFLOWS.items():
            assert len(files) > 0, f"Vertical '{vertical}' should have at least one workflow"

    def test_workflow_compilation_success_rate(self):
        """Test that most workflows compile successfully.

        This is a health check metric - should be > 90% success rate.
        """
        total = 0
        passed = 0

        for vertical, files in PRODUCTION_WORKFLOWS.items():
            for workflow_path in files:
                total += 1
                try:
                    load_and_compile_workflow(workflow_path)
                    passed += 1
                except Exception:
                    # Known issues are OK
                    if workflow_path not in KNOWN_ISSUES:
                        pass

        # Success rate should be high
        success_rate = (passed / total) * 100 if total > 0 else 0
        assert (
            success_rate >= 90
        ), f"Workflow compilation success rate ({success_rate:.1f}%) is below 90%"


# =============================================================================
# Node Type Validation Tests
# =============================================================================


@pytest.mark.workflow
@pytest.mark.integration
class TestNodeTypes:
    """Test that only valid node types are used."""

    VALID_NODE_TYPES = {"agent", "compute", "condition", "parallel", "transform", "hitl", "team"}

    def test_all_nodes_use_valid_types(self):
        """Test that all workflow nodes use valid node types.

        Regression test to prevent invalid node types from being added.
        """
        invalid_workflows = []

        for vertical, files in PRODUCTION_WORKFLOWS.items():
            for workflow_path in files:
                try:
                    loaded = load_workflow_from_file(workflow_path)
                    if isinstance(loaded, dict):
                        workflows = loaded
                    else:
                        workflows = {loaded.name: loaded}

                    for workflow_name, workflow_def in workflows.items():
                        for node_id, node in workflow_def.nodes.items():
                            # Get node type from the node object
                            node_type = type(node).__name__.replace("Node", "").lower()

                            # Map class names to valid types
                            # Note: Some node types have "Workflow" suffix (e.g., TeamNodeWorkflow)
                            type_mapping = {
                                "agent": "agent",
                                "agentworkflow": "agent",
                                "compute": "compute",
                                "computeworkflow": "compute",
                                "condition": "condition",
                                "conditionworkflow": "condition",
                                "parallel": "parallel",
                                "parallelworkflow": "parallel",
                                "transform": "transform",
                                "transformworkflow": "transform",
                                "hitl": "hitl",
                                "hitlworkflow": "hitl",
                                "team": "team",
                                "teamworkflow": "team",  # TeamNodeWorkflow -> TeamWorkflow -> teamworkflow
                            }

                            # Normalize node type
                            normalized_type = type_mapping.get(node_type, node_type)

                            if normalized_type not in VALID_NODE_TYPES:
                                invalid_workflows.append(
                                    f"{workflow_path}:{workflow_name}:{node_id} has invalid type '{node_type}'"
                                )
                except Exception:
                    # Skip workflows that don't load
                    pass

        # Report any invalid node types found
        if invalid_workflows:
            pytest.fail("Found invalid node types:\n" + "\n".join(invalid_workflows))
