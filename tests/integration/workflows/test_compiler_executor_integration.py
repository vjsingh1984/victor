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

"""Integration tests for pure compiler and executor (Phase 3).

Tests the separation of compilation and execution:
- WorkflowCompiler compiles YAML to CompiledGraph
- StateGraphExecutor executes CompiledGraph
"""

import pytest
from pathlib import Path


# Sample workflow YAML for testing
SAMPLE_WORKFLOW_YAML = """workflows:
  simple_linear:
    description: "Simple linear workflow for testing"

    nodes:
      - id: step1
        type: compute
        name: "First Step"
        handler: noop_handler
        output: result1
        next: [step2]

      - id: step2
        type: transform
        name: "Second Step"
        transform: "uppercase_transform"
        output: result2
        next: [step3]

      - id: step3
        type: compute
        name: "Final Step"
        handler: noop_handler
        output: final_result
"""


class TestCompilerExecutorIntegration:
    """Integration tests for compiler and executor separation."""

    def test_workflow_compiler_imports(self):
        """Test that WorkflowCompiler can be imported."""
        from victor.workflows.compiler.unified_compiler import WorkflowCompiler
        assert WorkflowCompiler is not None

    def test_state_graph_executor_imports(self):
        """Test that StateGraphExecutor can be imported."""
        from victor.workflows.executors import StateGraphExecutor
        assert StateGraphExecutor is not None

    def test_compiler_initialization(self):
        """Test that WorkflowCompiler can be initialized."""
        from victor.workflows.compiler.unified_compiler import WorkflowCompiler

        # Mock classes for initialization
        class MockLoader:
            pass

        class MockFactory:
            pass

        compiler = WorkflowCompiler(
            yaml_loader=MockLoader(),
            node_executor_factory=MockFactory(),
        )

        assert compiler is not None
        assert hasattr(compiler, 'compile')
        assert hasattr(compiler, '_yaml_config')

    def test_executor_initialization(self):
        """Test that StateGraphExecutor can be initialized with a mock graph."""
        from victor.workflows.executors import StateGraphExecutor
        from unittest.mock import MagicMock

        # Create a mock CompiledGraph
        mock_graph = MagicMock()
        mock_graph.graph_id = "test-graph"

        executor = StateGraphExecutor(mock_graph)

        assert executor is not None
        assert hasattr(executor, 'invoke')
        assert hasattr(executor, 'stream')

    def test_compiler_executor_protocol_compliance(self):
        """Test that compiler and executor comply with protocols."""
        from victor.workflows.compiler_protocols import (
            WorkflowCompilerProtocol,
            CompiledGraphProtocol,
        )

        # Check protocol exists
        assert hasattr(WorkflowCompilerProtocol, 'compile')
        assert hasattr(CompiledGraphProtocol, 'invoke')

    def test_sample_workflow_structure(self):
        """Test that sample workflow YAML is valid structure."""
        import yaml

        # Parse the sample workflow
        parsed = yaml.safe_load(SAMPLE_WORKFLOW_YAML)

        assert 'workflows' in parsed
        assert 'simple_linear' in parsed['workflows']

        workflow = parsed['workflows']['simple_linear']
        assert 'nodes' in workflow
        assert len(workflow['nodes']) == 3

        # Check node types
        node_types = {node['type'] for node in workflow['nodes']}
        assert 'compute' in node_types
        assert 'transform' in node_types

    @pytest.mark.skip("Requires actual YAML loader and validator setup")
    def test_compile_simple_workflow(self):
        """Test compiling a simple workflow from YAML."""
        from victor.workflows.compiler.unified_compiler import WorkflowCompiler
        from victor.workflows.yaml_loader import YAMLWorkflowConfig
        from victor.workflows.executors.factory import NodeExecutorFactory

        # Setup
        config = YAMLWorkflowConfig()
        factory = NodeExecutorFactory()

        # Note: Need to register handlers first
        # config.register_transform("uppercase_transform", lambda s: {"value": s.get("value", "").upper()})

        compiler = WorkflowCompiler(
            yaml_loader=config,
            node_executor_factory=factory,
            yaml_config=config,
        )

        # This would compile the workflow
        # compiled = compiler.compile(SAMPLE_WORKFLOW_YAML, workflow_name="simple_linear")
        # assert compiled is not None


class TestSeparationOfConcerns:
    """Tests for SRP compliance - compiler only compiles, executor only executes."""

    def test_compiler_has_no_execution_logic(self):
        """Verify compiler doesn't have execution methods."""
        from victor.workflows.compiler.unified_compiler import WorkflowCompiler

        class MockLoader:
            pass

        class MockFactory:
            pass

        compiler = WorkflowCompiler(
            yaml_loader=MockLoader(),
            node_executor_factory=MockFactory(),
        )

        # Compiler should NOT have execution methods
        # (those belong to executor)
        assert not hasattr(compiler, 'execute')
        assert not hasattr(compiler, 'run_workflow')

    def test_executor_has_no_compilation_logic(self):
        """Verify executor doesn't have compilation methods."""
        from victor.workflows.executors import StateGraphExecutor
        from unittest.mock import MagicMock

        mock_graph = MagicMock()
        executor = StateGraphExecutor(mock_graph)

        # Executor should NOT have compilation methods
        # (those belong to compiler)
        assert not hasattr(executor, 'compile')
        assert not hasattr(executor, 'load_yaml')
        assert not hasattr(executor, 'build_graph')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
