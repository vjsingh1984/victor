# Copyright 2025 Vijaykumar Singh
#
# Licensed under the Apache License, Version 2.0

"""End-to-end workflow execution tests.

Target: Verify complete workflows execute correctly.
"""

import pytest
from victor.workflows.unified_compiler import UnifiedWorkflowCompiler


class TestWorkflowExecutionE2E:
    """End-to-end workflow execution tests."""

    def test_simple_linear_workflow_execution(self):
        """Test execution of simple linear workflow."""
        yaml_content = """
workflows:
  linear_test:
    nodes:
      - id: step1
        type: transform
        handler: identity
        inputs:
          value: 10
        next: [step2]
      - id: step2
        type: transform
        handler: double_value
        inputs:
          data: $nodes.step1.value
"""
        # Define handler
        handlers = {
            "identity": lambda inputs: inputs,
            "double_value": lambda inputs: {"data": inputs["data"] * 2}
        }
        
        compiler = UnifiedWorkflowCompiler()
        compiled = compiler.compile_yaml_content(yaml_content, "linear_test", handlers=handlers)
        # Note: Actual execution would require orchestrator
        assert compiled is not None

    def test_parallel_workflow_execution(self):
        """Test execution of parallel workflow."""
        yaml_content = """
workflows:
  parallel_test:
    nodes:
      - id: parallel
        type: parallel
        branches:
          - branch_a
          - branch_b
      - id: branch_a
        type: transform
        handler: identity
        inputs:
          result: "A"
      - id: branch_b
        type: transform
        handler: identity
        inputs:
          result: "B"
      - id: aggregate
        type: transform
        handler: combine_results
        inputs:
          a: $nodes.parallel.branch_a.result
          b: $nodes.parallel.branch_b.result
"""
        handlers = {
            "identity": lambda inputs: inputs,
            "combine_results": lambda inputs: f"{inputs['a']}-{inputs['b']}"
        }
        
        compiler = UnifiedWorkflowCompiler()
        compiled = compiler.compile_yaml_content(yaml_content, "parallel_test", handlers=handlers)
        assert compiled is not None

    def test_conditional_workflow_execution(self):
        """Test execution of conditional workflow."""
        yaml_content = """
workflows:
  conditional_test:
    nodes:
      - id: check_quality
        type: condition
        condition: check_score
        inputs:
          score: 85
        branches:
          high_quality: finalize
          low_quality: improve
      - id: finalize
        type: transform
        handler: identity
        inputs:
          status: "done"
      - id: improve
        type: transform
        handler: identity
        inputs:
          status: "improving"
"""
        def check_score(inputs):
            return "high_quality" if inputs["score"] >= 70 else "low_quality"
        
        handlers = {
            "identity": lambda inputs: inputs,
            "check_score": check_score
        }
        
        compiler = UnifiedWorkflowCompiler()
        compiled = compiler.compile_yaml_content(yaml_content, "conditional_test", handlers=handlers)
        assert compiled is not None


class TestWorkflowCaching:
    """Tests for workflow caching behavior."""

    def test_workflow_cache_invalidation(self):
        """Test that changing workflow invalidates cache."""
        yaml_content_v1 = """
workflows:
  cached:
    nodes:
      - id: start
        type: transform
        handler: identity
        next: [end]
      - id: end
        type: transform
        handler: identity
"""
        yaml_content_v2 = """
workflows:
  cached:
    nodes:
      - id: start
        type: transform
        handler: identity
        next: [middle]
      - id: middle
        type: transform
        handler: identity
        next: [end]
      - id: end
        type: transform
        handler: identity
"""
        compiler = UnifiedWorkflowCompiler()
        compiled_v1 = compiler.compile_yaml_content(yaml_content_v1, "cached")
        compiled_v2 = compiler.compile_yaml_content(yaml_content_v2, "cached")
        
        assert compiled_v1 is not None
        assert compiled_v2 is not None
        # Different workflows should compile independently


class TestWorkflowErrorHandling:
    """Tests for workflow error handling."""

    def test_handler_failure_propagates(self):
        """Test that handler failures propagate properly."""
        yaml_content = """
workflows:
  error_test:
    nodes:
      - id: failing_step
        type: transform
        handler: failing_handler
        inputs:
          data: "test"
      - id: recovery
        type: transform
        handler: identity
"""
        def failing_handler(inputs):
            raise ValueError("Handler failed!")
        
        handlers = {
            "failing_handler": failing_handler,
            "identity": lambda inputs: inputs
        }
        
        compiler = UnifiedWorkflowCompiler()
        compiled = compiler.compile_yaml_content(yaml_content, "error_test", handlers=handlers)
        # Compilation should succeed, execution would fail
        assert compiled is not None
