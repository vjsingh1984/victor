# Workflow Editor Test Fixtures

This directory contains sample workflow YAML files for testing the visual workflow editor.

## Fixture Files

### `simple_linear.yaml`
A simple linear workflow with sequential agent nodes. Used for testing basic import/export.

### `team_formations.yaml`
Contains all 8 team formation types (parallel, sequential, pipeline, hierarchical, consensus, round_robin, dynamic, custom). Used for comprehensive formation testing.

### `conditional_branches.yaml`
Workflow with multiple conditional branches. Used for testing connection mapping and branch visualization.

### `parallel_execution.yaml`
Workflow with parallel execution nodes. Used for testing parallel node visualization and connection handling.

### `recursion_test.yaml`
Workflow with nested teams to test recursion depth limits. Used for testing depth validation.

### `hitl_workflow.yaml`
Workflow with human-in-the-loop nodes. Used for testing HITL node rendering.

## Adding New Fixtures

When adding new fixture files:

1. Use descriptive names that indicate what the fixture tests
2. Include comments explaining the purpose of the workflow
3. Keep fixtures minimal but realistic
4. Test edge cases and error conditions
5. Document any special characteristics in this README

## Fixture Usage

```python
def test_with_fixture():
    from victor.workflows import load_workflow_from_file

    workflow = load_workflow_from_file(
        "tests/integration/workflow_editor/fixtures/simple_linear.yaml"
    )

    assert workflow is not None
```
