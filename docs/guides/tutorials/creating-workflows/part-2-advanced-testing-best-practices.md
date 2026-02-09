# Creating Custom Workflows for Victor AI - Part 2

**Part 2 of 2:** Advanced Features, Testing, Best Practices, and Examples

---

## Navigation

- [Part 1: YAML & Python Workflows](part-1-yaml-python.md)
- **[Part 2: Advanced, Testing, Best Practices](#)** (Current)
- [**Complete Guide**](../CREATING_WORKFLOWS.md)

---

## Advanced Workflow Features

### Parallel Execution

Execute nodes in parallel:

```yaml
# workflow.yaml
name: parallel_analysis
nodes:
  analyze_code:
    type: llm
    prompt: "Analyze this code: {code}"
  run_tests:
    type: tool
    tool: pytest
  check_coverage:
    type: tool
    tool: coverage

edges:
  - [start, [analyze_code, run_tests, check_coverage]]
  - [[analyze_code, run_tests, check_coverage], end]
```

### Conditional Branching

Branch based on conditions:

```yaml
# conditional_workflow.yaml
name: conditional_deployment
nodes:
  check_tests:
    type: tool
    tool: pytest
  deploy_prod:
    type: tool
    tool: deploy_prod
  deploy_staging:
    type: tool
    tool: deploy_staging

edges:
  - [start, check_tests]
  - [check_tests, deploy_prod, condition: "{tests_passed} == true"]
  - [check_tests, deploy_staging, condition: "{tests_passed} == false"]
```

### Error Handling

Handle errors gracefully:

```python
# python_workflow.py
from victor.framework import StateGraph, Node

async def error_handler(state):
    """Handle errors in workflow."""
    if state.get("error"):
        # Log error
        logger.error(f"Workflow error: {state['error']}")

        # Retry or fallback
        if state.get("retry_count", 0) < 3:
            state["retry_count"] = state.get("retry_count", 0) + 1
            return "retry_node"
        else:
            return "error_node"

    return "next_node"
```

---

## Workflow Testing

### Unit Testing Workflows

```python
# test_workflows.py
import pytest
from victor.framework import StateGraph

def test_simple_workflow():
    """Test simple workflow execution."""
    graph = StateGraph("test_workflow")

    graph.add_node("process", process_node)
    graph.add_edge("start", "process")
    graph.add_edge("process", "end")

    result = graph.invoke({"data": "test"})
    assert result["status"] == "success"
```

### Integration Testing

```python
def test_workflow_integration():
    """Test workflow with real tools."""
    result = workflow.run(
        input_data={"query": "test"},
        use_real_tools=True
    )
    assert result["success"] is True
```

---

## Best Practices

1. **Keep workflows simple**: Avoid overly complex workflows
2. **Use descriptive names**: Make node and edge names clear
3. **Handle errors**: Always include error handling
4. **Test thoroughly**: Unit and integration tests
5. **Document well**: Clear documentation for complex workflows
6. **Version control**: Track workflow changes in git
7. **Monitor execution**: Track workflow performance

---

## Examples

### Example 1: Code Review Workflow

```yaml
# code_review.yaml
name: code_review
nodes:
  analyze_code:
    type: llm
    prompt: "Review this code for bugs: {code}"
  suggest_improvements:
    type: llm
    prompt: "Suggest improvements for: {code}"
  generate_report:
    type: template
    template: "review_report.md"

edges:
  - [start, analyze_code]
  - [analyze_code, suggest_improvements]
  - [suggest_improvements, generate_report]
  - [generate_report, end]
```

### Example 2: Deployment Workflow

```python
# deployment_workflow.py
from victor.framework import StateGraph

async def run_tests(state):
    """Run test suite."""
    result = await run_command("pytest")
    state["tests_passed"] = result.returncode == 0
    return state

async def deploy(state):
    """Deploy to environment."""
    if state["tests_passed"]:
        await run_command(f"deploy --env={state['environment']}")
    return state

# Create workflow
graph = StateGraph("deployment")
graph.add_node("run_tests", run_tests)
graph.add_node("deploy", deploy)

graph.add_edge("start", "run_tests")
graph.add_conditional_edge(
    "run_tests",
    lambda s: "deploy" if s["tests_passed"] else "end"
)
graph.add_edge("deploy", "end")
```

---

## Conclusion

Custom workflows enable powerful automation in Victor AI. By combining YAML and Python workflows,
  you can create sophisticated multi-step processes that combine AI operations with traditional code.

**Next Steps:**
- Explore [Workflow Examples](../../examples/workflows/)
- Read [StateGraph API](../../reference/framework/README.md)
- Learn [Advanced Patterns](../ADVANCED_WORKFLOW_PATTERNS.md)

---

**Reading Time:** 2 min
**Last Updated:** February 01, 2026
