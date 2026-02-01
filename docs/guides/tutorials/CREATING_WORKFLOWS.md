# Creating Custom Workflows for Victor AI

This comprehensive guide teaches you how to create custom workflows for Victor AI.

## Table of Contents

1. [Introduction](#introduction)
2. [Workflow Architecture](#workflow-architecture)
3. [YAML Workflows](#yaml-workflows)
4. [Python Workflows](#python-workflows)
5. [Advanced Workflow Features](#advanced-workflow-features)
6. [Workflow Testing](#workflow-testing)
7. [Best Practices](#best-practices)
8. [Examples](#examples)

## Introduction

### What are Workflows?

Workflows are reusable, multi-step processes that automate complex tasks. They can:
- Chain multiple AI operations
- Mix AI with deterministic code
- Handle human-in-the-loop interactions
- Parallelize independent tasks
- Implement complex branching logic

### Workflow Types

1. **YAML Workflows**: Declarative, easy to read and modify
2. **Python Workflows**: Programmatic, full flexibility
3. **StateGraph Workflows**: LangGraph-compatible, complex state management
4. **HITL Workflows**: Human-in-the-loop, interactive workflows

### Why Create Custom Workflows?

- **Automation**: Automate repetitive multi-step tasks
- **Consistency**: Ensure processes are followed consistently
- **Efficiency**: Combine multiple operations efficiently
- **Reusability**: Share workflows across teams
- **Maintainability**: Declarative workflows are easier to maintain

## Workflow Architecture

### Workflow Components

```
┌─────────────────────────────────────┐
│         Workflow Engine              │
├─────────────────────────────────────┤
│  ┌──────────┐    ┌──────────────┐  │
│  │  Nodes   │───▶│  Transitions │  │
│  └──────────┘    └──────────────┘  │
│  ┌──────────┐    ┌──────────────┐  │
│  │  Edges   │───▶│  Checkpoints │  │
│  └──────────┘    └──────────────┘  │
│  ┌──────────┐    ┌──────────────┐  │
│  │  State   │───▶│     Events   │  │
│  └──────────┘    └──────────────┘  │
└─────────────────────────────────────┘
```

### Node Types

- **agent**: LLM-powered agent node
- **compute**: Deterministic computation node
- **condition**: Branching logic node
- **parallel**: Parallel execution node
- **transform**: Data transformation node
- **hitl**: Human-in-the-loop node
- **end**: Workflow termination node

## YAML Workflows

### Basic YAML Workflow

```yaml
# workflows/code_review.yaml

workflows:
  code_review:
    description: "Comprehensive code review workflow"

    nodes:
      - id: analyze_code
        type: agent
        role: code_reviewer
        goal: |
          Analyze the provided code for:
          1. Code quality and style
          2. Security vulnerabilities
          3. Performance issues
          4. Best practices adherence
        tool_budget: 10
        next: [check_findings]

      - id: check_findings
        type: condition
        condition: has_critical_issues
        branches:
          "yes": request_fixes
          "no": generate_report

      - id: request_fixes
        type: agent
        role: code_reviewer
        goal: |
          Generate specific fix recommendations for the critical issues found.
          Provide code examples and explanations.
        tool_budget: 5
        next: [generate_report]

      - id: generate_report
        type: compute
        handler: generate_review_report
        inputs:
          analysis: $ctx.analyze_code.result
          fixes: $ctx.request_fixes.result
        next: [end]

      - id: end
        type: end
```

### Advanced YAML Workflow with Parallel Execution

```yaml
# workflows/comprehensive_analysis.yaml

workflows:
  comprehensive_analysis:
    description: "Parallel code analysis workflow"

    nodes:
      - id: start
        type: start
        next: [parallel_analysis]

      - id: parallel_analysis
        type: parallel
        nodes:
          - id: security_scan
            type: agent
            role: security_analyst
            goal: "Scan for security vulnerabilities"
            tool_budget: 10

          - id: performance_analysis
            type: agent
            role: performance_expert
            goal: "Analyze performance bottlenecks"
            tool_budget: 10

          - id: quality_check
            type: agent
            role: quality_auditor
            goal: "Check code quality and style"
            tool_budget: 10

          - id: documentation_check
            type: agent
            role: documentation_reviewer
            goal: "Review documentation completeness"
            tool_budget: 5
        next: [aggregate_results]

      - id: aggregate_results
        type: compute
        handler: aggregate_analysis_results
        inputs:
          security: $ctx.parallel_analysis.security_scan.result
          performance: $ctx.parallel_analysis.performance_analysis.result
          quality: $ctx.parallel_analysis.quality_check.result
          documentation: $ctx.parallel_analysis.documentation_check.result
        next: [prioritize_issues]

      - id: prioritize_issues
        type: agent
        role: prioritization_agent
        goal: |
          Prioritize findings from all analysis dimensions.
          Consider:
          1. Severity and impact
          2. Ease of fixing
          3. Dependencies between issues
          Generate prioritized action plan.
        tool_budget: 5
        next: [end]

      - id: end
        type: end
```

### Workflow with HITL (Human-in-the-Loop)

```yaml
# workflows/deployment_approval.yaml

workflows:
  deployment_approval:
    description: "Deployment workflow with human approval"

    nodes:
      - id: prepare_deployment
        type: agent
        role: devops_engineer
        goal: |
          Prepare deployment package:
          1. Check all tests pass
          2. Verify documentation is complete
          3. Generate deployment checklist
        tool_budget: 10
        next: [security_check]

      - id: security_check
        type: agent
        role: security_analyst
        goal: "Perform security pre-deployment check"
        tool_budget: 5
        next: [approve_deployment]

      - id: approve_deployment
        type: hitl
        interaction: approval
        prompt: |
          Review the deployment plan:
          {prepare_deployment.result}

          Security check results:
          {security_check.result}

          Approve deployment?
        options:
          - "approve"
          - "reject"
          - "request_changes"
        next:
          approve: deploy
          reject: cancel_deployment
          request_changes: make_changes

      - id: deploy
        type: agent
        role: deployment_agent
        goal: "Execute deployment plan"
        tool_budget: 15
        next: [verify_deployment]

      - id: verify_deployment
        type: agent
        role: qa_agent
        goal: "Verify deployment success"
        tool_budget: 5
        next: [end]

      - id: make_changes
        type: agent
        role: developer
        goal: "Implement requested changes"
        tool_budget: 10
        next: [prepare_deployment]

      - id: cancel_deployment
        type: compute
        handler: log_deployment_cancellation
        next: [end]

      - id: end
        type: end
```

## Python Workflows

### Basic Python Workflow

```python
# workflows/code_review_workflow.py

from victor.framework import Agent, Task, State, StateGraph
from typing import TypedDict, Dict, Any

class CodeReviewState(TypedDict):
    """State for code review workflow."""
    code: str
    analysis: Dict[str, Any]
    critical_issues: bool
    fixes: Dict[str, Any]
    report: Dict[str, Any]

def create_code_review_workflow() -> StateGraph:
    """Create code review workflow."""

    workflow = StateGraph(CodeReviewState)

    # Node: Analyze code
    async def analyze_code(state: CodeReviewState) -> CodeReviewState:
        """Analyze code for issues."""
        agent = await Agent.create(
            role="code_reviewer",
            goal="Analyze code for quality, security, and performance"
        )

        result = await agent.run(
            f"Analyze this code:\n\n{state['code']}\n\n"
            "Check for:\n"
            "1. Code quality and style\n"
            "2. Security vulnerabilities\n"
            "3. Performance issues\n"
            "4. Best practices"
        )

        state["analysis"] = {
            "content": result.content,
            "issues": result.metadata.get("issues", [])
        }
        state["critical_issues"] = any(
            issue.get("severity") == "CRITICAL"
            for issue in state["analysis"]["issues"]
        )

        return state

    # Node: Request fixes
    async def request_fixes(state: CodeReviewState) -> CodeReviewState:
        """Generate fix recommendations."""
        agent = await Agent.create(role="code_reviewer")

        critical_issues = [
            issue for issue in state["analysis"]["issues"]
            if issue.get("severity") == "CRITICAL"
        ]

        result = await agent.run(
            f"Generate fix recommendations for:\n"
            f"{critical_issues}\n\n"
            "Provide code examples and explanations."
        )

        state["fixes"] = {
            "recommendations": result.content,
            "examples": result.metadata.get("fix_examples", [])
        }

        return state

    # Node: Generate report
    async def generate_report(state: CodeReviewState) -> CodeReviewState:
        """Generate final review report."""
        report = {
            "analysis": state["analysis"],
            "fixes": state.get("fixes", {}),
            "summary": {
                "total_issues": len(state["analysis"]["issues"]),
                "critical": sum(
                    1 for issue in state["analysis"]["issues"]
                    if issue.get("severity") == "CRITICAL"
                ),
                "high": sum(
                    1 for issue in state["analysis"]["issues"]
                    if issue.get("severity") == "HIGH"
                ),
                "has_fixes": "fixes" in state
            }
        }

        state["report"] = report
        return state

    # Conditional: Check for critical issues
    def has_critical_issues(state: CodeReviewState) -> str:
        """Check if critical issues exist."""
        return "request_fixes" if state["critical_issues"] else "generate_report"

    # Add nodes to workflow
    workflow.add_node("analyze_code", analyze_code)
    workflow.add_node("request_fixes", request_fixes)
    workflow.add_node("generate_report", generate_report)

    # Add edges
    workflow.set_entry_point("analyze_code")
    workflow.add_conditional_edges(
        "analyze_code",
        has_critical_issues,
        {
            "request_fixes": "request_fixes",
            "generate_report": "generate_report"
        }
    )
    workflow.add_edge("request_fixes", "generate_report")
    workflow.set_finish_point("generate_report")

    return workflow
```

### Workflow with Custom Handlers

```python
# workflows/analysis_workflow.py

from victor.framework.workflow_engine import WorkflowEngine
from victor.framework import State
from typing import Dict, Any

class AnalysisWorkflowEngine(WorkflowEngine):
    """Custom workflow engine with custom handlers."""

    def __init__(self):
        super().__init__()
        self.register_handlers()

    def register_handlers(self):
        """Register custom workflow handlers."""
        self.register_handler("aggregate_results", self.aggregate_results)
        self.register_handler("prioritize_issues", self.prioritize_issues)
        self.register_handler("generate_report", self.generate_report)

    async def aggregate_results(
        self,
        state: State,
        **kwargs
    ) -> Dict[str, Any]:
        """Aggregate parallel analysis results."""
        security = state.get("security", {})
        performance = state.get("performance", {})
        quality = state.get("quality", {})
        documentation = state.get("documentation", {})

        aggregated = {
            "security": security,
            "performance": performance,
            "quality": quality,
            "documentation": documentation,
            "summary": {
                "total_issues": (
                    len(security.get("issues", [])) +
                    len(performance.get("issues", [])) +
                    len(quality.get("issues", []))
                ),
                "critical_count": sum([
                    len([i for i in security.get("issues", []) if i.get("severity") == "CRITICAL"]),
                    len([i for i in performance.get("issues", []) if i.get("severity") == "CRITICAL"]),
                    len([i for i in quality.get("issues", []) if i.get("severity") == "CRITICAL"])
                ])
            }
        }

        return aggregated

    async def prioritize_issues(
        self,
        state: State,
        **kwargs
    ) -> Dict[str, Any]:
        """Prioritize issues across all dimensions."""
        all_issues = []

        # Collect all issues
        for category in ["security", "performance", "quality"]:
            category_issues = state.get(category, {}).get("issues", [])
            for issue in category_issues:
                issue["category"] = category
                all_issues.append(issue)

        # Sort by severity
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        all_issues.sort(key=lambda x: severity_order.get(x.get("severity", "LOW"), 4))

        return {
            "prioritized_issues": all_issues,
            "action_plan": self._generate_action_plan(all_issues)
        }

    async def generate_report(
        self,
        state: State,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate final analysis report."""
        return {
            "title": "Comprehensive Code Analysis Report",
            "summary": state.get("aggregated_results", {}),
            "action_plan": state.get("prioritized_issues", {}),
            "timestamp": self._get_timestamp()
        }

    def _generate_action_plan(self, issues: list) -> list:
        """Generate prioritized action plan."""
        return [
            {
                "priority": i + 1,
                "issue": issue["description"],
                "severity": issue["severity"],
                "category": issue["category"],
                "estimated_effort": issue.get("effort", "unknown")
            }
            for i, issue in enumerate(issues[:10])  # Top 10 issues
        ]

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
```

## Advanced Workflow Features

### Feature 1: Workflow Caching

```python
from victor.workflows.cache import WorkflowCache
from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

# Create workflow with caching
compiler = UnifiedWorkflowCompiler(
    cache_config={
        "enabled": True,
        "backend": "sqlite",  # or "redis", "memory"
        "ttl": 3600  # 1 hour
    }
)

# Workflow will be cached after first execution
workflow = compiler.compile("my_workflow")

# Subsequent executions use cache
result1 = await workflow.invoke({"input": "data"})
result2 = await workflow.invoke({"input": "data"})  # From cache
```

### Feature 2: Workflow Checkpointing

```python
# Execute workflow with checkpointing
result = await workflow.invoke(
    initial_state,
    config={
        "thread_id": "unique_thread_id",
        "checkpoint_ns": "my_namespace"
    }
)

# Resume from checkpoint
resumed_result = await workflow.invoke(
    None,  # Don't provide state
    config={
        "thread_id": "unique_thread_id",
        "checkpoint_ns": "my_namespace"
    }
)
```

### Feature 3: Workflow Streaming

```python
# Stream workflow execution
async for event in workflow.stream(initial_state):
    if event["type"] == "node_complete":
        print(f"Node {event['node']} completed")
        print(f"Result: {event['result']}")
    elif event["type"] == "workflow_complete":
        print(f"Workflow completed: {event['result']}")
```

### Feature 4: Error Handling

```yaml
# workflows/error_handling.yaml

workflows:
  resilient_workflow:
    description: "Workflow with error handling"

    nodes:
      - id: risky_operation
        type: agent
        role: executor
        goal: "Execute risky operation"
        on_error:
          retry: 3
          backoff: exponential
          fallback: error_handler
        next: [next_step]

      - id: error_handler
        type: agent
        role: error_recovery_specialist
        goal: "Handle and recover from error"
        next: [end]
```

### Feature 5: Workflow Monitoring

```python
from victor.observability.metrics import MetricsRegistry

# Register workflow metrics
metrics = MetricsRegistry.get_instance()

# Track workflow execution time
workflow_timer = metrics.create_timer(
    "workflow.execution_time",
    tags={"workflow": "my_workflow"}
)

# Track workflow success rate
workflow_counter = metrics.create_counter(
    "workflow.executions",
    tags={"workflow": "my_workflow"}
)

# Use in workflow
async def execute_with_metrics():
    with workflow_timer.time():
        try:
            result = await workflow.invoke(state)
            workflow_counter.increment(tags={"status": "success"})
            return result
        except Exception as e:
            workflow_counter.increment(tags={"status": "error"})
            raise
```

## Workflow Testing

### Unit Testing Workflow Nodes

```python
# tests/unit/workflows/test_code_review_workflow.py

import pytest
from workflows.code_review_workflow import create_code_review_workflow

@pytest.mark.asyncio
async def test_analyze_code_node():
    """Test analyze code node."""
    workflow = create_code_review_workflow()

    state = {
        "code": "def hello(): print('hi')",
        "analysis": {},
        "critical_issues": False,
        "fixes": {},
        "report": {}
    }

    result = await workflow.nodes["analyze_code"](state)

    assert "analysis" in result
    assert "content" in result["analysis"]
```

### Integration Testing Workflows

```python
# tests/integration/workflows/test_full_workflow.py

import pytest
from workflows.code_review_workflow import create_code_review_workflow

@pytest.mark.asyncio
async def test_full_code_review_workflow():
    """Test complete workflow."""
    workflow = create_code_review_workflow()
    compiled = workflow.compile()

    initial_state = {
        "code": """
def authenticate(username, password):
    query = f"SELECT * FROM users WHERE username='{username}'"
    return execute(query)
""",
        "analysis": {},
        "critical_issues": False,
        "fixes": {},
        "report": {}
    }

    result = await compiled.invoke(initial_state)

    assert "report" in result
    assert result["report"]["summary"]["critical"] > 0
```

## Best Practices

### 1. Keep Workflows Modular

```yaml
# Good: Modular workflow
workflows:
  code_review:
    nodes:
      - id: analyze
        next: [report]

  security_scan:
    nodes:
      - id: scan
        next: [report]

# Bad: Monolithic workflow
workflows:
  everything:
    nodes:
      - id: do_everything  # Too complex
```

### 2. Use Appropriate Node Types

```yaml
# Agent node: For LLM tasks
- id: analyze
  type: agent
  role: analyzer

# Compute node: For deterministic operations
- id: calculate
  type: compute
  handler: calculate_metrics

# Condition node: For branching
- id: check
  type: condition
  condition: has_errors
```

### 3. Handle Errors Gracefully

```yaml
# Add error handling to critical nodes
- id: critical_operation
  type: agent
  on_error:
    retry: 3
    fallback: error_recovery
```

### 4. Document Workflows

```yaml
workflows:
  documented_workflow:
    description: |
      Comprehensive code review workflow.

      This workflow:
      1. Analyzes code for issues
      2. Checks for critical vulnerabilities
      3. Generates fix recommendations
      4. Produces final report

      Usage:
        victor workflow run code_review --code-path src/
```

### 5. Test Workflows

```python
# Test each node independently
@pytest.mark.asyncio
async def test_node():
    result = await workflow.nodes["node_name"](state)
    assert result["expected_key"] == "expected_value"

# Test full workflow
@pytest.mark.asyncio
async def test_workflow():
    result = await workflow.invoke(initial_state)
    assert result["final_key"] == "final_value"
```

## Examples

### Example 1: Documentation Generation Workflow

```yaml
# workflows/generate_docs.yaml

workflows:
  generate_docs:
    description: "Generate comprehensive documentation"

    nodes:
      - id: analyze_codebase
        type: agent
        role: documentation_specialist
        goal: "Analyze codebase structure and components"
        tool_budget: 10
        next: [generate_readme]

      - id: generate_readme
        type: agent
        role: documentation_specialist
        goal: "Generate comprehensive README.md"
        tool_budget: 15
        next: [generate_api_docs]

      - id: generate_api_docs
        type: agent
        role: documentation_specialist
        goal: "Generate API documentation for all modules"
        tool_budget: 20
        next: [create_diagrams]

      - id: create_diagrams
        type: agent
        role: documentation_specialist
        goal: "Create architecture diagrams using Mermaid or PlantUML"
        tool_budget: 10
        next: [end]
```

### Example 2: Testing Workflow

```yaml
# workflows/test_generation.yaml

workflows:
  generate_tests:
    description: "Generate comprehensive test suite"

    nodes:
      - id: analyze_code
        type: agent
        role: test_engineer
        goal: "Analyze code to identify testable components"
        tool_budget: 10
        next: [generate_unit_tests]

      - id: generate_unit_tests
        type: agent
        role: test_engineer
        goal: "Generate unit tests for all functions and classes"
        tool_budget: 20
        next: [generate_integration_tests]

      - id: generate_integration_tests
        type: agent
        role: test_engineer
        goal: "Generate integration tests for API endpoints and workflows"
        tool_budget: 15
        next: [run_tests]

      - id: run_tests
        type: compute
        handler: run_test_suite
        inputs:
            test_command: "pytest tests/"
        next: [end]
```

## Conclusion

Custom workflows enable you to automate complex, multi-step processes with Victor AI. Use YAML workflows for simplicity and Python workflows for flexibility.

For more examples, see:
- `victor/workflows/` - Built-in workflows
- `examples/workflows/` - Workflow examples
- `docs/guides/workflow-quickstart.md` - Workflow quickstart

Happy workflow building! 🔄

---

**Last Updated:** February 01, 2026
**Reading Time:** 2 min
