# Creating Workflows - Part 1

**Part 1 of 4:** Workflow Basics, Conditions, and Parallel Execution

---

## Navigation

- **[Part 1: Basics & Parallel](#)** (Current)
- [Part 2: HITL & Escape Hatches](part-2-hitl-escape.md)
- [Part 3: Testing & Examples](part-3-testing-examples.md)
- [Part 4: Reference](part-4-reference.md)
- [**Complete Guide**](../create-workflow.md)

---
# Creating Workflows in Victor

Learn how to build powerful multi-agent workflows using Victor's YAML-first workflow system. This tutorial walks you
  through creating a complete code review workflow with conditions,
  parallel execution, and human-in-the-loop approvals.

## What You Will Build

By the end of this tutorial, you will have created a **code review workflow** that:

- Analyzes code for issues using an AI agent
- Runs security and style checks in parallel
- Routes tasks based on code quality conditions
- Requires human approval before merging
- Generates a final review report

## Prerequisites

- Victor installed (`pip install victor-ai`)
- Basic understanding of YAML syntax
- Familiarity with Victor's agent system

**Time estimate: 45 minutes**

---

## 1. Workflow Basics

### YAML Structure Overview

Victor workflows are defined in YAML files with a consistent structure:

```yaml
workflows:
  workflow_name:
    description: "What this workflow does"
    metadata:
      key: value
    nodes:
      - id: node_id
        type: agent|compute|condition|parallel|transform|hitl
        # node-specific configuration
        next:
          - next_node_id
```text

### Node Types Summary

| Type | Purpose | Uses LLM |
|------|---------|----------|
| `agent` | Execute tasks using AI agents with tools | Yes |
| `compute` | Execute tools directly without LLM reasoning | No |
| `condition` | Branch workflow based on conditions | No |
| `parallel` | Execute multiple nodes simultaneously | Varies |
| `transform` | Transform data between nodes | No |
| `hitl` | Pause for human approval or input | No |
| `team` | Spawn multi-agent teams | Yes |

### How Workflows Execute

1. **Start**: Execution begins at the first node (or specified `start_node`)
2. **Sequential**: Nodes execute in order based on `next` references
3. **Branching**: Condition nodes route to different paths based on context
4. **Parallel**: Parallel nodes execute children simultaneously, then merge results
5. **Human Gates**: HITL nodes pause execution until human responds
6. **Completion**: Workflow ends when reaching a node with no `next` or an `END` type

---

## 2. Step-by-Step: Simple Workflow

Let's start with a basic two-node workflow.

### Step 1: Create the YAML File

Create a file `workflows/simple_review.yaml`:

```yaml
workflows:
  simple_review:
    description: "Basic code review workflow"
    nodes:
      - id: analyze
        type: agent
        role: code_reviewer
        goal: "Analyze the provided code for issues and best practice violations"
        tool_budget: 15
        allowed_tools:
          - read
          - grep
          - code_search
        output_key: analysis_result
        next:
          - report

      - id: report
        type: agent
        role: reporter
        goal: "Generate a comprehensive review report based on the analysis"
        tool_budget: 5
        allowed_tools:
          - read
        output_key: review_report
```

### Step 2: Add Metadata

Metadata provides context for the workflow system:

```yaml
workflows:
  simple_review:
    description: "Basic code review workflow"
    metadata:
      category: code_quality
      version: "1.0"
      author: "team"
      tags:
        - review
        - quality
    nodes:
      # ... nodes from above
```text

### Step 3: Configure Node Details

Each agent node supports these key properties:

```yaml
- id: analyze                    # Unique identifier
  type: agent                    # Node type
  name: "Code Analyzer"          # Human-readable name (optional)
  role: code_reviewer            # Agent role: researcher, executor, reviewer, writer, etc.
  goal: "Analyze code..."        # Task description (supports $ctx.variable substitution)
  tool_budget: 15                # Maximum tool calls allowed
  allowed_tools:                 # Specific tools to enable
    - read
    - grep
  input_mapping:                 # Map context keys to agent inputs
    file_path: target_file
  output_key: analysis_result    # Store output under this key
  timeout_seconds: 120           # Optional execution timeout
  llm_config:                    # Optional LLM overrides
    temperature: 0.3
  next:
    - report
```

### Step 4: Run the Workflow

Execute your workflow using the CLI:

```bash
# Validate the workflow
victor workflow validate workflows/simple_review.yaml

# Run the workflow
victor workflow run workflows/simple_review.yaml --workflow simple_review \
  --input '{"target_file": "src/main.py"}'
```text

Or programmatically in Python:

```python
from victor.workflows.unified_compiler import compile_and_execute

result = await compile_and_execute(
    "workflows/simple_review.yaml",
    initial_state={"target_file": "src/main.py"},
    workflow_name="simple_review"
)

print(result.state.get("review_report"))
```

---

## 3. Adding Conditions

Conditions allow workflows to branch based on runtime state.

### Condition Nodes

```yaml
workflows:
  conditional_review:
    description: "Review with quality-based routing"
    nodes:
      - id: analyze
        type: agent
        role: code_reviewer
        goal: "Analyze code quality and count issues"
        tool_budget: 15
        output_key: analysis
        next:
          - check_quality

      - id: check_quality
        type: condition
        condition: "code_quality_check"    # References escape hatch function
        branches:
          excellent: approve_merge
          good: approve_merge
          acceptable: minor_fixes
          needs_improvement: major_fixes

      - id: minor_fixes
        type: agent
        role: fixer
        goal: "Apply minor code improvements"
        tool_budget: 10
        next:
          - final_report

      - id: major_fixes
        type: agent
        role: fixer
        goal: "Address major issues found in analysis"
        tool_budget: 20
        next:
          - re_analyze

      - id: re_analyze
        type: agent
        role: code_reviewer
        goal: "Re-analyze code after fixes"
        tool_budget: 10
        next:
          - check_quality

      - id: approve_merge
        type: agent
        role: approver
        goal: "Approve code for merging"
        tool_budget: 3
        next:
          - final_report

      - id: final_report
        type: agent
        role: reporter
        goal: "Generate final review summary"
        tool_budget: 5
```text

### Escape Hatch Functions

Conditions reference Python functions defined in `escape_hatches.py`. Create one in your vertical or workflow directory:

```python
# escape_hatches.py
"""Escape hatches for review workflows."""

from typing import Any, Dict


def code_quality_check(ctx: Dict[str, Any]) -> str:
    """Assess code quality based on analysis results.

    Args:
        ctx: Workflow context containing:
            - analysis (dict): Analysis results from the analyzer

    Returns:
        "excellent", "good", "acceptable", or "needs_improvement"
    """
    analysis = ctx.get("analysis", {})

    # Extract metrics from analysis
    errors = analysis.get("errors", 0)
    warnings = analysis.get("warnings", 0)
    issues = analysis.get("issues", [])

    if errors == 0 and warnings == 0:
        return "excellent"

    if errors == 0 and warnings <= 3:
        return "good"

    if errors <= 2:
        return "acceptable"

    return "needs_improvement"


def tests_passed(ctx: Dict[str, Any]) -> str:
    """Check if tests are passing.

    Args:
        ctx: Workflow context with test_results

    Returns:
        "true" or "false"
    """
    test_results = ctx.get("test_results", {})
    failed = test_results.get("failed", 0)
    return "true" if failed == 0 else "false"


# Registry for the YAML loader
CONDITIONS = {
    "code_quality_check": code_quality_check,
    "tests_passed": tests_passed,
}

TRANSFORMS = {}
```

### Using Escape Hatches

When compiling, provide the condition registry:

```python
from victor.workflows.unified_compiler import UnifiedWorkflowCompiler
from escape_hatches import CONDITIONS, TRANSFORMS

compiler = UnifiedWorkflowCompiler()
graph = compiler.compile_yaml(
    "workflows/conditional_review.yaml",
    workflow_name="conditional_review",
    condition_registry=CONDITIONS,
    transform_registry=TRANSFORMS,
)

result = await graph.invoke({"target_file": "src/main.py"})
```text


**Reading Time:** 5 min
**Last Updated:** February 08, 2026**

---

## See Also

- [Documentation Home](../../README.md)


## 4. Parallel Execution
