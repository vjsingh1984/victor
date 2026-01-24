# Workflows User Guide

Complete guide to Victor's YAML workflow system for automating multi-step tasks.

## Overview

Victor's workflow system enables you to define complex, multi-step automation as declarative YAML files. Workflows combine LLM-powered agents, computational operations, conditional branching, parallel execution, and human-in-the-loop approvals into cohesive pipelines.

### Key Features

- **YAML-First Design**: Define workflows declaratively with Python escape hatches for complex logic
- **Six Node Types**: agent, compute, condition, parallel, transform, hitl
- **Domain Verticals**: Pre-built workflows for Coding, DevOps, RAG, Data Analysis, and Research
- **UnifiedWorkflowCompiler**: Single compilation pipeline with two-level caching
- **Checkpointing**: Resume interrupted workflows from saved state

### When to Use Workflows

| Use Case | Example |
|----------|---------|
| Multi-step processes | Code review with linting, security scan, and approval |
| Repeatable automation | CI/CD deployment with rollback |
| Complex decision trees | Bug investigation with branching diagnosis |
| Parallel operations | Running multiple analysis tools simultaneously |
| Human oversight | Approval gates before production deployment |

## Quick Start

### Running a Built-in Workflow

```bash
# Run a code review workflow
victor workflow run code_review

# Run with custom parameters
victor workflow run deploy --env staging --version 1.2.0
```

### Programmatic Execution

```python
from victor.workflows.unified_compiler import UnifiedWorkflowCompiler
from pathlib import Path

# Create compiler with caching
compiler = UnifiedWorkflowCompiler(enable_caching=True, cache_ttl=3600)

# Compile from YAML file
compiled = compiler.compile_yaml(Path("workflow.yaml"), "my_workflow")

# Execute the workflow
result = await compiled.invoke({"input": "data"})

# Check execution result
if result.success:
    print(f"Completed: {result.state}")
else:
    print(f"Failed: {result.error}")
```

## YAML Workflow Syntax

### Basic Structure

```yaml
workflows:
  workflow_name:
    description: "What this workflow does"

    metadata:
      version: "1.0"
      author: "victor"
      vertical: coding  # coding, devops, rag, dataanalysis, research

    # Optional: Service dependencies
    services:
      project_db:
        type: sqlite
        config:
          path: $ctx.project_dir/.victor/project.db

    nodes:
      - id: first_node
        type: agent
        name: "Human-readable name"
        role: researcher
        goal: "What this node should accomplish"
        tool_budget: 20
        tools: [read, grep, code_search]
        output: result_key
        next: [second_node]

      - id: second_node
        type: condition
        name: "Decision Point"
        condition: "check_result"
        branches:
          "success": success_node
          "failure": failure_node
```

### Variable References

Workflows support context variable references:

| Syntax | Description | Example |
|--------|-------------|---------|
| `$ctx.key` | Context variable | `$ctx.source_directory` |
| `$env.KEY` | Environment variable | `$env.ANTHROPIC_API_KEY` |
| `{key}` | Template substitution in goal/prompt | `{review_findings}` |

## Node Types

### Agent Nodes

Agent nodes use LLM reasoning to perform tasks. They can use tools and make decisions.

```yaml
- id: analyze_code
  type: agent
  name: "Analyze Code Quality"
  role: reviewer              # Agent's role identity
  goal: |                     # Detailed instructions
    Review the code changes for:
    1. Logic errors and bugs
    2. Security vulnerabilities
    3. Performance issues
    4. Code style violations

    Analysis Results: {lint_results}
  tool_budget: 30             # Maximum tool calls allowed
  tools: [read, grep, code_search, shell]  # Available tools
  llm_config:                 # Optional LLM settings
    temperature: 0.3
    model_hint: claude-3-sonnet
  input_mapping:              # Map context to inputs
    changes: pr_changes
  output: review_findings     # Key to store result
  next: [categorize]          # Next node(s)
```

**Key Properties:**

| Property | Required | Description |
|----------|----------|-------------|
| `id` | Yes | Unique node identifier |
| `role` | Yes | Agent's role (researcher, executor, reviewer, planner, writer, analyst) |
| `goal` | Yes | Instructions for what the agent should do |
| `tool_budget` | No | Maximum tool invocations (default: 20) |
| `tools` | No | List of available tools |
| `llm_config` | No | Temperature, model_hint, max_tokens |
| `output` | No | Context key to store result |
| `next` | No | Next node ID(s) |

### Compute Nodes

Compute nodes execute deterministic operations without LLM reasoning. Use them for:
- Running shell commands (linters, tests, builds)
- Data transformations
- API calls to external services
- File operations

```yaml
- id: run_tests
  type: compute
  name: "Run Test Suite"
  handler: retry_with_backoff   # Optional handler for execution
  tools: [shell]
  inputs:
    command: $ctx.test_command
    coverage: true
  output: test_results
  constraints:
    llm_allowed: false          # Block LLM usage
    network_allowed: true       # Allow network access
    write_allowed: true         # Allow file writes
    timeout: 300                # Timeout in seconds
  next: [check_tests]
```

**Constraint Options:**

| Constraint | Default | Description |
|------------|---------|-------------|
| `llm_allowed` | false | Whether LLM calls are permitted |
| `network_allowed` | false | Whether network access is permitted |
| `write_allowed` | false | Whether file writes are permitted |
| `timeout` | 120 | Maximum execution time in seconds |

### Condition Nodes

Condition nodes branch execution based on context state or escape hatch functions.

```yaml
- id: check_tests
  type: condition
  name: "Check Test Results"
  condition: "tests_passing"    # Escape hatch function name
  branches:
    "passing": deploy
    "failing": fix_tests
    "no_tests": generate_tests
```

Simple conditions can use expressions:

```yaml
- id: check_count
  type: condition
  condition: "result_count >= 3"
  branches:
    "true": proceed
    "false": fallback
```

### Parallel Nodes

Parallel nodes execute multiple nodes concurrently and wait for all to complete.

```yaml
- id: parallel_analysis
  type: parallel
  name: "Run Parallel Checks"
  parallel_nodes: [lint_check, type_check, security_scan, complexity_analysis]
  join_strategy: all           # Wait for all nodes to complete
  next: [aggregate_results]

- id: lint_check
  type: compute
  name: "Run Linters"
  tools: [shell]
  inputs:
    commands:
      - $ctx.lint_command
      - $ctx.format_check_command
  output: lint_results
  constraints:
    llm_allowed: false
    timeout: 180

- id: type_check
  type: compute
  name: "Run Type Checker"
  tools: [shell]
  inputs:
    command: $ctx.type_check_command
  output: type_results
  constraints:
    llm_allowed: false
    timeout: 180
```

**Join Strategies:**

| Strategy | Description |
|----------|-------------|
| `all` | Wait for all parallel nodes to complete |
| `any` | Continue when any node completes |
| `majority` | Continue when >50% complete |

### Transform Nodes

Transform nodes perform simple data transformations without tools or LLM.

```yaml
- id: aggregate_results
  type: transform
  name: "Aggregate Analysis Results"
  transform: |
    total_issues = lint_issues + type_issues + security_issues
    has_blocking = security_critical > 0 or type_errors > 0
    status = "ready" if not has_blocking else "blocked"
  next: [decision_point]
```

### HITL (Human-in-the-Loop) Nodes

HITL nodes pause execution for human approval or input.

```yaml
- id: deployment_approval
  type: hitl
  name: "Deployment Approval"
  hitl_type: approval           # approval, input, or review
  prompt: |
    ## Ready for Deployment

    **Environment:** {target_env}
    **Version:** {deploy_version}
    **Changes:** {change_summary}

    Approve deployment?
  context_keys:                 # Keys to include in prompt
    - target_env
    - deploy_version
    - change_summary
  choices:                      # Available choices for input type
    - "Approve"
    - "Reject"
    - "Request Changes"
  timeout: 900                  # Timeout in seconds (15 min)
  fallback: abort               # Action on timeout: continue, abort, skip
  next: [handle_approval]
```

**HITL Types:**

| Type | Description |
|------|-------------|
| `approval` | Binary approve/reject decision |
| `input` | Multiple choice selection |
| `review` | Review with comments |

## Escape Hatches

When conditions or transforms are too complex for YAML, use Python escape hatches.

### Location

Each vertical has an escape hatches file:

```
victor/coding/escape_hatches.py
victor/devops/escape_hatches.py
victor/rag/escape_hatches.py
victor/dataanalysis/escape_hatches.py
victor/research/escape_hatches.py
```

### Example Escape Hatch

```python
# victor/coding/escape_hatches.py

def tests_passing(ctx: Dict[str, Any]) -> str:
    """Check if tests are passing.

    Args:
        ctx: Workflow context with keys:
            - test_results (dict): Test execution results
            - min_coverage (float): Minimum coverage threshold

    Returns:
        "passing", "failing", or "no_tests"
    """
    test_results = ctx.get("test_results", {})
    min_coverage = ctx.get("min_coverage", 0.8)

    if not test_results:
        return "no_tests"

    passed = test_results.get("passed", 0)
    failed = test_results.get("failed", 0)
    coverage = test_results.get("coverage", 0)

    if failed > 0:
        return "failing"

    if coverage < min_coverage:
        return "failing"

    if passed > 0:
        return "passing"

    return "no_tests"


def code_quality_check(ctx: Dict[str, Any]) -> str:
    """Assess code quality based on linting and static analysis."""
    lint_results = ctx.get("lint_results", {})
    type_check_results = ctx.get("type_check_results", {})

    lint_errors = lint_results.get("errors", 0)
    type_errors = type_check_results.get("errors", 0)

    if lint_errors == 0 and type_errors == 0:
        return "excellent"
    if lint_errors <= 3 and type_errors <= 2:
        return "acceptable"
    return "needs_improvement"
```

### Using Escape Hatches in YAML

```yaml
- id: check_quality
  type: condition
  condition: "code_quality_check"  # References escape hatch function
  branches:
    "excellent": fast_track_approval
    "acceptable": standard_review
    "needs_improvement": request_fixes
```

## Built-in Workflows by Vertical

### Coding Vertical

| Workflow | Description | File |
|----------|-------------|------|
| `code_review` | Comprehensive code review with linting, security, and AI analysis | `victor/coding/workflows/code_review.yaml` |
| `quick_review` | Fast review for small changes | `victor/coding/workflows/code_review.yaml` |
| `pr_review` | Pull request review with impact analysis | `victor/coding/workflows/code_review.yaml` |
| `feature_implementation` | End-to-end feature development with tests | `victor/coding/workflows/feature.yaml` |
| `bugfix` | Bug investigation and fix with regression tests | `victor/coding/workflows/feature.yaml` |
| `tdd` | Test-Driven Development with red-green-refactor cycle | `victor/coding/workflows/tdd.yaml` |
| `refactor` | Code refactoring with safety checks | `victor/coding/workflows/refactor.yaml` |

**Example: Code Review Workflow**

```yaml
workflows:
  code_review:
    nodes:
      - id: gather_changes
        type: compute
        tools: [shell]
        inputs:
          command: $ctx.diff_command
        output: changes
        next: [parallel_analysis]

      - id: parallel_analysis
        type: parallel
        parallel_nodes: [lint_check, type_check, security_scan]
        next: [ai_review]

      - id: ai_review
        type: agent
        role: reviewer
        goal: "Review code for logic errors, best practices, and performance"
        tool_budget: 30
        next: [human_approval]

      - id: human_approval
        type: hitl
        hitl_type: approval
        prompt: "Review and approve changes?"
```

### DevOps Vertical

| Workflow | Description | File |
|----------|-------------|------|
| `deploy` | Safe deployment with validation and rollback | `victor/devops/workflows/deploy.yaml` |
| `cicd` | Continuous integration and deployment pipeline | `victor/devops/workflows/deploy.yaml` |
| `container_setup` | Dockerfile creation and container configuration | `victor/devops/workflows/container_setup.yaml` |
| `container_quick` | Quick container build without security scan | `victor/devops/workflows/container_setup.yaml` |

**Example: Deployment Workflow**

```yaml
workflows:
  deploy:
    nodes:
      - id: validate_config
        type: compute
        inputs:
          config_path: $ctx.config_file
        next: [backup_current]

      - id: backup_current
        type: compute
        inputs:
          backup_type: full
        next: [approval_gate]

      - id: approval_gate
        type: hitl
        hitl_type: approval
        prompt: "Approve deployment to {target_env}?"
        next: [deploy]

      - id: deploy
        type: compute
        inputs:
          strategy: $ctx.deploy_strategy
        next: [health_check]

      - id: health_check
        type: compute
        inputs:
          endpoints: $ctx.health_endpoints
        next: [verify_health]

      - id: verify_health
        type: condition
        condition: "all_healthy"
        branches:
          "true": complete
          "false": rollback
```

### RAG Vertical

| Workflow | Description | File |
|----------|-------------|------|
| `document_ingest` | Ingest documents into vector store | `victor/rag/workflows/ingest.yaml` |
| `incremental_update` | Update index with new/modified documents | `victor/rag/workflows/ingest.yaml` |
| `rag_query` | Answer questions using retrieved context | `victor/rag/workflows/query.yaml` |
| `conversation` | Multi-turn RAG conversation | `victor/rag/workflows/query.yaml` |
| `agentic_rag` | RAG with agentic reasoning and tool use | `victor/rag/workflows/query.yaml` |
| `maintenance` | Index maintenance and optimization | `victor/rag/workflows/query.yaml` |

**Example: RAG Query Workflow**

```yaml
workflows:
  rag_query:
    nodes:
      - id: analyze_query
        type: agent
        role: analyst
        goal: "Understand query intent and key concepts"
        next: [parallel_search]

      - id: parallel_search
        type: parallel
        parallel_nodes: [dense_search, sparse_search, entity_search]
        next: [merge_results]

      - id: dense_search
        type: compute
        inputs:
          queries: $ctx.expanded_queries
          top_k: 20
        constraints:
          llm_allowed: false

      - id: merge_results
        type: compute
        inputs:
          fusion_method: reciprocal_rank
        next: [rerank]

      - id: rerank
        type: agent
        role: analyst
        goal: "Rerank results by semantic relevance"
        next: [generate_answer]

      - id: generate_answer
        type: agent
        role: writer
        goal: "Generate answer with inline citations"
```

### Research Vertical

| Workflow | Description | File |
|----------|-------------|------|
| `deep_research` | Comprehensive research with source validation | `victor/research/workflows/deep_research.yaml` |
| `quick_research` | Fast research for simple queries | `victor/research/workflows/deep_research.yaml` |
| `fact_check` | Systematic fact verification | `victor/research/workflows/fact_check.yaml` |
| `literature_review` | Academic literature review | `victor/research/workflows/literature_review.yaml` |
| `competitive_analysis` | Competitive market analysis | `victor/research/workflows/competitive_analysis.yaml` |

**Example: Deep Research Workflow**

```yaml
workflows:
  deep_research:
    nodes:
      - id: understand_query
        type: agent
        role: researcher
        goal: "Analyze research query and key concepts"
        next: [parallel_search]

      - id: parallel_search
        type: parallel
        parallel_nodes: [web_search, academic_search, code_search]
        next: [validate_sources]

      - id: validate_sources
        type: agent
        role: analyst
        goal: "Evaluate source credibility and relevance"
        next: [synthesize]

      - id: synthesize
        type: agent
        role: analyst
        goal: "Synthesize findings and identify patterns"
        next: [generate_report]

      - id: generate_report
        type: agent
        role: writer
        goal: "Generate comprehensive research report"
        tools: [write]
```

### Data Analysis Vertical

| Workflow | Description | File |
|----------|-------------|------|
| `eda_pipeline` | Exploratory data analysis | `victor/dataanalysis/workflows/eda_pipeline.yaml` |
| `data_cleaning` | Data cleaning and preprocessing | `victor/dataanalysis/workflows/data_cleaning.yaml` |
| `ml_pipeline` | Machine learning pipeline | `victor/dataanalysis/workflows/ml_pipeline.yaml` |
| `statistical_analysis` | Statistical analysis workflow | `victor/dataanalysis/workflows/statistical_analysis.yaml` |
| `automl_pipeline` | Automated machine learning | `victor/dataanalysis/workflows/automl_pipeline.yaml` |

## Execution

### Using WorkflowProvider

Each vertical provides a workflow provider for easy execution:

```python
from victor.coding.workflows import CodingWorkflowProvider

# Create provider
provider = CodingWorkflowProvider()

# Compile workflow
compiled = provider.compile_workflow("code_review")

# Execute
result = await compiled.invoke({
    "diff_command": "git diff HEAD~1",
    "test_command": "pytest tests/",
    "lint_command": "ruff check .",
})
```

### Using UnifiedWorkflowCompiler

For direct control over compilation and execution:

```python
from victor.workflows.unified_compiler import UnifiedWorkflowCompiler
from pathlib import Path

# Create compiler with configuration
compiler = UnifiedWorkflowCompiler(
    enable_caching=True,
    cache_ttl=3600,
    enable_checkpointing=True,
)

# Compile from YAML
compiled = compiler.compile_yaml(
    Path("victor/coding/workflows/code_review.yaml"),
    "code_review"
)

# Execute with checkpointing for resumption
result = await compiled.invoke(
    {"source_directory": "/path/to/code"},
    thread_id="review-123"  # For checkpointing
)

# Check cache statistics
stats = compiler.get_cache_stats()
print(f"Cache hits: {stats['hits']}, misses: {stats['misses']}")
```

### Streaming Execution

Stream intermediate states during execution:

```python
async for event in compiled.stream(initial_state):
    match event.type:
        case "node_start":
            print(f"Starting: {event.node_id}")
        case "node_complete":
            print(f"Completed: {event.node_id}")
        case "edge_taken":
            print(f"Transition: {event.source} -> {event.target}")
        case "complete":
            print(f"Final: {event.state}")
```

## Advanced Topics

### Workflow Scheduling

Schedule workflows for periodic execution:

```yaml
workflows:
  daily_report:
    schedule:
      cron: "0 9 * * *"         # Daily at 9 AM
      timezone: "America/New_York"
      catchup: false
      max_active_runs: 1

    execution:
      max_timeout_seconds: 3600
      max_iterations: 50

    nodes:
      # ... workflow nodes
```

See the [Scheduling Guide](../guides/workflow-development/scheduling.md) for details.

### Workflow Versioning

Version workflows for safe migrations:

```python
from victor.workflows.versioning import (
    WorkflowVersion,
    VersionedWorkflow,
    WorkflowVersionRegistry,
)

# Register versioned workflow
registry = WorkflowVersionRegistry()
registry.register(VersionedWorkflow(
    name="data_pipeline",
    version=WorkflowVersion(2, 0, 0),
    definition=workflow_def,
))

# Get specific version
v1 = registry.get("data_pipeline", "0.5.0")
latest = registry.get_latest("data_pipeline")
```

### Services Configuration

Define service dependencies that are started before workflow execution:

```yaml
services:
  # SQLite database
  project_db:
    type: sqlite
    config:
      path: $ctx.project_dir/.victor/project.db
      journal_mode: WAL
    lifecycle:
      start: auto
      cleanup: preserve

  # Vector store
  vector_store:
    type: lancedb
    config:
      path: $ctx.project_dir/.victor/vectors
    lifecycle:
      start: auto
      cleanup: preserve
```

### Batch Processing

Configure batch processing for handling multiple items:

```yaml
batch_config:
  batch_size: 10              # Items per batch
  max_concurrent: 4           # Parallel batches
  retry_strategy: end_of_batch
  max_retries: 3
```

## CLI Commands

```bash
# List available workflows
victor workflow list

# Show workflow details
victor workflow show code_review

# Validate workflow YAML
victor workflow validate path/to/workflow.yaml

# Run workflow
victor workflow run code_review

# Run with parameters
victor workflow run deploy --env production --version 2.0.0

# Start scheduler daemon
victor scheduler start

# Add scheduled workflow
victor scheduler add daily_report --cron "0 9 * * *"
```

## Best Practices

### 1. Use Compute Nodes for Deterministic Operations

Prefer compute nodes over agent nodes when:
- Running shell commands (git, linters, tests)
- Performing data transformations
- Calling external APIs with structured responses

```yaml
# Good: Compute for deterministic operations
- id: run_lint
  type: compute
  tools: [shell]
  inputs:
    command: "ruff check ."
  constraints:
    llm_allowed: false

# Bad: Using agent for simple shell commands
- id: run_lint
  type: agent
  goal: "Run the linter"
```

### 2. Set Appropriate Tool Budgets

Balance capability with cost:

```yaml
# Quick analysis: Lower budget
- id: quick_scan
  type: agent
  tool_budget: 10

# Deep analysis: Higher budget
- id: comprehensive_review
  type: agent
  tool_budget: 50
```

### 3. Use Parallel Nodes for Independent Operations

```yaml
# Run independent checks concurrently
- id: parallel_checks
  type: parallel
  parallel_nodes: [lint, typecheck, security, tests]
```

### 4. Add HITL Gates for Critical Operations

```yaml
# Require approval before destructive operations
- id: deploy_approval
  type: hitl
  hitl_type: approval
  prompt: "Deploy to production?"
  next: [execute_deploy]
```

### 5. Document Complex Escape Hatches

```python
def complex_routing_logic(ctx: Dict[str, Any]) -> str:
    """Route based on multiple factors.

    Decision matrix:
    - Critical issues + no tests -> block
    - Critical issues + passing tests -> review
    - No critical issues -> proceed

    Args:
        ctx: Must contain 'issues' and 'test_results'

    Returns:
        One of: "block", "review", "proceed"
    """
    # ... implementation
```

## Related Documentation

- [StateGraph DSL Guide](../guides/workflow-development/dsl.md) - Building workflow graphs programmatically
- [Scheduling Guide](../guides/workflow-development/scheduling.md) - Workflow scheduling and versioning
- [Tool Catalog](../reference/tools/catalog.md) - Available tools for workflows

---

*Last Updated: 2026-01-10*
