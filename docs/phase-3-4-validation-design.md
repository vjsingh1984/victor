# Graph Validation and Refinement System Design

**Phase 3.4: Dynamic Workflow Validation**

**Author:** Claude (AI Assistant)
**Date:** 2026-01-09
**Status:** Design Draft
**Related:** Phase 3.0-3.3 (Orchestration Prerequisites)

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Validation Layers](#validation-layers)
4. [Error Reporting](#error-reporting)
5. [Refinement Strategies](#refinement-strategies)
6. [Iterative Refinement Loop](#iterative-refinement-loop)
7. [Human-in-the-Loop](#human-in-the-loop)
8. [LLM Integration](#llm-integration)
9. [Implementation Plan](#implementation-plan)
10. [MVP Feature List](#mvp-feature-list)

---

## Overview

### Problem Statement

As Victor introduces **dynamically generated workflows** (Phase 3.5+), where LLMs generate workflow definitions in response to natural language requests, we face a critical challenge: **LLMs make mistakes**. Generated workflows may contain:

- Schema errors (missing fields, invalid types)
- Structural errors (cycles, orphan nodes, broken edges)
- Semantic errors (invalid node types, unknown tools)
- Security issues (dangerous tool combinations, resource abuse)

Without robust validation, these errors would cause runtime failures, wasted computation, or worse—security vulnerabilities.

### Solution: Multi-Layer Validation with Automated Refinement

This design describes a **comprehensive validation and refinement system** that:

1. **Validates** generated workflows across multiple layers (schema, structure, semantics, security)
2. **Reports** clear, actionable errors with location context
3. **Refines** workflows automatically using fix strategies
4. **Iterates** with LLM until validation passes or max attempts reached
5. **Engages humans** for approval on complex workflows
6. **Integrates** seamlessly with existing StateGraph and workflow infrastructure

### Key Design Principles

- **Fail Fast, Fix Fast**: Catch errors early, fix automatically when safe
- **Clear Feedback**: Error messages must guide LLM to correct solution
- **Safe Defaults**: When in doubt, use conservative defaults
- **Human Oversight**: Complex workflows require approval
- **Progressive Enhancement**: MVP first, extend later

### Integration Points

This system integrates with:

- **victor.framework.graph**: StateGraph validation and compilation
- **victor.workflows.unified_compiler**: YAML/definition compilation
- **victor.core.validation**: Pydantic-based validation framework
- **victor.workflows.validation.tool_validator**: Existing tool dependency validation
- **victor.framework.hitl**: Human-in-the-loop infrastructure

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    WORKFLOW GENERATION                          │
│                                                                   │
│  User Request → LLM → Initial Workflow Definition (YAML/JSON)  │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              VALIDATION AND REFINEMENT ENGINE                    │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Multi-Layer Validator (4 layers)                      │    │
│  │  ┌──────────────────────────────────────────────────┐  │    │
│  │  │ 1. Schema Validation   (Pydantic)                │  │    │
│  │  │ 2. Structure Validation (Graph analysis)          │  │    │
│  │  │ 3. Semantic Validation  (Node/Tool checks)        │  │    │
│  │  │ 4. Security Validation  (Resource/Safety checks)   │  │    │
│  │  └──────────────────────────────────────────────────┘  │    │
│  └──────────────────────────┬─────────────────────────────┘    │
│                             ▼                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Error Aggregator & Reporter                            │    │
│  │  - Categorize errors (schema/structure/semantic/sec)   │    │
│  │  - Group related errors                                │    │
│  │  - Prioritize by severity (critical/error/warning)     │    │
│  │  - Generate fix suggestions                            │    │
│  └──────────────────────────┬─────────────────────────────┘    │
│                             ▼                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Refinement Decision                                    │    │
│  │  IF valid THEN proceed to compilation                  │    │
│  │  ELSE IF auto-fixable THEN apply fixes                 │    │
│  │  ELSE IF iterations < max THEN request LLM refinement  │    │
│  │  ELSE IF complex workflow THEN request human approval  │    │
│  │  ELSE fail with detailed error report                  │    │
│  └──────────────────────────┬─────────────────────────────┘    │
└────────────────────────────┼────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ITERATIVE LOOP                              │
│                                                                   │
│  Refinement Request → LLM → Revised Workflow → Re-validate      │
│                          ↑                                        │
│                          └──── Until valid OR max iterations     │
└───────────────────────────────────────────────────────────────────┘
```

### Module Structure

```
victor/workflows/generation/
├── __init__.py                 # Public API exports
├── validator.py                # Multi-layer validator (main)
├── refiner.py                  # Automated refinement strategies
├── error_reporter.py           # Error aggregation and reporting
├── approval.py                 # Human-in-the-loop approval workflow
├── prompts.py                  # LLM prompt templates for refinement
└── types.py                    # Type definitions for validation
```

### Key Components

1. **WorkflowValidator (`validator.py`)**
   - Orchestrates 4 validation layers
   - Runs layers sequentially or in parallel (configurable)
   - Returns aggregated `WorkflowValidationResult`

2. **RefinementEngine (`refiner.py`)**
   - Applies automated fixes based on error categories
   - Tracks refinement history for convergence detection
   - Provides fallback strategies

3. **ErrorReporter (`error_reporter.py`)**
   - Aggregates errors from all layers
   - Groups related errors
   - Generates human-readable reports
   - Formats errors for LLM consumption

4. **ApprovalWorkflow (`approval.py`)**
   - Manages HITL approval process
   - Tracks approval history
   - Integrates with existing HITL infrastructure

5. **RefinementPrompts (`prompts.py`)**
   - Template system for LLM refinement requests
   - Few-shot examples library
   - Error-context formatting

---

## Validation Layers

### Layer 1: Schema Validation

**Purpose**: Validate YAML/JSON structure against expected schema

**Implementation**: Uses `victor.core.validation` (Pydantic-based)

**Checks**:
- ✅ Required fields present (`nodes`, `edges`, `entry_point`)
- ✅ Field types correct (`nodes` is list, `edges` is list, etc.)
- ✅ Field values within valid ranges (`tool_budget` > 0, `timeout` >= 0)
- ✅ No unknown fields (strict mode) or warn (lenient mode)
- ✅ String constraints (non-empty node IDs, valid edge types)
- ✅ List constraints (non-empty node list, unique node IDs)

**Schema Definition**:

```python
from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Any, Literal

class WorkflowNodeSchema(BaseModel):
    """Schema for individual workflow nodes."""
    id: str = Field(..., min_length=1, max_length=100)
    name: str = Field(..., min_length=1, max_length=200)
    type: Literal["agent", "compute", "condition", "parallel", "transform", "team", "hitl"]
    next_nodes: List[str] = Field(default_factory=list)

    # Type-specific fields
    role: Optional[str] = None
    goal: Optional[str] = None
    tool_budget: Optional[int] = Field(None, ge=0, le=500)
    tools: Optional[List[str]] = None
    condition: Optional[str] = None
    branches: Optional[Dict[str, str]] = None
    parallel_nodes: Optional[List[str]] = None
    join_strategy: Optional[Literal["all", "any", "merge"]] = None
    handler: Optional[str] = None
    transform: Optional[str] = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Node ID must be alphanumeric with underscores/hyphens."""
        import re
        if not re.match(r"^[a-zA-Z0-9_\-]+$", v):
            raise ValueError("Node ID must be alphanumeric (underscores/hyphens allowed)")
        return v

class WorkflowEdgeSchema(BaseModel):
    """Schema for workflow edges."""
    source: str = Field(..., min_length=1)
    target: str = Field(...)  # Can be node ID or dict for conditional
    type: Literal["normal", "conditional"] = "normal"
    condition: Optional[str] = None

class WorkflowDefinitionSchema(BaseModel):
    """Root schema for workflow definitions."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    entry_point: str = Field(..., min_length=1)
    nodes: List[WorkflowNodeSchema] = Field(..., min_length=1)
    edges: List[WorkflowEdgeSchema] = Field(..., min_length=0)

    # Optional workflow-level config
    max_iterations: Optional[int] = Field(None, ge=1, le=500)
    max_timeout_seconds: Optional[float] = Field(None, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("nodes")
    @classmethod
    def validate_node_ids_unique(cls, v: List[WorkflowNodeSchema]) -> List[WorkflowNodeSchema]:
        """Ensure all node IDs are unique."""
        ids = [node.id for node in v]
        if len(ids) != len(set(ids)):
            duplicates = [id for id in ids if ids.count(id) > 1]
            raise ValueError(f"Duplicate node IDs found: {set(duplicates)}")
        return v
```

**Error Examples**:

```python
# Missing required field
ValidationError(
    path="nodes[0]",
    message="Field required: 'type'",
    severity="critical"
)

# Invalid type
ValidationError(
    path="nodes[0].tool_budget",
    message="'tool_budget' must be integer, got string",
    severity="error"
)

# Invalid range
ValidationError(
    path="entry_point",
    message="Node 'start' not found in node definitions",
    severity="critical"
)
```

---

### Layer 2: Graph Structure Validation

**Purpose**: Validate workflow graph topology (cycles, reachability, references)

**Implementation**: Custom graph algorithms

**Checks**:

1. **Node Reachability**
   - ✅ All nodes reachable from entry point
   - ✅ No orphan nodes (except nodes with no incoming edges that are entry points)
   - ✅ Terminal nodes exist (paths to END)

2. **Edge References**
   - ✅ All edge sources reference valid nodes
   - ✅ All edge targets reference valid nodes (or END)
   - ✅ Conditional edges have condition functions
   - ✅ Conditional branches map to valid nodes

3. **Cycle Detection**
   - ✅ No unconditional cycles (A → B → A without condition)
   - ✅ Conditional cycles allowed if condition node in cycle
   - ✅ Cycle depth within limits (prevent infinite loops)

4. **Path Validation**
   - ✅ At least one path from entry to terminal
   - ✅ No dead-end nodes (nodes with no outgoing edges that aren't terminal)
   - ✅ Parallel nodes have proper join structure

**Algorithm Examples**:

```python
class GraphStructureValidator:
    """Validates workflow graph structure."""

    def __init__(self, max_cycle_depth: int = 10):
        self.max_cycle_depth = max_cycle_depth

    def validate(
        self,
        nodes: List[WorkflowNodeSchema],
        edges: List[WorkflowEdgeSchema],
        entry_point: str
    ) -> List[GraphValidationError]:
        """Run all structure validations."""
        errors = []

        # Build adjacency list
        graph = self._build_graph(nodes, edges)

        # Check reachability
        errors.extend(self._check_reachability(graph, entry_point, nodes))

        # Check edge references
        errors.extend(self._check_edge_references(graph, nodes, edges))

        # Check for invalid cycles
        errors.extend(self._check_cycles(graph, nodes))

        # Check for dead ends
        errors.extend(self._check_dead_ends(graph, nodes))

        return errors

    def _build_graph(
        self,
        nodes: List[WorkflowNodeSchema],
        edges: List[WorkflowEdgeSchema]
    ) -> Dict[str, List[str]]:
        """Build adjacency list representation."""
        graph = {node.id: [] for node in nodes}
        for edge in edges:
            if edge.source in graph:
                if isinstance(edge.target, str):
                    graph[edge.source].append(edge.target)
                elif isinstance(edge.target, dict):
                    # Conditional edge - add all branch targets
                    graph[edge.source].extend(edge.target.values())
        return graph

    def _check_reachability(
        self,
        graph: Dict[str, List[str]],
        entry_point: str,
        nodes: List[WorkflowNodeSchema]
    ) -> List[GraphValidationError]:
        """Check all nodes are reachable from entry point."""
        errors = []
        visited = set()
        queue = [entry_point]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            queue.extend(graph.get(current, []))

        # Find unreachable nodes
        node_ids = {node.id for node in nodes}
        unreachable = node_ids - visited - {"__end__"}

        for node_id in unreachable:
            errors.append(GraphValidationError(
                error_type="unreachable_node",
                message=f"Node '{node_id}' is not reachable from entry point '{entry_point}'",
                location=f"nodes[{node_id}]",
                severity="error",
                suggestion=f"Add edge from existing node to '{node_id}' or set as entry_point"
            ))

        return errors

    def _check_cycles(
        self,
        graph: Dict[str, List[str]],
        nodes: List[WorkflowNodeSchema]
    ) -> List[GraphValidationError]:
        """Detect invalid cycles (cycles without condition nodes)."""
        errors = []

        # Build node type lookup
        node_types = {node.id: node.type for node in nodes}
        condition_nodes = {n.id for n in nodes if n.type == "condition"}

        # Detect cycles using DFS
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: List[str]) -> Optional[List[str]]:
            """DFS to detect cycles, returns cycle path if found."""
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                if neighbor not in node_types:
                    continue  # Skip END marker
                if neighbor not in visited:
                    result = dfs(neighbor, path.copy())
                    if result:
                        return result
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:] + [neighbor]

            path.pop()
            rec_stack.remove(node)
            return None

        for node_id in graph:
            if node_id not in visited:
                cycle_path = dfs(node_id, [])
                if cycle_path:
                    # Check if cycle contains a condition node
                    has_condition = any(n in condition_nodes for n in cycle_path)

                    if not has_condition:
                        errors.append(GraphValidationError(
                            error_type="invalid_cycle",
                            message=f"Unconditional cycle detected: {' -> '.join(cycle_path)}",
                            location=f"edges",
                            severity="critical",
                            suggestion=f"Add a condition node in the cycle or restructure to avoid cycle"
                        ))

        return errors
```

**Error Examples**:

```python
GraphValidationError(
    error_type="unreachable_node",
    message="Node 'cleanup' is not reachable from entry point 'start'",
    location="nodes[cleanup]",
    severity="error",
    suggestion="Add edge from existing node to 'cleanup' or set as entry_point"
)

GraphValidationError(
    error_type="invalid_cycle",
    message="Unconditional cycle detected: step1 -> step2 -> step1",
    location="edges",
    severity="critical",
    suggestion="Add a condition node in the cycle or restructure to avoid cycle"
)
```

---

### Layer 3: Semantic Validation

**Purpose**: Validate node semantics (types, configurations, tools)

**Implementation**: Type-specific validators with registry pattern

**Checks**:

1. **Node Type Validation**
   - ✅ Agent nodes have `role` and `goal`
   - ✅ Compute nodes have `tools` or `handler`
   - ✅ Condition nodes have `branches` mapping
   - ✅ Parallel nodes have `parallel_nodes` list
   - ✅ Transform nodes have `transform` function
   - ✅ Team nodes have `members` and `team_formation`

2. **Node Configuration**
   - ✅ Agent roles valid (researcher, planner, executor, reviewer, writer)
   - ✅ Team formations valid (sequential, parallel, hierarchical, pipeline, consensus)
   - ✅ Join strategies valid (all, any, merge)
   - ✅ Tool budgets within range (1-500)
   - ✅ Timeouts positive (or None)

3. **Tool Availability**
   - ✅ All referenced tools exist in tool registry
   - ✅ Tools are enabled (not disabled)
   - ✅ Tool permissions compatible with node context

4. **Handler/Function Availability**
   - ✅ Compute handlers registered
   - ✅ Transform functions registered
   - ✅ Condition functions registered

**Implementation**:

```python
class SemanticValidator:
    """Validates workflow node semantics."""

    def __init__(
        self,
        tool_registry: Optional[ToolRegistryProtocol] = None,
        handler_registry: Optional[Dict[str, Any]] = None,
        strict_mode: bool = True
    ):
        self.tool_validator = ToolDependencyValidator(tool_registry, strict=strict_mode)
        self.handler_registry = handler_registry or {}
        self.strict_mode = strict_mode

        # Valid values for enum-like fields
        self.valid_agent_roles = {
            "researcher", "planner", "executor", "reviewer", "writer", "analyst"
        }
        self.valid_team_formations = {
            "sequential", "parallel", "hierarchical", "pipeline", "consensus"
        }
        self.valid_join_strategies = {"all", "any", "merge"}
        self.valid_node_types = {
            "agent", "compute", "condition", "parallel", "transform", "team", "hitl"
        }

    def validate(
        self,
        workflow_def: WorkflowDefinitionSchema
    ) -> List[SemanticValidationError]:
        """Run all semantic validations."""
        errors = []

        for node in workflow_def.nodes:
            errors.extend(self._validate_node(node))

        return errors

    def _validate_node(
        self,
        node: WorkflowNodeSchema
    ) -> List[SemanticValidationError]:
        """Validate a single node's semantics."""
        errors = []

        # Type-specific validation
        if node.type == "agent":
            errors.extend(self._validate_agent_node(node))
        elif node.type == "compute":
            errors.extend(self._validate_compute_node(node))
        elif node.type == "condition":
            errors.extend(self._validate_condition_node(node))
        elif node.type == "parallel":
            errors.extend(self._validate_parallel_node(node))
        elif node.type == "transform":
            errors.extend(self._validate_transform_node(node))
        elif node.type == "team":
            errors.extend(self._validate_team_node(node))

        return errors

    def _validate_agent_node(
        self,
        node: WorkflowNodeSchema
    ) -> List[SemanticValidationError]:
        """Validate agent node configuration."""
        errors = []

        # Required fields
        if not node.role:
            errors.append(SemanticValidationError(
                error_type="missing_required_field",
                message=f"Agent node '{node.id}' must specify 'role'",
                location=f"nodes[{node.id}]",
                severity="error",
                suggestion=f"Add 'role' field (valid: {self.valid_agent_roles})"
            ))
        elif node.role not in self.valid_agent_roles:
            errors.append(SemanticValidationError(
                error_type="invalid_role",
                message=f"Invalid agent role '{node.role}' in node '{node.id}'",
                location=f"nodes[{node.id}].role",
                severity="error",
                suggestion=f"Use one of: {self.valid_agent_roles}"
            ))

        if not node.goal:
            errors.append(SemanticValidationError(
                error_type="missing_required_field",
                message=f"Agent node '{node.id}' must specify 'goal'",
                location=f"nodes[{node.id}]",
                severity="error",
                suggestion="Add 'goal' field with task description"
            ))

        # Validate tool budget
        if node.tool_budget is not None and not (1 <= node.tool_budget <= 500):
            errors.append(SemanticValidationError(
                error_type="invalid_range",
                message=f"Tool budget {node.tool_budget} out of range [1, 500]",
                location=f"nodes[{node.id}].tool_budget",
                severity="error",
                suggestion="Set tool_budget between 1 and 500"
            ))

        # Validate tools if specified
        if node.tools:
            # Tool availability check
            tool_result = self.tool_validator.validate_tools_exist(
                node.tools,
                context=f"nodes[{node.id}]"
            )
            for tool_error in tool_result.errors:
                errors.append(SemanticValidationError(
                    error_type="tool_not_found",
                    message=tool_error.message,
                    location=f"nodes[{node.id}].tools",
                    severity="error" if self.strict_mode else "warning",
                    suggestion="Verify tool is registered or remove from tools list"
                ))

        return errors

    def _validate_compute_node(
        self,
        node: WorkflowNodeSchema
    ) -> List[SemanticValidationError]:
        """Validate compute node configuration."""
        errors = []

        # Must have tools OR handler
        has_tools = node.tools is not None and len(node.tools) > 0
        has_handler = node.handler is not None

        if not has_tools and not has_handler:
            errors.append(SemanticValidationError(
                error_type="missing_required_field",
                message=f"Compute node '{node.id}' must specify 'tools' or 'handler'",
                location=f"nodes[{node.id}]",
                severity="error",
                suggestion="Add 'tools' list or 'handler' function name"
            ))

        # Validate handler exists
        if has_handler and node.handler not in self.handler_registry:
            errors.append(SemanticValidationError(
                error_type="handler_not_found",
                message=f"Handler '{node.handler}' not registered",
                location=f"nodes[{node.id}].handler",
                severity="error",
                suggestion=f"Register handler or use one of: {list(self.handler_registry.keys())}"
            ))

        # Validate tools if specified
        if has_tools:
            tool_result = self.tool_validator.validate_tools_exist(
                node.tools,
                context=f"nodes[{node.id}]"
            )
            for tool_error in tool_result.errors:
                errors.append(SemanticValidationError(
                    error_type="tool_not_found",
                    message=tool_error.message,
                    location=f"nodes[{node.id}].tools",
                    severity="error" if self.strict_mode else "warning"
                ))

        return errors
```

---

### Layer 4: Security Validation

**Purpose**: Validate workflow doesn't violate security/safety constraints

**Implementation**: Rule-based security checker

**Checks**:

1. **Tool Combinations**
   - ✅ No dangerous tool combinations (e.g., `file_delete` + `bash_execute`)
   - ✅ No network tools in airgapped mode
   - ✅ File operations within allowed directories

2. **Resource Limits**
   - ✅ Total tool budget within workflow limits (default: 500)
   - ✅ Total timeout within workflow limits (default: 3600s)
   - ✅ Max iterations within safe limits (default: 50)
   - ✅ Parallel branches within limits (default: 10)

3. **Permission Checks**
   - ✅ Nodes don't exceed granted permissions
   - ✅ No privileged tools in unapproved workflows
   - ✅ File access within sandbox boundaries

4. **Infinite Loop Prevention**
   - ✅ Conditional cycles have max_iterations
   - ✅ Parallel nodes don't spawn unbounded sub-workflows
   - ✅ Recursive patterns have depth limits

**Implementation**:

```python
class SecurityValidator:
    """Validates workflow security and safety constraints."""

    def __init__(
        self,
        max_tool_budget: int = 500,
        max_timeout_seconds: float = 3600.0,
        max_iterations: int = 50,
        max_parallel_branches: int = 10,
        airgapped_mode: bool = False,
        allowed_directories: Optional[List[str]] = None
    ):
        self.max_tool_budget = max_tool_budget
        self.max_timeout_seconds = max_timeout_seconds
        self.max_iterations = max_iterations
        self.max_parallel_branches = max_parallel_branches
        self.airgapped_mode = airgapped_mode
        self.allowed_directories = allowed_directories or []

        # Dangerous tool combinations
        self.dangerous_combinations = [
            ({"file_delete", "file_write"}, "bash_execute"),
            ({"bash_execute"}, "file_system_modify"),
        ]

        # Network tools (blocked in airgapped mode)
        self.network_tools = {
            "web_search", "http_request", "api_call", "fetch_url"
        }

        # Privileged tools (require approval)
        self.privileged_tools = {
            "bash_execute", "file_delete", "system_command"
        }

    def validate(
        self,
        workflow_def: WorkflowDefinitionSchema
    ) -> List[SecurityValidationError]:
        """Run all security validations."""
        errors = []

        # Check resource limits
        errors.extend(self._check_resource_limits(workflow_def))

        # Check tool combinations
        errors.extend(self._check_tool_combinations(workflow_def))

        # Check airgapped mode constraints
        if self.airgapped_mode:
            errors.extend(self._check_airgapped_constraints(workflow_def))

        # Check for privileged tools
        errors.extend(self._check_privileged_tools(workflow_def))

        # Check infinite loop patterns
        errors.extend(self._check_infinite_loops(workflow_def))

        return errors

    def _check_resource_limits(
        self,
        workflow_def: WorkflowDefinitionSchema
    ) -> List[SecurityValidationError]:
        """Check workflow doesn't exceed resource limits."""
        errors = []

        # Sum tool budgets
        total_tool_budget = sum(
            node.tool_budget or 15
            for node in workflow_def.nodes
            if node.type == "agent"
        )

        if total_tool_budget > self.max_tool_budget:
            errors.append(SecurityValidationError(
                error_type="resource_limit_exceeded",
                message=f"Total tool budget {total_tool_budget} exceeds limit {self.max_tool_budget}",
                location="workflow",
                severity="error",
                suggestion=f"Reduce tool budgets or increase max_tool_budget"
            ))

        # Check parallel branches
        for node in workflow_def.nodes:
            if node.type == "parallel" and node.parallel_nodes:
                if len(node.parallel_nodes) > self.max_parallel_branches:
                    errors.append(SecurityValidationError(
                        error_type="resource_limit_exceeded",
                        message=f"Parallel node '{node.id}' has {len(node.parallel_nodes)} branches, exceeds limit {self.max_parallel_branches}",
                        location=f"nodes[{node.id}]",
                        severity="error",
                        suggestion=f"Reduce parallel branches or increase max_parallel_branches"
                    ))

        return errors

    def _check_tool_combinations(
        self,
        workflow_def: WorkflowDefinitionSchema
    ) -> List[SecurityValidationError]:
        """Check for dangerous tool combinations."""
        errors = []

        # Collect all tools used in workflow
        all_tools = set()
        for node in workflow_def.nodes:
            if node.tools:
                all_tools.update(node.tools)

        # Check dangerous combinations
        for tool_set, dangerous_tool in self.dangerous_combinations:
            if tool_set.issubset(all_tools) and dangerous_tool in all_tools:
                errors.append(SecurityValidationError(
                    error_type="dangerous_tool_combination",
                    message=f"Dangerous tool combination: {tool_set} + {dangerous_tool}",
                    location="workflow",
                    severity="critical",
                    suggestion=f"Remove {dangerous_tool} or restructure workflow to avoid combination"
                ))

        return errors

    def _check_infinite_loops(
        self,
        workflow_def: WorkflowDefinitionSchema
    ) -> List[SecurityValidationError]:
        """Check for potential infinite loops."""
        errors = []

        # Build graph to detect cycles
        graph = self._build_graph(workflow_def)

        # Find cycles
        cycles = self._find_cycles(graph)

        for cycle_path in cycles:
            # Check if cycle has max_iterations on any node
            has_iter_limit = any(
                node.type == "condition" or
                (workflow_def.metadata.get("max_iterations"))
                for node in workflow_def.nodes
                if node.id in cycle_path
            )

            if not has_iter_limit:
                errors.append(SecurityValidationError(
                    error_type="potential_infinite_loop",
                    message=f"Cycle without iteration limit: {' -> '.join(cycle_path)}",
                    location="workflow",
                    severity="error",
                    suggestion="Add max_iterations to workflow or condition node in cycle"
                ))

        return errors
```

---

## Error Reporting

### Error Aggregation

**Purpose**: Collect, categorize, and prioritize validation errors

**Data Structure**:

```python
@dataclass
class ValidationError:
    """Base class for all validation errors."""
    error_type: str  # Category: schema/structure/semantic/security
    message: str    # Human-readable description
    location: str   # Path to error location (JSON path format)
    severity: Literal["critical", "error", "warning", "info"]
    suggestion: Optional[str] = None  # How to fix
    value: Optional[Any] = None       # Actual value that caused error

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "error_type": self.error_type,
            "message": self.message,
            "location": self.location,
            "severity": self.severity,
            "suggestion": self.suggestion,
            "value": str(self.value) if self.value else None,
        }

@dataclass
class WorkflowValidationResult:
    """Aggregated result of all validation layers."""
    is_valid: bool
    schema_errors: List[ValidationError] = field(default_factory=list)
    structure_errors: List[ValidationError] = field(default_factory=list)
    semantic_errors: List[ValidationError] = field(default_factory=list)
    security_errors: List[ValidationError] = field(default_factory=list)

    @property
    def all_errors(self) -> List[ValidationError]:
        """Get all errors aggregated."""
        return (
            self.schema_errors +
            self.structure_errors +
            self.semantic_errors +
            self.security_errors
        )

    @property
    def critical_errors(self) -> List[ValidationError]:
        """Get only critical errors."""
        return [e for e in self.all_errors if e.severity == "critical"]

    @property
    def error_count(self) -> Dict[str, int]:
        """Get count of errors by severity."""
        counts = {"critical": 0, "error": 0, "warning": 0, "info": 0}
        for error in self.all_errors:
            counts[error.severity] += 1
        return counts

    def summary(self) -> str:
        """Get human-readable summary."""
        if self.is_valid:
            return "✓ Workflow validation passed"

        counts = self.error_count
        total = sum(counts.values())
        return (
            f"✗ Validation failed: {total} issues "
            f"({counts['critical']} critical, {counts['error']} errors, "
            f"{counts['warning']} warnings)"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "is_valid": self.is_valid,
            "summary": self.summary(),
            "error_counts": self.error_count,
            "errors": {
                "schema": [e.to_dict() for e in self.schema_errors],
                "structure": [e.to_dict() for e in self.structure_errors],
                "semantic": [e.to_dict() for e in self.semantic_errors],
                "security": [e.to_dict() for e in self.security_errors],
            }
        }

    def group_by_node(self) -> Dict[str, List[ValidationError]]:
        """Group errors by node ID."""
        grouped = {}
        for error in self.all_errors:
            # Extract node ID from location
            node_id = self._extract_node_id(error.location)
            if node_id not in grouped:
                grouped[node_id] = []
            grouped[node_id].append(error)
        return grouped

    def _extract_node_id(self, location: str) -> str:
        """Extract node ID from location path."""
        if "nodes[" in location:
            # Extract from "nodes[node_id]" or "nodes[0].field"
            import re
            match = re.search(r'nodes\[([^\]]+)\]', location)
            if match:
                return match.group(1)
        return "workflow"  # Workflow-level error
```

### Error Reporter

**Purpose**: Format validation results for different consumers (humans, LLMs, logs)

**Implementation**:

```python
class ErrorReporter:
    """Formats validation results for different consumers."""

    def human_report(self, result: WorkflowValidationResult) -> str:
        """Generate human-readable error report."""
        lines = []
        lines.append("=" * 80)
        lines.append("WORKFLOW VALIDATION REPORT")
        lines.append("=" * 80)
        lines.append(f"Status: {'VALID ✓' if result.is_valid else 'INVALID ✗'}")
        lines.append("")

        if not result.is_valid:
            # Critical errors first
            if result.critical_errors:
                lines.append("CRITICAL ERRORS (must fix):")
                lines.append("-" * 80)
                for error in result.critical_errors:
                    lines.append(f"  • {error.location}: {error.message}")
                    if error.suggestion:
                        lines.append(f"    💡 Suggestion: {error.suggestion}")
                lines.append("")

            # Other errors by category
            for category, errors in [
                ("Schema", result.schema_errors),
                ("Structure", result.structure_errors),
                ("Semantic", result.semantic_errors),
                ("Security", result.security_errors),
            ]:
                if errors and not any(e.severity == "critical" for e in errors):
                    lines.append(f"{category.upper()} ERRORS:")
                    lines.append("-" * 80)
                    for error in errors:
                        if error.severity != "critical":
                            severity_emoji = {
                                "error": "❌",
                                "warning": "⚠️ ",
                                "info": "ℹ️ "
                            }.get(error.severity, "")
                            lines.append(f"  {severity_emoji} {error.location}: {error.message}")
                            if error.suggestion:
                                lines.append(f"    💡 {error.suggestion}")
                    lines.append("")

            # Summary
            counts = result.error_count
            lines.append("-" * 80)
            lines.append(f"Total: {sum(counts.values())} issues")
            lines.append(f"  Critical: {counts['critical']}")
            lines.append(f"  Errors: {counts['error']}")
            lines.append(f"  Warnings: {counts['warning']}")
            lines.append("=" * 80)

        return "\n".join(lines)

    def llm_report(
        self,
        result: WorkflowValidationResult,
        include_fixes: bool = True
    ) -> str:
        """Generate LLM-friendly error report for refinement."""
        if result.is_valid:
            return "Validation passed. No changes needed."

        lines = []
        lines.append("VALIDATION FAILED - FIXES REQUIRED:")
        lines.append("")

        # Group errors by type for LLM
        grouped = {
            "schema": result.schema_errors,
            "structure": result.structure_errors,
            "semantic": result.semantic_errors,
            "security": result.security_errors,
        }

        for category, errors in grouped.items():
            if not errors:
                continue

            lines.append(f"{category.upper()} ERRORS ({len(errors)}):")
            for error in errors:
                lines.append(f"  - Location: {error.location}")
                lines.append(f"    Error: {error.message}")
                if include_fixes and error.suggestion:
                    lines.append(f"    Fix: {error.suggestion}")
            lines.append("")

        return "\n".join(lines)

    def json_report(self, result: WorkflowValidationResult) -> str:
        """Generate JSON report for programmatic consumption."""
        import json
        return json.dumps(result.to_dict(), indent=2)

    def compact_report(self, result: WorkflowValidationResult) -> str:
        """Generate compact one-line summary."""
        if result.is_valid:
            return "✓ Valid"

        counts = result.error_count
        return (
            f"✗ {counts['critical']} critical, "
            f"{counts['error']} errors, "
            f"{counts['warning']} warnings"
        )
```

---

## Refinement Strategies

### Automated Refinement

**Purpose**: Automatically fix common errors without LLM intervention

**Categories of Auto-Fixes**:

1. **Schema Fixes**
   - Add missing required fields with defaults
   - Remove invalid extra fields
   - Convert types (string → int for numeric fields)
   - Normalize values (trim strings, clamp ranges)

2. **Structure Fixes**
   - Remove orphan nodes
   - Add missing entry point
   - Connect dangling edges
   - Remove duplicate edges

3. **Semantic Fixes**
   - Replace invalid roles with closest valid role
   - Remove unknown tools from tool lists
   - Add missing handler placeholders
   - Infer node types from context

**Implementation**:

```python
class RefinementEngine:
    """Automated refinement strategies for workflow validation errors."""

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode

        # Default values for missing fields
        self.defaults = {
            "agent": {
                "role": "executor",
                "tool_budget": 15,
                "goal": "Execute task",
            },
            "compute": {
                "tools": [],
            },
            "condition": {
                "branches": {},
            },
        }

    def refine(
        self,
        workflow_def: WorkflowDefinitionSchema,
        validation_result: WorkflowValidationResult
    ) -> Tuple[WorkflowDefinitionSchema, List[str]]:
        """Apply automated refinements to workflow.

        Returns:
            Tuple of (refined_workflow, list_of_changes_made)
        """
        changes = []
        refined = copy.deepcopy(workflow_def)

        # Apply fixes by category
        if not self.strict_mode:
            # Schema fixes (safe to apply)
            refined, schema_changes = self._fix_schema_errors(
                refined, validation_result.schema_errors
            )
            changes.extend(schema_changes)

            # Structure fixes (safe to apply)
            refined, struct_changes = self._fix_structure_errors(
                refined, validation_result.structure_errors
            )
            changes.extend(struct_changes)

            # Semantic fixes (conservative)
            refined, sem_changes = self._fix_semantic_errors(
                refined, validation_result.semantic_errors
            )
            changes.extend(sem_changes)

        return refined, changes

    def _fix_schema_errors(
        self,
        workflow: WorkflowDefinitionSchema,
        errors: List[ValidationError]
    ) -> Tuple[WorkflowDefinitionSchema, List[str]]:
        """Fix schema-level errors."""
        changes = []

        for error in errors:
            if error.error_type == "missing_required_field":
                # Try to infer or add default
                if "goal" in error.location and error.severity != "critical":
                    # Add default goal for agent nodes
                    node = self._get_node_by_location(workflow, error.location)
                    if node:
                        node.goal = "Execute task"
                        changes.append(f"Added default goal to node '{node.id}'")

                elif "role" in error.location:
                    node = self._get_node_by_location(workflow, error.location)
                    if node and node.type == "agent":
                        node.role = "executor"
                        changes.append(f"Set default role 'executor' for node '{node.id}'")

            elif error.error_type == "invalid_type":
                # Try type conversion
                if "tool_budget" in error.location:
                    node = self._get_node_by_location(workflow, error.location)
                    if node and isinstance(node.tool_budget, str):
                        try:
                            node.tool_budget = int(node.tool_budget)
                            changes.append(f"Converted tool_budget to int for node '{node.id}'")
                        except ValueError:
                            node.tool_budget = 15
                            changes.append(f"Set default tool_budget for node '{node.id}'")

        return workflow, changes

    def _fix_structure_errors(
        self,
        workflow: WorkflowDefinitionSchema,
        errors: List[ValidationError]
    ) -> Tuple[WorkflowDefinitionSchema, List[str]]:
        """Fix structure-level errors."""
        changes = []

        for error in errors:
            if error.error_type == "orphan_node":
                # Remove orphan node
                node_id = self._extract_node_id(error.location)
                workflow.nodes = [n for n in workflow.nodes if n.id != node_id]
                changes.append(f"Removed orphan node '{node_id}'")

            elif error.error_type == "missing_entry_point":
                # Set first node as entry point
                if workflow.nodes:
                    workflow.entry_point = workflow.nodes[0].id
                    changes.append(f"Set '{workflow.nodes[0].id}' as entry_point")

        return workflow, changes

    def _fix_semantic_errors(
        self,
        workflow: WorkflowDefinitionSchema,
        errors: List[ValidationError]
    ) -> Tuple[WorkflowDefinitionSchema, List[str]]:
        """Fix semantic-level errors (conservative)."""
        changes = []

        for error in errors:
            if error.error_type == "tool_not_found" and error.severity != "critical":
                # Remove unknown tool
                node = self._get_node_by_location(workflow, error.location)
                if node and node.tools:
                    tool_name = error.value
                    if tool_name in node.tools:
                        node.tools.remove(tool_name)
                        changes.append(f"Removed unknown tool '{tool_name}' from node '{node.id}'")

            elif error.error_type == "invalid_role":
                # Replace with closest valid role
                node = self._get_node_by_location(workflow, error.location)
                if node and node.role:
                    # Simple similarity mapping
                    role_mapping = {
                        "developer": "executor",
                        "coder": "executor",
                        "planner": "planner",
                        "research": "researcher",
                    }
                    if node.role.lower() in role_mapping:
                        old_role = node.role
                        node.role = role_mapping[node.role.lower()]
                        changes.append(f"Changed role '{old_role}' to '{node.role}' in node '{node.id}'")

        return workflow, changes
```

---

## Iterative Refinement Loop

### Loop Design

**Purpose**: Coordinate iterative refinement with LLM until validation passes

**Flow**:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Generate Initial Workflow                                 │
│    LLM generates workflow from user request                  │
└────────────┬────────────────────────────────────────────────┘
             ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Validate                                                   │
│    Run all 4 validation layers                               │
└────────────┬────────────────────────────────────────────────┘
             ▼
       ┌─────┴─────┐
       │ Valid?    │
       └─────┬─────┘
         Yes │   No
             ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Check Refinement Options                                  │
│    - Can auto-fix? → Apply fixes, revalidate                 │
│    - Need LLM refinement? → Go to step 4                     │
│    - Max iterations reached? → Fail or request approval      │
└────────────┬────────────────────────────────────────────────┘
             ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. LLM Refinement                                            │
│    - Format errors for LLM                                   │
│    - Provide few-shot examples                               │
│    - Request revised workflow                                │
└────────────┬────────────────────────────────────────────────┘
             ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Check Convergence                                         │
│    - Error count decreased? → Continue                      │
│    - No improvement? → Try alternative strategy              │
│    - Diminishing returns? → Request human approval           │
└────────────┬────────────────────────────────────────────────┘
             ▼
         Return to step 2
```

**Implementation**:

```python
class IterativeRefinementLoop:
    """Manages iterative refinement of workflows with LLM."""

    def __init__(
        self,
        validator: WorkflowValidator,
        refiner: RefinementEngine,
        llm_client: Any,  # LLM provider interface
        max_iterations: int = 5,
        convergence_threshold: float = 0.3  # 30% improvement required
    ):
        self.validator = validator
        self.refiner = refiner
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    async def refine(
        self,
        initial_workflow: WorkflowDefinitionSchema,
        user_request: str
    ) -> Tuple[WorkflowDefinitionSchema, RefinementHistory]:
        """Run iterative refinement loop.

        Returns:
            Tuple of (final_workflow, refinement_history)
        """
        history = RefinementHistory()
        current_workflow = initial_workflow
        previous_error_count = float('inf')

        for iteration in range(self.max_iterations):
            # Validate current workflow
            result = self.validator.validate(current_workflow)
            history.add_iteration(iteration, current_workflow, result)

            # Check if valid
            if result.is_valid:
                logger.info(f"Workflow validated successfully after {iteration} iterations")
                return current_workflow, history

            # Check convergence
            error_count = len(result.all_errors)
            improvement = (previous_error_count - error_count) / previous_error_count

            if iteration > 0 and improvement < self.convergence_threshold:
                logger.warning(
                    f"Convergence stalled at iteration {iteration}: "
                    f"only {improvement*100:.1f}% improvement"
                )
                # Try alternative strategy or break
                if iteration < self.max_iterations - 1:
                    # Try more aggressive refinement
                    current_workflow = await self._aggressive_refinement(
                        current_workflow, result, user_request
                    )
                    continue
                else:
                    break

            previous_error_count = error_count

            # Try auto-fix first
            if not result.critical_errors:
                logger.info("Attempting automated refinement...")
                refined_workflow, changes = self.refiner.refine(current_workflow, result)
                if changes:
                    logger.info(f"Applied {len(changes)} automated fixes")
                    current_workflow = refined_workflow
                    continue

            # Request LLM refinement
            logger.info(f"Requesting LLM refinement (iteration {iteration + 1})...")
            refinement_prompt = self._create_refinement_prompt(
                current_workflow, result, user_request, iteration
            )
            current_workflow = await self._llm_refine(refinement_prompt)

        # Max iterations reached without validation
        logger.error(f"Failed to validate after {self.max_iterations} iterations")
        return current_workflow, history

    async def _llm_refine(
        self,
        prompt: str
    ) -> WorkflowDefinitionSchema:
        """Request LLM to refine workflow."""
        # Call LLM with refinement prompt
        response = await self.llm_client.complete(prompt)

        # Parse LLM response as workflow
        # (Assumes LLM returns YAML/JSON workflow)
        return self._parse_workflow(response)

    def _create_refinement_prompt(
        self,
        workflow: WorkflowDefinitionSchema,
        validation_result: WorkflowValidationResult,
        user_request: str,
        iteration: int
    ) -> str:
        """Create refinement prompt for LLM."""
        from victor.workflows.generation.prompts import RefinementPromptBuilder

        builder = RefinementPromptBuilder()
        return builder.build(
            user_request=user_request,
            current_workflow=workflow,
            validation_result=validation_result,
            iteration=iteration,
            examples=self._get_relevant_examples(validation_result)
        )

    def _get_relevant_examples(
        self,
        validation_result: WorkflowValidationResult
    ) -> List[Dict[str, Any]]:
        """Get few-shot examples relevant to current errors."""
        # Group errors by type
        error_types = {e.error_type for e in validation_result.all_errors}

        # Retrieve examples for these error types
        examples = []
        for error_type in error_types:
            examples.extend(self._load_examples_for_error(error_type))

        return examples[:5]  # Limit to 5 examples
```

### Convergence Detection

**Metrics**:
- Error count reduction (primary)
- Severity reduction (critical → error → warning)
- Structural similarity (avoid complete rewrites)

**Strategies**:
1. **Standard refinement**: Present all errors, request fixes
2. **Focused refinement**: Present only errors of one category
3. **Incremental refinement**: Fix errors in batches (schema → structure → semantic → security)
4. **Aggressive refinement**: Remove problematic nodes, simplify structure

---

## Human-in-the-Loop

### Approval Workflow

**Purpose**: Require human approval for complex or risky workflows

**Triggers for Approval**:
- Workflow has > 10 nodes
- Workflow uses privileged tools
- Validation failed after max iterations
- Workflow contains parallel execution
- Workflow has conditional cycles
- User requests manual review

**Implementation**:

```python
class WorkflowApprovalManager:
    """Manages human-in-the-loop approval for workflows."""

    def __init__(
        self,
        approval_threshold_nodes: int = 10,
        require_approval_for_privileged: bool = True,
        approval_backend: Optional[Any] = None  # HITL backend
    ):
        self.approval_threshold_nodes = approval_threshold_nodes
        self.require_approval_for_privileged = require_approval_for_privileged
        self.approval_backend = approval_backend

    async def request_approval(
        self,
        workflow: WorkflowDefinitionSchema,
        validation_result: WorkflowValidationResult,
        context: Dict[str, Any]
    ) -> ApprovalResult:
        """Request human approval for workflow.

        Returns:
            ApprovalResult with decision and optional feedback
        """
        # Check if approval is required
        if not self._requires_approval(workflow, validation_result):
            return ApprovalResult(approved=True, auto_approved=True)

        # Generate approval request
        request = self._create_approval_request(workflow, validation_result, context)

        # Send to approval backend
        if self.approval_backend:
            result = await self.approval_backend.request_approval(request)
        else:
            # Fallback to CLI approval
            result = await self._cli_approval(request)

        # Record approval
        self._record_approval(workflow, result)

        return result

    def _requires_approval(
        self,
        workflow: WorkflowDefinitionSchema,
        validation_result: WorkflowValidationResult
    ) -> bool:
        """Check if workflow requires human approval."""
        # Complex workflow
        if len(workflow.nodes) >= self.approval_threshold_nodes:
            return True

        # Uses privileged tools
        if self.require_approval_for_privileged:
            for node in workflow.nodes:
                if node.tools:
                    privileged_tools = {"bash_execute", "file_delete", "system_command"}
                    if any(t in privileged_tools for t in node.tools):
                        return True

        # Validation failed
        if not validation_result.is_valid:
            return True

        # Parallel execution
        has_parallel = any(n.type == "parallel" for n in workflow.nodes)
        if has_parallel:
            return True

        return False

    def _create_approval_request(
        self,
        workflow: WorkflowDefinitionSchema,
        validation_result: WorkflowValidationResult,
        context: Dict[str, Any]
    ) -> WorkflowApprovalRequest:
        """Create approval request with workflow details."""
        return WorkflowApprovalRequest(
            workflow_id=workflow.name,
            workflow_description=workflow.description,
            workflow_yaml=workflow.to_yaml(),
            validation_summary=validation_result.summary(),
            validation_errors=validation_result.to_dict(),
            node_count=len(workflow.nodes),
            has_parallel=any(n.type == "parallel" for n in workflow.nodes),
            has_privileged_tools=self._has_privileged_tools(workflow),
            user_request=context.get("user_request", ""),
            metadata=context
        )

    async def _cli_approval(
        self,
        request: WorkflowApprovalRequest
    ) -> ApprovalResult:
        """CLI-based approval (fallback)."""
        print("\n" + "=" * 80)
        print("WORKFLOW APPROVAL REQUIRED")
        print("=" * 80)
        print(f"Workflow: {request.workflow_id}")
        print(f"Description: {request.workflow_description}")
        print(f"Nodes: {request.node_count}")
        print("")
        print("Validation Summary:")
        print(request.validation_summary)
        print("")
        print("Workflow Structure:")
        print(self._format_workflow_summary(request))
        print("=" * 80)

        while True:
            response = input("Approve workflow? (y/n/v) ").strip().lower()
            if response == 'y':
                return ApprovalResult(
                    approved=True,
                    approved_by="user",
                    timestamp=time.time()
                )
            elif response == 'n':
                feedback = input("Reason for rejection: ").strip()
                return ApprovalResult(
                    approved=False,
                    feedback=feedback,
                    rejected_by="user",
                    timestamp=time.time()
                )
            elif response == 'v':
                # Show full workflow YAML
                print("\n--- Workflow YAML ---")
                print(request.workflow_yaml)
                print("--- End YAML ---\n")
            else:
                print("Please enter 'y' (yes), 'n' (no), or 'v' (view workflow)")
```

---

## LLM Integration

### Prompt Engineering

**Purpose**: Structure prompts for effective LLM refinement

**Components**:

1. **Context**: User request, workflow purpose
2. **Errors**: Formatted validation errors with suggestions
3. **Current Workflow**: Current YAML/JSON workflow
4. **Examples**: Few-shot examples of similar fixes
5. **Instructions**: Clear guidance on what to fix
6. **Output Format**: Expected output structure

**Prompt Template**:

```python
class RefinementPromptBuilder:
    """Builds prompts for LLM workflow refinement."""

    def build(
        self,
        user_request: str,
        current_workflow: WorkflowDefinitionSchema,
        validation_result: WorkflowValidationResult,
        iteration: int,
        examples: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Build refinement prompt."""
        sections = []

        # Section 1: Task Description
        sections.append(self._task_description(user_request, iteration))

        # Section 2: Validation Errors
        sections.append(self._validation_errors(validation_result))

        # Section 3: Current Workflow
        sections.append(self._current_workflow(current_workflow))

        # Section 4: Few-Shot Examples (if provided)
        if examples:
            sections.append(self._examples_section(examples))

        # Section 5: Instructions
        sections.append(self._instructions())

        # Section 6: Output Format
        sections.append(self._output_format())

        return "\n\n".join(sections)

    def _task_description(self, user_request: str, iteration: int) -> str:
        """Task description section."""
        return f"""# Task: Fix Workflow Validation Errors

You are helping to fix a workflow that failed validation. The workflow was generated to fulfill this user request:

**User Request:** {user_request}

This is refinement iteration #{iteration + 1}. Previous attempts to fix this workflow have failed. Please carefully address all validation errors."""

    def _validation_errors(self, result: WorkflowValidationResult) -> str:
        """Format validation errors for LLM."""
        error_reporter = ErrorReporter()
        return f"""# Validation Errors

The workflow has {len(result.all_errors)} validation errors:

{error_reporter.llm_report(result)}

**Critical:** You must fix all critical and error-level issues. Warnings should be addressed if possible."""

    def _current_workflow(self, workflow: WorkflowDefinitionSchema) -> str:
        """Show current workflow YAML."""
        import yaml
        return f"""# Current Workflow (Invalid)

```yaml
{yaml.dump(workflow.to_dict(), default_flow_style=False)}
```"""

    def _examples_section(self, examples: List[Dict[str, Any]]) -> str:
        """Few-shot examples section."""
        sections = []
        for i, example in enumerate(examples, 1):
            sections.append(f"""## Example {i}

**Error:** {example['error']}

**Original:**
```yaml
{example['original']}
```

**Fixed:**
```yaml
{example['fixed']}
```

**Explanation:** {example['explanation']}
""")

        return "\n".join(sections)

    def _instructions(self) -> str:
        """Clear instructions for LLM."""
        return """# Instructions

1. Review the validation errors carefully
2. Understand what each error means
3. Fix the workflow by addressing each error
4. Maintain the workflow's original purpose (fulfill user request)
5. Preserve correct parts of the workflow
6. Output ONLY the fixed workflow YAML (no explanations outside YAML)

**Key Guidelines:**
- All node IDs must be unique
- All edge sources/targets must reference valid nodes
- Entry point must be a valid node ID
- Agent nodes must have 'role' and 'goal'
- Compute nodes must have 'tools' or 'handler'
- Condition nodes must have 'branches' mapping
- Tool budgets must be integers between 1-500
- No unconditional cycles (add conditions if needed)
- All tools must exist in the tool registry

**Do NOT:**
- Change the workflow's purpose
- Remove nodes unless they cause errors
- Make assumptions about tools that don't exist
- Add unnecessary complexity"""

    def _output_format(self) -> str:
        """Specify expected output format."""
        return """# Output Format

Output ONLY the fixed workflow in YAML format. Start your response with the workflow 'name' field.

Example:
```yaml
name: my_workflow
description: "Fixed workflow"
entry_point: start
nodes:
  ...
edges:
  ...
```

Do NOT include any explanations, comments, or text outside the YAML."""
```

### Few-Shot Example Library

**Structure**:

```python
# Examples for common error patterns

REFINEMENT_EXAMPLES = {
    "missing_entry_point": {
        "error": "ValidationError: Entry point 'start' not found in nodes",
        "original": """
name: data_pipeline
nodes:
  - id: fetch
    type: agent
    role: researcher
    goal: "Fetch data"
edges:
  - source: fetch
    target: __end__
entry_point: start  # INVALID: 'start' node doesn't exist
""",
        "fixed": """
name: data_pipeline
nodes:
  - id: fetch
    type: agent
    role: researcher
    goal: "Fetch data"
edges:
  - source: fetch
    target: __end__
entry_point: fetch  # FIXED: Use existing node as entry point
""",
        "explanation": "The entry_point must reference an existing node ID. Changed 'start' to 'fetch'."
    },

    "unconditional_cycle": {
        "error": "GraphValidationError: Unconditional cycle detected: step1 -> step2 -> step1",
        "original": """
name: iterative_process
nodes:
  - id: step1
    type: compute
    tools: [process_a]
  - id: step2
    type: compute
    tools: [process_b]
edges:
  - source: step1
    target: step2
  - source: step2
    target: step1  # INVALID: Unconditional cycle
entry_point: step1
""",
        "fixed": """
name: iterative_process
nodes:
  - id: step1
    type: compute
    tools: [process_a]
  - id: step2
    type: compute
    tools: [process_b]
  - id: should_continue
    type: condition
    condition: check_continuation
    branches:
      continue: step1
      done: __end__
edges:
  - source: step1
    target: step2
  - source: step2
    target: should_continue
entry_point: step1
""",
        "explanation": "Added a condition node to break the cycle. The condition checks if processing should continue or finish."
    },

    "missing_agent_fields": {
        "error": "SemanticValidationError: Agent node 'worker' must specify 'role' and 'goal'",
        "original": """
name: task_runner
nodes:
  - id: worker
    type: agent
    # Missing 'role' and 'goal'
    tool_budget: 20
edges:
  - source: worker
    target: __end__
entry_point: worker
""",
        "fixed": """
name: task_runner
nodes:
  - id: worker
    type: agent
    role: executor  # FIXED: Added role
    goal: "Execute the assigned task"  # FIXED: Added goal
    tool_budget: 20
edges:
  - source: worker
    target: __end__
entry_point: worker
""",
        "explanation": "Agent nodes require 'role' (what type of agent) and 'goal' (what task to perform)."
    },

    "invalid_tool_reference": {
        "error": "SemanticValidationError: Tool 'magic_wand' not found in registry",
        "original": """
name: automation
nodes:
  - id: do_magic
    type: compute
    tools: [magic_wand]  # INVALID: Tool doesn't exist
edges:
  - source: do_magic
    target: __end__
entry_point: do_magic
""",
        "fixed": """
name: automation
nodes:
  - id: do_magic
    type: compute
    tools: [bash_execute]  # FIXED: Use existing tool
edges:
  - source: do_magic
    target: __end__
entry_point: do_magic
""",
        "explanation": "Replaced non-existent tool 'magic_wand' with 'bash_execute' which exists in the tool registry."
    }
}
```

---

## Implementation Plan

### Module Breakdown

#### 1. `victor/workflows/generation/types.py`

**Purpose**: Type definitions for validation system

**LOC Estimate**: 300-400 lines

**Key Types**:
```python
- ValidationError (base)
- SchemaValidationError
- StructureValidationError
- SemanticValidationError
- SecurityValidationError
- WorkflowValidationResult (aggregated)
- RefinementHistory
- ApprovalResult
```

**Dependencies**:
- `victor.core.validation` (reuse ValidationSeverity, ValidationIssue)

---

#### 2. `victor/workflows/generation/validator.py`

**Purpose**: Main validator orchestrating all 4 layers

**LOC Estimate**: 600-800 lines

**Key Classes**:
```python
- WorkflowValidator (main facade)
- SchemaValidator (Layer 1)
- GraphStructureValidator (Layer 2)
- SemanticValidator (Layer 3)
- SecurityValidator (Layer 4)
```

**Dependencies**:
- `victor.core.validation` (Pydantic schemas)
- `victor.workflows.definition` (WorkflowDefinition types)
- `victor.workflows.validation.tool_validator` (reuse tool validation)

---

#### 3. `victor/workflows/generation/error_reporter.py`

**Purpose**: Format and aggregate validation errors

**LOC Estimate**: 400-500 lines

**Key Classes**:
```python
- ErrorReporter (formatters for different consumers)
- ErrorAggregator (groups and prioritizes errors)
```

**Dependencies**:
- `victor.workflows.generation.types`

---

#### 4. `victor/workflows/generation/refiner.py`

**Purpose**: Automated refinement strategies

**LOC Estimate**: 500-600 lines

**Key Classes**:
```python
- RefinementEngine (main facade)
- SchemaRefiner (fixes schema errors)
- StructureRefiner (fixes structure errors)
- SemanticRefiner (fixes semantic errors)
```

**Dependencies**:
- `victor.workflows.generation.types`
- `victor.workflows.generation.validator`

---

#### 5. `victor/workflows/generation/approval.py`

**Purpose**: Human-in-the-loop approval workflow

**LOC Estimate**: 400-500 lines

**Key Classes**:
```python
- WorkflowApprovalManager
- ApprovalRequest
- ApprovalResult
- ApprovalHistory (tracks approvals)
```

**Dependencies**:
- `victor.framework.hitl` (existing HITL infrastructure)
- `victor.workflows.generation.types`

---

#### 6. `victor/workflows/generation/prompts.py`

**Purpose**: LLM prompt templates and few-shot examples

**LOC Estimate**: 600-800 lines (including examples library)

**Key Classes**:
```python
- RefinementPromptBuilder
- ExampleLibrary (few-shot examples)
- PromptTemplate (base template system)
```

**Dependencies**:
- `victor.workflows.generation.types`

---

#### 7. `victor/workflows/generation/__init__.py`

**Purpose**: Public API exports

**LOC Estimate**: 100-150 lines

**Exports**:
```python
# Validators
from .validator import WorkflowValidator

# Refinement
from .refiner import RefinementEngine, IterativeRefinementLoop

# Error reporting
from .error_reporter import ErrorReporter

# Approval
from .approval import WorkflowApprovalManager, ApprovalResult

# Types
from .types import (
    ValidationError,
    WorkflowValidationResult,
    RefinementHistory,
)
```

---

### Total Implementation Effort

| Module | LOC | Time (days) | Dependencies |
|--------|-----|-------------|--------------|
| types.py | 300-400 | 1-2 | victor.core.validation |
| validator.py | 600-800 | 3-4 | types, tool_validator, graph analysis |
| error_reporter.py | 400-500 | 2-3 | types |
| refiner.py | 500-600 | 3-4 | types, validator |
| approval.py | 400-500 | 2-3 | types, hitl |
| prompts.py | 600-800 | 2-3 | types |
| __init__.py | 100-150 | 0.5 | - |
| **Total** | **2900-3750** | **14-20** | |

**Risk Factors**:
- Graph structure validation complexity (medium)
- LLM prompt engineering iteration (medium)
- HITL integration complexity (low)

---

## MVP Feature List

### MVP Scope (Phase 3.4a)

**Must Have** (Core validation):
1. ✅ Schema validation (Pydantic-based)
2. ✅ Basic structure validation (reachability, cycles)
3. ✅ Semantic validation (node types, tools)
4. ✅ Error aggregation and reporting (human-readable)
5. ✅ Automated refinement (schema fixes only)
6. ✅ Iterative refinement loop (basic)
7. ✅ LLM integration (prompt templates)

**Nice to Have** (Enhanced features):
8. ⭐ Security validation (resource limits)
9. ⭐ Advanced structure validation (dead ends, parallel joins)
10. ⭐ Automated semantic refinement (tool removal, role mapping)
11. ⭐ HITL approval (complex workflows)
12. ⭐ Convergence detection and optimization

**Future Enhancements** (Phase 3.4b+):
13. 🚀 Workflow similarity detection (avoid redundant refinements)
14. 🚀 Learning from refinement history
15. 🚀 Automatic test generation for validated workflows
16. 🚀 Workflow diff visualization (before/after refinement)
17. 🚀 Multi-LLM consensus for validation

---

### MVP Success Criteria

**Functional**:
- ✅ Catch 95%+ of schema errors
- ✅ Catch 90%+ of structure errors
- ✅ Catch 85%+ of semantic errors
- ✅ Reduce validation failures by 80% through automated refinement
- ✅ Achieve validation convergence in < 3 iterations (average)

**Non-Functional**:
- ✅ Validation completes in < 1s for workflows with < 50 nodes
- ✅ Error reports clearly identify problem location
- ✅ LLM refinement prompts are < 2000 tokens (cost-effective)
- ✅ No regression in existing workflow validation

---

## Integration with Existing Systems

### UnifiedWorkflowCompiler Integration

```python
# In victor/workflows/unified_compiler.py

class UnifiedWorkflowCompiler:
    def __init__(self, ...):
        # Existing initialization
        ...
        # NEW: Add validation for dynamically generated workflows
        self._workflow_validator = None

    def enable_validation(self, strict_mode: bool = True) -> None:
        """Enable workflow validation for generated workflows."""
        from victor.workflows.generation import WorkflowValidator

        self._workflow_validator = WorkflowValidator(strict_mode=strict_mode)

    def compile_definition(
        self,
        definition: WorkflowDefinition,
        validate: bool = True,  # NEW parameter
        **kwargs
    ) -> CachedCompiledGraph:
        """Compile with optional validation."""
        # NEW: Validate if enabled
        if validate and self._workflow_validator:
            result = self._workflow_validator.validate(definition)
            if not result.is_valid:
                from victor.workflows.generation import ErrorReporter
                reporter = ErrorReporter()
                raise ValueError(
                    f"Workflow validation failed:\n"
                    f"{reporter.human_report(result)}"
                )

        # Existing compilation logic
        ...
```

### StateGraph Integration

```python
# In victor/framework/graph.py

class StateGraph:
    def compile(
        self,
        validate: bool = True,  # NEW parameter
        **kwargs
    ) -> CompiledGraph:
        """Compile with optional validation."""
        # NEW: Validate before compilation
        if validate:
            from victor.workflows.generation import WorkflowValidator

            validator = WorkflowValidator()
            # Convert StateGraph to schema for validation
            workflow_schema = self._to_workflow_schema()
            result = validator.validate(workflow_schema)

            if not result.is_valid:
                from victor.workflows.generation import ErrorReporter
                reporter = ErrorReporter()
                raise ValueError(
                    f"Graph validation failed:\n"
                    f"{reporter.human_report(result)}"
                )

        # Existing compilation logic
        ...
```

---

## Testing Strategy

### Unit Tests

**Coverage Target**: 90%+

**Test Modules**:
```
tests/unit/workflows/generation/
├── test_types.py               # Type serialization/deserialization
├── test_validator.py           # All 4 validation layers
├── test_error_reporter.py      # Error formatting and aggregation
├── test_refiner.py             # Refinement strategies
├── test_approval.py            # HITL approval workflow
├── test_prompts.py             # Prompt template generation
└── fixtures/
    ├── invalid_workflows/      # Test cases for each error type
    └── valid_workflows/        # Valid workflow examples
```

**Test Cases**:
- Schema validation: 50+ test cases (field types, ranges, constraints)
- Structure validation: 30+ test cases (cycles, reachability, dead ends)
- Semantic validation: 40+ test cases (node types, tools, handlers)
- Security validation: 20+ test cases (resource limits, dangerous combinations)
- Refinement: 30+ test cases (auto-fixes, LLM refinement)
- Error reporting: 15+ test cases (formatting, aggregation)

### Integration Tests

**Scenarios**:
1. End-to-end workflow generation and validation
2. Iterative refinement loop convergence
3. HITL approval workflow
4. Integration with UnifiedWorkflowCompiler
5. Integration with StateGraph

### Performance Tests

**Benchmarks**:
- Validation throughput (workflows/second)
- Error aggregation performance
- Large workflow validation (1000+ nodes)
- Memory usage during validation

---

## Open Questions

1. **LLM Provider Selection**: Which LLM for refinement? (Claude, GPT-4, local models?)
2. **Cost Limits**: Max tokens per refinement session?
3. **Timeouts**: How long to wait for LLM refinement response?
4. **Fallback**: What if LLM fails to refine? (Request human, fail gracefully, retry?)
5. **Validation Strictness**: Default strict vs lenient mode?
6. **Caching**: Cache validation results for identical workflows?
7. **Metrics**: Track validation success rates, common errors, refinement patterns?

---

## Future Enhancements

### Phase 3.4b: Advanced Validation

1. **Workflow Learning**: Learn from refinement patterns to pre-empt common errors
2. **Test Generation**: Automatically generate tests for validated workflows
3. **Diff Visualization**: Show before/after workflow changes
4. **Multi-LLM Consensus**: Use multiple LLMs for validation confidence
5. **Workflow Optimization**: Suggest structural improvements (parallelization, merging)

### Phase 3.5+: Dynamic Workflow Generation

Once validation is robust, enable:
1. Natural language → Workflow generation
2. Workflow composition from sub-workflows
3. Workflow template library
4. Workflow recommendation engine
5. A/B testing for workflow variants

---

## Conclusion

This design provides a **comprehensive, production-ready validation and refinement system** for dynamically generated workflows. The multi-layer approach ensures thorough validation while automated refinement and iterative loops minimize manual intervention.

**Key Strengths**:
- ✅ Comprehensive validation (4 layers covering all error types)
- ✅ Clear error reporting (actionable feedback for LLMs and humans)
- ✅ Automated refinement (reduces manual effort)
- ✅ Iterative improvement (converges on valid workflows)
- ✅ Human oversight (approval for complex/risky workflows)
- ✅ Integration-ready (works with existing StateGraph and UnifiedWorkflowCompiler)

**Implementation Priority**:
1. **Phase 1** (Week 1-2): Schema validation + error reporting
2. **Phase 2** (Week 3-4): Structure + semantic validation
3. **Phase 3** (Week 5-6): Automated refinement + iterative loop
4. **Phase 4** (Week 7-8): Security validation + HITL approval
5. **Phase 5** (Week 9-10): Testing, documentation, integration

**Estimated Timeline**: 10 weeks for full implementation (2.5 months)

This system will enable Victor to safely transition from **static workflows** (hand-written YAML) to **dynamic workflows** (LLM-generated) while maintaining reliability, security, and correctness.
