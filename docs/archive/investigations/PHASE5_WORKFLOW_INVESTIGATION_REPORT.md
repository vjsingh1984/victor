# Phase 5 Workflow Investigation Report

**Date**: 2025-01-24
**Phase**: Workflow Consolidation
**Status**: Investigation Complete

---

## Executive Summary

Investigated workflow duplication across verticals to identify consolidation opportunities. Found **two distinct workflow systems** with overlapping functionality:

1. **Team-based workflows** (`victor/workflows/templates/`) - Multi-agent formations
2. **Node-based workflows** (vertical directories) - Traditional DAG workflows

**Key Finding**: The framework templates directory already exists with organized templates. The consolidation opportunity is in creating **shared stage templates** and **configuration overlays** rather than complete workflow duplication.

---

## Workflow Systems Comparison

### System 1: Team-Based Workflows (Formation-Based)

**Location**: `victor/workflows/templates/`

**Format**: Team formation definitions with multi-agent collaboration

**Structure**:
```yaml
name: code_review_parallel
display_name: "Code Review Team (Parallel)"
description: "Parallel multi-agent code review with specialized reviewers"
formation: "parallel"  # or "pipeline", "consensus", "hierarchical"

members:
  - id: "security_reviewer"
    role: "researcher"
    name: "Security Reviewer"
    goal: "Review code for security vulnerabilities..."
    expertise: ["security", "vulnerabilities", "owasp"]
    allowed_tools: ["read", "grep", "code_search"]
    tool_budget: 30

config:
  merge_strategy: "dict"
  merge_mode: "team_wins"
```

**Characteristics**:
- Multi-agent teams with specialized roles
- Formations: parallel, pipeline, consensus, hierarchical
- Tool budgets per agent
- Personality and backstory for each agent
- Built for LLM-powered collaboration

**Examples**:
- `coding/code_review_parallel.yaml` - 4 specialist reviewers (222 lines)
- `research/literature_review_pipeline.yaml` - 3-stage research (172 lines)
- `devops/deployment_pipeline.yaml` - 4-stage deployment (200+ lines)

### System 2: Node-Based Workflows (DAG-Based)

**Location**: `victor/{vertical}/workflows/`

**Format**: StateGraph-style DAG with typed nodes

**Structure**:
```yaml
workflows:
  code_review:
    description: "Comprehensive automated code review with feedback"

    nodes:
      - id: gather_changes
        type: compute  # or agent, transform, condition, hitl
        name: "Gather Code Changes"
        tools: [shell]
        inputs:
          command: $ctx.diff_command
        output: changes
        constraints: [llm, write]
        timeout: 60
        next: [check_changes]

      - id: ai_review
        type: agent
        name: "AI Code Review"
        role: reviewer
        goal: "Perform detailed code review..."
        tool_budget: 30
        tools: [read, grep, code_search]
        llm_config:
          temperature: 0.4
        output: review_findings
```

**Characteristics**:
- Typed nodes: agent, compute, transform, condition, hitl, parallel
- StateGraph execution engine
- Fine-grained tool and LLM constraints
- Compute/agent node rationale clearly documented
- Built for deterministic execution with selective LLM use

**Examples**:
- `victor/coding/workflows/code_review.yaml` - Full DAG workflow (809 lines)
- `victor/research/workflows/literature_review.yaml` - Research pipeline (489 lines)
- `victor/devops/workflows/deploy.yaml` - Deployment workflow (400+ lines)

---

## Duplication Analysis

### 1. Code Review Workflows

| Aspect | Node-Based | Team-Based |
|--------|-----------|------------|
| **File** | `victor/coding/workflows/code_review.yaml` | `victor/workflows/templates/coding/code_review_parallel.yaml` |
| **Lines** | 809 | 222 |
| **Approach** | DAG with compute/agent nodes | Multi-agent parallel team |
| **Stages** | 5+ stages with detailed nodes | 4 parallel reviewers |
| **Flexibility** | High (custom DAG structure) | Medium (formation-based) |

**Common Elements**:
- Security review focus
- Performance analysis
- Code quality assessment
- Documentation review
- Parallel execution of reviews

### 2. Literature Review Workflows

| Aspect | Node-Based | Team-Based |
|--------|-----------|------------|
| **File** | `victor/research/workflows/literature_review.yaml` | `victor/workflows/templates/research/literature_review_pipeline.yaml` |
| **Lines** | 489 | 172 |
| **Approach** | DAG with 6 stages | 3-member pipeline team |
| **Stages** | Scope → Search → Screen → Review → Extract → Synthesize | Search → Analyze → Synthesize |
| **Flexibility** | High (detailed screening, HITL) | Medium (formation-based) |

**Common Elements**:
- Literature search phase
- Paper analysis
- Synthesis of findings
- Quality assessment
- Citation formatting

### 3. Deployment Workflows

| Aspect | Node-Based | Team-Based |
|--------|-----------|------------|
| **File** | `victor/devops/workflows/deploy.yaml` | `victor/workflows/templates/devops/deployment_pipeline.yaml` |
| **Lines** | 400+ | 200+ |
| **Approach** | DAG with validation and rollback | 4-member pipeline team |
| **Stages** | Validate → Backup → Deploy → Verify → Rollback | Risk → Plan → Deploy → Verify |
| **Flexibility** | High (detailed validation loops) | Medium (formation-based) |

**Common Elements**:
- Pre-deployment validation
- Deployment execution
- Post-deployment verification
- Rollback capability
- Health checks

---

## Common Stage Patterns Identified

### Pattern 1: Read-Modify-Write (RMW)

**Used In**: Coding, Refactoring, Bugfix workflows

**Stages**:
1. **Read**: Gather context (read files, search codebase)
2. **Modify**: Make changes (edit, write, refactor)
3. **Verify**: Test changes (lint, type_check, test)

**Node-Based Example**:
```yaml
- id: gather_context  # Read
  type: compute
  tools: [read, search, grep]

- id: apply_changes  # Modify
  type: agent
  tools: [edit, write]

- id: verify_changes  # Verify
  type: compute
  tools: [lint, test]
```

**Team-Based Equivalent**:
```yaml
formation: "sequential"
members:
  - role: "researcher"  # Read
  - role: "executor"    # Modify
  - role: "tester"      # Verify
```

### Pattern 2: Pipeline (Sequential Stages)

**Used In**: Deployment, Literature Review, Data Processing

**Stages**:
1. Stage 1 → Stage 2 → Stage 3 → Stage 4

**Characteristics**:
- Each stage outputs to next stage
- Clear handoff between stages
- Can rollback to any stage

### Pattern 3: Parallel Analysis

**Used In**: Code Review, Testing, Data Analysis

**Stages**:
- Parallel execution of independent analyses
- Join/aggregation of results

**Node-Based Example**:
```yaml
- id: parallel_analysis
  type: parallel
  parallel_nodes: [lint_check, type_check, security_scan]
  join_strategy: all
```

**Team-Based Example**:
```yaml
formation: "parallel"
members: [security_reviewer, performance_reviewer, quality_reviewer]
```

### Pattern 4: Analysis Workflow

**Used In**: Research, Data Analysis, Competitive Analysis

**Stages**:
1. **Discover**: Gather data/sources
2. **Analyze**: Process and extract insights
3. **Report**: Generate summary/recommendations

---

## Consolidation Opportunities

### Opportunity 1: Shared Stage Templates ✅ HIGH PRIORITY

**Approach**: Define reusable stage templates that both systems can reference

**Implementation**:
```yaml
# victor/workflows/templates/common_stages.yaml
stage_templates:
  read_stage:
    type: agent
    role: reader
    tools: [read, search, grep]
    llm_config:
      temperature: 0.2

  modify_stage:
    type: agent
    role: executor
    tools: [edit, write]
    llm_config:
      temperature: 0.1

  verify_stage:
    type: compute
    tools: [lint, test, type_check]
    constraints: [llm]
```

**Benefits**:
- Reduce stage definition duplication
- Consistent behavior across workflows
- Easier to update common patterns

**Effort**: 2-3 days

### Opportunity 2: Configuration Overlays ✅ HIGH PRIORITY

**Approach**: Allow workflows to extend base templates with overrides

**Implementation**:
```yaml
# victor/coding/workflows/security_review.yaml
extends: "code_review_parallel"

overrides:
  name: "Security-Focused Code Review"
  members:
    # Extend security reviewer
    - id: "security_reviewer"
      tool_budget: 50  # Increased from 30

    # Remove other reviewers
    - remove: ["performance_reviewer", "quality_reviewer"]
```

**Benefits**:
- Eliminate full workflow duplication
- Easy to create specialized variants
- Clear inheritance structure

**Effort**: 3-4 days (requires template registry)

### Opportunity 3: Common Node Library ✅ MEDIUM PRIORITY

**Approach**: Create reusable node definitions library

**Implementation**:
```python
# victor/workflows/nodes/common_nodes.py
COMMON_NODES = {
    "git_diff": {
        "type": "compute",
        "tools": ["shell"],
        "constraints": ["llm", "write"],
        "handler": "git_diff_handler"
    },
    "lint_check": {
        "type": "compute",
        "tools": ["shell"],
        "constraints": ["llm", "write"],
        "handler": "lint_handler"
    },
    "security_scan": {
        "type": "compute",
        "tools": ["shell"],
        "constraints": ["llm", "write"],
        "handler": "security_scan_handler"
    }
}
```

**Benefits**:
- Consistent node behavior
- Easier workflow construction
- Centralized node testing

**Effort**: 2-3 days

### Opportunity 4: Workflow Template Registry ✅ HIGH PRIORITY

**Approach**: Create registry for loading and extending templates

**Implementation**:
```python
# victor/workflows/template_registry.py
class WorkflowTemplateRegistry:
    def load_templates_from_yaml(self, path: Path):
        """Load workflow templates from YAML."""

    def get_template(self, name: str) -> WorkflowDefinition:
        """Get workflow template by name."""

    def extend_template(self, base: str, overrides: Dict) -> Workflow:
        """Extend base template with overrides."""
```

**Benefits**:
- Centralized template management
- Dynamic template loading
- Template extension and composition

**Effort**: 2-3 days

---

## Recommended Implementation Strategy

### Phase 5A: Stage Template Library (1-2 days)

1. Create `victor/workflows/templates/common_stages.yaml`
2. Define reusable stage templates:
   - `read_stage`: Read/gather context
   - `modify_stage`: Make changes
   - `verify_stage`: Test/validate changes
   - `deploy_stage`: Deployment operations
   - `analyze_stage`: Data/code analysis
   - `report_stage`: Generate reports

3. Update `victor/workflows/registry.py` to load stage templates
4. Document stage template usage

### Phase 5B: Template Registry System (2-3 days)

1. Create `victor/workflows/template_registry.py`
2. Implement `WorkflowTemplateRegistry` class
3. Add YAML loading and validation
4. Implement template extension with deep merge
5. Add error handling for missing templates
6. Write comprehensive tests

### Phase 5C: Migrate Representative Workflows (2-3 days)

1. Select 2-3 workflows per vertical for migration:
   - Coding: `code_review.yaml`, `bugfix.yaml`
   - DevOps: `deploy.yaml`
   - Research: `literature_review.yaml`
   - DataAnalysis: `data_cleaning.yaml`

2. For each workflow:
   - Identify base template or common stages
   - Create overlay YAML with `extends:` key
   - Add `overrides:` section for vertical-specific changes
   - Test that migrated workflow works correctly

3. Document migration pattern for remaining workflows

### Phase 5D: Testing and Validation (1-2 days)

1. Create test suite for template system:
   - `tests/unit/workflows/test_template_registry.py`
   - `tests/unit/workflows/test_stage_templates.py`
   - `tests/integration/workflows/test_workflow_templates_e2e.py`

2. Test scenarios:
   - Template loading from YAML
   - Template extension with overrides
   - Deep merge of stages and tools
   - Error handling for missing templates
   - Backward compatibility with existing workflows

3. Performance testing:
   - Template load time
   - Extended workflow instantiation
   - Memory usage with many templates

---

## Risk Assessment

### Risk 1: Breaking Existing Workflows

**Impact**: High
**Probability**: Medium
**Mitigation**:
- Keep existing workflows as-is initially
- Add new template system alongside current system
- Gradual migration with backward compatibility
- Feature flag for template system

### Risk 2: Template Complexity

**Impact**: Medium
**Probability**: High
**Mitigation**:
- Clear documentation with examples
- Simple overlay syntax
- Validation during template loading
- Error messages for common mistakes

### Risk 3: Performance Overhead

**Impact**: Medium
**Probability**: Low
**Mitigation**:
- Lazy loading of templates
- Cache loaded templates
- Benchmark template extension
- Optimize deep merge algorithm

---

## Success Criteria

Phase 5 will be considered successful when:

1. ✅ Stage template library created with 6+ common stages
2. ✅ Template registry system implemented and tested
3. ✅ At least 8 workflows migrated to use overlays (2 per vertical)
4. ✅ Test coverage > 80% for template system
5. ✅ Documentation complete with examples
6. ✅ No breaking changes to existing workflows
7. ✅ Performance impact < 10% for template loading

---

## Next Steps

1. **Create Stage Template Library** (Task 2)
   - Define common stage templates
   - Add to framework templates directory

2. **Implement Template Registry** (Task 2)
   - Create `WorkflowTemplateRegistry` class
   - Add YAML loading and extension

3. **Migrate Workflows** (Task 3)
   - Select representative workflows
   - Create overlay definitions
   - Test and validate

4. **Add Tests** (Task 4)
   - Unit tests for registry
   - Integration tests for templates
   - End-to-end tests for workflows

---

**Document Version**: 1.0
**Last Updated**: 2025-01-24
**Status**: Investigation Complete - Ready for Implementation
