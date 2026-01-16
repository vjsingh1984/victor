# Advanced Team Formations

Beyond the 5 basic team formation patterns (Sequential, Parallel, Hierarchical, Pipeline, Consensus), Victor provides advanced formation strategies that enable dynamic coordination, adaptive selection, and hybrid multi-phase workflows.

## Overview

Advanced formations enable teams to:

- **Adapt mid-execution**: Switch coordination patterns based on progress, dependencies, and conflicts
- **Select intelligently**: Use AI-powered analysis to choose optimal formations for task characteristics
- **Combine patterns**: Execute complex multi-phase workflows that combine multiple formations

## Formation Types

### 1. Dynamic Formation

Automatically switches formation based on execution progress and detected patterns.

**Use Cases:**
- Tasks where coordination needs evolve during execution
- Unknown dependency structure upfront
- Situations requiring adaptability to conflicts or time pressure
- Complex problem-solving with changing requirements

**How It Works:**
1. Starts with an initial formation (default: parallel)
2. Monitors execution for triggers (dependencies, conflicts, consensus needs)
3. Switches to appropriate formation when triggers detected
4. Tracks phases: Exploration → Execution → Resolution

**Triggers:**
- `dependencies_emerge`: Switch to sequential
- `conflict_detected`: Switch to consensus
- `consensus_needed`: Switch to consensus
- `time_pressure`: Switch to parallel for speed
- `quality_concerns`: Switch to consensus for validation

**Example:**

```yaml
teams:
  - name: adaptive_review_team
    formation: dynamic
    dynamic_config:
      initial_formation: parallel
      max_switches: 5
      enable_auto_detection: true
      switching_rules:
        - trigger: dependencies_emerge
          target_formation: sequential
        - trigger: conflict_detected
          target_formation: consensus
        - trigger: time_pressure
          target_formation: parallel
    members:
      - id: reviewer_1
        role: reviewer
        goal: "Review code for security issues"
      - id: reviewer_2
        role: reviewer
        goal: "Review code for quality issues"
      - id: reviewer_3
        role: reviewer
        goal: "Review code for performance issues"
```

**Performance Characteristics:**
- **Latency**: Moderate (overhead from monitoring and switching)
- **Quality**: High (adapts to ensure best coordination)
- **Resource Usage**: Moderate to high (may execute multiple formations)
- **Best For**: Complex tasks with unknown structure

### 2. Adaptive Formation

AI-powered formation selection based on task analysis and characteristics.

**Use Cases:**
- Heterogeneous task mix (varying complexity, dependencies)
- Need for optimal formation selection without manual configuration
- Environments with diverse task types
- When you want data-driven formation selection

**How It Works:**
1. Analyzes task characteristics (complexity, urgency, uncertainty, dependencies)
2. Scores each formation based on characteristics
3. Selects highest-scoring formation
4. Executes with selected formation (with fallback option)

**Criteria Analyzed:**
- `complexity`: Task complexity based on length and structure
- `deadline`: Time urgency from context and keywords
- `resource_availability`: Number of available agents
- `dependency_level`: Presence of dependencies in task
- `collaboration_needed`: Need for team collaboration
- `uncertainty`: Ambiguity in task requirements

**Example:**

```yaml
teams:
  - name: smart_development_team
    formation: adaptive
    adaptive_config:
      criteria:
        - complexity
        - deadline
        - dependency_level
        - collaboration_needed
      default_formation: parallel
      fallback_formation: sequential
      use_ml: false  # Set to true for ML-based analysis
      # Optional: Custom scoring weights
      scoring_weights:
        parallel:
          complexity: 0.8
          deadline: 0.7
          dependency_level: -0.5
          collaboration_needed: 0.6
        sequential:
          complexity: 0.5
          deadline: 0.4
          dependency_level: 0.9
          collaboration_needed: 0.5
        consensus:
          complexity: 0.4
          deadline: 0.2
          dependency_level: 0.3
          collaboration_needed: 0.9
    members:
      - id: developer
        role: coder
        goal: "Implement the feature"
      - id: tester
        role: reviewer
        goal: "Test and validate"
```

**Performance Characteristics:**
- **Latency**: Low to moderate (analysis overhead)
- **Quality**: High (optimal formation selection)
- **Resource Usage**: Low (executes single formation)
- **Best For**: Mixed task types requiring optimization

### 3. Hybrid Formation

Multi-phase execution that combines multiple formations in sequence.

**Use Cases:**
- Complex workflows with distinct phases
- Need for different coordination patterns at different stages
- Structured processes (explore → analyze → validate)
- Research and development workflows

**How It Works:**
1. Executes phases in sequence
2. Each phase uses specified formation
3. Context and results flow between phases
4. Optional: Stop on phase failure or continue

**Phase Features:**
- `formation`: Formation to use in phase
- `goal`: Objective of this phase
- `duration_budget`: Time limit for phase (optional)
- `iteration_limit`: Max iterations for phase (optional)
- `completion_criteria`: When to advance to next phase

**Example:**

```yaml
teams:
  - name: research_team
    formation: hybrid
    hybrid_config:
      enable_phase_logging: true
      stop_on_first_failure: false
      phases:
        - formation: parallel
          goal: "Gather information from multiple sources rapidly"
          duration_budget: 60.0  # 1 minute exploration
        - formation: sequential
          goal: "Analyze findings in depth, building on each other's work"
          iteration_limit: 3
        - formation: consensus
          goal: "Validate final conclusions and reach agreement"
          completion_criteria: "All members satisfied with conclusions"
    members:
      - id: researcher_1
        role: researcher
        goal: "Gather and analyze data"
      - id: researcher_2
        role: researcher
        goal: "Cross-validate findings"
      - id: synthesizer
        role: planner
        goal: "Synthesize final report"
```

**Common Hybrid Patterns:**

**Research Pattern:**
```yaml
phases:
  - formation: parallel
    goal: "Broad exploration"
  - formation: sequential
    goal: "Deep analysis"
  - formation: consensus
    goal: "Validation"
```

**Development Pattern:**
```yaml
phases:
  - formation: parallel
    goal: "Parallel implementation of components"
  - formation: pipeline
    goal: "Integration and testing stages"
  - formation: hierarchical
    goal: "Manager coordination and deployment"
```

**Review Pattern:**
```yaml
phases:
  - formation: parallel
    goal: "Multiple perspectives on review"
  - formation: consensus
    goal: "Agreement on issues found"
  - formation: sequential
    goal: "Fix validation in sequence"
```

**Performance Characteristics:**
- **Latency**: High (executes multiple phases)
- **Quality**: Very high (structured multi-phase approach)
- **Resource Usage**: High (multiple formation executions)
- **Best For**: Complex workflows with clear phases

## Configuration Reference

### Dynamic Formation Configuration

```yaml
formation: dynamic
dynamic_config:
  # Required
  initial_formation: parallel  # sequential|parallel|hierarchical|pipeline|consensus

  # Optional
  max_switches: 5  # Maximum formation switches (0-20, default: 5)
  enable_auto_detection: true  # Auto-detect triggers (default: true)

  # Switching rules
  switching_rules:
    - trigger: dependencies_emerge
      target_formation: sequential
    - trigger: conflict_detected
      target_formation: consensus
    - trigger: consensus_needed
      target_formation: consensus
    - trigger: time_pressure
      target_formation: parallel
```

### Adaptive Formation Configuration

```yaml
formation: adaptive
adaptive_config:
  # Required
  criteria:
    - complexity
    - deadline
    - resource_availability
  default_formation: parallel
  fallback_formation: sequential

  # Optional
  use_ml: false  # Use ML model for analysis (default: false)

  # Optional: Custom scoring weights
  scoring_weights:
    parallel:
      complexity: 0.8  # -1.0 to 1.0
      deadline: 0.7
      dependency_level: -0.5
    sequential:
      complexity: 0.5
      deadline: 0.4
      dependency_level: 0.9
```

### Hybrid Formation Configuration

```yaml
formation: hybrid
hybrid_config:
  # Required
  phases:
    - formation: parallel
      goal: "Phase 1 objective"
      # Optional
      duration_budget: 30.0  # seconds
      iteration_limit: 5
      completion_criteria: "All members agree"

    - formation: sequential
      goal: "Phase 2 objective"

  # Optional
  enable_phase_logging: true  # Log phase transitions (default: true)
  stop_on_first_failure: false  # Stop on phase failure (default: false)
```

## Comparison Table

| Formation | Latency | Quality | Adaptability | Complexity | Best For |
|-----------|---------|---------|--------------|------------|----------|
| **Dynamic** | Moderate | High | Very High | Medium | Evolving tasks, unknown structure |
| **Adaptive** | Low-Moderate | High | High | Low | Mixed task types, optimization |
| **Hybrid** | High | Very High | Low | High | Structured multi-phase workflows |
| **Basic** | Low | Medium | None | Low | Simple, predictable tasks |

## Best Practices

### When to Use Dynamic Formation

✅ **Use when:**
- Task structure unknown upfront
- Dependencies may emerge during execution
- Team may encounter conflicts requiring consensus
- Have time for potential re-execution

❌ **Avoid when:**
- Task structure is well-known
- Execution time is critical
- Simple, straightforward tasks

### When to Use Adaptive Formation

✅ **Use when:**
- Handling diverse task types
- Want optimal formation selection
- Have training data for ML (optional)
- Need single-pass execution

❌ **Avoid when:**
- Task always uses same formation
- Want explicit control over formation
- Very simple tasks

### When to Use Hybrid Formation

✅ **Use when:**
- Workflow has natural phases
- Different stages need different coordination
- Quality is more important than speed
- Can structure task into clear phases

❌ **Avoid when:**
- Simple single-phase tasks
- Execution time critical
- Unclear phase boundaries

## Performance Tuning

### Dynamic Formation Tuning

```yaml
# For faster execution (less switching)
dynamic_config:
  max_switches: 2  # Reduce switches
  enable_auto_detection: false  # Disable auto-detection

# For better quality (more switching)
dynamic_config:
  max_switches: 10  # Allow more switches
  enable_auto_detection: true  # Enable detection
  switching_rules:  # Comprehensive rules
    - trigger: dependencies_emerge
      target_formation: sequential
    - trigger: conflict_detected
      target_formation: consensus
    - trigger: consensus_needed
      target_formation: consensus
    - trigger: time_pressure
      target_formation: parallel
    - trigger: quality_concerns
      target_formation: consensus
```

### Adaptive Formation Tuning

```yaml
# For speed (simple heuristic)
adaptive_config:
  use_ml: false  # Disable ML (faster)
  criteria:  # Fewer criteria
    - complexity
    - deadline

# For quality (comprehensive analysis)
adaptive_config:
  use_ml: true  # Enable ML (if available)
  criteria:  # More criteria
    - complexity
    - deadline
    - resource_availability
    - dependency_level
    - collaboration_needed
    - uncertainty
  scoring_weights:  # Custom weights
    parallel:
      complexity: 0.9
      deadline: 0.8
```

### Hybrid Formation Tuning

```yaml
# For faster execution
hybrid_config:
  stop_on_first_failure: true  # Stop early on failure
  phases:
    - formation: parallel
      goal: "Fast exploration"
      duration_budget: 10.0  # Short budget

# For better quality
hybrid_config:
  stop_on_first_failure: false  # Continue through all phases
  enable_phase_logging: true  # Better observability
  phases:
    - formation: parallel
      goal: "Thorough exploration"
      duration_budget: 60.0  # Longer budget
    - formation: consensus
      goal: "Validate with consensus"
```

## Monitoring and Observability

### Dynamic Formation Metadata

```python
results = await coordinator.execute_task(task, context)

metadata = results["member_results"]["adaptive_formation"].metadata

# Access formation history
formation_history = metadata["formation_history"]
for entry in formation_history:
    print(f"Switched from {entry['from_formation']} to {entry['to_formation']}")
    print(f"  Trigger: {entry['trigger']}")
    print(f"  Phase: {entry['phase']}")

# Access phase transitions
phase_transitions = metadata["phase_transitions"]
for entry in phase_transitions:
    print(f"Phase: {entry['phase']} at {entry['timestamp']}")
```

### Adaptive Formation Metadata

```python
results = await coordinator.execute_task(task, context)

metadata = results["member_results"]["adaptive_formation"].metadata

# Access selection details
selected_formation = metadata["selected_formation"]
task_characteristics = metadata["task_characteristics"]
formation_scores = metadata["formation_scores"]

print(f"Selected: {selected_formation}")
print(f"Characteristics: {task_characteristics}")
print(f"Scores: {formation_scores}")
```

### Hybrid Formation Metadata

```python
results = await coordinator.execute_task(task, context)

metadata = results["member_results"]["hybrid_formation"].metadata

# Access phase results
phase_results = metadata["phase_results"]
for phase_result in phase_results:
    print(f"Phase {phase_result['phase']}: {phase_result['goal']}")
    print(f"  Formation: {phase_result['formation']}")
    print(f"  Duration: {phase_result['duration_seconds']:.2f}s")
```

## Migration Guide

### From Basic to Dynamic Formation

**Before (basic parallel):**
```yaml
teams:
  - name: review_team
    formation: parallel
    members: [...]
```

**After (dynamic):**
```yaml
teams:
  - name: review_team
    formation: dynamic
    dynamic_config:
      initial_formation: parallel
      max_switches: 3
      switching_rules:
        - trigger: conflict_detected
          target_formation: consensus
    members: [...]
```

### From Basic to Adaptive Formation

**Before (manual formation selection):**
```yaml
# Different teams for different task types
teams:
  - name: simple_team
    formation: parallel
    members: [...]

  - name: complex_team
    formation: hierarchical
    members: [...]
```

**After (adaptive):**
```yaml
# Single adaptive team
teams:
  - name: smart_team
    formation: adaptive
    adaptive_config:
      criteria: [complexity, deadline]
      default_formation: parallel
      fallback_formation: hierarchical
    members: [...]
```

### From Manual to Hybrid Formation

**Before (manual workflow):**
```yaml
# Execute separate workflows manually
workflow1:
  formation: parallel
  ...

workflow2:
  formation: sequential
  ...

workflow3:
  formation: consensus
  ...
```

**After (hybrid):**
```yaml
teams:
  - name: integrated_workflow
    formation: hybrid
    hybrid_config:
      phases:
        - formation: parallel
          goal: "Phase 1"
        - formation: sequential
          goal: "Phase 2"
        - formation: consensus
          goal: "Phase 3"
    members: [...]
```

## Troubleshooting

### Dynamic Formation Issues

**Problem: Too many switches**
- Solution: Reduce `max_switches` or disable `enable_auto_detection`
- Solution: Review `switching_rules` to reduce triggers

**Problem: Not switching when expected**
- Solution: Verify `enable_auto_detection: true`
- Solution: Check switching rules match expected triggers
- Solution: Review logs for trigger detection

### Adaptive Formation Issues

**Problem: Always selecting same formation**
- Solution: Adjust `scoring_weights` to differentiate formations
- Solution: Verify criteria match task characteristics
- Solution: Enable `use_ml` for more sophisticated analysis

**Problem: Selecting inappropriate formation**
- Solution: Review `default_formation` and `fallback_formation`
- Solution: Customize `scoring_weights` for your use case
- Solution: Add more specific criteria

### Hybrid Formation Issues

**Problem: Phases not progressing**
- Solution: Check phase `duration_budget` and `iteration_limit`
- Solution: Verify `completion_criteria` is achievable
- Solution: Review previous phase results in context

**Problem: Poor performance**
- Solution: Enable `stop_on_first_failure: true`
- Solution: Reduce phase budgets
- Solution: Simplify phase structure

## Examples

See example workflows in:
- `victor/coding/workflows/examples/adaptive_team.yaml`
- `victor/research/workflows/examples/dynamic_research.yaml`
- `victor/devops/workflows/examples/hybrid_deployment.yaml`

## API Reference

### Python API

```python
from victor.workflows.advanced_formations import (
    DynamicFormation,
    AdaptiveFormation,
    HybridFormation,
    HybridPhase,
)

# Dynamic Formation
formation = DynamicFormation(
    initial_formation="parallel",
    max_switches=5,
    enable_auto_detection=True,
)

# Adaptive Formation
formation = AdaptiveFormation(
    criteria=["complexity", "deadline", "dependency_level"],
    default_formation="parallel",
    fallback_formation="sequential",
)

# Hybrid Formation
formation = HybridFormation(
    phases=[
        HybridPhase(formation="parallel", goal="Explore"),
        HybridPhase(formation="sequential", goal="Analyze"),
        HybridPhase(formation="consensus", goal="Validate"),
    ],
)

# Use with coordinator
from victor.teams import create_coordinator

coordinator = create_coordinator()
coordinator.set_formation("dynamic")  # or "adaptive", "hybrid"
result = await coordinator.execute_task(task, context)
```

## References

- [Team Formations Overview](../teams/README.md)
- [Workflow Configuration](./workflow_configuration.md)
- [Team YAML Schema](../config/teams/README.md)
- [Formation Strategies API](../api/formations.md)
