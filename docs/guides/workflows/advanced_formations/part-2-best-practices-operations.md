# Advanced Team Formations Guide - Part 2

**Part 2 of 2:** Best Practices, Performance Tuning, Monitoring, Migration, Troubleshooting, Examples, and API Reference

---

## Navigation

- [Part 1: Formation Types & Config](part-1-formation-types-config.md)
- **[Part 2: Best Practices & Operations](#)** (Current)
- [**Complete Guide](../advanced_formations.md)**

---
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

- [Team Formations Overview](../teams/collaboration.md)
- [Workflow Configuration](../guides/workflow-development/dsl.md)
- [Team YAML Schema](../teams/team_templates.md)
- [Formation Strategies API](./ml_formation_selection.md)

---

## See Also

- [Documentation Home](../../README.md)


**Last Updated:** February 01, 2026
**Reading Time:** 5 minutes
