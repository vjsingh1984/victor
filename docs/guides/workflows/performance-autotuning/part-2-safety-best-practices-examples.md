# Team Workflow Performance Auto-Tuning - Part 2

**Part 2 of 2:** Safety and Rollback, Best Practices, Examples, Troubleshooting

---

## Navigation

- [Part 1: Overview, Strategies, API](part-1-overview-cli-api-ab-testing.md)
- **[Part 2: Safety, Best Practices, Examples](#)** (Current)
- [**Complete Guide**](../performance_autotuning.md)

---

## Safety and Rollback

### Automatic Rollback

The PerformanceAutotuner includes automatic rollback on regression detection:

```python
# Auto-rollback on regression
tuner = PerformanceAutotuner(
    auto_rollback=True,
    rollback_threshold=-0.05  # 5% regression threshold
)

# If performance drops by 5% after optimization, auto-rollback
result = tuner.apply_optimizations(team_id)
```

### Manual Rollback

Manually rollback to previous configuration:

```python
# Rollback specific optimization
tuner.rollback_optimization(
    team_id="team-1",
    optimization_id="opt-123"
)

# Rollback all optimizations
tuner.rollback_all(team_id="team-1")
```

### Safety Checks

Before applying optimizations, the tuner runs safety checks:

```python
# Configure safety checks
tuner = PerformanceAutotuner(
    safety_checks=[
        "validate_syntax",
        "check_dependencies",
        "verify_permissions",
        "dry_run"
    ]
)
```

---

## Best Practices

### 1. Start with Low-Risk Optimizations

Begin with low-risk optimizations like timeout tuning and tool budget adjustment:

```python
# Low-risk first
tuner.suggest_optimizations(
    team_id="team-1",
    max_risk="low"
)
```

### 2. Use A/B Testing

Always validate optimizations with A/B testing before permanent application:

```python
# Run A/B test
ab_result = tuner.run_ab_test(
    team_id="team-1",
    optimization=optimization,
    duration_hours=24,
    sample_size=100
)

# Check results before applying
if ab_result.improvement > 0.05:
    tuner.apply_optimization(team_id, optimization)
```

### 3. Monitor After Applying

Continuously monitor performance after applying optimizations:

```python
# Set up monitoring
tuner.monitor(
    team_id="team-1",
    check_interval=3600,  # Every hour
    alert_threshold=-0.05  # Alert on 5% regression
)
```

### 4. Keep History

Maintain optimization history for rollback and analysis:

```python
# View history
history = tuner.get_history(team_id="team-1")

# Export for analysis
tuner.export_history(
    team_id="team-1",
    output_file="optimization_history.json"
)
```

---

## Examples

### Example 1: Basic Auto-Tuning

```python
from victor.teams import PerformanceAutotuner

# Create tuner
tuner = PerformanceAutotuner()

# Get suggestions for team
suggestions = tuner.suggest_optimizations(team_id="team-1")

# Review and apply
for suggestion in suggestions:
    print(f"Optimization: {suggestion.type}")
    print(f"Expected improvement: {suggestion.expected_improvement}%")

    # Apply if acceptable
    if suggestion.expected_improvement > 10:
        tuner.apply_optimization(team_id, suggestion)
```

### Example 2: A/B Testing

```python
# Run A/B test
ab_result = tuner.run_ab_test(
    team_id="team-1",
    optimization={"type": "team_size", "value": 3},
    control_config={"team_size": 2},
    test_config={"team_size": 3},
    duration_hours=24,
    metrics=["latency", "cost", "success_rate"]
)

# Analyze results
if ab_result.significant:
    print(f"Improvement: {ab_result.improvement}%")
    print(f"Confidence: {ab_result.confidence}")

    if ab_result.improvement > 0:
        tuner.apply_optimization("team-1", ab_result.optimization)
```

### Example 3: Custom Optimization Strategy

```python
# Define custom strategy
class CustomStrategy(OptimizationStrategy):
    def analyze(self, metrics):
        # Custom analysis logic
        return suggestions

    def apply(self, optimization):
        # Custom application logic
        pass

# Use custom strategy
tuner = PerformanceAutotuner(
    strategies=[CustomStrategy()]
)
```

---

## Troubleshooting

### No Optimizations Suggested

**Problem**: Tuner returns no suggestions.

**Solutions**:
1. **Check data availability**: Ensure sufficient metrics data
2. **Lower risk tolerance**: Allow medium/high risk optimizations
3. **Check constraints**: Review optimization constraints

```python
# Allow higher risk
suggestions = tuner.suggest_optimizations(
    team_id="team-1",
    max_risk="high"
)
```

### Regression After Optimization

**Problem**: Performance decreased after optimization.

**Solutions**:
1. **Rollback**: Use automatic or manual rollback
2. **Review metrics**: Check which metrics regressed
3. **Adjust strategy**: Try different optimization strategy

```python
# Rollback
tuner.rollback_optimization(team_id, optimization_id)

# Try different strategy
suggestions = tuner.suggest_optimizations(
    team_id="team-1",
    strategies=["tool_budget", "timeout"]
)
```

### A/B Test Not Conclusive

**Problem**: A/B test results not significant.

**Solutions**:
1. **Extend duration**: Run test longer
2. **Increase sample**: Larger sample size
3. **Check variance**: High variance may require more data

```python
# Extend test
ab_result = tuner.run_ab_test(
    team_id="team-1",
    optimization=optimization,
    duration_hours=48,  # Longer duration
    sample_size=200  # Larger sample
)
```

---

## References

- [Performance Monitoring](../../operations/observability/performance_monitoring.md)
- [Team Formation](../team_nodes.md)
- [Configuration Reference](../../reference/configuration/README.md)

---

## See Also

- [Documentation Home](../../README.md)


**Last Updated:** 2025-01-15
**Reading Time:** 15 min (all parts)
