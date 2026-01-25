# Team Workflow Performance Auto-Tuning

**Last Updated:** 2025-01-15

This guide covers automatic performance optimization for team workflows in Victor, including the PerformanceAutotuner system, optimization strategies, A/B testing, and best practices.

## Table of Contents

- [Overview](#overview)
- [How Auto-Tuning Works](#how-auto-tuning-works)
- [Optimization Strategies](#optimization-strategies)
- [Using the CLI](#using-the-cli)
- [Python API](#python-api)
- [A/B Testing](#ab-testing)
- [Safety and Rollback](#safety-and-rollback)
- [Best Practices](#best-practices)
- [Examples](#examples)

---

## Overview

The PerformanceAutotuner system automatically analyzes team execution metrics and suggests optimizations to improve performance, reduce costs, and increase reliability.

### Key Features

- **Automatic Analysis**: Analyzes historical metrics to identify bottlenecks
- **Intelligent Suggestions**: Recommends optimizations based on patterns and benchmarks
- **A/B Testing**: Validates optimizations before permanent application
- **Rollback Safety**: Automatically reverts optimizations on regression detection
- **Multi-Strategy**: Supports 6 optimization strategies (team sizing, formation, budget, etc.)

### Expected Improvements

Based on benchmark data, auto-tuning typically provides:

| Optimization Type | Avg Improvement | Risk Level |
|-------------------|-----------------|------------|
| Formation Selection | 20-50% | Medium |
| Team Sizing | 10-20% | Low |
| Tool Budget | 5-15% | Low |
| Timeout Tuning | 5-10% | Very Low |

**Overall Target**: 10-20% performance improvement through automated optimization.

---

## How Auto-Tuning Works

### 1. Metrics Collection

Team execution metrics are automatically collected by `TeamMetricsCollector`:

```python
from victor.workflows.team_metrics import get_team_metrics_collector

collector = get_team_metrics_collector()

# Metrics are collected automatically during team execution
# No manual intervention required
```

**Collected Metrics**:
- Execution duration (avg, P95)
- Success rate
- Tool usage per member
- Formation performance
- Team size efficiency

### 2. Performance Analysis

The `PerformanceAnalyzer` processes metrics to identify issues:

```python
from victor.workflows.performance_autotuner import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()
analyzer.load_metrics_from_file("metrics.json")

# Analyze specific team
insights = analyzer.analyze_team_workflow(team_id="my_team")

# Each insight includes:
# - Bottleneck identification
# - Severity score (0-1)
# - Current vs baseline values
# - Actionable recommendations
```

**Identified Bottlenecks**:
- `slow_execution`: Average duration above baseline
- `high_latency_outliers`: P95 latency exceeds threshold
- `low_success_rate`: Success rate below threshold
- `excessive_tool_usage`: Tool calls above baseline
- `suboptimal_formation_<type>`: Formation performing poorly vs benchmark
- `oversized_team`: Too many members showing diminishing returns

### 3. Optimization Generation

Based on insights, optimization strategies generate suggestions:

```python
from victor.workflows.performance_autotuner import PerformanceAutotuner

autotuner = PerformanceAutotuner()

# Get prioritized suggestions
suggestions = autotuner.suggest_optimizations(
    team_id="my_team",
    current_config={"formation": "sequential", "member_count": 5}
)

# Suggestions sorted by priority (CRITICAL > HIGH > MEDIUM > LOW)
# and expected improvement × confidence
```

### 4. Application with Validation

Optimizations are applied with optional A/B testing:

```python
result = await autotuner.apply_optimizations(
    team_id="my_team",
    optimizations=suggestions,
    workflow_config=current_config,
    enable_ab_testing=True  # Validate before permanent application
)

# Result includes:
# - Success status
# - Before/after metrics
# - Actual improvement percentage
# - Validation status (passed/failed/inconclusive)
# - Rollback configuration
```

---

## Optimization Strategies

### 1. Team Sizing

**Detects**: Oversized teams showing diminishing returns

**Optimization**: Reduce member count to optimal range (3-5 members)

**Impact**: 10-20% improvement, Low risk

```yaml
# Before
member_count: 8

# After
member_count: 4
```

**When to apply**:
- Per-member execution time > 30s
- Team size > 7 members
- High coordination overhead

---

### 2. Formation Selection

**Detects**: Formation performing poorly vs benchmark

**Optimization**: Switch to more suitable formation

**Impact**: 20-50% improvement, Medium risk

```yaml
# Before: Consensus (slow for task type)
formation: consensus
members: 5

# After: Parallel (faster for independent tasks)
formation: parallel
members: 3
```

**Formation Selection Guide**:

| Current | Switch To | When |
|---------|-----------|------|
| Consensus | Parallel | Speed needed, low quality risk |
| Sequential | Pipeline | Staged refinement needed |
| Sequential | Parallel | Tasks are independent |
| Any | Sequential | Strong dependencies emerge |

---

### 3. Tool Budget

**Detects**: Excessive tool usage vs baseline

**Optimization**: Reduce budget to optimal level

**Impact**: 5-15% cost improvement, Low risk

```yaml
# Before
tool_budget: 50

# After
tool_budget: 30  # Reduced by 40%
```

**Budget Guidelines**:

| Task Type | Recommended Budget |
|-----------|-------------------|
| Quick exploration | 5-15 |
| Standard development | 20-40 |
| Complex features | 40-60 |
| Large refactoring | 60-100 |

---

### 4. Timeout Tuning

**Detects**: High P95 latency causing timeouts

**Optimization**: Set timeout to P95 + 20% buffer

**Impact**: 5-10% reliability improvement, Very low risk

```yaml
# Before
timeout_seconds: 300  # Fixed timeout

# After: Based on actual P95 latency
timeout_seconds: 180  # P95 (150s) + 20% buffer
```

**Benefits**:
- Faster failure detection
- Better resource utilization
- Improved user experience

---

### 5. Member Selection (Advanced)

**Detects**: Underperforming members based on execution metrics

**Optimization**: Replace or reconfigure problematic members

**Impact**: 10-30% improvement, Medium risk

```yaml
# Before: All members identical
members:
  - id: member1
  - id: member2
  - id: member3

# After: Optimized per-member config
members:
  - id: member1
    tool_budget_multiplier: 1.5  # High performer gets more budget
  - id: member2
    tool_budget_multiplier: 0.8  # Reduce budget for low performer
  - id: member3
    enable: false  # Disable underperforming member
```

---

### 6. Parallelization (Advanced)

**Detects**: Sequential tasks that could run in parallel

**Optimization**: Split into parallel sub-teams

**Impact**: 30-50% improvement, High risk

```yaml
# Before: Single sequential team
team_formation: sequential
members: 5

# After: Parallel sub-teams
team_formation: hierarchical
members:
  - id: coordinator
    role: manager
  - id: sub_team_1
    formation: parallel
    members: 2
  - id: sub_team_2
    formation: parallel
    members: 2
```

---

## Using the CLI

### Installation

The autotune CLI is included with Victor:

```bash
# Already installed as part of victor-ai
python scripts/workflows/autotune.py --help
```

### Commands

#### 1. Analyze Performance

```bash
python scripts/workflows/autotune.py analyze \
    --team-id code_review_team \
    --metrics metrics.json \
    --output analysis_report.json
```

**Output**:
```
================================================================================
  Performance Analysis
================================================================================
Team ID: code_review_team

Found 3 performance issue(s):

1. Slow Execution
   Severity: [██████████] 0.85/1.0
   Current:  75.00
   Baseline: 30.00
   Impact:   0.85

   Average execution time (75.0s) is 150.0% above baseline. Consider
   optimizing formation or reducing team size.
```

#### 2. Get Suggestions

```bash
python scripts/workflows/autotune.py suggest \
    --team-id my_team \
    --config workflow.yaml \
    --metrics metrics.json \
    --output suggestions.json
```

#### 3. Apply Optimizations (Interactive)

```bash
python scripts/workflows/autotune.py apply \
    --team-id my_team \
    --config workflow.yaml \
    --interactive
```

**Interactive Mode**:
```
Generated 2 optimization suggestion(s):

1. 🟠 Reduce team size from 8 to 4
   Type:     team_sizing
   Priority: high
   Expected: +15.0% improvement
   Confidence: 0.80

Apply suggestion #1? (y/n/v/q):
```

#### 4. Apply Optimizations (Automatic)

```bash
python scripts/workflows/autotune.py apply \
    --team-id my_team \
    --config workflow.yaml \
    --auto \
    --ab-test
```

#### 5. Benchmark Before/After

```bash
python scripts/workflows/autotune.py benchmark \
    --before workflow_old.yaml \
    --after workflow_new.yaml
```

#### 6. Rollback

```bash
python scripts/workflows/autotune.py rollback \
    --team-id my_team \
    --index 0  # Latest optimization
```

---

## Python API

### Basic Usage

```python
from victor.workflows.performance_autotuner import (
    PerformanceAnalyzer,
    PerformanceAutotuner,
)

# 1. Analyze performance
analyzer = PerformanceAnalyzer()
analyzer.load_metrics_from_file("metrics.json")

insights = analyzer.analyze_team_workflow(team_id="my_team")

for insight in insights:
    print(f"{insight.bottleneck}: {insight.recommendation}")

# 2. Get suggestions
autotuner = PerformanceAutotuner()
suggestions = autotuner.suggest_optimizations(
    team_id="my_team",
    current_config={"formation": "sequential", "member_count": 5}
)

for suggestion in suggestions:
    print(f"{suggestion.priority}: {suggestion.description}")

# 3. Apply optimization
result = await autotuner.apply_optimizations(
    team_id="my_team",
    optimizations=suggestions,
    workflow_config=current_config,
    enable_ab_testing=True
)

print(f"Improvement: {result.improvement_percentage:.1f}%")
```

### Advanced Configuration

```python
from victor.workflows.performance_autotuner import (
    PerformanceAutotuner,
    TeamSizingStrategy,
    FormationSelectionStrategy,
    ToolBudgetStrategy,
)

# Custom strategies
custom_strategies = [
    TeamSizingStrategy(),
    FormationSelectionStrategy(),
    ToolBudgetStrategy(),
]

# Custom autotuner configuration
autotuner = PerformanceAutotuner(
    strategies=custom_strategies,
    ab_test_threshold=10.0,  # Require 10% improvement
    enable_auto_rollback=True,  # Auto-rollback on regression
)

# Get and apply suggestions
suggestions = autotuner.suggest_optimizations(team_id="my_team")
result = await autotuner.apply_optimizations(
    team_id="my_team",
    optimizations=suggestions,
    workflow_config=config,
    enable_ab_testing=True
)

# Check optimization history
history = autotuner.get_optimization_history(team_id="my_team")
for entry in history:
    print(f"{entry['timestamp']}: {entry['optimization']['description']}")
```

### Custom Optimization Strategy

```python
from victor.workflows.performance_autotuner import OptimizationStrategy

class MyCustomStrategy(OptimizationStrategy):
    """Custom optimization strategy."""

    def suggest(self, insights, current_config):
        """Generate suggestions based on insights."""
        suggestions = []

        # Your custom logic here
        for insight in insights:
            if insight.bottleneck == "my_custom_bottleneck":
                suggestions.append(OptimizationSuggestion(
                    type=OptimizationType.MEMBER_SELECTION,
                    priority=OptimizationPriority.HIGH,
                    description="My custom optimization",
                    current_config=current_config,
                    suggested_config={"custom_param": "new_value"},
                    expected_improvement=20.0,
                    confidence=0.8,
                    risk_level=0.3,
                ))

        return suggestions

    def apply(self, config, suggestion):
        """Apply suggestion to config."""
        new_config = copy.deepcopy(config)
        new_config.update(suggestion.suggested_config)
        return new_config

# Use custom strategy
autotuner = PerformanceAutotuner(strategies=[MyCustomStrategy()])
```

---

## A/B Testing

### How A/B Testing Works

When enabled, optimizations are validated before permanent application:

1. **Baseline Measurement**: Run current configuration N times
2. **Optimized Measurement**: Run optimized configuration N times
3. **Statistical Analysis**: Compare metrics using statistical tests
4. **Validation**: Pass if improvement ≥ threshold (default: 5%)
5. **Application**: Apply only if validation passes

### Configuration

```python
autotuner = PerformanceAutotuner(
    ab_test_threshold=5.0,  # Minimum 5% improvement
    enable_auto_rollback=True,  # Rollback on regression
)

result = await autotuner.apply_optimizations(
    team_id="my_team",
    optimizations=suggestions,
    workflow_config=config,
    enable_ab_testing=True,  # Enable A/B testing
)

# Check validation result
if result.validation_status == "passed":
    print("Optimization validated and applied")
elif result.validation_status == "inconclusive":
    print("Improvement detected but below threshold")
else:
    print("Optimization failed validation")
```

### Validation Status Values

- `passed`: Improvement ≥ threshold, optimization applied
- `failed`: Improvement < 0% (regression), not applied
- `inconclusive`: 0% < improvement < threshold, not applied
- `skipped`: A/B testing disabled
- `dry_run`: Simulation only, no actual changes

### Statistical Testing

The A/B testing framework uses:

- **Welch's t-test**: Compare means with unequal variance
- **Mann-Whitney U test**: Non-parametric alternative
- **Bootstrap confidence intervals**: Robust estimation

**Confidence Level**: 95% (α = 0.05)

**Minimum Sample Size**: 10 runs per configuration

---

## Safety and Rollback

### Automatic Rollback

Auto-rollback is enabled by default and triggers when:

1. **Regression Detected**: After-optimization performance < baseline
2. **Error Rate Increase**: Success rate drops by > 10%
3. **Timeout Increase**: P95 latency increases by > 50%

```python
autotuner = PerformanceAutotuner(
    enable_auto_rollback=True,  # Default: True
)

# Optimization will auto-rollback if regression detected
result = await autotuner.apply_optimizations(...)
```

### Manual Rollback

```python
# Rollback latest optimization
success = await autotuner.rollback_optimization(team_id="my_team")

# Rollback specific optimization
success = await autotuner.rollback_optimization(
    team_id="my_team",
    rollback_index=0  # Index in history
)
```

### Rollback Configuration

Each optimization stores a rollback configuration:

```python
result = await autotuner.apply_optimizations(...)

# Access rollback config
rollback_config = result.rollback_config

# Save for manual recovery
with open("rollback_config.json", "w") as f:
    json.dump(rollback_config, f)
```

### Optimization History

Track all applied optimizations:

```python
history = autotuner.get_optimization_history(team_id="my_team")

for entry in history:
    print(f"Timestamp: {entry['timestamp']}")
    print(f"Optimization: {entry['optimization']['description']}")
    print(f"Improvement: {entry['improvement_percentage']:.1f}%")
    print(f"Status: {entry['validation_status']}")
```

---

## Best Practices

### 1. Start with Analysis

Always analyze before optimizing:

```bash
# Step 1: Analyze
python scripts/workflows/autotune.py analyze --team-id my_team --metrics metrics.json

# Step 2: Review insights
# Step 3: Get suggestions
python scripts/workflows/autotune.py suggest --team-id my_team --config workflow.yaml

# Step 4: Apply (with validation)
python scripts/workflows/autotune.py apply --team-id my_team --config workflow.yaml --ab-test
```

### 2. Use Interactive Mode for Critical Workflows

```bash
# Interactive mode lets you review and approve each suggestion
python scripts/workflows/autotune.py apply --team-id critical_team --interactive
```

### 3. Enable A/B Testing in Production

```python
autotuner = PerformanceAutotuner()
result = await autotuner.apply_optimizations(
    team_id="production_team",
    optimizations=suggestions,
    workflow_config=config,
    enable_ab_testing=True,  # Always enable in production
)
```

### 4. Set Appropriate Thresholds

```python
# Conservative: Require 10% improvement
autotuner_conservative = PerformanceAutotuner(ab_test_threshold=10.0)

# Aggressive: Accept 3% improvement
autotuner_aggressive = PerformanceAutotuner(ab_test_threshold=3.0)

# Default: 5% improvement
autotuner_default = PerformanceAutotuner(ab_test_threshold=5.0)
```

### 5. Monitor and Iterate

```python
# 1. Apply optimization
result = await autotuner.apply_optimizations(...)

# 2. Monitor for 24-48 hours
# 3. Collect new metrics
# 4. Re-analyze
new_insights = analyzer.analyze_team_workflow(team_id="my_team")

# 5. Iterate
```

### 6. Keep Rollback Configs

```bash
# Backup before applying
cp workflow.yaml workflow.yaml.backup

# Apply optimization
python scripts/workflows/autotune.py apply --team-id my_team --config workflow.yaml

# If needed, restore backup
cp workflow.yaml.backup workflow.yaml
```

### 7. Test in Staging First

```python
# 1. Test in staging
staging_result = await autotuner.apply_optimizations(
    team_id="staging_team",
    optimizations=suggestions,
    workflow_config=staging_config,
    enable_ab_testing=True,
)

# 2. If successful, apply to production
if staging_result.improvement_percentage > 10:
    prod_result = await autotuner.apply_optimizations(
        team_id="production_team",
        optimizations=suggestions,
        workflow_config=production_config,
        enable_ab_testing=True,
    )
```

---

## Examples

### Example 1: Optimizing a Code Review Team

**Initial State**:
```yaml
team_id: code_review_team
formation: parallel
member_count: 5
tool_budget: 50
timeout_seconds: 300
```

**Analysis**:
- Slow execution detected (75s vs 30s baseline)
- Excessive tool usage (50 calls vs 15 baseline)
- Formation suboptimal for task type

**Suggestions**:
1. Switch to pipeline formation (+35% expected)
2. Reduce tool budget to 30 (+10% expected)
3. Reduce team size to 3 (+12% expected)

**Application**:
```bash
python scripts/workflows/autotune.py apply \
    --team-id code_review_team \
    --config workflow.yaml \
    --interactive
```

**Result**:
```yaml
formation: pipeline
member_count: 3
tool_budget: 30
timeout_seconds: 180
```

**Improvement**: 42% faster, 38% cost reduction

---

### Example 2: Auto-Tuning a Research Team

**Initial State**:
```yaml
team_id: research_team
formation: consensus
member_count: 8
tool_budget: 75
```

**Python Script**:
```python
from victor.workflows.performance_autotuner import (
    PerformanceAutotuner,
    PerformanceAnalyzer,
)

# Load metrics
analyzer = PerformanceAnalyzer()
analyzer.load_metrics_from_file("research_metrics.json")

# Analyze
insights = analyzer.analyze_team_workflow("research_team")
print(f"Found {len(insights)} bottlenecks")

# Generate suggestions
autotuner = PerformanceAutotuner()
suggestions = autotuner.suggest_optimizations(
    team_id="research_team",
    current_config={
        "formation": "consensus",
        "member_count": 8,
        "tool_budget": 75
    }
)

print(f"Generated {len(suggestions)} suggestions")

# Apply top suggestion automatically
result = await autotuner.apply_optimizations(
    team_id="research_team",
    optimizations=[suggestions[0]],
    workflow_config=current_config,
    enable_ab_testing=True,
)

if result.success and result.improvement_percentage > 10:
    print(f"Success! Improved by {result.improvement_percentage:.1f}%")
else:
    # Rollback
    await autotuner.rollback_optimization("research_team")
```

**Outcome**:
- Formation: consensus → hierarchical
- Team size: 8 → 5
- Improvement: 28% faster

---

### Example 3: Continuous Auto-Tuning

```python
import asyncio
from victor.workflows.performance_autotuner import PerformanceAutotuner

async def continuous_autotune(team_id: str, interval_hours: int = 24):
    """Continuously auto-tune team workflow."""

    autotuner = PerformanceAutotuner()

    while True:
        print(f"\n[{datetime.now()}] Auto-tuning {team_id}")

        # Get suggestions
        suggestions = autotuner.suggest_optimizations(team_id)

        if not suggestions:
            print("No optimizations needed")
        else:
            print(f"Found {len(suggestions)} suggestions")

            # Apply with validation
            result = await autotuner.apply_optimizations(
                team_id=team_id,
                optimizations=suggestions,
                workflow_config=current_config,
                enable_ab_testing=True,
            )

            if result.success:
                print(f"Applied: {result.optimization.description}")
                print(f"Improvement: {result.improvement_percentage:.1f}%")
            else:
                print(f"Failed: {result.error}")

        # Wait for next interval
        await asyncio.sleep(interval_hours * 3600)

# Run continuous auto-tuning
asyncio.run(continuous_autotune("my_team", interval_hours=24))
```

---

## Troubleshooting

### Issue: No optimizations suggested

**Cause**: Metrics don't show significant issues

**Solution**:
- Check if metrics are being collected: `TeamMetricsCollector.get_instance().get_summary()`
- Verify team_id matches
- Lower A/B test threshold temporarily

### Issue: A/B test always fails

**Cause**: Insufficient sample size or high variance

**Solution**:
- Increase sample size: run more team executions
- Check for external factors (load, network)
- Use lower threshold for validation

### Issue: Optimization caused regression

**Cause**: Rare edge case or changed workload

**Solution**:
```python
# Rollback immediately
await autotuner.rollback_optimization(team_id="my_team")

# Report issue
# Investigate logs
# Re-analyze with new metrics
```

### Issue: Rollback failed

**Cause**: Missing rollback configuration

**Solution**:
```bash
# Restore from manual backup
cp workflow.yaml.backup workflow.yaml

# Check optimization history
python scripts/workflows/autotune.py rollback --team-id my_team --list
```

---

## References

- [Team Node Performance Guide](./team_node_performance.md) - Performance characteristics
- [Team Node Usage Guide](./team_nodes.md) - Team node basics
- [Workflow DSL Documentation](../guides/workflow-development/dsl.md) - YAML workflow syntax
- [Formation Pattern Reference](./advanced_formations.md) - Formation patterns

---

**Last Updated**: 2025-01-15

For questions or contributions, please open an issue on GitHub.
