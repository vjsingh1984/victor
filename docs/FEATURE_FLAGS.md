# Feature Flags System

## Overview

Victor 1.0.0 includes a comprehensive feature flag system for gradual rollout of new capabilities. This enables safe deployment of experimental features, instant rollback without deployment, and A/B testing.

## Available Feature Flags

### Planning Capabilities

#### `hierarchical_planning_enabled`
- **Description**: Enable hierarchical task decomposition and planning engine
- **Default**: `False`
- **Rollout Strategy**: Gradual
- **Since**: v1.0.0
- **Stable**: No
- **Dependencies**: None
- **Metadata**:
  - Max depth: 5
  - Min subtasks: 2
  - Max subtasks: 10

```python
from victor.feature_flags import get_feature_flag_manager

manager = get_feature_flag_manager()
if manager.is_enabled("hierarchical_planning_enabled"):
    # Use hierarchical planning
    pass
```

### Memory Systems

#### `enhanced_memory_enabled`
- **Description**: Enable episodic and semantic memory with consolidation
- **Default**: `False`
- **Rollout Strategy**: Gradual
- **Since**: v1.0.0
- **Stable**: No
- **Dependencies**: None
- **Metadata**:
  - Episodic max episodes: 1000
  - Episodic recall threshold: 0.3
  - Semantic max facts: 5000
  - Semantic query threshold: 0.25

### Dynamic Skills

#### `dynamic_skills_enabled`
- **Description**: Enable runtime tool discovery and skill composition
- **Default**: `False`
- **Rollout Strategy**: Gradual
- **Since**: v1.0.0
- **Stable**: No
- **Dependencies**: None
- **Metadata**:
  - Max tools: 20
  - Min compatibility: 0.5
  - Auto composition: True

#### `self_improvement_enabled`
- **Description**: Enable proficiency tracking and self-improvement loops
- **Default**: `False`
- **Rollout Strategy**: Gradual
- **Since**: v1.0.0
- **Stable**: No
- **Dependencies**: `dynamic_skills_enabled`
- **Metadata**:
  - Window size: 100
  - Decay rate: 0.95
  - Min samples: 5

### Multimodal Capabilities

#### `multimodal_vision_enabled`
- **Description**: Enable vision/image processing capabilities
- **Default**: `False`
- **Rollout Strategy**: Gradual
- **Since**: v1.0.0
- **Stable**: No
- **Dependencies**: None
- **Metadata**:
  - Supported formats: png, jpg, jpeg, gif, webp
  - Max image size: 10 MB

#### `multimodal_audio_enabled`
- **Description**: Enable audio/speech processing capabilities
- **Default**: `False`
- **Rollout Strategy**: Gradual
- **Since**: v1.0.0
- **Stable**: No
- **Dependencies**: None
- **Metadata**:
  - Supported formats: wav, mp3, ogg, flac
  - Max audio size: 25 MB

### Dynamic Personas

#### `dynamic_personas_enabled`
- **Description**: Enable dynamic persona management and adaptation
- **Default**: `False`
- **Rollout Strategy**: Gradual
- **Since**: v1.0.0
- **Stable**: No
- **Dependencies**: None
- **Metadata**:
  - Max personas: 10
  - Adaptation threshold: 0.7
  - Switching cooldown: 300 seconds

### Performance Optimizations

#### `lazy_loading_enabled`
- **Description**: Enable lazy component loading for faster initialization
- **Default**: `True`
- **Rollout Strategy**: Immediate
- **Since**: v1.0.0
- **Stable**: Yes
- **Dependencies**: None
- **Metadata**:
  - Load on access: True
  - Preload critical: True

#### `parallel_execution_enabled`
- **Description**: Enable parallel tool and workflow execution
- **Default**: `True`
- **Rollout Strategy**: Immediate
- **Since**: v1.0.0
- **Stable**: Yes
- **Dependencies**: None
- **Metadata**:
  - Max workers: 10
  - Timeout: 300 seconds
  - Chunk size: 5

## Configuration Methods

### 1. Environment Variables (Highest Priority)

Set environment variables to enable/disable features:

```bash
# Enable hierarchical planning
export VICTOR_FEATURE_HIERARCHICAL_PLANNING_ENABLED=true

# Enable enhanced memory
export VICTOR_FEATURE_ENHANCED_MEMORY_ENABLED=true

# Disable a feature
export VICTOR_FEATURE_DYNAMIC_SKILLS_ENABLED=false
```

**Note**: Requires process restart to take effect.

### 2. Settings File

Add to `~/.victor/profiles.yaml`:

```yaml
feature_flags:
  hierarchical_planning_enabled: true
  enhanced_memory_enabled: true
  dynamic_skills_enabled: false
  self_improvement_enabled: false
  multimodal_vision_enabled: false
  multimodal_audio_enabled: false
  dynamic_personas_enabled: false
  lazy_loading_enabled: true
  parallel_execution_enabled: true
```

### 3. Runtime API (Python)

Use the FeatureFlagManager for hot-reload without restart:

```python
from victor.feature_flags import get_feature_flag_manager

manager = get_feature_flag_manager()

# Check if feature is enabled
if manager.is_enabled("hierarchical_planning_enabled"):
    # Use feature
    pass

# Enable feature at runtime
manager.set_flag("enhanced_memory_enabled", True)

# Disable feature
manager.set_flag("dynamic_skills_enabled", False)

# Reset to default
manager.reset_flag("hierarchical_planning_enabled")

# Get all flags
flags = manager.get_all_flags()
for flag_name, flag_value in flags.items():
    print(f"{flag_name}: {flag_value}")
```

### 4. Rollout Strategies

#### Immediate Rollout
Enable for all users immediately:
```python
manager.set_flag("lazy_loading_enabled", True)
```

#### Gradual Rollout
Monitor metrics and gradually increase exposure:
```python
# Start with 10% of users
from victor.feature_flags.resolvers import StagedRolloutResolver

resolver = StagedRolloutResolver(rollout_percentage=10.0)
enabled = resolver.resolve("new_feature", default=False)

# After monitoring, increase to 50%
resolver.set_rollout_percentage(50.0)

# Finally, 100%
resolver.set_rollout_percentage(100.0)
```

#### A/B Testing
Compare different implementations:
```python
from victor.feature_flags.resolvers import ABTestingResolver

resolver = ABTestingResolver(
    variants=["control", "treatment_a", "treatment_b"],
    weights=[0.5, 0.25, 0.25]  # 50% control, 25% each treatment
)

variant = resolver.resolve_variant("experiment_1")
if variant == "control":
    # Use current implementation
    pass
elif variant == "treatment_a":
    # Try new implementation A
    pass
elif variant == "treatment_b":
    # Try new implementation B
    pass
```

## Dependency Management

Features can depend on other features. Dependencies must be enabled first:

```python
from victor.feature_flags import get_feature_flag_manager

manager = get_feature_flag_manager()

# This will raise an error because dynamic_skills_enabled is not enabled
try:
    manager.set_flag("self_improvement_enabled", True)
except ValueError as e:
    print(f"Error: {e}")
    # Error: Cannot enable self_improvement_enabled: dependencies not satisfied: dynamic_skills_enabled

# Enable dependency first
manager.set_flag("dynamic_skills_enabled", True)

# Now enable dependent feature
manager.set_flag("self_improvement_enabled", True)
```

## Audit Logging

All flag changes are logged for audit and debugging:

```python
from victor.feature_flags import get_feature_flag_manager

manager = get_feature_flag_manager()

# Make some changes
manager.set_flag("hierarchical_planning_enabled", True)

# Get audit log
log = manager.get_audit_log("hierarchical_planning_enabled", limit=10)
for entry in log:
    print(f"{entry['timestamp']}: {entry['flag_name']} = {entry['new_value']} (source: {entry['source']})")

# Clear audit log
manager.clear_audit_log()
```

## Callbacks

Register callbacks to react to flag changes:

```python
from victor.feature_flags import get_feature_flag_manager

def on_flag_change(flag_name: str, value: bool):
    print(f"Flag {flag_name} changed to {value}")
    # Take action (e.g., reload config, clear cache, etc.)

manager = get_feature_flag_manager()

# Register callback for specific flag
manager.on_flag_changed("hierarchical_planning_enabled", on_flag_change)

# Register callback for all flags
manager.on_flag_changed("*", on_flag_change)
```

## Monitoring and Metrics

### Feature Metrics

Track feature usage and performance:

```python
from victor.observability.metrics_config import get_feature_metrics_registry

registry = get_feature_metrics_registry()

# Get metrics for a feature
metrics = registry.get_or_create("hierarchical_planning_enabled")

# Record execution
metrics.record_execution(duration_ms=150, success=True)

# Record user activity
metrics.record_user_activity(user_count=42)

# Get summary
summary = metrics.get_summary()
print(summary)
# {
#     "feature": "hierarchical_planning_enabled",
#     "enabled": True,
#     "executions": 100,
#     "errors": 2,
#     "error_rate": 0.02,
#     "avg_duration_ms": 145.5,
#     "active_users": 42
# }
```

### Health Checks

Monitor feature health and detect degradation:

```python
from victor.observability.health_checks import FeatureHealthChecker

checker = FeatureHealthChecker()

# Add health checks
checker.add_resource_check(
    cpu_threshold=80.0,
    memory_threshold=85.0,
    disk_threshold=90.0
)

checker.add_performance_check(
    component_name="hierarchical_planning",
    latency_threshold_ms=1000.0,
    error_rate_threshold=0.05
)

# Run all checks
results = await checker.check_all()
for result in results:
    print(f"{result.name}: {result.status.value}")
```

## Best Practices

### 1. Gradual Rollout

Always roll out new features gradually:

1. Enable for internal testing (5%)
2. Monitor metrics for 24-48 hours
3. Increase to 25% if metrics look good
4. Continue monitoring
5. Gradually increase to 50%, 75%, 100%

### 2. Monitor Metrics

Track these metrics during rollout:
- Error rate (should not increase)
- Latency (should not degrade)
- User satisfaction (should improve or stay same)
- Feature adoption (should increase gradually)

### 3. Have Rollback Plan

Always be ready to disable features instantly:

```python
# Emergency disable
manager.set_flag("experimental_feature", False, source="emergency_rollback")
```

### 4. Use Feature Gates

Wrap new features in flag checks:

```python
from victor.feature_flags import get_feature_flag_manager

manager = get_feature_flag_manager()

def my_function():
    # Old behavior
    result = old_implementation()

    # New behavior (guarded by flag)
    if manager.is_enabled("new_feature_enabled"):
        result = new_implementation()

    return result
```

### 5. Document Dependencies

Clearly document feature dependencies in code:

```python
from victor.feature_flags import get_feature_flag_manager

manager = get_feature_flag_manager()

# This feature requires enhanced memory to be enabled
if not manager.is_enabled("enhanced_memory_enabled"):
    raise RuntimeError("enhanced_memory_enabled must be enabled first")

if manager.is_enabled("advanced_memory_feature"):
    # Use advanced memory features
    pass
```

### 6. Test with Flags Disabled

Ensure your code works with all combinations of flags:

```python
import pytest
from victor.feature_flags import get_feature_flag_manager

def test_feature_disabled():
    manager = get_feature_flag_manager()
    manager.set_flag("new_feature", False)

    # Test old behavior
    result = my_function()
    assert result == expected_old_result

def test_feature_enabled():
    manager = get_feature_flag_manager()
    manager.set_flag("new_feature", True)

    # Test new behavior
    result = my_function()
    assert result == expected_new_result
```

## Troubleshooting

### Feature Not Enabling

**Problem**: Feature flag shows as disabled even after setting it.

**Solutions**:
1. Check dependencies are enabled:
```python
manager = get_feature_flag_manager()
deps = manager.get_flag_metadata("my_feature")["dependencies"]
for dep in deps:
    print(f"{dep}: {manager.is_enabled(dep)}")
```

2. Clear cache and retry:
```python
manager.clear_cache()
manager.is_enabled("my_feature")
```

3. Check for environment variable conflicts:
```bash
env | grep VICTOR_FEATURE_
```

### Performance Degradation

**Problem**: System slowed down after enabling a feature.

**Solutions**:
1. Check health status:
```python
from victor.observability.health_checks import FeatureHealthChecker

checker = FeatureHealthChecker()
results = await checker.check_all()
```

2. Disable feature immediately:
```python
manager.set_flag("problematic_feature", False)
```

3. Review metrics:
```python
from victor.observability.metrics_config import get_feature_metrics_registry

registry = get_feature_metrics_registry()
summary = registry.get_or_create("problematic_feature").get_summary()
print(summary)
```

### High Error Rates

**Problem**: Error rate increased after enabling feature.

**Solutions**:
1. Check error metrics:
```python
from victor.observability.metrics_config import get_feature_metrics_registry

registry = get_feature_metrics_registry()
metrics = registry.get_or_create("problematic_feature")
print(f"Error rate: {metrics.get_summary()['error_rate']:.2%}")
```

2. Disable feature and investigate:
```python
manager.set_flag("problematic_feature", False)
```

3. Review logs for error patterns.

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Test with Feature Flags

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        feature_combination:
          - "all_off"
          - "hierarchical_planning_only"
          - "enhanced_memory_only"
          - "all_on"
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      - name: Configure feature flags
        run: |
          case "${{ matrix.feature_combination }}" in
            "all_off")
              export VICTOR_FEATURE_HIERARCHICAL_PLANNING_ENABLED=false
              export VICTOR_FEATURE_ENHANCED_MEMORY_ENABLED=false
              ;;
            "hierarchical_planning_only")
              export VICTOR_FEATURE_HIERARCHICAL_PLANNING_ENABLED=true
              export VICTOR_FEATURE_ENHANCED_MEMORY_ENABLED=false
              ;;
            "enhanced_memory_only")
              export VICTOR_FEATURE_HIERARCHICAL_PLANNING_ENABLED=false
              export VICTOR_FEATURE_ENHANCED_MEMORY_ENABLED=true
              ;;
            "all_on")
              export VICTOR_FEATURE_HIERARCHICAL_PLANNING_ENABLED=true
              export VICTOR_FEATURE_ENHANCED_MEMORY_ENABLED=true
              ;;
          esac
      - name: Run tests
        run: |
          pytest tests/ -v
```

## Further Reading

- [Monitoring Guide](guides/MONITORING.md) - Setting up observability
- [Architecture Documentation](../architecture/) - System design
- [Contributing Guidelines](../CONTRIBUTING.md) - Adding new features
