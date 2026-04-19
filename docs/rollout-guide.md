# Predictive Features Rollout Guide

This guide explains how to safely rollout the predictive tool selection features using feature flags and gradual rollout strategies.

## Overview

The predictive tool selection system includes three major enhancements:

1. **Hybrid Decision Service** - Fast, deterministic decision-making with LLM fallback
2. **Phase-Based Context Management** - Context optimization based on task phase
3. **Predictive Tool Selection** - Ensemble prediction with async preloading

All features are controlled by feature flags and can be rolled out gradually with instant rollback capability.

## Feature Flags

### Environment Variables

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `VICTOR_ENABLE_PREDICTIVE_TOOLS` | `false` | Master switch for all predictive features |
| `VICTOR_PREDICTIVE_ROLLOUT_PERCENTAGE` | `0` | Rollout percentage (0-100) |
| `VICTOR_ENABLE_HYBRID_DECISIONS` | `false` | Enable hybrid decision service |
| `VICTOR_ENABLE_PHASE_AWARE_CONTEXT` | `false` | Enable phase-based context management |
| `VICTOR_ENABLE_TOOL_PREDICTOR` | `false` | Enable ensemble tool prediction |
| `VICTOR_ENABLE_COOCCURRENCE_TRACKING` | `false` | Enable co-occurrence pattern learning |
| `VICTOR_ENABLE_TOOL_PRELOADING` | `false` | Enable async tool schema preloading |
| `VICTOR_PREDICTIVE_CONFIDENCE_THRESHOLD` | `0.6` | Minimum confidence for predictions (0.0-1.0) |

### Configuration Files

**`~/.victor/settings.yaml`**
```yaml
feature_flags:
  # Master switches
  enable_predictive_tools: true
  predictive_rollout_percentage: 10  # Start with 10%

  # Individual feature flags
  enable_hybrid_decisions: false
  enable_phase_aware_context: false
  enable_tool_predictor: true
  enable_cooccurrence_tracking: true
  enable_tool_preloading: true

  # Tuning
  predictive_confidence_threshold: 0.6
```

## Rollout Strategy

### 4-Week Gradual Rollout Plan

```
Week 1: Canary (1%)
├── Goal: Initial testing in production
├── Target: 1% of requests
├── Success Criteria: < 1% error rate, no regressions
└── Rollback: Set VICTOR_PREDICTIVE_ROLLOUT_PERCENTAGE=0

Week 2: Early Adopters (10%)
├── Goal: Broader testing
├── Target: 10% of requests
├── Success Criteria: < 2% error rate, 15% latency reduction
└── Rollback: Set VICTOR_PREDICTIVE_ROLLOUT_PERCENTAGE=1

Week 3: Beta (50%)
├── Goal: Majority of users
├── Target: 50% of requests
├── Success Criteria: < 3% error rate, 20% latency reduction
└── Rollback: Set VICTOR_PREDICTIVE_ROLLOUT_PERCENTAGE=10

Week 4: General Availability (100%)
├── Goal: All users
├── Target: 100% of requests
├── Success Criteria: < 5% error rate, 25% latency reduction
└── Rollback: Set VICTOR_PREDICTIVE_ROLLOUTPercentage=50
```

### Rollout Commands

```bash
# Week 1: Start canary (1%)
export VICTOR_ENABLE_PREDICTIVE_TOOLS=true
export VICTOR_PREDICTIVE_ROLLOUT_PERCENTAGE=1
export VICTOR_ENABLE_TOOL_PREDICTOR=true
export VICTOR_ENABLE_TOOL_PRELOADING=true

# Week 2: Increase to 10%
export VICTOR_PREDICTIVE_ROLLOUT_PERCENTAGE=10

# Week 3: Increase to 50%
export VICTOR_PREDICTIVE_ROLLOUT_PERCENTAGE=50

# Week 4: Full rollout
export VICTOR_PREDICTIVE_ROLLOUT_PERCENTAGE=100

# Instant rollback (anytime)
export VICTOR_ENABLE_PREDICTIVE_TOOLS=false
```

## Monitoring

### Key Metrics

| Metric | Target | Alert Threshold |
|--------|--------|----------------|
| Error Rate | < 5% | > 5% |
| Decision Latency | <100ms | > 200ms |
| Tool Prediction Accuracy | >80% | < 70% |
| Cache Hit Rate (L1) | >60% | < 40% |
| Preload Success Rate | >95% | < 90% |

### Viewing Metrics

```python
from victor.agent.planning.rollout_manager import RolloutManager

manager = RolloutManager()
summary = manager.get_metrics_summary()

print(f"Current Stage: {summary['current_stage']}")
print(f"Rollout Percentage: {summary['rollout_percentage']}%")
print(f"Total Requests: {summary['total_requests']}")
print(f"Error Rate: {summary['error_rate']:.2%}")
print(f"Avg Latency: {summary['avg_latency_ms']:.1f}ms")
```

### Dashboard Setup

```python
# Add to your monitoring dashboard
from victor.config.settings import Settings

settings = Settings()
flags = settings.feature_flags

# Check rollout status
effective = flags.get_effective_settings()
print(f"Predictive Enabled: {effective['predictive_tools_enabled']}")
print(f"Rollout Percentage: {effective['rollout_percentage']}%")
```

## Rollback Procedures

### Instant Rollback

**Environment Variable Method (Fastest)**
```bash
# Immediately disable all predictive features
export VICTOR_ENABLE_PREDICTIVE_TOOLS=false

# Restart services
systemctl restart victor
```

**Configuration File Method**
```yaml
# ~/.victor/settings.yaml
feature_flags:
  enable_predictive_tools: false
```

### Partial Rollback

**Disable Specific Components**
```bash
# Keep predictive tools but disable preloading
export VICTOR_ENABLE_TOOL_PRELOADING=false

# Keep predictive tools but disable co-occurrence tracking
export VICTOR_ENABLE_COOCCURRENCE_TRACKING=false
```

**Reduce Rollout Percentage**
```bash
# Reduce from 50% to 10%
export VICTOR_PREDICTIVE_ROLLOUT_PERCENTAGE=10
```

### Rollback Decision Tree

```
Error Rate > 5%?
├── YES → Rollback to previous stage
├── NO → Continue

Cache Hit Rate < 40%?
├── YES → Check preloading configuration
├── NO → Continue

Prediction Accuracy < 70%?
├── YES → Adjust confidence threshold
│   export VICTOR_PREDICTIVE_CONFIDENCE_THRESHOLD=0.7
├── NO → Continue

Latency Increased > 20%?
├── YES → Disable preloading
│   export VICTOR_ENABLE_TOOL_PRELOADING=false
├── NO → Continue
```

## Testing Before Rollout

### Unit Tests

```bash
# Run all predictive feature tests
python -m pytest tests/unit/agent/planning/ -v

# Run specific test suites
python -m pytest tests/unit/agent/planning/test_tool_predictor.py -v
python -m pytest tests/unit/agent/planning/test_cooccurrence_tracker.py -v
python -m pytest tests/unit/agent/planning/test_tool_preloader.py -v
```

### Integration Tests

```bash
# Run integration tests
python -m pytest tests/integration/agent/test_predictive_features_e2e.py -v
```

### Manual Testing Checklist

- [ ] Test with 1% rollout, verify 1% of requests use predictive
- [ ] Test error handling when predictor fails
- [ ] Test rollback by disabling feature flag
- [ ] Test cache hit rates
- [ ] Test preloading effectiveness
- [ ] Test with different task types (bugfix, feature, refactor)
- [ ] Test co-incident learning over multiple sessions

## Troubleshooting

### Issue: Predictive Features Not Working

**Check Feature Flags**
```python
from victor.config.settings import Settings

settings = Settings()
print(f"Enable Predictive: {settings.feature_flags.enable_predictive_tools}")
print(f"Rollout Percentage: {settings.feature_flags.predictive_rollout_percentage}")
```

**Check Logs**
```bash
# Look for predictive feature logs
grep -i "predictive" ~/.victor/logs/*.log | tail -50
```

### Issue: High Error Rate

**Check Confidence Threshold**
```bash
# Increase threshold to reduce predictions
export VICTOR_PREDICTIVE_CONFIDENCE_THRESHOLD=0.8
```

**Check Rollback Triggers**
```python
from victor.agent.planning.rollout_manager import RolloutManager

manager = RolloutManager()
summary = manager.get_metrics_summary()
print(f"Error Rate: {summary['error_rate']:.2%}")
```

### Issue: No Performance Improvement

**Check Preloading**
```bash
# Verify preloading is enabled
echo $VICTOR_ENABLE_TOOL_PRELOADING

# Check preload statistics
from victor.agent.planning.tool_preloader import ToolPreloader
preloader = ToolPreloader()
stats = preloader.get_statistics()
print(f"Preload Count: {stats['preload_count']}")
print(f"Background Loads: {stats['background_loads']}")
```

**Check Cache Hit Rates**
```python
selector = get_tool_selector()  # Your selector instance
stats = selector.get_statistics()
print(f"L1 Hit Rate: {stats['l1_hit_rate']:.2%}")
```

## Best Practices

### 1. Always Start with Canary
Never enable predictive features for all users immediately. Start with 1% canary deployment.

### 2. Monitor Metrics Closely
Check error rates, latency, and accuracy regularly during rollout. Set up alerts for thresholds.

### 3. Have Rollback Plan Ready
Know exactly how to rollback instantly before starting rollout. Test rollback procedures.

### 4. Gradual Percentage Increases
Increase rollout percentage in stages: 1% → 10% → 50% → 100%. Don't skip stages.

### 5. Document Each Stage
Document metrics, issues, and decisions at each rollout stage.

### 6. Test Rollback
Test rollback procedures during canary stage to ensure they work when needed.

## Success Criteria

### Week 1 (Canary - 1%)
- ✓ Error rate < 1%
- ✓ No regression in existing functionality
- ✓ Predictive features working for 1% of requests

### Week 2 (Early Adopters - 10%)
- ✓ Error rate < 2%
- ✓ 10-15% latency reduction
- ✓ Prediction accuracy > 75%

### Week 3 ( Beta - 50%)
- ✓ Error rate < 3%
- ✓ 15-20% latency reduction
- ✓ Prediction accuracy > 80%
- ✓ Cache hit rate > 50%

### Week 4 (General - 100%)
- ✓ Error rate < 5%
- ✓ 20-25% latency reduction
- ✓ Prediction accuracy > 80%
- ✓ Cache hit rate > 60%
- ✓ User satisfaction maintained

## Support

For issues or questions during rollout:
1. Check logs: `~/.victor/logs/*.log`
2. Check metrics: `RolloutManager.get_metrics_summary()`
3. Review this guide's troubleshooting section
4. Rollback if unsure, investigate after system is stable

## Appendix: Quick Reference

### Enable Predictive Features (Canary)
```bash
export VICTOR_ENABLE_PREDICTIVE_TOOLS=true
export VICTOR_PREDICTIVE_ROLLOUT_PERCENTAGE=1
export VICTOR_ENABLE_TOOL_PREDICTOR=true
export VICTOR_ENABLE_TOOL_PRELOADING=true
```

### Instant Rollback
```bash
export VICTOR_ENABLE_PREDICTIVE_TOOLS=false
```

### Check Status
```python
from victor.config.settings import Settings
settings = Settings()
effective = settings.feature_flags.get_effective_settings()
print(effective)
```

### View Metrics
```python
from victor.agent.planning.rollout_manager import RolloutManager
manager = RolloutManager()
print(manager.get_metrics_summary())
```
