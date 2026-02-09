# Coordinator Settings Guide - Part 2

**Part 2 of 2:** Best Practices, Troubleshooting, Examples, and Related Documentation

---

## Navigation

- [Part 1: Configuration](part-1-configuration.md)
- **[Part 2: Best Practices](#)** (Current)
- [**Complete Guide](../coordinator_settings.md)**

---
## Best Practices

### 1. Gradual Rollout

**Recommended Approach**:
1. Test in development environment first
2. Run validation scripts
3. Enable in staging with monitoring
4. Gradual rollout to production (percentage-based)
5. Monitor metrics closely
6. Full rollout after validation period

**Example**:
```bash
# Stage 1: Development
export VICTOR_USE_COORDINATOR_ORCHESTRATOR=true
victor chat --no-tui  # Test basic functionality

# Stage 2: Validation
python scripts/validate_coordinator_orchestrator.py --full-report

# Stage 3: Production (with backup)
python scripts/toggle_coordinator_orchestrator.py enable --backup

# Stage 4: Monitor
tail -f ~/.victor/metrics/victor.jsonl
```

### 2. Enable Observability

Always enable observability logging during rollout:

```bash
export VICTOR_ENABLE_OBSERVABILITY_LOGGING=true
export VICTOR_OBSERVABILITY_LOG_PATH=~/.victor/metrics/victor_rollout.jsonl
```

This allows you to:
- Track performance metrics
- Detect anomalies early
- Analyze coordinator interactions
- Debug issues post-mortem

### 3. Backup Before Changes

Always backup settings before enabling:

```bash
# Manual backup
cp ~/.victor/profiles.yaml ~/.victor/profiles.yaml.backup.$(date +%Y%m%d)

# Or use the toggle script with automatic backup
python scripts/toggle_coordinator_orchestrator.py enable --backup
```

### 4. Test Before Full Rollout

Run comprehensive tests:

```bash
# Quick validation
python scripts/validate_coordinator_orchestrator.py --quick

# Full validation with tests
python scripts/validate_coordinator_orchestrator.py --full-report --output report.json

# Run test suite
pytest tests/unit/agent/coordinators/ -v
pytest tests/integration/ -m "not slow" -v
```

### 5. Monitor Key Metrics

Track these metrics after rollout:

- **Performance**: Chat latency, tool execution time
- **Errors**: Coordinator failures, tool errors
- **Resources**: Memory usage, CPU usage
- **Functionality**: Feature parity with legacy

### 6. Have a Rollback Plan

Know how to rollback quickly:

```bash
# Quick rollback via environment variable
export VICTOR_USE_COORDINATOR_ORCHESTRATOR=false

# Or via toggle script
python scripts/toggle_coordinator_orchestrator.py disable --backup

# Or restore from backup
cp ~/.victor/profiles.yaml.backup.YYYYMMDD ~/.victor/profiles.yaml
```

### 7. Document Your Configuration

Keep documentation of your setup:

```yaml
# ~/.victor/profiles.yaml

# Add comments explaining configuration
use_coordinator_orchestrator: true  # Enabled 2025-01-14 after validation

profiles:
  production:
    provider: anthropic
    model: claude-sonnet-4-5
    # Temperature lowered for more deterministic outputs in production
    temperature: 0.3
```

### 8. Use Profiles for Different Environments

Create separate profiles for dev/staging/prod:

```yaml
profiles:
  development:
    provider: ollama
    model: qwen3-coder:30b
    use_coordinator_orchestrator: true  # Test new architecture

  staging:
    provider: anthropic
    model: claude-sonnet-4-5
    use_coordinator_orchestrator: true  # Validate in staging

  production:
    provider: anthropic
    model: claude-sonnet-4-5
    # Gradual rollout: only enable for specific users initially
    use_coordinator_orchestrator: false  # Legacy until validated
```

---

## Troubleshooting

### Issue: Feature Flag Not Taking Effect

**Symptoms**:
- Setting `use_coordinator_orchestrator: true` but legacy orchestrator still used

**Solutions**:

1. **Check environment variable override**:
```bash
echo $VICTOR_USE_COORDINATOR_ORCHESTRATOR
# If set, it overrides profiles.yaml
unset VICTOR_USE_COORDINATOR_ORCHESTRATOR
```

2. **Verify YAML syntax**:
```bash
python -c "import yaml; yaml.safe_load(open('~/.victor/profiles.yaml'))"
# Check for syntax errors
```

3. **Check file location**:
```bash
# Should be at ~/.victor/profiles.yaml, not project-local .victor/
ls -la ~/.victor/profiles.yaml
```

4. **Restart Victor**:
```bash
# Victor caches settings on startup
# Exit and restart
victor chat
```

### Issue: Performance Degradation

**Symptoms**:
- Slower response times after enabling coordinators

**Solutions**:

1. **Check if analytics is enabled** (can add overhead):
```yaml
analytics_enabled: false  # Try disabling
```

2. **Adjust tool selection strategy**:
```yaml
# For faster performance
tool_selection_strategy: "keyword"  # Faster than semantic
```

3. **Reduce metrics collection**:
```yaml
streaming_metrics_enabled: false
```

4. **Run performance benchmark**:
```bash
pytest tests/benchmark/test_orchestrator_refactoring_performance.py -v
```

### Issue: Coordinator Import Errors

**Symptoms**:
- `ImportError: cannot import name 'ConfigCoordinator'`

**Solutions**:

1. **Check Victor version**:
```bash
victor --version
# Should be 0.5.0 or higher
```

2. **Reinstall Victor**:
```bash
pip install --upgrade victor-ai
```

3. **Verify installation**:
```bash
python -c "from victor.agent.coordinators import ConfigCoordinator; print('OK')"
```

### Issue: Validation Failures

**Symptoms**:
- Validation script reports errors

**Solutions**:

1. **Run with verbose output**:
```bash
python scripts/validate_coordinator_orchestrator.py --verbose
```

2. **Check specific failures**:
```bash
python scripts/validate_coordinator_orchestrator.py --output report.json
cat report.json | jq '.results[] | select(.passed == false)'
```

3. **Run unit tests**:
```bash
pytest tests/unit/agent/coordinators/ -v
```

### Issue: Settings Not Persisting

**Symptoms**:
- Changes to profiles.yaml not saved

**Solutions**:

1. **Check file permissions**:
```bash
ls -la ~/.victor/profiles.yaml
# Should be writable by your user
```

2. **Check disk space**:
```bash
df -h ~/.victor
```

3. **Use toggle script** (handles errors gracefully):
```bash
python scripts/toggle_coordinator_orchestrator.py enable --backup
```

---

## Examples

### Example 1: Development Environment

```bash
# ~/.victor/profiles.yaml

use_coordinator_orchestrator: true
log_level: "DEBUG"

profiles:
  dev:
    provider: ollama
    model: qwen3-coder:30b
    temperature: 0.7
    # More aggressive tool selection for testing
    tool_selection:
      base_threshold: 0.4
      base_max_tools: 20
```

### Example 2: Production Environment

```bash
# ~/.victor/profiles.yaml

use_coordinator_orchestrator: true
log_level: "INFO"
enable_observability_logging: true
observability_log_path: "~/.victor/metrics/production.jsonl"

profiles:
  prod:
    provider: anthropic
    model: claude-sonnet-4-5
    temperature: 0.3
    # Conservative settings for production
    tool_selection:
      base_threshold: 0.6
      base_max_tools: 12
      model_size_tier: "cloud"
    # Limit resource usage
    checkpoint_max_per_session: 30
    streaming_metrics_history_size: 500
```

### Example 3: CI/CD Pipeline

```yaml
# .github/workflows/test.yml

name: Test with Coordinators
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install Victor
        run: pip install victor-ai

      - name: Enable Coordinator Orchestrator
        run: export VICTOR_USE_COORDINATOR_ORCHESTRATOR=true

      - name: Run Validation
        run: python scripts/validate_coordinator_orchestrator.py --full-report --output report.json

      - name: Upload Report
        uses: actions/upload-artifact@v2
        with:
          name: validation-report
          path: report.json
```

### Example 4: Gradual Rollout Script

```bash
#!/bin/bash
# gradual_rollout.sh

# Enable for 10% of users (example)
PERCENTAGE=10

# Generate random number 1-100
ROLL=$((RANDOM % 100 + 1))

if [ $ROLL -le $PERCENTAGE ]; then
    export VICTOR_USE_COORDINATOR_ORCHESTRATOR=true
    echo "Coordinator orchestrator ENABLED (user in $PERCENTAGE% rollout)"
else
    export VICTOR_USE_COORDINATOR_ORCHESTRATOR=false
    echo "Coordinator orchestrator DISABLED (user not in rollout)"
fi

# Start Victor
victor chat "$@"
```

---

## Related Documentation

- [Production Checklist](coordinator_rollback_checklist.md)
- [Monitoring Guide](coordinator_monitoring.md)
- [Migration Guide](../migration/orchestrator_refactoring_guide.md)
- [Architecture Overview](../architecture/coordinator_based_architecture.md)

---

**Document Version**: 1.0
**Last Updated**: 2025-01-14
**Next Review**: 2025-02-14

---

**Last Updated:** February 01, 2026
**Reading Time:** 6 minutes
