# Coordinator Orchestrator Production Enablement Checklist

**Version**: 1.0
**Date**: 2025-01-14
**Status**: READY FOR PRODUCTION
**Purpose**: Safe rollout and rollback procedures for coordinator-based orchestrator

---

## Table of Contents

1. [Pre-Rollout Checks](#pre-rollout-checks)
2. [Enablement Steps](#enablement-steps)
3. [Validation Tests](#validation-tests)
4. [Monitoring Requirements](#monitoring-requirements)
5. [Rollback Procedures](#rollback-procedures)
6. [Post-Rollout Verification](#post-rollout-verification)
7. [Emergency Contacts](#emergency-contacts)

---

## Pre-Rollout Checks

### Environment Preparation

- [ ] **Backup Current Configuration**
  ```bash
  # Backup settings
  cp ~/.victor/profiles.yaml ~/.victor/profiles.yaml.backup.$(date +%Y%m%d)
  cp .victor/init.md .victor/init.md.backup.$(date +%Y%m%d)  # if exists
  ```

- [ ] **Document Current State**
  - Current Victor version: `victor --version`
  - Current feature flag status: `grep use_coordinator_orchestrator ~/.victor/profiles.yaml`
  - Known issues or workarounds in use

- [ ] **Create Rollback Plan**
  - Identify specific commit/version to rollback to: `__________`
  - Document rollback steps in this checklist
  - Set timeline for rollback decision (e.g., 24 hours)

### Testing Requirements

- [ ] **Run Unit Tests**
  ```bash
  pytest tests/unit/agent/test_orchestrator.py -v
  pytest tests/unit/agent/coordinators/ -v
  ```
  Expected: All tests pass (85%+ coverage)

- [ ] **Run Integration Tests**
  ```bash
  pytest tests/integration/ -m "not slow" -v
  ```
  Expected: All integration tests pass

- [ ] **Run Performance Benchmarks**
  ```bash
  pytest tests/benchmark/test_orchestrator_refactoring_performance.py -v
  ```
  Expected: Coordinator overhead < 10% (currently 3-5%)

- [ ] **Test All Coordinators**
  ```bash
  python scripts/validate_coordinator_orchestrator.py --all-coordinators
  ```
  Expected: All 15 coordinators validate successfully

### Dependency Verification

- [ ] **Check Dependencies**
  ```bash
  pip list | grep -E "(pydantic|yaml|anthropic|openai)"
  ```
  Required: pydantic>=2.0, pydantic-settings, pyyaml

- [ ] **Verify Provider Support**
  ```bash
  victor providers list
  ```
  Expected: All configured providers available

- [ ] **Check Disk Space**
  ```bash
  df -h ~/.victor
  ```
  Required: At least 1GB free for logs and cache

---

## Enablement Steps

### Step 1: Enable Feature Flag

**Option A: Via Environment Variable (Recommended for Testing)**
```bash
export VICTOR_USE_COORDINATOR_ORCHESTRATOR=true
victor chat --no-tui
```

**Option B: Via profiles.yaml (Recommended for Production)**
```yaml
# ~/.victor/profiles.yaml
profiles:
  default:
    provider: anthropic
    model: claude-sonnet-4-5

# Add at top level
use_coordinator_orchestrator: true
```

**Option C: Via Toggle Script (Automated)**
```bash
python scripts/toggle_coordinator_orchestrator.py enable --backup
```

### Step 2: Validate Configuration

```bash
# Verify feature flag is enabled
python -c "from victor.config.settings import Settings; s = Settings(); print(f'Coordinators enabled: {s.use_coordinator_orchestrator}')"

# Run full validation
python scripts/validate_coordinator_orchestrator.py --full-report
```

Expected output:
```
✓ Feature flag enabled
✓ All coordinators loaded
✓ Configuration valid
✓ Performance baseline established
```

### Step 3: Start Victor with Coordinators

```bash
# Test in non-TUI mode first
victor chat --no-tui

# If successful, start TUI
victor chat
```

### Step 4: Run Smoke Tests

Test basic functionality:
```python
# In Victor CLI
1. "Hello" - Test basic chat
2. "Read victor/agent/orchestrator.py" - Test file reading
3. "Search for 'coordinator' in victor/agent/coordinators/" - Test code search
4. "List files in victor/tools/" - Test directory listing
```

Expected: All commands work without errors

---

## Validation Tests

### Automated Validation

```bash
# Run comprehensive validation
python scripts/validate_coordinator_orchestrator.py \
  --full-report \
  --output validation_report_$(date +%Y%m%d).json
```

This validates:
- ✓ All 15 coordinators initialize correctly
- ✓ Coordinator interactions work
- ✓ Backward compatibility maintained
- ✓ Performance metrics within threshold
- ✓ Memory usage acceptable
- ✓ No errors in logs

### Manual Validation Checklist

- [ ] **Chat Operations**
  - [ ] Basic chat request/response
  - [ ] Multi-turn conversation
  - [ ] Tool calling (file operations, code search)
  - [ ] Streaming responses
  - [ ] Error handling

- [ ] **Tool Operations**
  - [ ] Code search (keyword and semantic)
  - [ ] File read/write operations
  - [ ] Directory navigation
  - [ ] Command execution (if enabled)
  - [ ] Tool deduplication works

- [ ] **Session Management**
  - [ ] Session creation and persistence
  - [ ] Context window management
  - [ ] Conversation history
  - [ ] Checkpoint save/restore

- [ ] **Provider Operations**
  - [ ] Provider switching
  - [ ] Multiple providers in same session
  - [ ] Fallback on provider failure
  - [ ] Circuit breaker activation

- [ ] **Analytics and Metrics**
  - [ ] Metrics collection works
  - [ ] Analytics tracking enabled
  - [ ] Performance logs written
  - [ ] No data loss

### Performance Validation

```bash
# Run performance benchmarks
pytest tests/benchmark/test_orchestrator_refactoring_performance.py \
  --benchmark-json=performance_$(date +%Y%m%d).json \
  -v
```

**Acceptance Criteria**:
- Chat latency: < 10% increase vs legacy
- Tool execution: No change
- Memory overhead: < 100MB additional
- Coordinator initialization: < 500ms total

---

## Monitoring Requirements

### Key Metrics to Monitor

#### 1. Performance Metrics

| Metric | Threshold | Alert Level | Action |
|--------|-----------|-------------|--------|
| Chat latency increase | < 10% | Warning | Optimize coordinator |
| Coordinator init time | < 500ms | Warning | Profile slow coordinators |
| Memory overhead | < 100MB | Critical | Restart, investigate |
| Tool execution time | No change | Warning | Check coordinator overhead |

#### 2. Error Metrics

| Metric | Threshold | Alert Level | Action |
|--------|-----------|-------------|--------|
| Coordinator init failures | 0 | Critical | Rollback immediately |
| Tool execution errors | < 1% | Warning | Check tool coordinator |
| Provider switch failures | < 0.5% | Warning | Check provider coordinator |
| Session corruption | 0 | Critical | Rollback immediately |

#### 3. Functional Metrics

| Metric | Threshold | Alert Level | Action |
|--------|-----------|-------------|--------|
| Backward compatibility issues | 0 | Critical | Rollback immediately |
| Feature parity vs legacy | 100% | Warning | Document gap |
| Test coverage | > 85% | Warning | Improve tests |
| Coordinator interactions | All tested | Warning | Add integration tests |

### Monitoring Setup

```bash
# Enable observability logging
export VICTOR_ENABLE_OBSERVABILITY_LOGGING=true
export VICTOR_OBSERVABILITY_LOG_PATH=~/.victor/metrics/victor_coordinator_rollout.jsonl

# Start Victor
victor chat

# In another terminal, monitor logs
tail -f ~/.victor/metrics/victor_coordinator_rollout.jsonl | jq .
```

### Alert Configuration

**Critical Alerts** (Immediate Action Required):
- Coordinator initialization failures
- Session corruption
- Memory leaks (memory usage growing unbounded)
- Complete service failure

**Warning Alerts** (Investigate within 24 hours):
- Performance degradation > 10%
- Increased error rates
- Feature parity gaps discovered

---

## Rollback Procedures

### When to Rollback

**Immediate Rollback (Critical Issues)**:
- Coordinator initialization fails
- Data corruption or loss
- Security vulnerability discovered
- Complete service failure

**Scheduled Rollback (Warning Issues)**:
- Performance degradation > 15%
- Feature parity gaps blocking users
- Unacceptable error rates (> 5%)

### Rollback Steps

#### Option 1: Disable Feature Flag (Fastest)

```bash
# Via toggle script
python scripts/toggle_coordinator_orchestrator.py disable --restart

# Or manually
# Edit ~/.victor/profiles.yaml
# Set: use_coordinator_orchestrator: false

# Or via environment
export VICTOR_USE_COORDINATOR_ORCHESTRATOR=false
```

#### Option 2: Revert to Previous Version

```bash
# Install previous version
pip install victor-ai==0.5.0

# Or revert to specific commit
git checkout <previous-commit-tag>
pip install -e .
```

#### Option 3: Restore from Backup

```bash
# Restore settings backup
cp ~/.victor/profiles.yaml.backup.YYYYMMDD ~/.victor/profiles.yaml

# Restore context file
cp .victor/init.md.backup.YYYYMMDD .victor/init.md

# Restart Victor
```

### Rollback Verification

After rollback, verify:

```bash
# 1. Check feature flag is disabled
python -c "from victor.config.settings import Settings; s = Settings(); print(f'Coordinators: {s.use_coordinator_orchestrator}')"
# Expected: Coordinators: False

# 2. Run smoke tests
victor chat --no-tui
# Test: "Hello", file operations, code search

# 3. Check logs for errors
tail -100 ~/.victor/logs/victor.log
# Expected: No coordinator-related errors

# 4. Run tests
pytest tests/unit/agent/test_orchestrator.py -v
# Expected: All pass
```

### Rollback Documentation

After rollback, document:

- [ ] Reason for rollback
- [ ] Issues encountered
- [ ] Timestamp of rollback
- [ ] Users affected
- [ ] Root cause analysis
- [ ] Fix plan

---

## Post-Rollout Verification

### Day 1 Checks (First 24 Hours)

- [ ] **Monitor Error Rates**
  ```bash
  grep -i error ~/.victor/logs/victor.log | wc -l
  ```
  Expected: No increase in errors

- [ ] **Check Performance**
  ```bash
  # Parse observability logs
  cat ~/.victor/metrics/victor_coordinator_rollout.jsonl | \
    jq '.latency' | \
    awk '{sum+=$1; count++} END {print "Average:", sum/count "ms"}'
  ```
  Expected: < 10% increase vs baseline

- [ ] **User Feedback**
  - Survey internal users
  - Check GitHub issues
  - Review support tickets

- [ ] **Resource Usage**
  ```bash
  # Monitor memory
  ps aux | grep victor | awk '{print $6}'
  # Expected: < 100MB increase vs baseline
  ```

### Day 7 Checks (First Week)

- [ ] **Weekly Performance Review**
  - Compare metrics to pre-rollout baseline
  - Identify trends (improving or degrading)

- [ ] **Stability Assessment**
  - Number of crashes/failures
  - Number of rollbacks (should be 0)

- [ ] **Feature Parity Review**
  - Confirm all features work
  - Document any gaps found

### Day 30 Checks (First Month)

- [ ] **Monthly Performance Report**
  - Aggregate performance metrics
  - Compare to legacy orchestrator

- [ ] **Stability Metrics**
  - Uptime percentage
  - Mean time between failures

- [ ] **Decision Point**
  - Keep coordinator orchestrator enabled?
  - Proceed with cleanup (remove legacy code)?
  - Need additional tuning?

---

## Emergency Contacts

### Primary Contacts

| Role | Name | Contact | Timezone |
|------|------|---------|----------|
| Tech Lead | Vijay Singh | singhvjd@gmail.com | PST |
| Architect | [Name] | [email] | [timezone] |
| DevOps Lead | [Name] | [email] | [timezone] |

### Escalation Path

1. **Level 1: Issue Detected**
   - Document in GitHub issue
   - Tag relevant team members
   - Assess severity

2. **Level 2: Warning (> 1 hour unresolved)**
   - Notify tech lead
   - Begin investigation
   - Prepare rollback if needed

3. **Level 3: Critical (Immediate)**
   - Execute rollback
   - Notify all stakeholders
   - Post-mortem within 24 hours

### Communication Channels

- Slack: #victor-ops (if available)
- Email: singhvjd@gmail.com
- GitHub Issues: https://github.com/your-org/victor/issues

---

## Appendix

### Useful Commands

```bash
# Check coordinator status
python -c "from victor.config.settings import Settings; s = Settings(); print(s.use_coordinator_orchestrator)"

# List all coordinators
python -c "from victor.agent.coordinators import __all__; print(__all__)"

# View coordinator logs
tail -f ~/.victor/logs/victor.log | grep -i coordinator

# Run validation
python scripts/validate_coordinator_orchestrator.py --quick

# Toggle feature flag
python scripts/toggle_coordinator_orchestrator.py status

# Performance benchmark
pytest tests/benchmark/test_orchestrator_refactoring_performance.py -v

# Check test coverage
pytest --cov=victor.agent.coordinators --cov-report=html
```

### File Locations

| File | Location | Purpose |
|------|----------|---------|
| Settings | `~/.victor/profiles.yaml` | Feature flag configuration |
| Logs | `~/.victor/logs/victor.log` | Application logs |
| Metrics | `~/.victor/metrics/victor.jsonl` | Observability data |
| Validation Report | `validation_report_YYYYMMDD.json` | Validation results |
| Performance Baseline | `performance_YYYYMMDD.json` | Benchmark results |

### Related Documentation

- [Migration Guide](../migration/orchestrator_refactoring_guide.md)
- [Settings Guide](coordinator_settings.md)
- [Monitoring Guide](coordinator_monitoring.md)
- [Architecture Overview](../architecture/coordinator_based_architecture.md)

---

**Checklist Version**: 1.0
**Last Updated**: 2025-01-14
**Next Review**: 2025-02-14

**Sign-off**:

- [ ] Pre-rollout checks completed: ____________
- [ ] Enablement steps executed: ____________
- [ ] Validation tests passed: ____________
- [ ] Monitoring configured: ____________
- [ ] Post-rollout verification scheduled: ____________

---

## Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-01-14 | 1.0 | Initial creation | Vijay Singh |
