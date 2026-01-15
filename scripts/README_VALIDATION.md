# Quick Start Guide - Production Validation Suite

This guide provides quick commands to run the validation suite for the coordinator-based orchestrator.

## Pre-Rollout Commands

### 1. Smoke Tests (Quick Check)
```bash
# Run all smoke tests (< 5 minutes)
pytest tests/smoke/test_coordinator_smoke.py -v

# Run with smoke marker
pytest tests/smoke/ -v -m smoke

# Run specific test class
pytest tests/smoke/test_coordinator_smoke.py::TestCoordinatorSmokeTests -v
```

### 2. Load Tests (Performance Validation)
```bash
# Basic load test (10 concurrent users, 60 seconds)
python scripts/load_test_coordinators.py

# High-load test (50 users, 2 minutes)
python scripts/load_test_coordinators.py --concurrent-users 50 --duration 120

# Custom configuration
python scripts/load_test_coordinators.py \
  --concurrent-users 20 \
  --duration 90 \
  --operations 200 \
  --think-time 50 \
  --output /tmp/load_test.json
```

### 3. Production Validation (Comprehensive)
```bash
# Run full validation suite
python scripts/final_production_validation.py

# Generate HTML and JSON reports
python scripts/final_production_validation.py \
  --output /tmp/validation_report.html \
  --json /tmp/validation_metrics.json

# View HTML report
open /tmp/validation_report.html  # macOS
xdg-open /tmp/validation_report.html  # Linux
```

### 4. Pre-Rollout Checklist
```bash
# View checklist
cat docs/production/pre_rollout_final_checklist.md

# Open in editor for completion
vim docs/production/pre_rollout_final_checklist.md
```

## Rollout Commands

### 1. Deploy
```bash
# Use your deployment script
./scripts/deploy-production.sh
```

### 2. Post-Rollout Verification (IMMEDIATE)
```bash
# Run immediately after deployment
python scripts/post_rollout_verification.py

# Generate reports
python scripts/post_rollout_verification.py \
  --output /tmp/health_report.html \
  --json /tmp/health_status.json

# View health report
open /tmp/health_report.html  # macOS
```

### 3. Monitor System
```bash
# Check logs
tail -f /var/log/victor/production.log

# Check metrics (if using Prometheus)
curl http://localhost:9090/metrics

# Check health endpoint
curl http://localhost:8000/health
```

## Rollback Commands (If Needed)

### 1. Quick Rollback
```bash
# Toggle back to old orchestrator
python scripts/toggle_coordinator_orchestrator.py --mode orchestrator

# Restart services
systemctl restart victor
```

### 2. Full Rollback
```bash
# Use your rollback procedure
./scripts/rollback-production.sh

# Verify rollback
python scripts/post_rollout_verification.py
```

## Common Workflows

### Workflow 1: Pre-Commit Quick Check
```bash
# Run smoke tests before committing
pytest tests/smoke/ -v -m smoke

# If tests pass, commit
git add .
git commit -m "feat: coordinator implementation"
```

### Workflow 2: Pre-Merge Full Validation
```bash
# Run all tests
pytest tests/unit -v
pytest tests/integration -v -m "not slow"
pytest tests/smoke/ -v -m smoke

# Run linting
ruff check victor tests
black --check victor tests
mypy victor

# If all pass, merge
git merge feature-branch
```

### Workflow 3: Staging Validation
```bash
# Deploy to staging
./scripts/deploy-staging.sh

# Run load tests in staging
python scripts/load_test_coordinators.py --concurrent-users 50 --duration 120

# Run production validation
python scripts/final_production_validation.py --output staging_validation.html

# Review results
open staging_validation.html
```

### Workflow 4: Production Rollout
```bash
# 1. Complete checklist
vim docs/production/pre_rollout_final_checklist.md

# 2. Deploy to production
./scripts/deploy-production.sh

# 3. Run post-rollout verification (IMMEDIATELY)
python scripts/post_rollout_verification.py

# 4. Monitor for first hour
watch -n 10 'curl -s http://localhost:8000/health | jq'

# 5. Run follow-up verification (after 1 hour)
python scripts/post_rollout_verification.py --output hour1_health.html
```

## Exit Codes Reference

All scripts use standard exit codes:
- `0`: Success
- `1`: Non-critical failures
- `2`: Critical failures

### Using Exit Codes in Scripts
```bash
# Run validation and check result
python scripts/final_production_validation.py
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo "All validations passed - proceeding with rollout"
elif [ $EXIT_CODE -eq 1 ]; then
  echo "Some validations failed - review and decide"
else
  echo "Critical failures - aborting rollout"
  exit 1
fi
```

## Report Locations

### Default Report Paths
- Production Validation: `/tmp/production_validation_report.html`
- Load Test: `/tmp/load_test_report.json`
- Post-Rollout: `/tmp/post_rollout_report.html`

### Custom Report Paths
```bash
# Custom output directory
REPORT_DIR=/tmp/victor_validation_$(date +%Y%m%d_%H%M%S)
mkdir -p $REPORT_DIR

# Generate all reports
python scripts/final_production_validation.py --output $REPORT_DIR/validation.html
python scripts/load_test_coordinators.py --output $REPORT_DIR/load_test.json
python scripts/post_rollout_verification.py --output $REPORT_DIR/health.html

# List all reports
ls -lh $REPORT_DIR/
```

## Troubleshooting Quick Reference

### Issue: Import Errors
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Try importing coordinator
python -c "from victor.teams import create_coordinator; print('OK')"
```

### Issue: Tests Failing
```bash
# Run with verbose output
pytest tests/smoke/ -vv -s

# Run specific test with details
pytest tests/smoke/test_coordinator_smoke.py::TestCoordinatorSmokeTests::test_checkpoint_coordinator_creation -vv
```

### Issue: Performance Degradation
```bash
# Run profiler
python -m cProfile -o profile.stats scripts/load_test_coordinators.py
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"
```

### Issue: High Memory Usage
```bash
# Check memory usage
python -c "import psutil; p = psutil.Process(); print(f'Memory: {p.memory_info().rss / 1024 / 1024:.1f} MB')"

# Run memory profiler
python -m memory_profiler scripts/load_test_coordinators.py
```

## CI/CD Integration Examples

### Pre-Commit Hook
```bash
# .git/hooks/pre-commit
#!/bin/bash
echo "Running smoke tests..."
pytest tests/smoke/ -v -m smoke
if [ $? -ne 0 ]; then
  echo "Smoke tests failed - commit aborted"
  exit 1
fi
```

### GitHub Actions Step
```yaml
- name: Run Production Validation
  run: |
    pip install -e ".[dev]"
    python scripts/final_production_validation.py --json report.json
    python scripts/post_rollout_verification.py --json health.json
```

### Jenkins Pipeline Stage
```groovy
stage('Validation') {
  steps {
    sh 'pip install -e ".[dev]"'
    sh 'pytest tests/smoke/ -v -m smoke'
    sh 'python scripts/final_production_validation.py'
  }
}
```

## Additional Resources

- Full documentation: `docs/production/VALIDATION_SUITE_SUMMARY.md`
- Pre-rollout checklist: `docs/production/pre_rollout_final_checklist.md`
- Architecture: `docs/architecture/coordinators.md`
- Migration guide: `docs/MIGRATION_GUIDE.md`

## Support

If you encounter issues:
1. Check the error message in the report
2. Review the troubleshooting section above
3. Consult the full documentation
4. Contact the on-call engineer

---

**Last Updated**: 2025-01-14
**Version**: 1.0
