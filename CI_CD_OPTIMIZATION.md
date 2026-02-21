# Victor CI/CD Optimization Guide

## Overview

This document describes the optimized CI/CD pipeline for Victor. The new pipeline reduces total CI time from ~25 minutes to ~8 minutes through parallelization, caching, and smart path filtering.

## Workflow Architecture

### Main Workflows

| Workflow | Purpose | Duration | Triggers |
|----------|---------|----------|----------|
| `ci-fast.yml` | Quick checks (format, lint, typecheck) | ~1-2 min | All commits |
| `ci-test.yml` | Parallel test matrix with sharding | ~4 min | All commits |
| `ci-integration.yml` | Integration & VS Code tests | ~5 min | Path-triggered |
| `security.yml` | Security scanning | ~3 min | Scheduled + PRs |
| `validation.yml` | FEP & Vertical validation | ~2 min | Path-triggered |
| `build.yml` | Build all artifacts | ~5 min | Push + manual |
| `ci.yml` | Main orchestration | N/A | All commits |

### Workflow Dependencies

```
ci.yml (Main Orchestrator)
├── ci-fast.yml (parallel)
│   ├── format (black)
│   ├── lint (ruff)
│   ├── typecheck (mypy)
│   ├── imports
│   ├── guards
│   └── security-quick
├── ci-test.yml (parallel with ci-fast)
│   ├── test matrix (3 Python versions × 4 shards)
│   └── smoke-test
├── ci-integration.yml (path-triggered)
│   ├── integration tests
│   ├── vscode-extension
│   └── rust-native
├── security.yml (parallel)
│   ├── secret-scan
│   ├── dependency-audit
│   ├── code-security
│   ├── sast-scan
│   └── license-scan
├── validation.yml (path-triggered)
│   ├── fep-validation
│   └── vertical-validation
└── build.yml (after tests pass)
    ├── build-python
    ├── build-docker
    └── build-vscode
```

## Key Optimizations

### 1. Parallel Execution

All checks run in parallel where possible:

```yaml
# ci-fast.yml runs 6 jobs in parallel
jobs:
  format:    # Black check
  lint:      # Ruff check
  typecheck: # MyPy check
  imports:   # Import validation
  guards:    # CI guards
  security-quick:  # Trivy scan
```

### 2. Test Sharding

Tests are split across multiple shards for faster execution:

```yaml
# ci-test.yml
strategy:
  matrix:
    python-version: ["3.10", "3.11", "3.12"]
    shard: [1, 2, 3, 4]  # 4 shards per Python version
```

This creates 12 parallel test jobs (3 Python versions × 4 shards).

### 3. Smart Caching

Multi-layer caching strategy:

```yaml
# Pip dependencies cache
- name: Cache pip dependencies
  uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/pyproject.toml') }}

# Pytest cache
- name: Cache pytest cache
  uses: actions/cache@v4
  with:
    path: .pytest_cache
    key: ${{ runner.os }}-pytest-${{ github.run_id }}

# Docker layer cache
- name: Cache Docker layers
  uses: actions/cache@v4
  with:
    path: /tmp/.buildx-cache
    key: ${{ runner.os }}-buildx-${{ github.sha }}
```

### 4. Path-Aware Execution

Workflows only run when relevant files change:

```yaml
# ci-integration.yml only runs when these paths change
on:
  push:
    paths:
      - 'victor/**'
      - 'tests/integration/**'
      - 'tests/e2e/**'
```

### 5. Fail-Fast Strategy

Quick checks run first to catch common issues early:

```yaml
# ci-fast.yml completes in ~1-2 minutes
# If it fails, ci-test.yml is cancelled automatically
```

## Local Development

### Pre-commit Hooks

Install pre-commit hooks for local validation:

```bash
make install-dev  # Installs pre-commit
make pre-commit   # Run all hooks manually
```

This catches issues before pushing to CI.

### Test Parallelization

Run tests locally with the same parallelization as CI:

```bash
# Run tests in parallel
make test

# Run specific test shard
make test-split

# Run quick tests (skip slow ones)
make test-quick
```

## CI Workflow Usage

### Running CI Locally

```bash
# Run fast checks
make lint
make format-check

# Run tests
make test
make test-cov

# Run all checks
make pre-commit
```

### Triggering CI Workflows

```yaml
# Trigger on commit
git push origin main

# Trigger build workflow with commit message
git commit -m "[build] Add new feature"

# Trigger VS Code tests
git commit -m "[vscode] Fix extension bug"

# Manual trigger via GitHub UI
# Actions → Select workflow → Run workflow
```

## Performance Metrics

### Before Optimization

| Component | Duration |
|-----------|----------|
| Lint | 3 min |
| Tests (serial) | 15 min (3 versions × 5 min) |
| Security | 5 min |
| Guards | 2 min |
| **Total** | **~25 min** |

### After Optimization

| Component | Duration |
|-----------|----------|
| Fast checks (parallel) | 1-2 min |
| Tests (sharded, parallel) | 4 min |
| Security (parallel) | 3 min |
| **Total (parallel)** | **~8 min** |

**Improvement: 68% reduction in CI time**

## Monitoring CI Performance

### GitHub Actions Dashboard

1. Go to Actions tab in GitHub
2. Click on any workflow run
3. View timing for each job

### Identifying Bottlenecks

```bash
# Check workflow timing
gh run view <run-id>

# List recent runs with timing
gh run list --workflow=CI-Tests.yml
```

## Troubleshooting

### Common Issues

**Issue: Tests flaky on CI**
```yaml
# Increase timeout in .github/workflows/ci-test.yml
timeout-minutes: 20
```

**Issue: Docker build fails**
```yaml
# Clear cache manually
# Actions → Build Artifacts → Clear cache
```

**Issue: Pre-commit hooks fail**
```bash
# Update hooks
pre-commit autoupdate
pre-commit run --all-files
```

### Debug Mode

Enable debug logging:

```yaml
env:
  ACTIONS_STEP_DEBUG: true
  ACTIONS_RUNNER_DEBUG: true
```

## Best Practices

1. **Run pre-commit hooks locally** - Catches issues before CI
2. **Use path triggers** - Only run relevant workflows
3. **Keep caches warm** - Don't clear cache unless necessary
4. **Monitor CI times** - Watch for regressions
5. **Update dependencies** - Keep actions and tools updated

## Future Improvements

- [ ] Add test timing dashboard
- [ ] Implement smart test selection (based on changed files)
- [ ] Add performance regression detection
- [ ] Set up CI caching on self-hosted runners
- [ ] Add pytest-benchmark for performance tests
- [ ] Implement test result caching
