# Workflow Validation in CI/CD

This document describes how Victor validates workflows in the CI/CD pipeline and how to fix validation failures.

## Overview

Victor uses a comprehensive workflow validation system to ensure all YAML workflows are syntactically correct,
  compile successfully, and pass integration tests. This validation happens at multiple levels:

1. **Pre-commit hooks** - Local validation before committing
2. **CI workflow validation** - Automated validation on pull requests
3. **Performance regression tests** - Detect performance degradation
4. **Integration tests** - Full workflow execution tests

## CI/CD Workflows

### Workflow Validation Workflow

**File**: `.github/workflows/workflow-validation.yml`

**Triggers**:
- Push to `main` or `develop` branches
- Pull requests targeting `main` or `develop`
- Changes to workflow YAML files
- Manual trigger (`workflow_dispatch`)

**Platforms**: Ubuntu, macOS, Windows

**Steps**:
1. Checkout code
2. Install Victor with dependencies
3. Run workflow validation in CI mode (`--ci` flag)
4. Validate all production workflows
5. Run integration tests
6. Security scan for sensitive data
7. Generate validation report
8. Post PR comment with results

**Exit Codes**:
- `0` - All validations passed
- `1` - One or more validations failed

### Performance Regression Workflow

**File**: `.github/workflows/workflow-performance.yml`

**Triggers**:
- Pull requests targeting `main` or `develop`
- Changes to workflow files or compiler code
- Manual trigger

**What It Tests**:
- Workflow load time (YAML parsing)
- Workflow compile time (graph compilation)
- Total time (load + compile)

**Regression Threshold**:
- Fails if >10% slower than baseline
- Reports improvements if >10% faster than baseline
- Acceptable range: ±10% of baseline

**Tested Workflows**:
- `victor/coding/workflows/feature.yaml`
- `victor/coding/workflows/team_node_example.yaml`
- `victor/research/workflows/deep_research.yaml`
- `victor/benchmark/workflows/swe_bench.yaml`

## Local Development Workflow

### Pre-commit Validation

Before committing changes, run the pre-commit hook:

```bash
# Install pre-commit hooks (one-time setup)
pre-commit install

# Make changes to workflow files
git add victor/coding/workflows/feature.yaml

# Commit (pre-commit will validate automatically)
git commit -m "Update feature workflow"
```

The pre-commit hook runs automatically and prevents commits with invalid workflows.

### Manual Validation

You can also validate workflows manually:

```bash
# Validate a single workflow
bash scripts/hooks/validate_workflows.sh victor/coding/workflows/feature.yaml

# Validate multiple workflows
bash scripts/hooks/validate_workflows.sh victor/*/workflows/*.yaml

# Validate in CI mode (generates JSON output)
bash scripts/hooks/validate_workflows.sh --ci victor/*/workflows/*.yaml

# Enable verbose output
VICTOR_VERBOSE_VALIDATION=1 bash scripts/hooks/validate_workflows.sh victor/coding/workflows/feature.yaml
```

## CI Mode

The validation script supports CI mode for automated testing:

```bash
# Run in CI mode
bash scripts/hooks/validate_workflows.sh --ci victor/*/workflows/*.yaml
```

**CI Mode Features**:
- JSON output (`validation-results.json`)
- Human-readable summary (`validation-summary.txt`)
- Sarif output for GitHub annotations (`validation-results.sarif`)
- Structured error messages
- Exit codes for automation

**Output Format**:

```json
{
  "version": "1.0",
  "validation_results": [
    {
      "file": "victor/coding/workflows/feature.yaml",
      "status": "passed",
      "message": "Validation successful",
      "errors": "",
      "timestamp": "2025-01-15T10:30:00Z"
    }
  ],
  "summary": {
    "total_files": 25,
    "total_errors": 0,
    "total_warnings": 0,
    "elapsed_time_seconds": 12.5,
    "validation_timestamp": "2025-01-15T10:30:12Z"
  }
}
```

## Common Validation Failures

### 1. YAML Syntax Errors

**Error**:
```
YAML syntax error: mapping values are not allowed here
  in "victor/coding/workflows/feature.yaml", line 15, column 3
```

**Cause**: Invalid YAML syntax (indentation, colons, quotes, etc.)

**Fix**:
```yaml
# Wrong
workflows:
  feature:
    nodes:
  - id: start  # Bad indentation

# Correct
workflows:
  feature:
    nodes:
      - id: start  # Correct indentation
```

### 2. Unknown Node Types

**Error**:
```
Unknown node type: 'invalid_type'
Valid types: agent, compute, condition, parallel, transform, hitl, team
```

**Cause**: Using an invalid node type in workflow YAML

**Fix**:
```yaml
# Wrong
nodes:
  - id: start
    type: invalid_type  # Not a valid type

# Correct
nodes:
  - id: start
    type: agent  # Valid type
```

### 3. Missing Node References

**Error**:
```
Node 'start' references non-existent node 'next_step'
```

**Cause**: A node references another node that doesn't exist

**Fix**:
```yaml
# Wrong
nodes:
  - id: start
    type: agent
    next: [next_step]  # next_step doesn't exist

# Correct
nodes:
  - id: start
    type: agent
    next: [analyze]  # analyze exists

  - id: analyze
    type: agent
    next: []
```

### 4. Compilation Errors

**Error**:
```
Workflow validation failed: Error compiling workflow 'feature'
```

**Cause**: Workflow fails to compile (usually due to complex dependencies)

**Fix**:
1. Check for circular dependencies
2. Verify all node IDs are unique
3. Ensure conditional nodes have valid branches
4. Check team node member configuration

### 5. Performance Regressions

**Error**:
```
⚠ PERFORMANCE REGRESSIONS DETECTED:
  victor/coding/workflows/feature.yaml
    Baseline: 0.1234s
    Current:  0.1456s
    Change:   +18.0% (+0.0222s)
```

**Cause**: Workflow compilation is slower than baseline

**Fix**:
1. Review workflow YAML changes for complexity
2. Check compiler code changes for inefficiencies
3. Optimize node parsing logic
4. Reduce unnecessary node dependencies

## Adding Workflow Tests to CI

### Option 1: Add to Production Workflows List

Edit `tests/integration/workflows/test_workflow_yaml_validation.py`:

```python
PRODUCTION_WORKFLOWS = {
    "coding": [
        # ... existing workflows ...
        "victor/coding/workflows/my_new_workflow.yaml",  # Add here
    ],
}
```

### Option 2: Create Custom Test

Add a new test in `tests/integration/workflows/`:

```python
# tests/integration/workflows/test_my_workflow.py
import pytest
from victor.workflows import load_workflow_from_file
from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

def test_my_workflow_compiles():
    """Test that my workflow compiles successfully."""
    compiler = UnifiedWorkflowCompiler()
    workflows = load_workflow_from_file("victor/coding/workflows/my_new_workflow.yaml")

    for name, definition in workflows.items():
        compiled = compiler.compile_definition(definition)
        assert compiled is not None
```

### Option 3: Add to Performance Benchmarks

Edit `.github/workflows/workflow-performance.yml`:

```yaml
- name: Run current benchmarks
  run: |
    python - <<'EOF'
    # Add your workflow to the list
    test_workflows = [
        "victor/coding/workflows/feature.yaml",
        "victor/coding/workflows/my_new_workflow.yaml",  # Add here
    ]
    EOF
```

## Validation Status Badge

Add to your README.md:

```markdown
![Workflow Validation](https://github.com/YOUR_USERNAME/victor/actions/workflows/workflow-validation.yml/badge.svg)
```

This shows the current status of workflow validation in CI.

## Debugging CI Failures

### Enable Debug Logging

```yaml
# In .github/workflows/workflow-validation.yml
- name: Run workflow validation (CI mode)
  env:
    VICTOR_VERBOSE_VALIDATION: 1  # Add this
  run: |
    bash scripts/hooks/validate_workflows.sh --ci victor/*/workflows/*.yaml
```

### Download Validation Artifacts

1. Go to the failed workflow run in GitHub Actions
2. Scroll to "Artifacts" section
3. Download `workflow-validation-results-*`
4. Extract and review `validation-results.json`

### Local Reproduction

```bash
# Reproduce CI failure locally
bash scripts/hooks/validate_workflows.sh --ci victor/*/workflows/*.yaml

# Check the output
cat validation-results.json
cat validation-summary.txt
```

## Performance Regression Investigation

### Identify the Cause

```bash
# Check what changed
git diff origin/main...HEAD -- victor/coding/workflows/feature.yaml

# Compare load times
python - <<'EOF'
from victor.workflows import load_workflow_from_file
import time

start = time.perf_counter()
load_workflow_from_file("victor/coding/workflows/feature.yaml")
elapsed = time.perf_counter() - start
print(f"Load time: {elapsed:.4f}s")
EOF
```

### Profile Workflow Compilation

```python
# Create a profiling script
import cProfile
import pstats
from victor.workflows import load_workflow_from_file
from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

# Load workflow
workflows = load_workflow_from_file("victor/coding/workflows/feature.yaml")
compiler = UnifiedWorkflowCompiler()

# Profile compilation
profiler = cProfile.Profile()
profiler.enable()

for name, definition in workflows.items():
    compiler.compile_definition(definition)

profiler.disable()

# Print stats
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

## Best Practices

### 1. Keep Workflows Simple

- Avoid excessive node nesting
- Minimize circular dependencies
- Use clear, descriptive node IDs
- Document complex workflows

### 2. Test Locally First

```bash
# Always validate before pushing
bash scripts/hooks/validate_workflows.sh victor/coding/workflows/my_workflow.yaml

# Run integration tests
pytest tests/integration/workflows/test_workflow_yaml_validation.py -v
```

### 3. Use Pre-commit Hooks

```bash
# Install hooks (one-time)
pre-commit install

# Update hooks periodically
pre-commit autoupdate
```

### 4. Monitor Performance

- Keep an eye on CI build times
- Review performance regression comments
- Optimize slow workflows
- Profile bottlenecks

### 5. Write Tests

- Add tests for new workflows
- Test edge cases (empty workflows, single node, etc.)
- Test error conditions
- Mock external dependencies

## Troubleshooting

### Issue: Pre-commit Hook Not Running

**Symptoms**: Invalid workflows are committed despite pre-commit hook

**Solutions**:
```bash
# Check if hooks are installed
ls .git/hooks/pre-commit

# Reinstall hooks
pre-commit install --install-hooks

# Run manually
pre-commit run validate-workflows --all-files
```

### Issue: CI Passes But Workflows Fail Locally

**Symptoms**: CI validation passes, but `victor workflow validate` fails

**Causes**:
- Different Python versions
- Missing dependencies locally
- Environment variable differences

**Solutions**:
```bash
# Match CI environment
python3.11 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Run same command as CI
bash scripts/hooks/validate_workflows.sh --ci victor/*/workflows/*.yaml
```

### Issue: Flaky Performance Tests

**Symptoms**: Performance regression test fails intermittently

**Causes**:
- CI server load variations
- Network timing differences
- Non-deterministic code

**Solutions**:
1. Increase regression threshold (10% → 15%)
2. Run performance tests multiple times
3. Use median instead of mean
4. Exclude noisy workflows

## Related Documentation

- [Workflow Development Guide](../guides/workflow-development/dsl.md)
- [Testing Guide](../testing/TESTING_GUIDE.md)
- [CI/CD Overview](./README.md)
- [Performance Optimization](../performance/optimization_guide.md)

## Support

If you encounter issues with workflow validation:

1. Check existing GitHub issues for similar problems
2. Search the documentation for error messages
3. Ask for help in the Victor community
4. Open a new issue with:
   - Error message and stack trace
   - Workflow YAML file
   - Steps to reproduce
   - Environment details (OS, Python version)

---

**Last Updated:** February 01, 2026
**Reading Time:** 4 minutes
