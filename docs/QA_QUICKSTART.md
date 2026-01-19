# QA Framework Quick Start Guide

**Get started with Victor AI quality assurance in 5 minutes**

---

## Installation

Ensure you have development dependencies installed:

```bash
# Install Victor AI with dev dependencies
pip install -e ".[dev]"

# Verify installation
victor --version
```

---

## 30-Second QA Check

The fastest way to validate code quality:

```bash
make qa-fast
```

This runs:
- ✅ Unit tests (fast mode)
- ✅ Ruff linting
- ✅ Black formatting check
- ✅ Release readiness validation

**Expected output:**
```
Running quick QA validation...
==============================
[Progress indicators...]
✓ Unit Tests (Fast): PASS
✓ Ruff Linting: PASS
✓ Black Formatting: PASS
✓ Release Readiness: PASS

Quick QA validation complete!
```

---

## 5-Minute Comprehensive QA

For thorough validation before commits or releases:

```bash
make qa
```

This runs:
- ✅ All unit tests
- ✅ Integration tests
- ✅ Code coverage analysis
- ✅ Ruff, Black, Mypy checks
- ✅ Security scans (Bandit, Safety)
- ✅ Documentation build
- ✅ Performance benchmarks

**Expected duration:** 5-10 minutes

---

## Understanding QA Results

### Text Output (Default)

```
======================================================================
QA SUMMARY
======================================================================

Total Duration: 342.50s
Total Checks: 12
Passed: 11
Failed: 1
Warnings: 0
Success Rate: 91.7%

======================================================================
DETAILED RESULTS
======================================================================

✓ Unit Tests
  Status: PASS
  Duration: 45.20s

✓ Integration Tests
  Status: PASS
  Duration: 120.30s

✓ Code Coverage
  Status: PASS
  Duration: 50.10s
  Coverage: 72.5%

✓ Ruff Linting
  Status: PASS
  Duration: 8.50s
  Errors: 20

⚠ Mypy Type Checking
  Status: WARN
  Duration: 15.30s
  Errors: 100
```

### JSON Output

```bash
make qa-report
```

Generates `qa_report.json`:

```json
{
  "summary": {
    "total_duration": 342.50,
    "total_checks": 12,
    "passed": 11,
    "failed": 1,
    "warned": 0,
    "success_rate": 91.7
  },
  "results": {
    "Unit Tests": {
      "status": "PASS",
      "duration": 45.20,
      "exit_code": 0
    }
  }
}
```

### HTML Output

```bash
python scripts/run_full_qa.py --report html --output qa_report.html
open qa_report.html
```

---

## Common Workflows

### 1. Before Committing Code

```bash
# Quick validation
make qa-fast

# Format code if needed
make format

# Run tests for modified files
pytest tests/unit/ -v -k "test_name_pattern"
```

### 2. Before Creating Pull Request

```bash
# Comprehensive validation
make qa

# Generate coverage report
make test-cov
open htmlcov/index.html

# Check for security issues
make security-full
```

### 3. Before Release

```bash
# Complete validation
make release-validate

# View checklist
make release-checklist

# Generate detailed report
python scripts/run_full_qa.py --coverage --report json --output pre_release_qa.json
```

### 4. Investigating Failures

```bash
# Run specific test with details
pytest tests/unit/test_specific.py::test_name -v --tb=long

# Check linting issues
ruff check victor/ --output-format=text

# Type check specific module
mypy victor/agent/coordinators/ --no-error-summary
```

---

## Interpreting Results

### ✅ PASS (Green)

All checks passed. No action required.

### ⚠ WARN (Yellow)

Non-critical issues found. Review but not blocking:

- **Ruff warnings:** Style issues, not bugs
- **Mypy errors:** Type hints incomplete (cosmetic)
- **Coverage < 70%:** Consider adding tests

### ✗ FAIL (Red)

Critical issues that MUST be fixed:

- **Test failures:** Code is broken
- **Black formatting:** Code style inconsistent
- **Security issues:** Vulnerabilities present
- **Performance regression:** Slower than baseline

---

## Quick Reference

### Essential Commands

| Command | Duration | Purpose |
|---------|----------|---------|
| `make qa-fast` | 2-3 min | Quick validation |
| `make qa` | 5-10 min | Comprehensive QA |
| `make qa-report` | 2-3 min | Generate JSON report |
| `make test` | 1-2 min | Unit tests only |
| `make lint` | 30 sec | Code quality checks |
| `make security` | 1 min | Security scan |
| `make benchmark` | 2 min | Performance tests |

### Exit Codes

- `0` = All checks passed
- `1` = One or more checks failed

Use in scripts:
```bash
make qa-fast
if [ $? -eq 0 ]; then
    echo "QA passed, proceeding with deployment"
else
    echo "QA failed, aborting deployment"
    exit 1
fi
```

---

## Troubleshooting

### Issue: "Module not found" errors

**Solution:**
```bash
# Reinstall with dependencies
pip install -e ".[dev]"
```

### Issue: Tests timing out

**Solution:**
```bash
# Run tests in fast mode
make qa-fast

# Or skip specific slow tests
pytest tests/unit/ -v -m "not slow"
```

### Issue: Ruff formatting errors

**Solution:**
```bash
# Auto-fix formatting
make format
```

### Issue: Mypy errors

**Solution:**
```bash
# Check specific module only
mypy victor/agent/coordinators/ --no-error-summary

# Type errors are expected in non-critical modules
# Focus on strictly-typed core modules
```

### Issue: Security scan false positives

**Solution:**
```bash
# Review detailed reports
make security-full
cat security-reports/bandit-report.txt

# Add # nosec comments for validated false positives
```

---

## Next Steps

### Learn More

- **Full Documentation:** `docs/QA_FRAMEWORK.md`
- **Release Checklist:** `RELEASE_CHECKLIST.md`
- **Release Notes:** `docs/RELEASE_NOTES_0.5.1.md`

### Customize QA

- **Add custom checks:** Edit `tests/qa/test_comprehensive_qa.py`
- **Modify automation:** Edit `scripts/run_full_qa.py`
- **Update Makefile:** Add new targets

### Contribute

- **Report issues:** [GitHub Issues](https://github.com/vijayksingh/victor/issues)
- **Suggest improvements:** [GitHub Discussions](https://github.com/vijayksingh/victor/discussions)

---

## Tips and Tricks

### Speed Up QA

```bash
# Use fast mode for development
make qa-fast

# Run only specific test categories
pytest tests/unit/ -v -k "test_provider"

# Skip slow tests
pytest tests/ -v -m "not slow"
```

### Better Output

```bash
# JSON for parsing
make qa-report

# HTML for visualization
python scripts/run_full_qa.py --report html --output qa.html

# Verbose pytest output
pytest tests/unit/ -vv --tb=long
```

### Parallel Execution

```bash
# Run tests in parallel (requires pytest-xdist)
pip install pytest-xdist
pytest tests/unit/ -n auto
```

### Continuous Monitoring

```bash
# Watch mode for development
pip install pytest-watch
ptw tests/unit/ -v
```

---

## Best Practices

### ✅ DO

- Run `make qa-fast` before every commit
- Run `make qa` before creating PR
- Fix all test failures immediately
- Keep type errors minimal
- Monitor performance trends

### ❌ DON'T

- Skip QA validation for "small changes"
- Ignore test failures
- Commit without linting
- Skip security scans
- Release without full QA

---

**Quick Start Complete!** 🎉

You're ready to ensure quality with Victor AI's comprehensive QA framework.

For detailed information, see `docs/QA_FRAMEWORK.md`.

**Need Help?**
- GitHub: [github.com/vijayksingh/victor](https://github.com/vijayksingh/victor)
- Email: singhvjd@gmail.com
