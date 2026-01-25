# Testing Documentation

**Victor AI Coding Assistant - Test Coverage & Quality**

Welcome to the Victor testing documentation. This directory contains comprehensive information about test coverage, quality metrics, and improvement plans.

> **Note**: Coverage and test counts are time-bound snapshots. Verify against current reports before making decisions.

## Quick Links

- **[Test Coverage Report](test_coverage_report.md)** - Comprehensive coverage analysis with module-by-module breakdown
- **[Test Coverage Report](test_coverage_report.md)** - Visual coverage summary with heatmaps and priorities
- **[Test Improvement Plan](test_improvement_plan.md)** - Detailed actionable plan for improving coverage
- **[Coverage Trends](coverage_trends.md)** - Historical data and velocity metrics

## Quick Stats

### Current Coverage (as of 2026-01-14)

```
Overall Coverage:  11% (20,171 / 182,186 lines)
Total Tests:       ~21,496 tests
  - Unit Tests:    ~20,083 (93.5%)
  - Integration:   ~1,413 (6.5%)
  - Security:      75 tests
Pass Rate:         ~95%
Failing Tests:     7 integration tests
```

### Target Coverage (by 2026-03-31)

```
Overall Coverage:  80% (145,749 / 182,186 lines)
Total Tests:       ~25,000+ tests
  - Unit Tests:    ~23,000+
  - Integration:   ~2,000+
  - E2E Tests:     25+
  - Performance:   15+
Pass Rate:         98%+
Failing Tests:     0
```

## Coverage by Layer

```
Agent Layer:        11% (CRITICAL - Needs immediate attention)
  - orchestrator.py:        45% (3,298 missing lines)
  - coordinators/:          82% (average - GOOD)
  - action_authorizer.py:   0%  (114 missing lines - CRITICAL)

Framework Layer:     20% (HIGH PRIORITY)
  - unified_compiler.py:    14% (553 missing lines)
  - executor.py:            19% (436 missing lines)
  - handlers.py:            20% (272 missing lines)

Protocol Layer:      95% (EXCELLENT)
Core Layer:          70% (GOOD)
Tool Layer:          35% (MEDIUM)
Workflow Layer:      25% (HIGH PRIORITY)
Provider Layer:      15% (MEDIUM)
UI Layer:            0%  (LOW PRIORITY - expected)
```

## Top 5 Priority Gaps

1. **victor.agent.orchestrator.py** - 45% coverage (CRITICAL)
   - Impact: Core application logic
   - Effort: 5-7 days
   - Target: 60% coverage

2. **victor.framework.unified_compiler.py** - 14% coverage (HIGH)
   - Impact: Workflow compilation
   - Effort: 5-7 days
   - Target: 50% coverage

3. **victor.framework.executor.py** - 19% coverage (HIGH)
   - Impact: Workflow execution
   - Effort: 5-7 days
   - Target: 50% coverage

4. **victor.agent.action_authorizer.py** - 0% coverage (CRITICAL)
   - Impact: Security
   - Effort: 2-3 days
   - Target: 70% coverage

5. **victor.framework.handlers.py** - 20% coverage (HIGH)
   - Impact: Workflow handlers
   - Effort: 3-4 days
   - Target: 60% coverage

## Immediate Action Items

### Week 1-2 (Jan 14-24): Critical Fixes

- [ ] **Fix 7 Failing Integration Tests** (2-3 days)
  - test_tool_execution_tracking
  - test_context_budget_checking
  - test_compaction_execution
  - test_analytics_coordinator_data_collection
  - test_prompt_coordinator_building
  - test_chat_event_tracked
  - test_analytics_export_functionality

- [ ] **Increase Orchestrator Coverage to 60%** (5-7 days)
  - Add 40 new test cases
  - Cover chat flow, tool execution, error handling
  - Cover session management, configuration

- [ ] **Add Action Authorizer Tests** (2-3 days)
  - Add 20 new test cases
  - Target 70% coverage
  - Focus on security-critical paths

### Week 3-4 (Jan 25-Feb 7): High Impact Modules

- [ ] **Improve Workflow Compiler Coverage to 50%** (5-7 days)
- [ ] **Improve Workflow Executor Coverage to 50%** (5-7 days)
- [ ] **Add Tool Validation Tests to 70%** (2-3 days)
- [ ] **Add Provider Module Tests to 60%** (5-7 days)

## Running Tests

### Run All Tests

```bash
# Run all tests with coverage
pytest tests/ --cov=victor --cov-report=html --cov-report=term

# Open HTML coverage report
open htmlcov/index.html
```

### Run Specific Test Types

```bash
# Unit tests only
pytest tests/unit/ --cov=victor --cov-report=term

# Integration tests only
pytest tests/integration/ --cov=victor --cov-report=term

# Security tests only
pytest tests/unit/security/ --cov=victor --cov-report=term

# Coordinator tests only
pytest tests/unit/agent/coordinators/ --cov=victor.agent.coordinators --cov-report=term
```

### Run with Filters

```bash
# Run fast tests only (skip slow)
pytest -m "not slow" --cov=victor --cov-report=term

# Run integration tests only
pytest -m integration --cov=victor --cov-report=term

# Run specific test file
pytest tests/unit/agent/test_orchestrator.py --cov=victor.agent.orchestrator --cov-report=term

# Run specific test
pytest tests/unit/agent/test_orchestrator.py::test_chat_flow --cov=victor.agent.orchestrator --cov-report=term
```

### Generate Coverage Reports

```bash
# HTML report (detailed)
pytest --cov=victor --cov-report=html
open htmlcov/index.html

# JSON report (for CI/CD)
pytest --cov=victor --cov-report=json

# Terminal report (summary)
pytest --cov=victor --cov-report=term

# Combined reports
pytest --cov=victor --cov-report=html --cov-report=json --cov-report=term
```

## Test Organization

```
tests/
├── unit/ (661 files)
│   ├── agent/
│   │   ├── cache/              # Cache system tests
│   │   ├── coordinators/       # Coordinator tests
│   │   ├── orchestrator.py     # Orchestrator tests
│   │   └── ...
│   ├── framework/              # Framework tests
│   ├── core/                   # Core system tests
│   ├── providers/              # Provider tests
│   ├── tools/                  # Tool tests
│   └── security/               # Security tests
└── integration/ (81 files)
    ├── agent/                  # Agent integration tests
    │   ├── test_orchestrator_integration.py
    │   ├── test_consolidation_integration.py
    │   ├── test_cli_session_integration.py
    │   └── ...
    └── ...
```

## Test Markers

```python
@pytest.mark.unit           # Unit tests
@pytest.mark.integration    # Integration tests
@pytest.mark.slow           # Slow tests (can be skipped)
@pytest.mark.workflows      # Workflow tests
@pytest.mark.agents         # Agent tests
@pytest.mark.hitl           # Human-in-the-loop tests
```

## Coverage Targets

### By Priority

- **Critical Modules** (security, core): 80%+
- **High Priority** (workflow, framework): 70%+
- **Medium Priority** (tools, providers): 60%+
- **Low Priority** (UI, optional): 50%+

### By Layer

- **Protocol Layer**: 95%+ ✅ (ACHIEVED)
- **Core Layer**: 80%+
- **Agent Layer**: 70%+
- **Framework Layer**: 60%+
- **Workflow Layer**: 60%+
- **Tool Layer**: 60%+
- **Provider Layer**: 60%+
- **UI Layer**: 40% (lower priority)

## Quality Metrics

### Pass Rate Targets

- **Current**: 95%
- **Target (Feb 14)**: 97%
- **Target (Mar 31)**: 98%+

### Flaky Test Rate Targets

- **Current**: 0.14% (30 / 21,496)
- **Target**: <2%
- **Stretch Goal**: <0.1%

### Test Execution Time Targets

- **Current**: 35 min
- **Target (Feb 14)**: 45 min
- **Target (Mar 31)**: 60 min
- **Note**: Will increase as more tests added

## Coverage Badges

Add to README.md:

```markdown
![Coverage](https://img.shields.io/badge/coverage-11%25-red.svg)
![Tests](https://img.shields.io/badge/tests-21,496-blue.svg)
![Pass Rate](https://img.shields.io/badge/pass_rate-95%25-brightgreen.svg)
```

Target badges (end of Q1):

```markdown
![Coverage](https://img.shields.io/badge/coverage-80%25-brightgreen.svg)
![Tests](https://img.shields.io/badge/tests-25,000-blue.svg)
![Pass Rate](https://img.shields.io/badge/pass_rate-98%25-brightgreen.svg)
```

## CI/CD Integration

### GitHub Actions Workflow

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Run tests with coverage
        run: |
          pytest tests/ --cov=victor --cov-report=xml --cov-report=term

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
          flags: unittests
          fail_ci_if_error: false
```

## Resources

### Documentation

- [Testing Guide](TESTING_GUIDE.md) - Overall testing approach
- [Development Testing Guide](../contributing/testing.md) - How to write tests
- [Pytest Documentation](https://docs.pytest.org/) - Test framework docs
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/) - Coverage plugin docs

### Tools

- **pytest**: Test framework
- **pytest-cov**: Coverage plugin
- **pytest-asyncio**: Async test support
- **pytest-mock**: Mocking support
- **respx**: HTTP mocking
- **pytest-benchmark**: Performance testing (to be added)
- **hypothesis**: Property-based testing (to be added)

### Best Practices

1. **Write Tests First**: Test-driven development (TDD)
2. **Test Isolation**: Each test should be independent
3. **Clear Naming**: Test names should describe what they test
4. **Arrange-Act-Assert**: Structure tests clearly
5. **Mock External Dependencies**: Use mocks for external services
6. **Test Edge Cases**: Don't just test happy paths
7. **Keep Tests Fast**: Optimize slow tests or mark as `@pytest.mark.slow`
8. **Review Test Coverage**: Regularly review coverage reports
9. **Fix Flaky Tests**: Address flaky tests immediately
10. **Document Complex Tests**: Add comments for complex test logic

## Getting Help

### Questions About Testing?

- Check the [Testing Guide](TESTING_GUIDE.md)
- Review existing tests for examples
- Ask in team chat/Slack
- Create an issue for bugs or improvements

### Report Coverage Issues

If you find coverage gaps or issues:
1. Check if it's already documented in [Test Coverage Report](test_coverage_report.md)
2. If not, create an issue with:
   - Module name and path
   - Current coverage percentage
   - Lines missing (if known)
   - Priority (CRITICAL/HIGH/MEDIUM/LOW)
   - Reason for priority

### Contribute Tests

We welcome contributions! See:
1. [Development Testing Guide](../contributing/testing.md)
2. [Test Improvement Plan](test_improvement_plan.md)
3. Existing test files for examples

## Summary

Victor has a solid test foundation with ~21,496 tests and 11% coverage. The immediate focus is on:

1. **Fixing 7 failing integration tests**
2. **Increasing critical module coverage** (orchestrator, action authorizer)
3. **Improving workflow framework coverage** (compiler, executor)
4. **Adding comprehensive integration tests**
5. **Establishing sustainable testing practices**

With consistent effort and focus on high-value coverage, we aim to achieve 80% coverage by the end of Q1 2026.

---

**Last Updated**: 2026-01-14
**Next Review**: 2026-01-21
**Maintained By**: Victor Development Team
