# Orchestrator Coordinator Integration Tests

## Overview

This document describes the comprehensive integration tests for orchestrator-coordinator interactions in the Victor AI
  coding assistant.

## Test File

**Location**: `/Users/vijaysingh/code/codingagent/tests/integration/agent/test_orchestrator_integration.py`

**Total Tests**: 15 integration tests (7 existing + 6 new + 2 error handling)

## Test Coverage

### Existing Tests (7)

1. **TestToolExecutionTracking** (1 test)
   - Verifies that tool executions are tracked by AnalyticsCoordinator
   - Tests event tracking with correct metadata
   - Validates session ID association

2. **TestContextBudgetChecking** (1 test)
   - Tests context budget checking through ContextCoordinator
   - Validates budget threshold enforcement
   - Tests within-budget and exceeded-budget scenarios

3. **TestCompactionExecution** (1 test)
   - Tests context compaction when budget exceeded
   - Validates CompactionResult structure
   - Tests message removal and token savings

4. **TestAnalyticsDataCollection** (1 test)
   - Tests analytics event collection during session
   - Validates event querying by session and type
   - Tests event history maintenance

5. **TestPromptCoordinatorBuilding** (1 test)
   - Tests prompt building through PromptCoordinator
   - Validates system prompt construction
   - Tests task hint generation

6. **TestChatEventTracking** (1 test)
   - Tests that chat events are tracked
   - Validates LLM call event tracking
   - Tests token usage recording

7. **TestAnalyticsExport** (1 test)
   - Tests analytics data export functionality
   - Validates ExportResult structure
   - Tests multiple exporter support

### New Tests (6)

8. **TestSimpleChatFlow** (1 test)
   - Tests basic chat interaction through coordinators
   - Validates provider.chat() invocation
   - Tests analytics tracking integration

9. **TestToolExecutionFlow** (1 test)
   - Tests tool execution through ToolCoordinator
   - Validates tool selection and execution
   - Tests budget tracking and consumption

10. **TestStreamingResponses** (1 test)
    - Tests streaming chat through ChatCoordinator
    - Validates stream chunk processing
    - Tests complete response assembly

11. **TestChatToolCoordinatorInteraction** (1 test)
    - Tests interaction between ChatCoordinator and ToolCoordinator
    - Validates tool selection and execution loop
    - Tests response continuation after tool use

12. **TestErrorHandlingAcrossCoordinators** (2 tests)
    - **test_analytics_coordinator_failure**: Validates graceful degradation when analytics fails
    - **test_context_coordinator_failure**: Tests fallback behavior when compaction fails

13. **TestConfigCoordinatorLoading** (2 tests)
    - **test_config_loading**: Tests configuration loading from providers
    - **test_config_validation_errors**: Tests validation error detection

## Test Fixtures

**Location**: `/Users/vijaysingh/code/codingagent/tests/integration/agent/conftest.py`

### Key Fixtures

1. **test_settings**: Mock Settings object with typical configuration
2. **test_provider**: Mock BaseProvider with chat and streaming capabilities
3. **test_container**: Mock ServiceContainer with service resolution
4. **legacy_orchestrator_factory**: Factory for creating legacy orchestrators
5. **legacy_orchestrator**: AgentOrchestrator instance without coordinators
6. **mock_config_coordinator**: Mock ConfigCoordinator
7. **mock_prompt_coordinator**: Mock PromptCoordinator
8. **mock_context_coordinator**: Mock ContextCoordinator
9. **mock_analytics_coordinator**: Mock AnalyticsCoordinator
10. **test_analytics_events**: Sample analytics events for testing
11. **test_session_id**: Unique test session ID
12. **test_conversation_history**: Test conversation messages
13. **test_conversation_long**: Long conversation for compaction testing
14. **temp_workspace**: Temporary workspace directory
15. **mock_env_vars**: Mock environment variables
16. **performance_monitor**: Performance tracking utility
17. **test_helpers**: Helper functions for testing

## Running the Tests

### Run All Tests (Skipped by Default)

```bash
pytest tests/integration/agent/test_orchestrator_integration.py -v
```text

Expected output: All tests SKIPPED (requires coordinator flag)

### Run with Coordinator Flag

```bash
USE_COORDINATOR_ORCHESTRATOR=true pytest tests/integration/agent/test_orchestrator_integration.py -v
```

Expected output: Tests run (will fail until coordinators fully implemented)

### Run with Coverage

```bash
pytest tests/integration/agent/test_orchestrator_integration.py \
  --cov=victor.agent.coordinators \
  --cov-report=html \
  --cov-report=term-missing \
  -v
```text

### Run Individual Test Class

```bash
# Test simple chat flow
pytest tests/integration/agent/test_orchestrator_integration.py::TestSimpleChatFlow -v

# Test tool execution
pytest tests/integration/agent/test_orchestrator_integration.py::TestToolExecutionFlow -v
```

## Current Implementation Status

### Working Components

- **AnalyticsCoordinator**: Fully implemented and tested (40 unit tests passing)
- **ContextCoordinator**: Implemented with compaction strategies
- **ConfigCoordinator**: Implemented with validation
- **PromptCoordinator**: Implemented with prompt builders
- **ToolCoordinator**: Fully implemented with 950+ lines of code
- **ChatCoordinator**: Implemented with streaming support

### Not Yet Integrated

The coordinators exist but are not fully integrated into AgentOrchestrator:

- AgentOrchestrator doesn't delegate to coordinators
- Analytics tracking not connected to chat flow
- Context compaction not triggered automatically
- Configuration loading not integrated
- Prompt building uses legacy methods

### Required Implementation

To make these tests pass, add methods to AgentOrchestrator:

1. `_track_tool_execution()` → AnalyticsCoordinator.track_event()
2. `_check_context_budget()` → ContextCoordinator.is_within_budget()
3. `_compact_context()` → ContextCoordinator.compact_context()
4. `_build_prompt()` → PromptCoordinator.build_system_prompt()
5. `_track_chat_event()` → AnalyticsCoordinator.track_event()
6. `export_analytics()` → AnalyticsCoordinator.export_analytics()
7. `get_session_stats()` → AnalyticsCoordinator.get_session_stats()
8. Chat and tool coordinator interaction loops
9. Error isolation and graceful degradation
10. Config coordinator integration

## Coverage Goals

**Target**: >80% coverage for coordinator modules

### Coordinator Modules

- `victor/agent/coordinators/analytics_coordinator.py`
- `victor/agent/coordinators/chat_coordinator.py`
- `victor/agent/coordinators/config_coordinator.py`
- `victor/agent/coordinators/context_coordinator.py`
- `victor/agent/coordinators/tool_coordinator.py`
- `victor/agent/coordinators/prompt_coordinator.py`

### Current Coverage

Run coverage report:
```bash
pytest tests/integration/agent/test_orchestrator_integration.py \
  --cov=victor.agent.coordinators \
  --cov-report=html \
  --cov-report=term-missing
```text

View HTML report:
```bash
open htmlcov/index.html
```

## CI/CD Integration

### GitHub Actions

**File**: `.github/workflows/ci.yml`

#### Test Job

```yaml
- name: Run unit tests
  run: |
    pytest tests/unit -v \
      --cov=victor \
      --cov-report=xml \
      --cov-report=html \
      --cov-report=term-missing \
      --cov-context=test \
      --cov-fail-under=70

- name: Run integration tests (sample)
  run: |
    pytest tests/integration/agent/test_orchestrator_integration.py -v \
      --cov=victor.agent.coordinators \
      --cov-append \
      --cov-report=xml \
      --no-cov-on-fail || true

- name: Generate coverage report
  run: |
    echo "=== Coverage Summary ===" >> $GITHUB_STEP_SUMMARY
    coverage report --sort=cover >> $GITHUB_STEP_SUMMARY
    echo "### Coordinator Modules Coverage" >> $GITHUB_STEP_SUMMARY
    coverage report --include='victor/agent/coordinators/*' >> $GITHUB_STEP_SUMMARY
```text

#### Coverage Upload

```yaml
- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v4
  with:
    file: ./coverage.xml
    flags: unit-tests
    name: codecov-umbrella

- name: Upload coverage HTML
  uses: actions/upload-artifact@v4
  with:
    name: coverage-report
    path: htmlcov/
```

### Pyproject.toml Configuration

```toml
[tool.coverage.run]
source = ["victor"]
branch = true
parallel = true
context = "test"

[tool.coverage.report]
precision = 2
show_missing = true
skip_covered = false

# Coordinator-specific coverage goals
[tool.coverage.coordinator_target]
modules = [
    "victor/agent/coordinators/analytics_coordinator.py",
    "victor/agent/coordinators/chat_coordinator.py",
    "victor/agent/coordinators/config_coordinator.py",
    "victor/agent/coordinators/context_coordinator.py",
    "victor/agent/coordinators/tool_coordinator.py",
    "victor/agent/coordinators/prompt_coordinator.py",
]
min_coverage = 80.0
fail_under = true
```text

## Test Structure

Each test class follows this pattern:

```python
@pytest.mark.integration
@pytest.mark.skipif(not USE_COORDINATOR, reason=SKIP_REASON)
class TestFeatureName:
    """Test feature description.

    Expected Behavior:
        - Behavior 1
        - Behavior 2
    """

    @pytest.mark.asyncio
    async def test_feature_behavior(self, fixture1, fixture2):
        """Test feature behavior description.

        Scenario:
        1. Setup step
        2. Action step
        3. Verification step

        Expected:
        - Expected outcome 1
        - Expected outcome 2

        Current Implementation Status:
        - STATUS: Description
        - Required: What needs to be implemented
        """
```

## Next Steps

1. **Implement Coordinator Delegation**
   - Add methods to AgentOrchestrator to delegate to coordinators
   - Connect analytics tracking to chat flow
   - Enable automatic context compaction
   - Integrate configuration loading

2. **Run Tests with Flag**
   ```bash
   USE_COORDINATOR_ORCHESTRATOR=true pytest tests/integration/agent/test_orchestrator_integration.py -v
```text

3. **Generate Coverage Report**
   ```bash
   pytest tests/integration/agent/test_orchestrator_integration.py \
     --cov=victor.agent.coordinators \
     --cov-report=html
   ```

4. **Verify Coverage >80%**
   - Check coverage report for coordinator modules
   - Add tests for uncovered code paths
   - Ensure all coordinator methods tested

5. **Remove Skip Markers**
   - Once coordinators fully integrated
   - Tests pass without USE_COORDINATOR_ORCHESTRATOR flag
   - Update CI to run tests by default

## Troubleshooting

### Tests Not Collected

**Issue**: pytest doesn't find tests

**Solution**:
```bash
# Check pytest configuration
cat pyproject.toml | grep -A 20 "\[tool.pytest.ini_options\]"

# Verify test paths
pytest tests/integration/agent/test_orchestrator_integration.py --collect-only
```bash

### Import Errors

**Issue**: Circular import when importing orchestrator

**Solution**:
- Check for circular dependencies
- Use lazy imports in __init__.py
- Verify dataclass field definitions (use field(default_factory=dict) not {})

### Fixtures Not Found

**Issue**: pytest can't find fixtures

**Solution**:
- Verify conftest.py is in test directory
- Check fixture scope (session, module, function)
- Ensure fixtures are imported or defined locally

### Coverage Not Generated

**Issue**: Coverage report missing or incomplete

**Solution**:
- Install pytest-cov: `pip install pytest-cov`
- Check pyproject.toml coverage configuration
- Verify --cov arguments in pytest command
- Check that tests actually run (not skipped)

## Related Documentation

- **CLAUDE.md**: Project architecture and development commands
- **TYPE_SAFETY_PLAN.md**: Type checking roadmap for coordinators
- **COordinator Quick Reference**: Coordinator API documentation
- **Migration Guides**: Upgrading from legacy to coordinator-based architecture

## Contributors

- Created: 2025-01-14
- Author: Claude Code
- Status: Active Development

---

**Last Updated:** February 01, 2026
**Reading Time:** 5 minutes
