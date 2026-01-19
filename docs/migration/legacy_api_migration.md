# Legacy API Migration Guide

**Version**: 0.5.1
**Removal Target**: v0.7.0
**Status**: Active Migration Period

This guide helps you migrate away from deprecated APIs in the AgentOrchestrator that have been moved to `LegacyAPIMixin`. All deprecated methods issue clear `DeprecationWarning` messages with migration instructions.

## Table of Contents

- [Overview](#overview)
- [Category 1: Vertical Configuration](#category-1-vertical-configuration)
- [Category 2: Vertical Storage Protocol](#category-2-vertical-storage-protocol)
- [Category 3: Metrics and Analytics](#category-3-metrics-and-analytics)
- [Category 4: State and Conversation](#category-4-state-and-conversation)
- [Category 5: Protocol Methods](#category-5-protocol-methods)
- [Category 6: Provider and Model](#category-6-provider-and-model)
- [Category 7: Tool Access](#category-7-tool-access)
- [Category 8: System Prompt](#category-8-system-prompt)
- [Category 9: Message Access](#category-9-message-access)
- [Category 10: Search Routing](#category-10-search-routing)
- [Category 11: Health Check](#category-11-health-check)
- [Testing Your Migration](#testing-your-migration)
- [Timeline](#timeline)

## Overview

As part of Phase 4 of the orchestrator refactoring, 596 lines of legacy code have been consolidated into the `LegacyAPIMixin` class. This improves maintainability and provides a clear migration path away from deprecated APIs.

**Key Benefits**:
- Cleaner main orchestrator class
- Clear deprecation warnings for all legacy methods
- Centralized location for backward compatibility code
- SOLID compliance through coordinator-based alternatives

**How to Enable Warnings**:

```python
import warnings

# Show all deprecation warnings
warnings.filterwarnings("default", category=DeprecationWarning)

# Or treat them as errors to catch all deprecated usage
warnings.filterwarnings("error", category=DeprecationWarning)
```

## Category 1: Vertical Configuration

### `set_vertical_context()`

**Deprecated**: Use `VerticalContext.set_context()` instead.

**Before**:
```python
orchestrator.set_vertical_context(vertical_context)
```

**After**:
```python
from victor.core.events.vertical_context import VerticalContext

# Vertical context is now managed centrally
vertical_context = VerticalContext.set_context(orchestrator, context)
```

### `set_tiered_tool_config()`

**Deprecated**: ConfigurationManager handles this internally.

**Before**:
```python
orchestrator.set_tiered_tool_config(config)
```

**After**:
```python
# Configuration is now automatic through ConfigurationManager
# No manual intervention needed
```

### `set_workspace()`

**Deprecated**: Use `set_project_root()` from settings module.

**Before**:
```python
orchestrator.set_workspace(Path("/path/to/workspace"))
```

**After**:
```python
from victor.config.settings import set_project_root

set_project_root(Path("/path/to/workspace"))
```

## Category 2: Vertical Storage Protocol

### `apply_vertical_middleware()`

**Deprecated**: VerticalIntegrationAdapter handles this automatically.

**Before**:
```python
orchestrator.apply_vertical_middleware(middleware_list)
```

**After**:
```python
# Middleware is now applied automatically through VerticalIntegrationAdapter
# No manual intervention needed
```

### `apply_vertical_safety_patterns()`

**Deprecated**: VerticalIntegrationAdapter handles this automatically.

**Before**:
```python
orchestrator.apply_vertical_safety_patterns(patterns)
```

**After**:
```python
# Safety patterns are now applied automatically
# No manual intervention needed
```

### `get_middleware_chain()`, `set_middleware()`, `get_middleware()`, `set_safety_patterns()`, `get_safety_patterns()`

**Deprecated**: Use VerticalIntegrationAdapter for middleware management.

**Before**:
```python
chain = orchestrator.get_middleware_chain()
orchestrator.set_middleware(middleware_list)
middleware = orchestrator.get_middleware()
```

**After**:
```python
# Access through VerticalIntegrationAdapter
adapter = orchestrator.vertical_integration_adapter
# Middleware is managed internally
```

### `set_team_specs()`, `get_team_specs()`

**Deprecated**: Use TeamCoordinator methods directly.

**Before**:
```python
orchestrator.set_team_specs(specs)
specs = orchestrator.get_team_specs()
```

**After**:
```python
from victor.teams import TeamCoordinator

coordinator = orchestrator.team_coordinator
coordinator.set_team_specs(specs)
specs = coordinator.get_team_specs()
```

## Category 3: Metrics and Analytics

### `get_tool_usage_stats()`

**Deprecated**: Use MetricsCoordinator.get_tool_usage_stats().

**Before**:
```python
stats = orchestrator.get_tool_usage_stats()
```

**After**:
```python
stats = orchestrator.metrics_coordinator.get_tool_usage_stats()
```

### `get_token_usage()`

**Deprecated**: Use MetricsCoordinator.get_token_usage().

**Before**:
```python
token_usage = orchestrator.get_token_usage()
```

**After**:
```python
token_usage = orchestrator.metrics_coordinator.get_token_usage()
```

### `reset_token_usage()`

**Deprecated**: Use MetricsCoordinator.reset_token_usage().

**Before**:
```python
orchestrator.reset_token_usage()
```

**After**:
```python
orchestrator.metrics_coordinator.reset_token_usage()
```

### Stream Metrics Methods

**Deprecated**: Use MetricsCoordinator methods.

**Before**:
```python
metrics = orchestrator.get_last_stream_metrics()
summary = orchestrator.get_streaming_metrics_summary()
history = orchestrator.get_streaming_metrics_history(limit=10)
cost_summary = orchestrator.get_session_cost_summary()
cost_formatted = orchestrator.get_session_cost_formatted()
orchestrator.export_session_costs("/path/to/file.json", format="json")
```

**After**:
```python
from victor.agent.coordinators.metrics_coordinator import MetricsCoordinator

coordinator = orchestrator.metrics_coordinator
metrics = coordinator.get_last_stream_metrics()
summary = coordinator.get_streaming_metrics_summary()
history = coordinator.get_streaming_metrics_history(limit=10)
cost_summary = coordinator.get_session_cost_summary()
cost_formatted = coordinator.get_session_cost_formatted()
coordinator.export_session_costs("/path/to/file.json", format="json")
```

## Category 4: State and Conversation

### `get_conversation_stage()`

**Deprecated**: Use StateCoordinator.get_stage().

**Before**:
```python
stage = orchestrator.get_conversation_stage()
```

**After**:
```python
from victor.config.conversation import ConversationStage

stage_name = orchestrator.state_coordinator.get_stage()
stage = ConversationStage[stage_name] if stage_name else ConversationStage.INITIAL
```

### `get_stage_recommended_tools()`

**Deprecated**: Use StateCoordinator.get_stage_tools().

**Before**:
```python
tools = orchestrator.get_stage_recommended_tools()
```

**After**:
```python
tools = orchestrator.state_coordinator.get_stage_tools()
```

### `get_optimization_status()`

**Deprecated**: Use AnalyticsCoordinator.get_optimization_status().

**Before**:
```python
status = orchestrator.get_optimization_status()
```

**After**:
```python
from victor.agent.coordinators.analytics_coordinator import AnalyticsCoordinator

coordinator = AnalyticsCoordinator(exporters=[])
status = coordinator.get_optimization_status(
    context_compactor=orchestrator._context_compactor,
    usage_analytics=orchestrator._usage_analytics,
    sequence_tracker=orchestrator._sequence_tracker,
    # ... other components
)
```

### `get_observed_files()`

**Deprecated**: Use StateCoordinator.observed_files.

**Before**:
```python
files = orchestrator.get_observed_files()
```

**After**:
```python
files = orchestrator.state_coordinator.observed_files
```

### `get_modified_files()`

**Deprecated**: Access conversation_state.state directly.

**Before**:
```python
modified = orchestrator.get_modified_files()
```

**After**:
```python
if orchestrator.conversation_state and hasattr(orchestrator.conversation_state, "state"):
    modified = set(getattr(orchestrator.conversation_state.state, "modified_files", []))
else:
    modified = set()
```

## Category 5: Protocol Methods

### `get_tool_calls_count()`, `get_tool_budget()`, `get_iteration_count()`, `get_max_iterations()`

**Deprecated**: Use unified_tracker properties.

**Before**:
```python
calls = orchestrator.get_tool_calls_count()
budget = orchestrator.get_tool_budget()
iterations = orchestrator.get_iteration_count()
max_iterations = orchestrator.get_max_iterations()
```

**After**:
```python
tracker = orchestrator.unified_tracker
if tracker:
    calls = tracker.tool_calls_used
    budget = tracker.tool_budget
    iterations = tracker.iteration_count
    max_iterations = tracker.max_iterations
```

## Category 6: Provider and Model

### `current_provider`, `current_model` (properties)

**Deprecated**: Use direct attribute access.

**Before**:
```python
provider = orchestrator.current_provider
model = orchestrator.current_model
```

**After**:
```python
provider = orchestrator.provider_name
model = orchestrator.model
```

### `get_current_provider_info()`

**Deprecated**: Use ProviderManager.get_info().

**Before**:
```python
info = orchestrator.get_current_provider_info()
```

**After**:
```python
info = orchestrator.provider_manager.get_info()
# Add orchestrator-specific state if needed
info["tool_budget"] = orchestrator.unified_tracker.tool_budget
info["tool_calls_used"] = orchestrator.unified_tracker.tool_calls_used
```

## Category 7: Tool Access

### `get_available_tools()`

**Deprecated**: Use tools.list_tools().

**Before**:
```python
tools = orchestrator.get_available_tools()
```

**After**:
```python
if orchestrator.tools:
    tools = set(orchestrator.tools.list_tools())
else:
    tools = set()
```

### `is_tool_enabled()`

**Deprecated**: Use ToolAccessConfigCoordinator.is_tool_enabled().

**Before**:
```python
enabled = orchestrator.is_tool_enabled("read_file")
```

**After**:
```python
enabled = orchestrator.tool_access_coordinator.is_tool_enabled("read_file")
```

## Category 8: System Prompt

### `get_system_prompt()`, `set_system_prompt()`, `append_to_system_prompt()`

**Deprecated**: Use prompt_builder methods.

**Before**:
```python
prompt = orchestrator.get_system_prompt()
orchestrator.set_system_prompt("Custom prompt")
orchestrator.append_to_system_prompt("Additional content")
```

**After**:
```python
builder = orchestrator.prompt_builder
prompt = builder.build()
builder.set_custom_prompt("Custom prompt")
builder.append_content("Additional content")
```

## Category 9: Message Access

### `get_messages()`, `get_message_count()`

**Deprecated**: Access conversation directly.

**Before**:
```python
messages = orchestrator.get_messages()
count = orchestrator.get_message_count()
```

**After**:
```python
messages = [{"role": m.role, "content": m.content} for m in orchestrator.conversation.messages]
count = len(orchestrator.conversation.messages)
```

## Category 10: Search Routing

### `route_search_query()`, `get_recommended_search_tool()`

**Deprecated**: Use SearchCoordinator methods.

**Before**:
```python
route = orchestrator.route_search_query("query text")
tool = orchestrator.get_recommended_search_tool("query text")
```

**After**:
```python
route = orchestrator.search_coordinator.route_search_query("query text")
tool = orchestrator.search_coordinator.get_recommended_search_tool("query text")
```

## Category 11: Health Check

### `check_tool_selector_health()`

**Deprecated**: Use tool_selector.get_health().

**Before**:
```python
health = orchestrator.check_tool_selector_health()
```

**After**:
```python
health = orchestrator.tool_selector.get_health()
```

## Testing Your Migration

### Enable Deprecation Warnings in Tests

```python
import pytest
import warnings

@pytest.fixture(autouse=True)
def catch_deprecation_warnings():
    """Turn all DeprecationWarnings into errors during tests."""
    warnings.filterwarnings("error", category=DeprecationWarning)

def test_my_code(orchestrator):
    # This will fail if any deprecated APIs are used
    result = orchestrator.chat("test")
```

### Run Tests with Warnings

```bash
# Show all deprecation warnings
pytest tests/ -W default::DeprecationWarning

# Fail on any deprecation warning
pytest tests/ -W error::DeprecationWarning

# Count deprecated API usage
pytest tests/ -W default::DeprecationWarning 2>&1 | grep -c "DeprecationWarning"
```

### Gradual Migration Strategy

1. **Phase 1**: Enable warnings in development
   ```python
   import warnings
   warnings.filterwarnings("default", category=DeprecationWarning)
   ```

2. **Phase 2**: Audit and migrate critical paths
   - Focus on high-frequency code paths
   - Migrate one category at a time

3. **Phase 3**: Enable warnings in CI/CD
   ```yaml
   # .github/workflows/tests.yml
   - name: Run tests
     run: pytest -W error::DeprecationWarning
   ```

4. **Phase 4**: Clean up remaining warnings
   - Address all warnings before v0.7.0

## Timeline

| Version | Milestone | Action Required |
|---------|-----------|-----------------|
| v0.5.1 | Deprecation | Warnings issued for all legacy APIs |
| v0.6.0 | Soft Removal | LegacyAPIMixin marked for removal |
| v0.7.0 | Hard Removal | LegacyAPIMixin and all deprecated methods removed |

**Migration Period**: v0.5.1 - v0.7.0 (approximately 4-6 months)

**Recommendation**: Complete migration by v0.6.0 to allow testing buffer.

## Summary Statistics

- **Total deprecated methods**: 41
- **Lines consolidated**: 596
- **Line reduction**: 80% (from orchestrator class)
- **Categories**: 11
- **Removal target**: v0.7.0

## Need Help?

- Check inline deprecation warnings for specific migration guidance
- Review coordinator implementations for examples
- See `victor/agent/mixins/legacy_api.py` for deprecated method signatures
- Run tests with `-W default::DeprecationWarning` to identify usage

## Checklist

Use this checklist to track your migration progress:

- [ ] Enable deprecation warnings in development
- [ ] Audit codebase for deprecated API usage
- [ ] Migrate Category 1: Vertical Configuration
- [ ] Migrate Category 2: Vertical Storage Protocol
- [ ] Migrate Category 3: Metrics and Analytics
- [ ] Migrate Category 4: State and Conversation
- [ ] Migrate Category 5: Protocol Methods
- [ ] Migrate Category 6: Provider and Model
- [ ] Migrate Category 7: Tool Access
- [ ] Migrate Category 8: System Prompt
- [ ] Migrate Category 9: Message Access
- [ ] Migrate Category 10: Search Routing
- [ ] Migrate Category 11: Health Check
- [ ] Enable warnings in CI/CD pipeline
- [ ] Verify all tests pass without warnings
- [ ] Remove LegacyAPIMixin dependency (optional, pre-v0.7.0)
