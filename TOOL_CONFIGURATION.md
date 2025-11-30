# Tool Configuration Guide

This document explains how to configure tool budgets and enable/disable individual tools in Victor.

## Tool Call Budget

Victor now supports configurable tool call budgets to handle long-running operations that require many tool executions.

### Default Configuration

- **Default Budget**: 300 tool calls per session (increased from 20)
- **Warning Threshold**: 250 calls (warns when approaching limit)

### Configuration in Settings

The tool budget can be configured via environment variables or in the Settings class:

```python
# In victor/config/settings.py or via environment variables
TOOL_CALL_BUDGET=500  # Set custom budget
TOOL_CALL_BUDGET_WARNING_THRESHOLD=450  # Set warning threshold
```

### Budget Behavior

- **Warn at threshold**: When you hit the warning threshold, Victor will display a message showing current usage
- **Hard stop at budget**: When the budget is exhausted, no more tools will be executed
- **Per-session limit**: The budget resets for each new conversation session

## Tool Enable/Disable System

You can now selectively enable or disable tools via configuration files. This is useful for:

- **Security**: Disable potentially dangerous tools in production
- **Performance**: Reduce tool selection overhead by disabling unused tools
- **Debugging**: Isolate tool behavior by enabling only specific tools
- **Compliance**: Meet organizational requirements by restricting tool access

### Configuration File

Tool states are configured in `~/.victor/profiles.yaml` under the `tools` section.

### Format 1: Disable Specific Tools

```yaml
tools:
  disabled:
    - code_review
    - security_scan
    - docker
```

All tools are enabled by default except those listed in `disabled`.

### Format 2: Enable Only Specific Tools

```yaml
tools:
  enabled:
    - read_file
    - write_file
    - list_directory
    - execute_bash
    - edit_files
```

When using `enabled`, **all tools are disabled by default** except those listed.

### Format 3: Individual Tool Configuration

```yaml
tools:
  code_review:
    enabled: false
  security_scan:
    enabled: false
  read_file:
    enabled: true
  web_search:
    enabled: true
```

This format allows fine-grained control with additional metadata per tool.

### Mixed Configuration

You can combine formats:

```yaml
tools:
  disabled:
    - code_review  # Disable code_review
    - docker       # Disable docker

  web_search:
    enabled: false  # Also disable web_search using individual config
```

## Example: Minimal Tool Set

For lightweight operations or constrained environments:

```yaml
tools:
  enabled:
    - read_file
    - write_file
    - list_directory
    - execute_bash
```

## Example: Disable Problematic Tools

If certain tools are causing issues (like code_review in the analysis example):

```yaml
tools:
  disabled:
    - code_review  # Has validation issues with certain aspects
```

## Tool Registry API

Programmatic access to tool states (for advanced use cases):

```python
from victor.tools.base import ToolRegistry

registry = ToolRegistry()

# Enable/disable tools
registry.enable_tool("code_review")
registry.disable_tool("security_scan")

# Check tool state
is_enabled = registry.is_tool_enabled("code_review")

# Get all tool states
states = registry.get_tool_states()  # Returns Dict[str, bool]

# Set multiple states at once
registry.set_tool_states({
    "code_review": False,
    "security_scan": True,
    "docker": False,
})

# List only enabled tools
enabled_tools = registry.list_tools(only_enabled=True)

# List all tools (including disabled)
all_tools = registry.list_tools(only_enabled=False)
```

## Implementation Details

### File Locations

1. **Settings**: `victor/config/settings.py:113-114`
   - `tool_call_budget: int = 300`
   - `tool_call_budget_warning_threshold: int = 250`

2. **Tool Registry**: `victor/tools/base.py:181-370`
   - Enable/disable methods
   - State tracking
   - Filtered tool listing

3. **Orchestrator**: `victor/agent/orchestrator.py:398-453`
   - Configuration loading
   - Tool state management
   - Budget tracking

### Tool Execution

When a tool is disabled:

1. It **will not appear** in `list_tools(only_enabled=True)` (default)
2. It **will not appear** in tool selection for the LLM
3. Attempting to execute it directly returns an error: `Tool '{name}' is disabled`

### Configuration Loading

Tool configurations are loaded during orchestrator initialization:

1. Reads `~/.victor/profiles.yaml`
2. Parses `tools` section
3. Applies enable/disable states to registry
4. Logs disabled tools at INFO level

## Configuration Validation (NEW)

Victor now validates tool configurations on startup to prevent common mistakes:

### Validation Features

1. **Invalid Tool Names**: Warns about tool names that don't exist
   ```
   WARNING: Configuration contains invalid tool names in 'disabled' list: code_reviw, secuity_scan
   Available tools: analyze_docs, code_review, execute_bash, ...
   ```

2. **Missing Core Tools**: Warns if enabled list doesn't include core tools
   ```
   WARNING: 'enabled' list is missing recommended core tools: read_file, write_file
   This may limit agent functionality.
   ```

3. **Disabled Core Tools**: Warns when disabling essential tools
   ```
   WARNING: Disabling core tool 'execute_bash'. This may limit agent functionality.
   ```

4. **Tool Count Logging**: Shows how many tools are active
   ```
   INFO: Enabled tools: 28/33
   INFO: Disabled tools: code_review
   ```

### Validation Benefits

- **Catch typos**: Immediately see misspelled tool names
- **Avoid lockout**: Warned before accidentally disabling all tools
- **Better visibility**: See which tools are active at startup
- **Helpful messages**: Get list of available tools when configuration is invalid

## Troubleshooting

### Issue: Tools still executing after disabling

**Solution**: Check that:
1. Configuration file is at `~/.victor/profiles.yaml`
2. YAML syntax is correct (use `yamllint` to verify)
3. Tool names match exactly (case-sensitive)
4. Victor was restarted after configuration change
5. Check startup logs for validation warnings

### Issue: All tools disabled

**Solution**: If using `enabled` list format, ensure you've listed all required core tools:
- `read_file`
- `write_file`
- `list_directory`
- `execute_bash`
- `edit_files`

### Issue: Budget exhausted too quickly

**Solution**: Increase the budget in settings or via environment variable:

```bash
export TOOL_CALL_BUDGET=500
victor main "your command"
```

## Best Practices

1. **Start permissive**: Begin with all tools enabled, disable only as needed
2. **Log changes**: Keep track of which tools you disable and why
3. **Test configurations**: Verify tool behavior after changes
4. **Use format 1 for most cases**: Disabling specific tools is clearer than enabling a whitelist
5. **Monitor budget usage**: Watch for budget warnings and adjust limits accordingly

## Related Files

- `victor/config/settings.py` - Settings configuration
- `victor/tools/base.py` - Tool registry implementation
- `victor/agent/orchestrator.py` - Tool management and execution
- `~/.victor/profiles.yaml` - User configuration file

## Migration from Previous Version

Previous versions had a hardcoded 20-tool budget and no tool enable/disable feature.

**Before**:
```python
self.tool_budget = 20  # Hardcoded
```

**After**:
```python
self.tool_budget = getattr(settings, "tool_call_budget", 300)  # Configurable, default 300
```

No action required - the system automatically uses the new defaults. Optionally configure in `profiles.yaml` for custom behavior.
