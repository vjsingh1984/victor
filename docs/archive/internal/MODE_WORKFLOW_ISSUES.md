# Mode Workflow Issues - Framework-Level Fixes

Testing of EXPLORE, PLAN, and BUILD modes across OpenAI and DeepSeek providers revealed several framework-level issues that need to be addressed.

## Testing Summary

| Mode | Provider | Status | Issues |
|------|----------|--------|--------|
| EXPLORE | OpenAI | ✅ Works | Path resolution fixed |
| EXPLORE | DeepSeek | ✅ Works | Path resolution fixed |
| PLAN | OpenAI | ✅ Works | Exploration multiplier 2.5x applied |
| PLAN | DeepSeek | ⚠️ Partial | May timeout on complex tasks |
| BUILD | OpenAI | ✅ Works | Shell access, write tools, exploration multiplier 2.0x |
| BUILD | DeepSeek | ⚠️ Partial | May timeout on complex tasks |

## Implementation Status (December 2025)

All 5 issues have been implemented:

| Issue | Status | Commit |
|-------|--------|--------|
| Issue 1: shell_readonly in BUILD mode | ✅ Fixed | c8c7026 |
| Issue 2: Path resolution (subdirectory prefix) | ✅ Fixed | c8c7026 |
| Issue 3: Mode controller integration | ✅ Fixed | c8c7026 |
| Issue 4: Exploration limit for BUILD | ✅ Fixed | c8c7026 |
| Issue 5: Vertical tool filtering override | ✅ Fixed | c8c7026 |
| Additional: Stage filtering in BUILD mode | ✅ Fixed | 925cc50 |
| Additional: exploration_multiplier for BUILD | ✅ Fixed | 925cc50 |

## Identified Issues

### Issue 1: Shell Tool Resolved to shell_readonly in BUILD Mode

**Symptom**: In BUILD mode, the orchestrator resolves shell commands to `shell_readonly` instead of `shell`, preventing file creation.

**Error Message**:
```
Command 'mkdir' is not allowed in readonly mode. Allowed commands: ag, awk, cargo, cat, cmp, cut, date, df, diff, du, echo...
```

**Root Cause**: The `_resolve_shell_variant()` method in `orchestrator.py` checks `is_tool_enabled(ToolNames.SHELL)` which returns False because the vertical or framework hasn't properly enabled the shell tool.

**Location**: `victor/agent/orchestrator.py:2460-2475`

**Framework Fix**:
```python
# In _resolve_shell_variant(), also check mode controller
def _resolve_shell_variant(self, tool_name: str) -> str:
    from victor.agent.mode_controller import get_mode_controller

    mode_controller = get_mode_controller()
    config = mode_controller.config

    # If BUILD mode (allow_all_tools=True), prefer full shell
    if config.allow_all_tools and "shell" not in config.disallowed_tools:
        return ToolNames.SHELL

    # Existing fallback logic...
```

### Issue 2: Path Resolution Includes Subdirectory Prefix

**Symptom**: When working in a subdirectory, the LLM uses paths like `investor_homelab/utils` instead of `utils` (relative to cwd).

**Error Messages**:
```
Directory not found: investor_homelab/utils
File not found: investor_homelab/models/news_model.py
```

**Root Cause**: The project context or arch_summary includes full paths from a different root, and the LLM uses these paths directly.

**Location**: `victor/tools/filesystem.py` and tool responses

**Framework Fix**:
- Normalize paths in tool responses to be relative to current working directory
- Add path validation in tools to strip common prefix patterns
- Update `arch_summary` to use paths relative to cwd

### Issue 3: Mode Controller Not Integrated with Tool Enablement

**Symptom**: `ModeController.is_tool_allowed()` returns True for BUILD mode, but `orchestrator.is_tool_enabled()` returns False for certain tools.

**Root Cause**: Two separate systems control tool access:
1. `ModeController.is_tool_allowed()` - mode-based restrictions
2. `orchestrator.get_enabled_tools()` - vertical/session-based restrictions

These don't communicate, causing inconsistent behavior.

**Framework Fix**:
```python
# In orchestrator.py, integrate mode checks
def is_tool_enabled(self, tool_name: str) -> bool:
    from victor.agent.mode_controller import get_mode_controller

    # Check mode controller first
    mode_controller = get_mode_controller()
    if not mode_controller.is_tool_allowed(tool_name):
        return False

    # Then check session/vertical restrictions
    enabled = self.get_enabled_tools()
    return tool_name in enabled if enabled else True
```

### Issue 4: Exploration Limit Prevents BUILD Actions

**Symptom**: Even in BUILD mode, agents hit exploration limits and summarize findings instead of creating files.

**Message**:
```
⚠️ Reached exploration limit - summarizing findings...
```

**Root Cause**: The exploration limit check doesn't distinguish between exploration phases and action phases. In BUILD mode, tool calls for file creation shouldn't count toward exploration limit.

**Framework Fix**:
- Add task type awareness to exploration limit
- Write operations (edit_files, write_file, shell with mkdir) should not count toward exploration
- Only read/search operations should count

```python
# In unified_task_tracker.py
def should_force_completion(self) -> bool:
    # Don't force completion if last actions were write operations
    if self._last_tool_was_write_operation:
        return False

    # Existing logic...
```

### Issue 5: Vertical Tool Filtering Overrides Mode Settings

**Symptom**: When a vertical is active, its tool list may exclude tools that BUILD mode should have access to.

**Root Cause**: Vertical's `get_tools()` returns a restricted set, and this set is used as `_enabled_tools` in the orchestrator, overriding mode settings.

**Framework Fix**:
- Mode settings should be applied AFTER vertical filtering
- BUILD mode's `allow_all_tools=True` should expand the tool set
- Priority: Mode > Vertical > Session defaults

```python
# In framework shim or orchestrator setup
def get_effective_tools(self) -> Set[str]:
    vertical_tools = self.vertical.get_tools() if self.vertical else None
    mode_config = get_mode_controller().config

    if mode_config.allow_all_tools:
        # Mode allows all - use full tool set
        return self.get_available_tools()
    elif vertical_tools:
        # Apply mode restrictions to vertical tools
        return vertical_tools - mode_config.disallowed_tools
    else:
        return self.get_available_tools() - mode_config.disallowed_tools
```

## Recommended Implementation Order

1. **Issue 1 (shell_readonly)**: Critical - BUILD mode is broken without this
2. **Issue 3 (mode integration)**: High - Foundation for mode-aware tool selection
3. **Issue 5 (vertical override)**: High - Ensures mode settings are respected
4. **Issue 4 (exploration limit)**: Medium - Improves BUILD mode behavior
5. **Issue 2 (path resolution)**: Low - Cosmetic but improves UX

## Testing Verification

After implementing fixes, run these tests:

```bash
# Simple BUILD test
cd /path/to/test/project
victor chat --mode build --provider openai --model gpt-4o \
  "Create file utils/constants.py with TIMEOUT=30"

# Medium BUILD test
victor chat --mode build --provider deepseek --model deepseek-reasoner \
  "Add a new RSSNewsClient class to utils/rss_client.py"

# Complex BUILD test
victor chat --mode build --provider openai --model gpt-4o \
  "Refactor the web_search_client.py to add caching with Redis"
```

## Related Files

- `victor/agent/mode_controller.py` - Mode configuration
- `victor/agent/orchestrator.py` - Tool resolution and enablement
- `victor/agent/unified_task_tracker.py` - Exploration limits
- `victor/tools/filesystem.py` - Path handling
- `victor/verticals/*.py` - Vertical tool configuration
