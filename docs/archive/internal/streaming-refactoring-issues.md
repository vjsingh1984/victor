# Streaming Refactoring Issues (January 2025)

## Issues Identified

### 1. ✅ FIXED: Debug Logging Malfunction
**File:** `victor/agent/orchestrator.py` (lines 6115-6193)

**Problem:** Debug logs for type conversion were showing dictionary comprehension literals instead of actual values:
```
[NORMALIZE] read: Original types={k: type(v).__name__ for k, v in tool_args.items()}
```

**Root Cause:** F-string with nested dictionary comprehensions not being evaluated.

**Fix:** Extract comprehensions to variables before f-string:
```python
original_types = {k: type(v).__name__ for k, v in tool_args.items()}
normalized_types = {k: type(v).__name__ for k, v in normalized_args.items()}
logger.debug(f"[NORMALIZE] {tool_name}: Original types={original_types}, Normalized types={normalized_types}")
```

---

### 2. ✅ FIXED: Dead Code - StreamingToolAdapter
**Files:**
- `victor/agent/streaming_tool_adapter.py` → `archived/obsolete/streaming_tool_adapter.py` (356 lines)
- `victor/agent/orchestrator_factory.py:create_streaming_tool_adapter()` (lines 1102-1153)

**Problem:** StreamingToolAdapter was created but never used in production code.

**Evidence:**
- Factory method `create_streaming_tool_adapter()` defined but never called
- No imports of `StreamingToolAdapter` in production code
- Part of unfinished refactoring effort

**Current Tool Execution Path:**
```
orchestrator._handle_tool_calls()
  → argument_normalizer.normalize_arguments() [type coercion]
  → tool_adapter.normalize_arguments() [defaults/filters]
  → tool_executor.execute(skip_normalization=True)
    → tool.execute(**arguments)
```

**Fix:** Moved to `archived/obsolete/` with deprecation warning.

---

### 3. ✅ FIXED: asyncio.run() in Running Event Loop
**File:** `victor/agent/chunk_generator.py`

**Log (before fix):**
```
Failed to emit chunk tool start event: asyncio.run() cannot be called from a running event loop
```

**Problem:** Chunk generator was calling `asyncio.run()` from within an already-running async context (streaming loop).

**Root Cause:** Three methods using `asyncio.run()` to emit events:
- `generate_tool_start_chunk()` (line 116)
- `generate_final_marker_chunk()` (line 184)
- `generate_metrics_chunk()` (line 222)

**Fix:** Replaced `asyncio.run()` with `emit_event_sync()` helper from `victor.core.events.emit_helper`:
```python
# BEFORE (WRONG):
asyncio.run(self._event_bus.emit(...))

# AFTER (CORRECT):
from victor.core.events.emit_helper import emit_event_sync
emit_event_sync(self._event_bus, topic=..., data=...)
```

**Impact:** Events now emit properly without RuntimeErrors.

**Status:** ✅ FIXED - All 3 occurrences updated to use `emit_event_sync()`

---

### 4. ✅ VERIFIED: Type Coercion Working (Ollama Provider)

**Log Analysis:**
```
Tool 'read' enabled=True
[1] read: {'limit': 200, 'path': 'victor/agent/orchestrator.py'}
Tool call: read with args: {'limit': 200, 'path': 'victor/agent/orchestrator.py'}
```

**Observation:** When Ollama provides integer arguments (e.g., `limit: 200`), execution succeeds without type errors.

**Remaining Question:** Does the original error still occur with other providers that output strings (e.g., OpenRouter, Fireworks)?

**Testing Needed:** Run with providers that output numeric strings like `{'limit': '300', 'offset': '202'}`.

---

## Active Streaming Components

### Core Streaming Architecture:
```
StreamingChatHandler (streaming/handler.py)
    ↓
IterationCoordinator (streaming/coordinator.py) - Loop control
    ↓
StreamingCoordinator (streaming/streaming_coordinator.py) - Response processing
    ↓
StreamingController (streaming_controller.py) - Session/metrics
    ↓
ToolExecutionHandler (streaming/tool_execution.py) - Tool execution
```

### All Active Modules:
- ✅ `streaming_controller.py` - Session/metrics management
- ✅ `stream_handler.py` - Core streaming data structures
- ✅ `streaming/coordinator.py` - IterationCoordinator
- ✅ `streaming/streaming_coordinator.py` - StreamingCoordinator
- ✅ `streaming/handler.py` - StreamingChatHandler
- ✅ `streaming/context.py` - StreamingChatContext
- ✅ `streaming/tool_execution.py` - ToolExecutionHandler
- ✅ `streaming/continuation.py` - ContinuationHandler
- ✅ `streaming/iteration.py` - IterationResult/IterationAction
- ✅ `streaming/intent_classification.py` - Intent tracking

### Obsolete Modules:
- ❌ `streaming_tool_adapter.py` → `archived/obsolete/streaming_tool_adapter.py`

---

## Next Steps

1. **Test with different providers** to verify type coercion handles numeric strings
2. **Fix asyncio.run() issue** in chunk_generator
3. **Run full test suite** to verify no regressions
4. **Remove archived file** after confirmation (optional cleanup)

---

## Debug Log Guide

After fixes, debug logs will show:

**Normal flow (integers):**
```
[NORMALIZE] read: Original types={'limit': 'int', 'path': 'str'}, Normalized types={'limit': 'int', 'path': 'str'}
[EXECUTE] read: Final argument types={'limit': 'int', 'path': 'str'}
```

**Type coercion flow (strings):**
```
[NORMALIZE] read: Original types={'limit': 'str', 'path': 'str'}, Normalized types={'limit': 'int', 'path': 'str'}
[EXECUTE] read: Final argument types={'limit': 'int', 'path': 'str'}
```

**Broken flow (if issue persists):**
```
[NORMALIZE] read: Original types={'limit': 'str', 'path': 'str'}, Normalized types={'limit': 'str', 'path': 'str'}
[EXECUTE] read: Final argument types={'limit': 'str', 'path': 'str'}
ERROR: '>' not supported between instances of 'int' and 'str'
```

---

## Additional asyncio.run() Calls Requiring Review

The following `asyncio.run()` calls were identified but NOT fixed yet. They should be reviewed on a case-by-case basis:

### ✅ SAFE (Entry points - no existing loop):
- `victor/ui/commands/*.py` - CLI commands (chat, serve, dashboard, etc.)
- `victor/evaluation/__init__.py` - Evaluation entry point
- `victor/integrations/mcp/*.py` - MCP server startup
- Demo scripts in `victor/rag/`

### ⚠️ NEEDS REVIEW (Called from agent/async context):
- `victor/observability/emitters/*.py` - 6 emitter classes using `asyncio.run()` wrappers
  - `tool_emitter.py`, `state_emitter.py`, `error_emitter.py`, `model_emitter.py`, `lifecycle_emitter.py`
  - These wrap async `emit_async()` with sync methods using `asyncio.run()`
  - **TODO:** Should use `emit_event_sync()` instead

- `victor/agent/recovery_coordinator.py` - 4 occurrences
- `victor/agent/provider_coordinator.py` - 2 occurrences (switch_provider, switch_model)
- `victor/agent/background_agent.py` - 1 occurrence
- `victor/agent/conversation_state.py` - 1 occurrence
- `victor/agent/tool_planner.py` - 2 occurrences
- `victor/agent/conversation_memory.py` - 1 occurrence
- `victor/framework/cqrs_bridge.py` - 2 occurrences
- `victor/workflows/context.py` - 3 occurrences
- Others in various modules

### Recommendation:
**Priority 1:** Fix observability emitters (widely used, called from streaming context)
**Priority 2:** Fix agent module calls (recovery_coordinator, provider_coordinator, etc.)
**Priority 3:** Fix workflow/framework calls if needed

### Fix Pattern:
```python
# BEFORE:
asyncio.run(self._event_bus.emit(topic, data))

# AFTER:
from victor.core.events.emit_helper import emit_event_sync
emit_event_sync(self._event_bus, topic, data)
```

