# Duplicate Code Verification List

**Purpose:** Verified duplicate candidates with recommendations
**Generated:** 2026-01-10
**Status:** VERIFIED

---

## Summary

| Category | Count | Action |
|----------|-------|--------|
| False Positives (Intentional) | 6 classes | None - different implementations by design |
| True Duplicates | 2 classes | Consolidate |
| Potential Consolidation | 3 classes | Consider base class or protocol |

---

## FALSE POSITIVES - No Action Required

These classes share names but have **intentionally different implementations** for different domains/contexts.

### 1. ValidationResult (6 locations) - DIFFERENT

Each serves a different validation domain with domain-specific fields:

| File | Purpose | Key Fields |
|------|---------|------------|
| `framework/adaptation/protocols.py` | Generic validation | is_valid, errors, warnings, metadata |
| `framework/adaptation/types.py` | Adaptation validation | + suggestions field |
| `framework/patterns/types.py` | Pattern validation | + quality_score, safety_score |
| `optimization/validator.py` | Optimization validation | + speedup, cost_reduction, functional_equivalence |
| `workflows/generation/requirements.py` | Requirement validation | + recommendations, score |
| `workflows/generation/types.py` | Workflow generation | + schema_errors, structure_errors, semantic_errors |

**Verdict:** KEEP SEPARATE - Domain-specific validation requires different fields.

---

### 2. CacheEntry (4 locations) - DIFFERENT

Each caches different types of data:

| File | Caches | Key Fields |
|------|--------|------------|
| `agent/tool_result_cache.py` | Tool results | embedding, args_hash, expires_at |
| `framework/graph_cache.py` | Compiled graphs | graph_hash, compiled |
| `tools/cache_manager.py` | Generic values | value, ttl, accessed_at |
| `workflows/cache.py` | Node results | node_id, result |

**Verdict:** KEEP SEPARATE - Different cache subsystems need different metadata.

---

### 3. CheckpointManager (3 locations) - DIFFERENT

Each manages different checkpoint types:

| File | Manages | Mechanism |
|------|---------|-----------|
| `agent/checkpoints.py` | Git checkpoints | Git stash |
| `framework/graph.py` | Graph state | CheckpointerProtocol |
| `storage/checkpoints/manager.py` | Conversation state | SQLite/memory backend |

**Verdict:** KEEP SEPARATE - Different checkpoint domains.

---

### 4. Event (3 locations) - DIFFERENT

Each serves a different event system:

| File | Purpose | Key Fields |
|------|---------|------------|
| `core/event_sourcing.py` | Domain events (CQRS) | event_id, timestamp, immutable |
| `core/events/protocols.py` | Distributed messaging | topic, correlation_id, source |
| `framework/events.py` | Agent execution | type, content, tool_name, progress |

**Verdict:** KEEP SEPARATE - Different event systems (CQRS vs messaging vs observability).

---

### 5. ValidationError (3 locations) - DIFFERENT

One is an exception, others are dataclasses:

| File | Type | Purpose |
|------|------|---------|
| `core/errors.py` | **Exception** | Raised on validation failure |
| `workflows/generation/requirements.py` | Dataclass | Validation issue record |
| `workflows/generation/types.py` | Dataclass | Detailed validation error |

**Verdict:** KEEP SEPARATE - Exception vs data class are fundamentally different.

---

### 6. ExecutionResult (4 locations) - DIFFERENT

Each represents execution in different contexts:

| File | Context | Key Fields |
|------|---------|------------|
| `framework/graph.py` | Graph execution | state, iterations, node_history |
| `framework/workflow_engine.py` | Workflow execution | final_state, checkpoints, hitl_requests |
| `workflows/.../state_graph_executor.py` | Migration wrapper | final_state, metrics |
| `workflows/sandbox_executor.py` | Sandbox execution | output, error, exit_code |

**Verdict:** KEEP SEPARATE - Different execution contexts need different result fields.
**Note:** `state_graph_executor.py` says "Minimal implementation for migration phase" - review if still needed.

---

## TRUE DUPLICATES - Action Required

### 1. TaskContext - CONSOLIDATE

**Real duplicate found:**

| File | Status |
|------|--------|
| `victor/agent/tool_coordinator.py:75` | **DUPLICATE** |
| `victor/agent/coordinators/tool_coordinator.py:85` | **PRIMARY** (more complete) |

**Evidence:**
- Same class name, same purpose (tool selection context)
- `coordinators/tool_coordinator.py` has additional fields: `conversation_depth`, `conversation_history`
- `tool_coordinator.py` appears to be an older/simpler version

**Action:**
1. Deprecate `victor/agent/tool_coordinator.py`
2. Update imports to use `victor/agent/coordinators/tool_coordinator.py`
3. Add deprecation warning if backward compatibility needed

**Other TaskContext locations (KEEP):**
- `framework/patterns/protocols.py` - Different purpose (pattern recommendation)
- `framework/patterns/types.py` - Dataclass version of above
- `agent/prompt_coordinator.py` - Different purpose (prompt building)

---

### 2. ToolExecutionResult - CONSOLIDATE

**Similar implementations that could share base class:**

| File | Purpose | Key Fields |
|------|---------|------------|
| `coordinators/tool_coordinator.py:142` | Single tool result | tool_name, success, result, error, elapsed_ms |
| `tool_executor.py:109` | Single tool result | + correlation_id, cached, retry_count |

**Streaming versions (KEEP SEPARATE):**
- `streaming/iteration.py` - Batch results with chunks
- `streaming/tool_execution.py` - Streaming phase result

**Action:**
1. Create base `ToolExecutionResult` in `victor/tools/results.py`
2. Have both implementations extend the base
3. Or consolidate to single class in `tool_executor.py` (more complete)

---

## POTENTIAL CONSOLIDATION - Consider

### 1. Message - Consider Base Class

| File | Key Fields | Notes |
|------|------------|-------|
| `providers/base.py` | role, content, tool_calls, tool_call_id | **Most canonical** - LLM API format |
| `context/manager.py` | role, content, tokens, priority | Adds context mgmt fields |
| `ui/tui/session.py` | role, content, timestamp | Adds UI fields |

**Recommendation:** Consider making `providers/base.py:Message` the base class, with extensions for context management and UI.

---

### 2. ProviderProtocol - Consolidate to Core

| File | Purpose | Methods |
|------|---------|---------|
| `core/protocols.py` | Base provider interface | name, supports_tools, chat |
| `streaming/continuation.py` | Minimal for continuation | chat only |
| `framework/protocols.py` | Provider management | current_provider, switch |

**Recommendation:** `streaming/continuation.py` should import from `core/protocols.py` instead of defining its own.

---

### 3. ToolCallRecord - Consider Base Class

| File | Purpose | Notes |
|------|---------|-------|
| `background_agent.py` | Agent tracking | id, status, start/end time |
| `tool_loop_detector.py` | Loop analysis | arguments_hash, result_hash |
| `tracing/tool_calls.py` | Observability | call_id, parent_span_id |

**Recommendation:** Consider creating `victor/tools/types.py:ToolCallRecord` as base class.

---

## Action Items

### Immediate (P0)
- [ ] Consolidate TaskContext: Remove `victor/agent/tool_coordinator.py` duplicate

### Short-term (P1)
- [ ] Consolidate ToolExecutionResult: Create base class or merge implementations
- [ ] Fix ProviderProtocol: Update `streaming/continuation.py` to use `core/protocols.py`

### Long-term (P2)
- [ ] Consider Message base class consolidation
- [ ] Consider ToolCallRecord base class
- [ ] Review `state_graph_executor.py` ExecutionResult migration status

---

## Verification Complete

All 11 duplicate class candidates have been verified:
- 6 are false positives (intentionally different)
- 2 require consolidation action
- 3 are candidates for future refactoring

*Verified by comparing actual implementations on 2026-01-10*
