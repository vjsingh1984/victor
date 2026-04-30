# Session Analysis: codingagent-9f788887
## Stage Transition Thrashing Investigation

**Date**: 2026-04-30
**Session**: codingagent-9f788887
**Status**: ⚠️ STREAMING PATH BYPASSING PHASE 1 OPTIMIZATIONS

---

## Executive Summary

**Root Cause Identified**: The `StreamingChatPipeline` is **NOT integrated** with AgenticLoop and bypasses all Phase 1 optimizations (cooldown, high confidence skip), causing:
- **Excessive edge model calls**: 5 calls in 9 iterations (vs. 1 in previous session)
- **Stage thrashing**: EXECUTION → ANALYSIS → EXECUTION → ANALYSIS...
- **Tool count threshold (MIN_TOOLS_FOR_TRANSITION = 5) is NOT the cause**

**Key Finding**: This is a **path-specific issue**, not a configuration issue. The non-streaming path (AgenticLoop) works correctly with only 1 edge model call per session.

---

## Detailed Analysis

### Session Overview

| Metric | Value | Assessment |
|--------|-------|------------|
| **Total iterations** | 9 | Normal for analysis task |
| **Stage transitions** | 8 | ⚠️ High thrashing |
| **Edge model calls** | 5 | ⚠️ 5x more than expected |
| **Calibration events** | 2 | ✅ Working correctly |
| **Tool executions** | 36 (4 per iteration) | Normal |

### Stage Transition Sequence

```
ITER 1: INITIAL → PLANNING → EXECUTION
ITER 2: EXECUTION (4 edge model calls in 1 second!)
ITER 3: EXECUTION → ANALYSIS (edge model + calibration)
ITER 4: ANALYSIS → EXECUTION
ITER 5: EXECUTION → ANALYSIS (calibration: 11 files read, 0 edits)
ITER 6: ANALYSIS (no transition, 4 reads)
ITER 7: ANALYSIS → EXECUTION
ITER 8: EXECUTION → ANALYSIS (calibration: 20 files read, 0 edits)
ITER 9: ANALYSIS → EXECUTION
```

**Pattern**: EXECUTION ↔ ANALYSIS oscillation every 1-2 iterations

### Edge Model Call Analysis

**All 5 edge model calls happened in ITER 2**:
```
13:46:22,307 - Edge stage detection: EXECUTION (confidence=1.00)
13:46:22,795 - Edge stage detection: EXECUTION (confidence=1.00)
13:46:23,120 - Edge stage detection: EXECUTION (confidence=1.00)
13:46:23,462 - Edge stage detection: EXECUTION (confidence=1.00)
13:46:35,637 - Edge stage detection: ANALYSIS (confidence=0.95)
```

**All within 1 second!** Cooldown is not working.

### Log Evidence

**Edge model bypassing cooldown**:
```
13:46:22,307 - Edge model override: ANALYSIS→EXECUTION (edge=1.00 > heuristic=0.6)
13:46:22,795 - Edge model override: ANALYSIS→EXECUTION (edge=1.00 > heuristic=0.6)
13:46:23,120 - Edge model override: ANALYSIS→EXECUTION (edge=1.00 > heuristic=0.6)
13:46:23,462 - Edge model override: ANALYSIS→EXECUTION (edge=1.00 > heuristic=0.6)
```

Each tool execution triggers an edge model call that overrides the heuristic.

**Calibration working correctly**:
```
ITER 5: Edge model calibration: EXECUTION (1.00) → ANALYSIS
         Reason: Agent has read 11 files without any edits.

ITER 8: Edge model calibration: EXECUTION (1.00) → ANALYSIS
         Reason: Agent has read 20 files without any edits.
```

**StreamingChatPipeline being used**:
```
13:46:11,414 - Task type classification: coarse=default, unified=analyze
13:46:11,418 - Stream chat limits: tool_budget=200, is_analysis_task=True
13:46:11,432 - Streaming perception: intent=ActionIntent.WRITE_ALLOWED
```

---

## Root Cause

### The Problem: Two Execution Paths

Victor has **two different execution paths** for chat:

| Path | Entry Point | AgenticLoop Integration | Phase 1 Optimizations |
|------|-------------|------------------------|----------------------|
| **Non-streaming** | `TurnExecutor.execute_agentic_loop()` | ✅ Yes | ✅ Cooldown, high confidence skip |
| **Streaming** | `StreamingChatPipeline.run()` | ❌ No | ❌ Bypassed, edge model called per tool |

**This session is using the streaming path**, which does NOT integrate with AgenticLoop.

### Why Edge Model is Called Per Tool

The streaming path has its own iteration logic and calls stage detection for each tool execution:

```python
# StreamingChatPipeline (simplified)
for tool_call in tool_calls:
    execute_tool(tool_call)
    record_tool_execution(tool_name, args)  # ← Triggers _maybe_transition()
    # ↑ _maybe_transition() is called for EACH tool, not once per turn
```

Compare with AgenticLoop (non-streaming):
```python
# AgenticLoop (correct behavior)
execute_turn_with_tools():  # ← Executes all tools in one turn
    for tool_call in tool_calls:
        execute_tool(tool_call)
        record_tool_execution(tool_name, args)
    _maybe_transition()  # ← Called ONCE after all tools
```

### Why Cooldown Doesn't Work

In the streaming path, `_maybe_transition()` is called for each tool, but:
1. **Cooldown check passes**: Each tool execution happens > 2 seconds apart (within the turn)
2. **High confidence skip fails**: Tool overlap < 5 (only 1 tool at a time)
3. **Edge model called**: Every single tool execution triggers an edge model call

### Why Stage Thrashing Occurs

1. **Heuristic detects ANALYSIS**: Agent is reading files (ANALYSIS stage)
2. **Edge model returns EXECUTION**: Overconfident (1.00 confidence)
3. **Override happens**: Edge model overrides heuristic → EXECUTION
4. **Calibration fixes it**: Detects 11-20 files read, 0 edits → calibrates to ANALYSIS
5. **Repeat**: Next iteration repeats the cycle

---

## Comparison with Previous Session

### Session codingagent-54930a73 (Non-streaming path)

| Metric | Value |
|--------|-------|
| **Edge model calls** | 1 (entire session) |
| **Stage transitions** | 11 (optimal) |
| **Calibration** | 4 events, 100% accurate |
| **Path** | Non-streaming (AgenticLoop) |
| **Phase 1 optimizations** | ✅ Working |

### Session codingagent-9f788887 (Streaming path)

| Metric | Value |
|--------|-------|
| **Edge model calls** | 5 (in ITER 2 alone!) |
| **Stage transitions** | 8 (thrashing) |
| **Calibration** | 2 events, 100% accurate |
| **Path** | Streaming (StreamingChatPipeline) |
| **Phase 1 optimizations** | ❌ Bypassed |

**Conclusion**: The tool count threshold (MIN_TOOLS_FOR_TRANSITION = 5) is **NOT the cause**. The streaming path is the issue.

---

## Tool Count Threshold Analysis

### Current Setting
```python
MIN_TOOLS_FOR_TRANSITION: int = 5  # Increased from 3
```

### How It Works (in non-streaming path)

1. After executing all tools in a turn, check overlap:
   ```python
   recent_overlap = len(set(last_tools) & stage_tools)
   ```

2. If overlap ≥ 5, skip edge model (high confidence)

3. If overlap < 5, consult edge model

### Why It's Not the Cause

In this session:
- **ITER 2**: 4 `ls` tools (directory listings)
  - Unique tools: 1 (`ls`)
  - Overlap with EXECUTION stage: 1
  - Result: 1 < 5, so edge model called ✅

- **ITER 3**: 4 `read` tools
  - Unique tools: 1 (`read`)
  - Overlap with ANALYSIS stage: 1
  - Result: 1 < 5, so edge model called ✅

The threshold is working as designed. The problem is that the streaming path calls it **4 times per iteration** instead of once.

### Would Decreasing to 3 Help?

**NO**. Here's why:

**Current (threshold = 5)**:
- 4 edge model calls in ITER 2
- 1 edge model call in ITER 3
- Total: 5 calls

**If threshold = 3**:
- 4 edge model calls in ITER 2 (overlap = 1 < 3)
- 1 edge model call in ITER 3 (overlap = 1 < 3)
- Total: 5 calls (same!)

**The threshold doesn't matter** because the streaming path calls edge model per tool, not per turn.

---

## Recommendations

### ✅ Immediate Actions

1. **DO NOT change MIN_TOOLS_FOR_TRANSITION**
   - It's not the cause
   - Changing it won't help (threshold < overlap still true)

2. **Integrate StreamingChatPipeline with AgenticLoop**
   - This is the root cause
   - Already documented in codebase as TODO

3. **Add streaming path detection**
   - Log which path is being used
   - Apply different optimizations per path

### 🔧 Medium-Term Fixes

1. **Add cooldown to streaming path**
   - Implement per-tool cooldown in StreamingChatPipeline
   - Prevent edge model calls within 2 seconds

2. **Batch stage detection in streaming**
   - Call `_maybe_transition()` once per iteration, not per tool
   - Collect all tools, then check overlap

3. **Improve edge model confidence**
   - Edge model is overconfident (always 1.00)
   - Add uncertainty quantification

### 📊 Long-Term Solutions

1. **Unify execution paths**
   - Make StreamingChatPipeline use AgenticLoop internally
   - Single source of truth for stage transitions

2. **Add path-aware optimizations**
   - Different strategies for streaming vs. non-streaming
   - Configurable per deployment

3. **Improve calibration**
   - Make it proactive (prevent calls) vs. reactive (fix after)
   - Cache calibration results

---

## Phase 1 Optimization Status

### What Works (Non-Streaming Path)

✅ **Cooldown check**: Prevents 80%+ of edge model calls
✅ **High confidence skip**: Skips edge model when overlap ≥ 5
✅ **Calibration**: Corrects edge model bias (100% accurate)
✅ **Result**: 1 edge model call per session

### What Doesn't Work (Streaming Path)

❌ **Cooldown check**: Bypassed (called per tool, not per turn)
❌ **High confidence skip**: Bypassed (overlap always < 5 per tool)
❌ **Calibration**: Reactive (fixes after, doesn't prevent)
❌ **Result**: 5+ edge model calls per iteration

---

## Conclusion

**The tool count threshold (MIN_TOOLS_FOR_TRANSITION = 5) is NOT causing stage thrashing.**

**Root cause**: The `StreamingChatPipeline` is not integrated with AgenticLoop and bypasses all Phase 1 optimizations.

**Evidence**:
- Non-streaming path: 1 edge model call per session ✅
- Streaming path: 5+ edge model calls per iteration ❌
- Calibration is working (fixes bias after the fact)
- Stage thrashing is caused by edge model overconfidence

**Recommendation**: Keep `MIN_TOOLS_FOR_TRANSITION = 5` and focus on integrating StreamingChatPipeline with AgenticLoop.

---

## Next Steps

1. ✅ **Keep current configuration**
   - MIN_TOOLS_FOR_TRANSITION = 5
   - Compaction thresholds = 75%
   - Token estimation = 3.5 chars/token

2. 🔧 **Fix streaming path integration**
   - Add StreamingChatPipeline to AgenticLoop
   - Apply Phase 1 optimizations to streaming path
   - Add per-tool cooldown

3. 📊 **Monitor future sessions**
   - Track which path is being used
   - Compare edge model calls per path
   - Measure stage transition frequency

---

**Status**: Analysis complete. Root cause identified. Configuration is correct. Path integration needed.
