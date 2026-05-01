# Stage Transition Thrashing - Quick Reference

## Problem

Victor thrashes between ANALYSIS and EXECUTION stages, calling the edge model (Ollama) 30-40 times per task instead of 3-5. This wastes 30-40% of task time on redundant API calls.

## Root Causes

1. **Aggressive threshold**: `MAX_READS_WITHOUT_EDIT = 5` forces EXECUTION after just 5 files
2. **Redundant edge calls**: Edge model called after EVERY tool result, even during cooldown
3. **Wrong metric**: Counts unique files, not consecutive reads
4. **No intent awareness**: Doesn't distinguish exploration vs. bug-fix tasks
5. **Biased model**: Edge model always returns EXECUTION with 1.00 confidence

## Solution Overview

### Phase 1: Critical Fixes (2-3 hours) ⚡

1. **Add cooldown check before edge model call**
   - Location: `state_machine.py:_maybe_transition()`
   - Impact: Prevents 80%+ of edge model calls

2. **Track consecutive reads instead of unique files**
   - Location: `state_machine.py:ConversationState`
   - Impact: More accurate "stuck in loop" detection

3. **Use consecutive reads in force transition logic**
   - Location: `state_machine.py:_should_force_execution_transition()`
   - Change: `files_read >= 5` → `consecutive_reads >= 12`

### Phase 2: Intent-Aware Thresholds (2-3 hours) 🎯

4. **Add ActionIntent to state machine**
   - Location: `state_machine.py:ConversationStateMachine.__init__()`
   - Add: `action_intent` parameter and `set_action_intent()` method

5. **Use intent-aware thresholds**
   - `write_allowed`: 8 consecutive reads
   - `read_only`: 30 consecutive reads
   - `default`: 12 consecutive reads

6. **Wire intent through creation path**
   - Location: `controller.py` and `runtime_builders.py`

### Phase 3: Edge Model Calibration (1-2 hours) 🔧

7. **Add confidence calibration**
   - Downgrade EXECUTION → ANALYSIS if reading 10+ files without edits
   - Correct for intent bias (read_only tasks shouldn't be EXECUTION)

### Phase 4: Additional Optimizations (1-2 hours) ⚙️

8. **Add minimum interval between compactions**
   - 60 seconds minimum between compactions

9. **Increase synthesis checkpoint threshold**
   - Change: 3 → 8 consecutive reads

## Files Modified

- `victor/agent/conversation/state_machine.py` (Primary)
- `victor/agent/conversation/controller.py`
- `victor/agent/factory/runtime_builders.py`
- `victor/agent/tool_pipeline.py` (if applicable)

## Expected Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Edge model calls | 30-40 | 3-5 | **-90%** |
| Stage transitions | 8-12 | 2-3 | **-75%** |
| Task completion time | 100% | 70% | **-30% faster** |
| Ollama API calls | 40-50 | 5-8 | **-85%** |

## Documentation

- **Analysis**: `STAGE_TRANSITION_THRASHING_ANALYSIS.md`
- **Implementation**: `STAGE_TRANSITION_FIX_IMPLEMENTATION.md`
- **Additional Issues**: `ADDITIONAL_ISSUES_FROM_LOG_ANALYSIS.md`
- **Action Plan**: `CONSOLIDATED_ACTION_PLAN.md`

## Testing

```bash
# Run unit tests
pytest tests/unit/agent/test_conversation_state.py -v

# Run integration tests
pytest tests/integration/test_stage_transition_fixes.py -v

# Run with feature flag
VICTOR_USE_INTENT_AWARE_STAGE_THRESHOLDS=true victor chat
```

## Rollout

1. Week 1: Internal testing
2. Week 2: 10% of sessions
3. Week 3: 50% of sessions
4. Week 4: 100% of sessions

## Status

- [ ] Phase 1: Critical fixes
- [ ] Phase 2: Intent-aware thresholds
- [ ] Phase 3: Edge model calibration
- [ ] Phase 4: Additional optimizations
- [ ] Testing complete
- [ ] Rollout complete

---

**Generated**: 2026-04-30
**Session**: codingagent-9f50c710
**Log File**: ~/.victor/logs/victor.log
