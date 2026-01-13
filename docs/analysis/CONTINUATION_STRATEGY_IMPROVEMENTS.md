# Continuation Strategy Analysis and Improvements

## Problem Summary

**Current Issue**: The agent is starving the LLM from tool calling even though the LLM wants to perform additional discovery. The continuation strategy intervenes too aggressively, forcing synthesis before the LLM has completed its exploration.

### Evidence from Logs

```
ITER 20: LLM exploring with tool calls (read, ls, graph)
ITER 21: "Cumulative prompt interventions (5) reached threshold - nudging synthesis"
ITER 22: LLM continues exploring
ITER 23: Another "nudging synthesis" intervention
ITER 24: LLM still exploring
ITER 25: "Prompting for tool calls (1/6) - Encouraging tool usage"
```

### Root Cause

The `_cumulative_prompt_interventions` counter (`victor/agent/streaming/continuation.py:357`) is a **global session counter** that:
- Increments on every continuation prompt
- Triggers synthesis nudges at 5 interventions
- Forces synthesis at 8 interventions
- **Does NOT distinguish between**:
  - Simple vs complex prompts
  - Active tool usage vs stuck loops
  - Progress being made vs spinning wheels

---

## Current Architecture Analysis

### Continuation Flow

```
LLM Response (no tool calls)
    ↓
Intent Classification
    ↓
ContinuationStrategy.determine_continuation_action()
    ↓
Checks cumulative_prompt_interventions >= 5?
    ├─ Yes → Nudge synthesis
    └─ No → Continue normally
```

### Key Files

| File | Purpose | Lines |
|------|---------|-------|
| `continuation_strategy.py:413-458` | Cumulative intervention check | 46 |
| `streaming/continuation.py:357` | Increment counter | 1 |
| `streaming/intent_classification.py:164` | Counter tracking | 1 |

---

## Proposed Strategy: Adaptive Progress-Based Continuation

### Core Principle

**Replace global intervention counter with progress-based metrics that distinguish between:**
1. **Productive exploration** (tool calls, new files read, progress)
2. **Spinning loops** (repeated queries without progress)
3. **Task complexity** (simple vs medium vs complex)

---

## Strategy 1: Progress Metrics (Recommended)

### Concept

Track **progress velocity** instead of absolute intervention count:
- Files read per minute
- Unique tools used
- New information discovered
- Moving forward vs revisiting

### Implementation

```python
@dataclass
class ProgressMetrics:
    """Track exploration progress."""
    files_read: Set[str] = field(default_factory=set)
    files_revisited: Set[str] = field(default_factory=set)
    tools_used: Set[str] = field(default_factory=set)
    iterations_without_tools: int = 0
    last_tool_call_iteration: int = 0
    stuck_patterns: List[str] = field(default_factory=list)

    @property
    def progress_velocity(self) -> float:
        """Calculate files read per iteration."""
        total_iters = self.last_tool_call_iteration or 1
        return len(self.files_read) / max(total_iters, 1)

    @property
    def is_making_progress(self) -> bool:
        """Determine if exploration is productive."""
        # Progress if: used tools recently OR reading new files
        recent_tools = (self.iterations_without_tools < 3)
        new_discovery = (len(self.files_read) > len(self.files_revisited))
        return recent_tools or new_discovery

    @property
    def is_stuck_loop(self) -> bool:
        """Detect unproductive cycling."""
        return (self.iterations_without_tools > 5 and
                len(self.files_revisited) > len(self.files_read))
```

### Adaptive Thresholds

```python
def get_continuation_action(metrics: ProgressMetrics) -> str:
    """Dynamic continuation based on progress."""

    # Making progress → allow continued exploration
    if metrics.is_making_progress:
        return "continue"

    # Stuck loop → force synthesis
    if metrics.is_stuck_loop:
        return "force_synthesis"

    # No tools but reading new content → gentle nudge
    if metrics.iterations_without_tools > 3:
        return "nudge_synthesis"

    return "continue"
```

---

## Strategy 2: Task Complexity Classification

### Concept

Classify tasks upfront and apply different thresholds:

| Complexity | Tool Budget | Max Iterations | Intervention Threshold |
|------------|-------------|----------------|------------------------|
| **Simple** | 10 | 5 | 2 interventions |
| **Medium** | 25 | 15 | 5 interventions |
| **Complex** | 100 | 50 | 10 interventions |

### Implementation

```python
def classify_task_complexity(prompt: str, initial_context: Dict) -> str:
    """Classify task complexity before execution."""

    # Simple questions
    if _is_simple_qa(prompt):
        return "simple"

    # Code analysis tasks
    if _is_analysis_task(prompt):
        context_size = initial_context.get('context_size', 0)
        if context_size > 100000:  # Large codebase
            return "complex"
        return "medium"

    # Multi-step tasks
    if _has_multiple_deliverables(prompt):
        return "complex"

    return "medium"
```

---

## Strategy 3: Token Budget with Soft Limits

### Concept

Use token-based budgets instead of iteration counts:

```python
@dataclass
class TokenBudget:
    """Token-based continuation limits."""

    # Percentage of context window
    exploration_budget: float = 0.25      # 25% for exploration
    synthesis_budget: float = 0.15         # 15% for final output
    buffer: float = 0.60                   # 60% buffer

    # Soft limits (warnings, not hard stops)
    exploration_soft_limit: float = 0.20   # Nudge at 20%
    synthesis_soft_limit: float = 0.10     # Nudge at 10%

    def should_nudge_synthesis(self, tokens_used: int, context_window: int) -> bool:
        """Check if synthesis nudge is needed."""
        usage_ratio = tokens_used / context_window
        return usage_ratio > self.exploration_soft_limit

    def should_force_synthesis(self, tokens_used: int, context_window: int) -> bool:
        """Check if synthesis should be forced."""
        usage_ratio = tokens_used / context_window
        return usage_ratio > self.exploration_budget
```

---

## Strategy 4: Hybrid Approach (Best of All Worlds)

### Combines All Strategies

```python
class AdaptiveContinuationStrategy:
    """Hybrid continuation strategy with multiple signals."""

    def determine_continuation_action(
        self,
        progress_metrics: ProgressMetrics,
        task_complexity: str,
        token_budget: TokenBudget,
        iteration: int,
    ) -> Dict[str, Any]:
        """Multi-factor continuation decision."""

        # Signal 1: Progress velocity (weight: 40%)
        progress_score = self._calculate_progress_score(progress_metrics)

        # Signal 2: Token budget (weight: 30%)
        token_score = self._calculate_token_score(token_budget)

        # Signal 3: Task complexity (weight: 20%)
        complexity_adjustment = self._get_complexity_multiplier(task_complexity)

        # Signal 4: Intervention count (weight: 10%)
        intervention_penalty = min(self._interventions / 20, 1.0)

        # Combined score (0-1, higher = continue)
        continuation_score = (
            progress_score * 0.40 +
            token_score * 0.30 +
            complexity_adjustment * 0.20 -
            intervention_penalty * 0.10
        )

        # Decision thresholds
        if continuation_score > 0.7:
            return {"action": "continue"}
        elif continuation_score > 0.4:
            return {"action": "continue_with_nudge"}
        elif continuation_score > 0.2:
            return {"action": "nudge_synthesis"}
        else:
            return {"action": "force_synthesis"}
```

---

## Implementation Plan

### Phase 1: Progress Metrics (High Priority)

**Files to modify:**
1. `victor/agent/streaming/continuation.py` - Add `ProgressMetrics` class
2. `victor/agent/continuation_strategy.py` - Update decision logic
3. `victor/agent/orchestrator.py` - Track progress per session

**Changes:**
```python
# Replace cumulative_prompt_interventions check with:
if not metrics.is_making_progress:
    if metrics.iterations_without_tools > 5:
        return {"action": "force_synthesis"}
    elif metrics.iterations_without_tools > 3:
        return {"action": "nudge_synthesis"}
```

### Phase 2: Task Complexity Classification (Medium Priority)

**Files to modify:**
1. `victor/agent/task_analyzer.py` - Add complexity classification
2. `victor/config/settings.py` - Add complexity thresholds

**Example:**
```python
COMPLEXITY_LIMITS = {
    "simple": {"max_iterations": 5, "tool_budget": 10},
    "medium": {"max_iterations": 15, "tool_budget": 25},
    "complex": {"max_iterations": 50, "tool_budget": 100},
}
```

### Phase 3: Token Budget Integration (Low Priority)

**Files to modify:**
1. `victor/agent/streaming/continuation.py` - Add token tracking
2. Reuse existing `get_provider_limits()` from `config_loaders.py`

---

## Quick Fix (Immediate Relief)

While implementing the full strategy, apply this **immediate fix**:

**File**: `victor/agent/continuation_strategy.py:418`

**Change:**
```python
# OLD (aggressive):
if cumulative_interventions >= 5 and is_analysis_task:

# NEW (progress-aware):
file_count = len(read_files)
unique_files = len(set(read_files))  # Dedupe
progress_ratio = unique_files / max(cumulative_interventions, 1)

# Only nudge if:
# - High interventions (>8) OR
# - Low progress (<0.5 files per intervention) AND >5 interventions
if (cumulative_interventions >= 8 or
    (progress_ratio < 0.5 and cumulative_interventions >= 5)) and is_analysis_task:
```

This allows:
- **Active exploration** to continue (reading new files)
- **Only nudges** when progress is slow
- **Forces synthesis** at 8+ interventions (was 5)

---

## Configuration-Driven Thresholds

Add to `victor/config/continuation_config.yaml`:

```yaml
continuation:
  # Progress-based thresholds
  progress:
    min_files_per_intervention: 0.5  # At least 1 new file per 2 iterations
    max_revisits_before_nudge: 3      # Nudge after 3 revisits without new files

  # Complexity-based limits
  complexity:
    simple:
      max_interventions: 2
      max_iterations: 5
      tool_budget: 10
    medium:
      max_interventions: 5
      max_iterations: 15
      tool_budget: 25
    complex:
      max_interventions: 10
      max_iterations: 50
      tool_budget: 100

  # Token budgets (percentage of context window)
  token_budgets:
    exploration: 0.25    # 25% for reading/exploring
    synthesis: 0.15      # 15% for final output
    soft_limit: 0.20     # Nudge when exceeding 20%
```

---

## Testing Strategy

### Test Cases

1. **Simple Prompt**: "What is 2+2?"
   - Expected: Complete in 1-2 iterations
   - Old: Would work fine
   - New: Same performance

2. **Medium Prompt**: "Read config.py and explain it"
   - Expected: 2-5 iterations, completes naturally
   - Old: May nudge at 5
   - New: Completes without nudging

3. **Complex Prompt**: "Analyze entire Victor architecture"
   - Expected: 20-50 iterations with active tool use
   - Old: Nudges at 5, forces at 8 ❌
   - New: Allows exploration until progress slows ✅

4. **Stuck Loop**: LLM re-reading same files
   - Expected: Detect and force synthesis
   - Old: Eventually forces (too late)
   - New: Detects pattern, forces earlier ✅

---

## Summary

| Issue | Current | Proposed |
|-------|---------|----------|
| **Simple prompts** | Works fine | Same performance |
| **Medium prompts** | May over-intervene | Natural completion |
| **Complex prompts** | ❌ Interrupted too early | ✅ Completes exploration |
| **Stuck loops** | Eventually stops | Detects and stops early |
| **Token efficiency** | Fixed iteration limits | Adaptive to context size |

---

## Next Steps

1. **Immediate**: Apply quick fix to continuation_strategy.py
2. **Short-term**: Implement ProgressMetrics class
3. **Medium-term**: Add task complexity classification
4. **Long-term**: Full hybrid approach with token budgets

**Estimated effort**:
- Quick fix: 15 minutes
- Progress metrics: 2-3 hours
- Complexity classification: 1-2 hours
- Full implementation: 1 day

---

## Implementation Status: ✅ COMPLETE

**Completion Date**: 2025-01-12
**Test Results**: 4004 passed, 1 skipped

### Phase 1: Progress Metrics ✅ COMPLETE

**Implementation Details:**
- Created `ProgressMetrics` dataclass in `victor/agent/streaming/continuation.py`
- Tracks: files read, revisits, tools used, iterations without tools, progress velocity, stuck loops
- Integrated into `AgentOrchestrator` for per-session tracking
- Updated continuation strategy to use progress-based decisions

**Key Features:**
```python
@dataclass
class ProgressMetrics:
    unique_files_read: int = 0
    total_file_reads: int = 0
    tools_used_count: int = 0
    iterations_without_tools: int = 0
    consecutive_file_revisits: int = 0

    @property
    def progress_velocity(self) -> float:
        """Files read per iteration."""
        return self.unique_files_read / max(self.total_iterations, 1)

    @property
    def is_making_progress(self) -> bool:
        """Agent is discovering new content or using tools."""
        return self.unique_files_read > len(self._get_revisited_files())

    @property
    def is_stuck_loop(self) -> bool:
        """Agent is cycling without progress."""
        return self.consecutive_file_revisits >= 3
```

### Phase 2: Task Complexity Classification ✅ COMPLETE

**Implementation Details:**
- Added complexity classification to `victor/agent/task_analyzer.py`
- Created complexity-based thresholds in `victor/config/settings.py`
- Four complexity levels: SIMPLE, MEDIUM, COMPLEX, GENERATION

**Thresholds:**
```python
COMPLEXITY_LIMITS = {
    "simple": {"max_interventions": 5, "max_iterations": 10},
    "medium": {"max_interventions": 10, "max_iterations": 25},
    "complex": {"max_interventions": 20, "max_iterations": 50},
    "generation": {"max_interventions": 15, "max_iterations": 35},
}
```

### Phase 3: Token Budget Integration ✅ COMPLETE

**Implementation Details:**
- Created `TokenBudget` dataclass with soft/hard limits
- Integrated with `get_provider_limits()` for model-specific context windows
- Dynamic warnings based on actual context window size

**Key Features:**
```python
@dataclass
class TokenBudget:
    context_window: int = 128000
    soft_limit_pct: float = 0.30  # Nudge at 30%
    hard_limit_pct: float = 0.70  # Force at 70%

    def should_nudge_synthesis(self, tokens_used: int) -> bool:
        return (tokens_used / self.context_window) >= self.soft_limit_pct

    def should_force_synthesis(self, tokens_used: int) -> bool:
        return (tokens_used / self.context_window) >= self.hard_limit_pct
```

### Phase 4: Hybrid Multi-Factor Scoring ✅ COMPLETE

**Implementation Details:**
- Created `ContinuationSignals` dataclass combining all signals
- Implemented weighted scoring system with 5 factors
- Score 0.0-1.0 determines action (continue/nudge_synthesis/force_synthesis)

**Weighted Scoring:**
```python
weights = {
    "progress_velocity": 0.30,      # 30% - Most important
    "stuck_loop_penalty": 0.25,     # 25% - Strong penalty
    "token_budget": 0.20,           # 20% - Context pressure
    "intervention_ratio": 0.15,     # 15% - Intervention count
    "complexity_adjustment": 0.10,  # 10% - Task complexity
}

# Decision thresholds:
# - Score >= 0.6: continue exploration
# - Score >= 0.3: nudge synthesis
# - Score < 0.3: force synthesis
```

### Files Modified

| File | Changes |
|------|---------|
| `victor/agent/streaming/continuation.py` | Added TokenBudget, ContinuationSignals, ProgressMetrics enhancements |
| `victor/agent/continuation_strategy.py` | Integrated hybrid scoring into decision logic |
| `victor/agent/orchestrator.py` | Initialize token budget from provider limits |
| `victor/config/settings.py` | Added complexity-based continuation thresholds |
| `tests/unit/agent/test_continuation_strategy.py` | Updated mock fixtures |
| `tests/unit/agent/test_continuation_loop_fix.py` | Updated mock fixtures |

### Test Results

```
========== 4004 passed, 1 skipped, 808 warnings in 133.89s ===========
```

All continuation strategy tests pass:
- `test_continuation_strategy.py`: 27 tests passed
- `test_continuation_loop_fix.py`: 19 tests passed

### Benefits Achieved

1. **Prevents premature synthesis** - Agent continues exploration when making progress
2. **Detects stuck loops** - Identifies and terminates unproductive cycling
3. **Adapts to task complexity** - Simple tasks complete quickly, complex tasks get adequate exploration
4. **Token-aware** - Respects model context windows with dynamic thresholds
5. **Multi-factor intelligence** - Combines progress, tokens, interventions, and complexity

### Optional Future Enhancements

While the core implementation is complete, optional future enhancements could include:
- RL-based learning of optimal continuation prompts
- More sophisticated token budget management
- Dynamic weight adjustment based on task performance
- User feedback integration for threshold tuning
