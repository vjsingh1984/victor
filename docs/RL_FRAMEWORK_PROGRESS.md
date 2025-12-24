# RL Framework Migration - Progress Report

**Status:** Phase 1 Complete (4/10 tasks) ✅
**Date:** 2025-12-18

---

## ✅ Phase 1 Complete: Framework Infrastructure

### What Was Built

1. **Directory Structure**
   ```
   victor/agent/rl/
   ├── __init__.py              ✅ Framework exports
   ├── base.py                  ✅ BaseLearner, RLOutcome, RLRecommendation
   ├── coordinator.py           ✅ RLCoordinator with unified SQLite
   └── learners/
       ├── __init__.py          ✅ Learners package
       └── continuation_patience.py  ✅ New learner implementation
   ```

2. **Base Classes (`victor/agent/rl/base.py`)**
   - `RLOutcome` - Standardized outcome data model
   - `RLRecommendation` - Standardized recommendation format
   - `BaseLearner` - Abstract base with common infrastructure:
     - SQLite database access
     - Provider adapter integration
     - Telemetry collection hooks
     - Statistics export

3. **RLCoordinator (`victor/agent/rl/coordinator.py`)**
   - **Centralized coordination** for all learners
   - **Unified SQLite database** at `~/.victor/graph/graph.db` (reuses existing graph DB)
   - **Lazy learner initialization** to avoid circular imports
   - **Shared outcomes table** for cross-learner analysis
   - **Telemetry table** for monitoring
   - **Global singleton** pattern via `get_rl_coordinator()`

4. **ContinuationPatienceLearner** (NEW)
   - Learns optimal `continuation_patience` per provider:model:task
   - Currently static values (DeepSeek: 5, Claude: 3) will be refined
   - Tracks false positives (flagged as stuck but wasn't) vs true positives
   - Adjusts patience based on FP rate and missed stuck loop rate
   - Range: [1, 15] prompts

### Database Schema

**Core Tables** (in `~/.victor/graph/graph.db`):

```sql
-- Shared outcomes table for all learners
CREATE TABLE rl_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    learner_name TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    task_type TEXT NOT NULL,
    vertical TEXT NOT NULL,
    success INTEGER NOT NULL,
    quality_score REAL NOT NULL,
    metadata TEXT,
    timestamp TEXT NOT NULL,
    created_at REAL DEFAULT (julianday('now'))
);

-- Telemetry for monitoring
CREATE TABLE rl_telemetry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    learner_name TEXT NOT NULL,
    event_type TEXT NOT NULL,
    data TEXT,
    timestamp TEXT NOT NULL
);

-- Continuation patience stats
CREATE TABLE continuation_patience_stats (
    context_key TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    task_type TEXT NOT NULL,
    current_patience INTEGER NOT NULL,
    total_sessions INTEGER DEFAULT 0,
    false_positives INTEGER DEFAULT 0,
    true_positives INTEGER DEFAULT 0,
    missed_stuck_loops INTEGER DEFAULT 0,
    last_updated TEXT
);
```

### Usage Example

```python
from victor.agent.rl.coordinator import get_rl_coordinator
from victor.agent.rl.base import RLOutcome

# Get coordinator (lazy init)
coordinator = get_rl_coordinator()

# Record outcome after session
outcome = RLOutcome(
    provider="deepseek",
    model="deepseek-chat",
    task_type="analysis",
    success=True,
    quality_score=0.75,
    metadata={
        "flagged_as_stuck": True,
        "actually_stuck": False,  # False positive
        "eventually_made_progress": True,
    },
    vertical="coding",
)

coordinator.record_outcome("continuation_patience", outcome, "coding")

# Get recommendation
recommendation = coordinator.get_recommendation(
    "continuation_patience",
    provider="deepseek",
    model="deepseek-chat",
    task_type="analysis",
)

if recommendation:
    patience = recommendation.value
    confidence = recommendation.confidence
    print(f"Use patience={patience} (confidence={confidence:.2f})")
```

### Key Design Decisions

1. **Reuse graph.db instead of separate RL database**
   - Consolidates all persistent data
   - Avoids multiple database files
   - Uses RL-prefixed tables to avoid conflicts

2. **Lazy learner initialization**
   - Avoids circular import issues
   - Only creates learners when first accessed
   - Reduces startup time

3. **Shared outcomes table**
   - All learners record to same table
   - Enables cross-learner analysis
   - Simplifies telemetry and monitoring

4. **Provider adapter integration**
   - Learners can access provider baselines
   - Start with static values, refine over time
   - Falls back to baseline if no learning data

---

## ⏳ Phase 2 Pending: Migrate Existing Learners

### What Needs to Be Done

1. **Migrate ContinuationPromptLearner**
   - Move from `victor/agent/continuation_learner.py`
   - To `victor/agent/rl/learners/continuation_prompts.py`
   - Inherit from `BaseLearner`
   - Convert JSON storage to SQLite tables
   - Migrate existing learned data from `~/.victor/rl_data/continuation_rl.json`

2. **Migrate SemanticThresholdLearner**
   - Move from `victor/codebase/semantic_threshold_learner.py`
   - To `victor/agent/rl/learners/semantic_threshold.py`
   - Convert JSON storage to SQLite
   - Migrate data from `~/.victor/rl_data/semantic_threshold_rl.json`

3. **Migrate RLModelSelector**
   - Move from `victor/agent/rl_model_selector.py`
   - To `victor/agent/rl/learners/model_selector.py`
   - Refactor as `ModelSelectorLearner(BaseLearner)`
   - Convert JSON storage to SQLite
   - Migrate data from `~/.victor/rl_data/model_selector_q.json`

### Migration Strategy

For each learner:
1. Create new file in `victor/agent/rl/learners/`
2. Inherit from `BaseLearner`
3. Implement `_ensure_tables()` - create SQLite tables
4. Implement `record_outcome()` - record to SQLite
5. Implement `get_recommendation()` - query from SQLite
6. Implement `_compute_reward()` - reward function
7. Add data migration utility to convert JSON → SQLite
8. Keep old file for backward compatibility (deprecated)
9. Update all import sites progressively

---

## ⏳ Phase 3 Pending: Update Integration Points

### Orchestrator Integration

**File:** `victor/agent/orchestrator.py`

**Current (Siloed):**
```python
from victor.agent.continuation_learner import ContinuationPromptLearner

self._continuation_learner = ContinuationPromptLearner()
learned_val = self._continuation_learner.get_recommendation(...)
```

**Target (Framework):**
```python
from victor.agent.rl.coordinator import get_rl_coordinator

self._rl_coordinator = get_rl_coordinator()
learned_val = self._rl_coordinator.get_recommendation(
    "continuation_prompts", provider, model, task_type
)
```

**Additional Changes:**
1. Record continuation patience outcomes for new learner
2. Query continuation patience recommendation
3. Use RL-enhanced provider capabilities

### Tool Integration

**File:** `victor/tools/code_search_tool.py`

**Current:**
```python
from victor.codebase.semantic_threshold_learner import SemanticThresholdLearner

learner = SemanticThresholdLearner()
threshold = learner.get_recommendation(...)
```

**Target:**
```python
from victor.agent.rl.coordinator import get_rl_coordinator

coordinator = get_rl_coordinator()
recommendation = coordinator.get_recommendation(
    "semantic_threshold", embedding_model, task_type, "semantic_code_search"
)
threshold = recommendation.value if recommendation else default
```

### All Import Sites (from graph database analysis)

1. **ContinuationPromptLearner** → orchestrator.py, show_continuation_rl.py
2. **SemanticThresholdLearner** → code_search_tool.py, p4_dashboard.py, show_semantic_threshold_rl.py
3. **RLModelSelector** → orchestrator.py, fastapi_server.py, server.py, slash_commands.py, utils.py

---

## ⏳ Phase 4 Pending: Provider Adapter Integration

### Goal

Make provider capabilities RL-aware so they return learned values instead of static baselines.

**File:** `victor/protocols/provider_adapter.py`

**Add method:**
```python
def get_param_with_rl(
    self,
    param_name: str,
    provider: str,
    model: str,
    task_type: str,
) -> Any:
    """Get parameter value, using RL-learned value if available.

    Priority:
    1. RL-learned value (if confidence > 0.5)
    2. Provider static baseline (from capabilities)
    3. Global default
    """
    if param_name not in self.rl_learnable_params:
        return getattr(self, param_name)

    from victor.agent.rl.coordinator import get_rl_coordinator

    coordinator = get_rl_coordinator()
    recommendation = coordinator.get_recommendation(
        param_name, provider, model, task_type
    )

    if recommendation and recommendation.confidence > 0.5:
        logger.debug(
            f"RL: Using learned {param_name}={recommendation.value} "
            f"(confidence={recommendation.confidence:.2f})"
        )
        return recommendation.value

    # Fall back to static baseline
    return getattr(self, param_name)
```

**Update orchestrator:**
```python
# OLD
patience_threshold = self.tool_calling_caps.continuation_patience

# NEW
patience_threshold = self.tool_calling_caps.get_param_with_rl(
    "continuation_patience",
    self.provider.name,
    self.model,
    task_type
)
```

---

## ⏳ Phase 5 Pending: Data Migration

### Migrate Existing JSON Data to SQLite

1. **ContinuationPromptLearner data**
   - Source: `~/.victor/rl_data/continuation_rl.json`
   - Target: `continuation_prompts_stats` table in graph.db

2. **SemanticThresholdLearner data**
   - Source: `~/.victor/rl_data/semantic_threshold_rl.json`
   - Target: `semantic_threshold_stats` table in graph.db

3. **RLModelSelector data**
   - Source: `~/.victor/rl_data/model_selector_q.json`
   - Target: `model_selector_stats` table in graph.db

**Migration Script:**
```python
# scripts/migrate_rl_data.py
from victor.agent.rl.coordinator import get_rl_coordinator
import json

def migrate_continuation_prompts():
    """Migrate continuation_rl.json to SQLite."""
    with open("~/.victor/rl_data/continuation_rl.json") as f:
        data = json.load(f)

    coordinator = get_rl_coordinator()
    learner = coordinator.get_learner("continuation_prompts")

    # Migrate stats
    for key, stats in data["stats"].items():
        # Convert to SQLite row
        ...

    # Migrate outcomes
    for outcome_data in data["outcomes"]:
        outcome = RLOutcome.from_dict(outcome_data)
        learner.record_outcome(outcome)
```

---

## Testing Plan

### Unit Tests

1. **test_rl_coordinator.py**
   - Test coordinator initialization
   - Test learner registration
   - Test outcome recording
   - Test recommendation retrieval
   - Test metrics export

2. **test_continuation_patience_learner.py**
   - Test table creation
   - Test outcome recording
   - Test patience adjustment logic
   - Test recommendation with/without data
   - Test provider baseline fallback

3. **test_rl_base.py**
   - Test RLOutcome serialization
   - Test RLRecommendation
   - Test BaseLearner abstract methods

### Integration Tests

1. **test_orchestrator_rl_integration.py**
   - Test orchestrator uses coordinator
   - Test continuation patience learning
   - Test continuation prompts learning
   - Test data persistence

2. **test_tool_rl_integration.py**
   - Test code_search_tool uses coordinator
   - Test semantic threshold learning

### End-to-End Tests

1. Run session with DeepSeek
2. Record stuck loop false positive
3. Verify patience increased
4. Run another session
5. Verify learned patience used

---

## Success Metrics

### Before Migration (Current State)

| Aspect | Status |
|--------|--------|
| RL centralization | ❌ Siloed in components |
| Unified database | ❌ Multiple JSON files |
| Cross-vertical learning | ❌ None |
| Provider adapter integration | ❌ Static values only |
| Continuation patience learning | ❌ Static per provider |
| Framework-level RL | ❌ Application-level only |

### After Phase 1 (Current)

| Aspect | Status |
|--------|--------|
| RL centralization | ⏳ Infrastructure created |
| Unified database | ✅ Single SQLite DB (graph.db) |
| Cross-vertical learning | ⏳ Infrastructure ready |
| Provider adapter integration | ⏳ Planned |
| Continuation patience learning | ✅ Implemented |
| Framework-level RL | ✅ Base classes created |

### After Full Migration (Target)

| Aspect | Status |
|--------|--------|
| RL centralization | ✅ Single coordinator |
| Unified database | ✅ All data in SQLite |
| Cross-vertical learning | ✅ Automatic propagation |
| Provider adapter integration | ✅ RL-enhanced capabilities |
| Continuation patience learning | ✅ Fully integrated |
| Framework-level RL | ✅ All learners migrated |

---

## Next Steps (Prioritized)

### Immediate (Next Session)

1. **Migrate ContinuationPromptLearner** - Highest priority, used by orchestrator
2. **Update orchestrator integration** - Test with real sessions
3. **Create data migration script** - Preserve existing learning

### Short Term (This Week)

4. **Migrate SemanticThresholdLearner** - Used by code search
5. **Migrate RLModelSelector** - Used across multiple layers
6. **Update all import sites** - Orchestrator, tools, API, UI

### Medium Term (Next Week)

7. **Provider adapter RL integration** - `get_param_with_rl()` method
8. **Cross-vertical learning** - Share insights across verticals
9. **Comprehensive testing** - Unit, integration, e2e tests

### Long Term (Next Month)

10. **Add more learners** - QualityThresholdLearner, GroundingStrictnessLearner
11. **Dashboard integration** - Visualize learned values in P4 dashboard
12. **Documentation** - User guide, API docs, examples

---

## Files Created This Session

1. `victor/agent/rl/__init__.py` - Framework exports
2. `victor/agent/rl/base.py` - Base classes (370 lines)
3. `victor/agent/rl/coordinator.py` - RLCoordinator (290 lines)
4. `victor/agent/rl/learners/__init__.py` - Learners package
5. `victor/agent/rl/learners/continuation_patience.py` - New learner (290 lines)

**Total:** ~950 lines of new framework code

---

## Summary

✅ **Phase 1 Complete:** Framework infrastructure is in place with unified SQLite storage, centralized coordination, and the first new learner (ContinuationPatienceLearner) implemented.

⏳ **Next Priority:** Migrate existing learners to framework and update orchestrator integration to start collecting real learning data.

🎯 **Goal:** Framework-level RL that benefits all verticals, with learned values complementing provider adapter baselines.
