# Database Schema Redesign Proposal

## Executive Summary

This document proposes a strategic redesign of Victor's database schema to achieve:
- **Semantic consistency** - Table names reflect their purpose
- **Professional naming** - Industry-standard conventions
- **Logical grouping** - Related tables share prefixes
- **Reduced redundancy** - Consolidate overlapping tables
- **Scalability** - Schema supports future growth

---

## Current State Analysis

### Issues Identified

| Issue | Examples | Impact |
|-------|----------|--------|
| **Inconsistent naming** | `rl_outcomes` vs `tool_selector_outcomes` | Confusion about relationships |
| **Verbose prefixes** | `continuation_patience_stats`, `grounding_threshold_history` | Long, hard to remember |
| **Mixed patterns** | `*_q_values`, `*_stats`, `*_history` suffixes | No clear taxonomy |
| **Redundant tables** | 3 separate `*_q_values` patterns per learner | Schema bloat |
| **Unclear scope** | `sessions` (TUI? Conversation? Auth?) | Ambiguous purpose |

### Current Table Count: 39 tables

```
Learner-specific tables: 27 (69%)
Core system tables: 8 (21%)
Metadata tables: 4 (10%)
```

---

## Proposed Schema Design

### Design Principles

1. **Prefix by domain** - `rl_`, `agent_`, `ui_`, `graph_`
2. **Verb-noun pattern** - Describes what the table stores
3. **Singular nouns** - `rl_outcome` not `rl_outcomes`
4. **No redundant suffixes** - Remove `_stats`, `_history` where implied
5. **Max 3 words** - Keep names concise

---

## Proposed Table Structure

### 1. REINFORCEMENT LEARNING DOMAIN (`rl_`)

**Core RL Tables:**

| Current Name | Proposed Name | Purpose |
|--------------|---------------|---------|
| `rl_outcomes` | `rl_outcome` | All learner outcomes (central fact table) |
| `rl_telemetry` | `rl_metric` | Performance metrics & monitoring |
| - | `rl_learner` | Learner registry (name, type, config) |

**Q-Learning Tables (Consolidated):**

| Current Names | Proposed Name | Purpose |
|---------------|---------------|---------|
| `mode_transition_q_values` | `rl_q_value` | Unified Q-value storage |
| `model_selector_q_values` | ↑ | All learners use same table |
| `tool_selector_q_values` | ↑ | Partitioned by `learner_id` |
| `cache_eviction_q_values` | ↑ | |
| `workflow_q_values` | ↑ | |

**Schema for `rl_q_value`:**
```sql
CREATE TABLE rl_q_value (
    id INTEGER PRIMARY KEY,
    learner_id TEXT NOT NULL,      -- 'mode_transition', 'tool_selector', etc.
    state_key TEXT NOT NULL,        -- Encoded state
    action_key TEXT NOT NULL,       -- Encoded action
    q_value REAL DEFAULT 0.5,
    visit_count INTEGER DEFAULT 0,
    last_updated TEXT,
    UNIQUE(learner_id, state_key, action_key)
);
CREATE INDEX idx_rl_q_learner ON rl_q_value(learner_id);
```

**Learner History (Consolidated):**

| Current Names | Proposed Name | Purpose |
|---------------|---------------|---------|
| `mode_transition_history` | `rl_transition` | State transitions with rewards |
| `grounding_threshold_history` | ↑ | |
| `quality_weight_history` | ↑ | |
| `prompt_template_history` | ↑ | |
| `cache_eviction_history` | ↑ | |
| `curriculum_history` | ↑ | |

**Schema for `rl_transition`:**
```sql
CREATE TABLE rl_transition (
    id INTEGER PRIMARY KEY,
    learner_id TEXT NOT NULL,
    from_state TEXT,
    to_state TEXT,
    action TEXT,
    reward REAL,
    metadata TEXT,                  -- JSON for learner-specific data
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX idx_rl_trans_learner ON rl_transition(learner_id, created_at);
```

**Learner Parameters:**

| Current Names | Proposed Name | Purpose |
|---------------|---------------|---------|
| `continuation_patience_stats` | `rl_param` | Learned parameters |
| `continuation_prompts_stats` | ↑ | |
| `semantic_threshold_stats` | ↑ | |
| `grounding_threshold_params` | ↑ | |
| `grounding_threshold_stats` | ↑ | |
| `quality_weights` | ↑ | |
| `model_selector_state` | ↑ | |

**Schema for `rl_param`:**
```sql
CREATE TABLE rl_param (
    id INTEGER PRIMARY KEY,
    learner_id TEXT NOT NULL,
    param_key TEXT NOT NULL,        -- 'patience', 'threshold', 'weight', etc.
    param_value REAL,
    context TEXT,                   -- Provider/model/task context
    sample_count INTEGER DEFAULT 0,
    confidence REAL DEFAULT 0.5,
    updated_at TEXT,
    UNIQUE(learner_id, param_key, context)
);
```

**Cross-Vertical Learning:**

| Current Name | Proposed Name | Purpose |
|--------------|---------------|---------|
| `cross_vertical_patterns` | `rl_pattern` | Discovered patterns |
| `cross_vertical_applications` | `rl_pattern_use` | Pattern application tracking |

**Task-Specific Q-Values (Consolidated):**

| Current Names | Proposed Name | Purpose |
|---------------|---------------|---------|
| `mode_transition_task_stats` | `rl_task_stat` | Task-level statistics |
| `model_selector_task_q_values` | ↑ | |
| `tool_selector_task_q_values` | ↑ | |
| `cache_eviction_tool_values` | ↑ | |
| `tool_selector_outcomes` | ↑ | |

---

### 2. AGENT DOMAIN (`agent_`)

**Team Execution:**

| Current Name | Proposed Name | Purpose |
|--------------|---------------|---------|
| `team_composition_stats` | `agent_team_config` | Team composition Q-values |
| `team_execution_history` | `agent_team_run` | Team execution records |

**Workflow Execution:**

| Current Name | Proposed Name | Purpose |
|--------------|---------------|---------|
| `workflow_executions` | `agent_workflow_run` | Workflow execution records |

**Prompt Templates:**

| Current Name | Proposed Name | Purpose |
|--------------|---------------|---------|
| `prompt_template_styles` | `agent_prompt_style` | Prompt style definitions |
| `prompt_template_elements` | `agent_prompt_element` | Prompt components |

**Curriculum & Policy:**

| Current Name | Proposed Name | Purpose |
|--------------|---------------|---------|
| `curriculum_stages` | `agent_curriculum_stage` | Learning curriculum |
| `curriculum_metrics` | `agent_curriculum_metric` | Curriculum performance |
| `policy_checkpoints` | `agent_policy_snapshot` | Policy state snapshots |

---

### 3. UI DOMAIN (`ui_`)

| Current Name | Proposed Name | Purpose |
|--------------|---------------|---------|
| `sessions` | `ui_session` | TUI session persistence |
| `failed_signatures` | `ui_failed_call` | Failed tool call signatures |

---

### 4. SYSTEM DOMAIN (`sys_`)

| Current Name | Proposed Name | Purpose |
|--------------|---------------|---------|
| `_db_metadata` | `sys_metadata` | Database metadata |
| `schema_version` | `sys_schema_version` | Schema versioning |

---

## Consolidation Summary

### Before: 39 tables
### After: 18 tables (54% reduction)

| Domain | Before | After | Reduction |
|--------|--------|-------|-----------|
| RL Core | 2 | 3 | +1 (added registry) |
| RL Q-Values | 5 | 1 | -4 (consolidated) |
| RL History | 6 | 1 | -5 (consolidated) |
| RL Params | 7 | 1 | -6 (consolidated) |
| RL Task Stats | 5 | 1 | -4 (consolidated) |
| RL Patterns | 2 | 2 | 0 |
| Agent | 7 | 7 | 0 |
| UI | 2 | 2 | 0 |
| System | 2 | 2 | 0 |
| **Total** | **39** | **18** | **-21** |

---

## Migration Strategy

### Phase 1: Create New Schema
```sql
-- Create new consolidated tables
CREATE TABLE rl_q_value (...);
CREATE TABLE rl_transition (...);
CREATE TABLE rl_param (...);
CREATE TABLE rl_task_stat (...);
```

### Phase 2: Data Migration
```sql
-- Migrate Q-values from all learners
INSERT INTO rl_q_value (learner_id, state_key, action_key, q_value, visit_count, last_updated)
SELECT 'mode_transition', state_key, action_key, q_value, visit_count, last_updated
FROM mode_transition_q_values;

INSERT INTO rl_q_value (learner_id, state_key, action_key, q_value, visit_count, last_updated)
SELECT 'model_selector', state_key, action_key, q_value, visit_count, last_updated
FROM model_selector_q_values;
-- ... repeat for other learners
```

### Phase 3: Update Code
1. Create `victor/core/schema.py` with table definitions
2. Update learners to use consolidated tables
3. Add `learner_id` column to queries
4. Update `RLCoordinator` to use new schema

### Phase 4: Deprecate Old Tables
```sql
-- After verification, drop old tables
DROP TABLE mode_transition_q_values;
DROP TABLE model_selector_q_values;
-- ... etc
```

---

## New Learner Interface

With consolidated tables, learners become simpler:

```python
class BaseLearner:
    """Unified learner interface using consolidated schema."""

    def __init__(self, learner_id: str, db: DatabaseManager):
        self.learner_id = learner_id
        self.db = db

    def get_q_value(self, state: str, action: str) -> float:
        row = self.db.query_one(
            "SELECT q_value FROM rl_q_value WHERE learner_id=? AND state_key=? AND action_key=?",
            (self.learner_id, state, action)
        )
        return row[0] if row else 0.5

    def update_q_value(self, state: str, action: str, new_q: float):
        self.db.execute("""
            INSERT INTO rl_q_value (learner_id, state_key, action_key, q_value, visit_count, last_updated)
            VALUES (?, ?, ?, ?, 1, datetime('now'))
            ON CONFLICT(learner_id, state_key, action_key) DO UPDATE SET
                q_value = excluded.q_value,
                visit_count = visit_count + 1,
                last_updated = excluded.last_updated
        """, (self.learner_id, state, action, new_q))

    def record_transition(self, from_state: str, to_state: str, action: str, reward: float, metadata: dict = None):
        self.db.execute("""
            INSERT INTO rl_transition (learner_id, from_state, to_state, action, reward, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (self.learner_id, from_state, to_state, action, reward, json.dumps(metadata)))
```

---

## Benefits

1. **Simpler queries** - Single table per concept
2. **Easier analytics** - Cross-learner analysis without JOINs across many tables
3. **Reduced maintenance** - Fewer tables to manage
4. **Consistent naming** - Easy to understand and remember
5. **Scalable** - Add new learners without new tables
6. **Professional** - Industry-standard naming conventions

---

## Implementation Priority

| Priority | Task | Effort |
|----------|------|--------|
| P0 | Fix duplicate `template_stats` in recovery/prompts.py | 1 hour |
| P1 | Create consolidated schema (SQL) | 2 hours |
| P1 | Write migration script | 4 hours |
| P2 | Update `BaseLearner` interface | 4 hours |
| P2 | Update individual learners | 8 hours |
| P3 | Remove deprecated tables | 1 hour |

**Total estimated effort: 20 hours**

---

## Appendix: Full Proposed Schema

```sql
-- ===========================================
-- REINFORCEMENT LEARNING DOMAIN
-- ===========================================

-- Learner registry
CREATE TABLE rl_learner (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,  -- 'q_learning', 'bandit', 'policy_gradient'
    config TEXT,         -- JSON configuration
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT
);

-- Central outcome storage (fact table)
CREATE TABLE rl_outcome (
    id INTEGER PRIMARY KEY,
    learner_id TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    task_type TEXT,
    vertical TEXT DEFAULT 'coding',
    success INTEGER,
    quality_score REAL,
    metadata TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (learner_id) REFERENCES rl_learner(id)
);
CREATE INDEX idx_outcome_learner ON rl_outcome(learner_id, created_at);
CREATE INDEX idx_outcome_context ON rl_outcome(provider, model, task_type);

-- Unified Q-value storage
CREATE TABLE rl_q_value (
    id INTEGER PRIMARY KEY,
    learner_id TEXT NOT NULL,
    state_key TEXT NOT NULL,
    action_key TEXT NOT NULL,
    q_value REAL DEFAULT 0.5,
    visit_count INTEGER DEFAULT 0,
    last_updated TEXT,
    UNIQUE(learner_id, state_key, action_key)
);
CREATE INDEX idx_q_learner ON rl_q_value(learner_id);

-- State transition history
CREATE TABLE rl_transition (
    id INTEGER PRIMARY KEY,
    learner_id TEXT NOT NULL,
    from_state TEXT,
    to_state TEXT,
    action TEXT,
    reward REAL,
    metadata TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX idx_trans_learner ON rl_transition(learner_id, created_at);

-- Learned parameters
CREATE TABLE rl_param (
    id INTEGER PRIMARY KEY,
    learner_id TEXT NOT NULL,
    param_key TEXT NOT NULL,
    param_value REAL,
    context TEXT,
    sample_count INTEGER DEFAULT 0,
    confidence REAL DEFAULT 0.5,
    updated_at TEXT,
    UNIQUE(learner_id, param_key, context)
);

-- Task-level statistics
CREATE TABLE rl_task_stat (
    id INTEGER PRIMARY KEY,
    learner_id TEXT NOT NULL,
    task_type TEXT NOT NULL,
    stat_key TEXT NOT NULL,
    stat_value REAL,
    sample_count INTEGER DEFAULT 0,
    updated_at TEXT,
    UNIQUE(learner_id, task_type, stat_key)
);

-- Cross-vertical patterns
CREATE TABLE rl_pattern (
    id TEXT PRIMARY KEY,
    task_type TEXT NOT NULL,
    pattern_name TEXT,
    avg_quality REAL,
    confidence REAL,
    source_verticals TEXT,
    recommendation TEXT,
    sample_count INTEGER DEFAULT 0,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE rl_pattern_use (
    id INTEGER PRIMARY KEY,
    pattern_id TEXT NOT NULL,
    target_vertical TEXT,
    success INTEGER,
    quality_score REAL,
    applied_at TEXT,
    FOREIGN KEY (pattern_id) REFERENCES rl_pattern(id)
);

-- Telemetry/metrics
CREATE TABLE rl_metric (
    id INTEGER PRIMARY KEY,
    learner_id TEXT,
    metric_type TEXT NOT NULL,
    metric_value REAL,
    metadata TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- ===========================================
-- AGENT DOMAIN
-- ===========================================

-- Team configuration learning
CREATE TABLE agent_team_config (
    id INTEGER PRIMARY KEY,
    config_key TEXT UNIQUE NOT NULL,
    formation TEXT NOT NULL,
    role_counts TEXT NOT NULL,
    task_category TEXT,
    execution_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    avg_quality REAL DEFAULT 0.5,
    avg_duration REAL DEFAULT 0,
    q_value REAL DEFAULT 0.5,
    updated_at TEXT
);

-- Team execution history
CREATE TABLE agent_team_run (
    id INTEGER PRIMARY KEY,
    team_id TEXT NOT NULL,
    task_category TEXT,
    formation TEXT,
    role_counts TEXT,
    member_count INTEGER,
    budget_used INTEGER,
    tools_used INTEGER,
    success INTEGER,
    quality_score REAL,
    duration_seconds REAL,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Workflow execution
CREATE TABLE agent_workflow_run (
    id INTEGER PRIMARY KEY,
    workflow_name TEXT NOT NULL,
    task_type TEXT,
    success INTEGER,
    duration_seconds REAL,
    quality_score REAL,
    vertical TEXT,
    mode TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Prompt styles
CREATE TABLE agent_prompt_style (
    id TEXT PRIMARY KEY,
    style_name TEXT NOT NULL,
    description TEXT,
    template TEXT,
    success_rate REAL DEFAULT 0.5,
    usage_count INTEGER DEFAULT 0,
    updated_at TEXT
);

-- Prompt elements
CREATE TABLE agent_prompt_element (
    id INTEGER PRIMARY KEY,
    style_id TEXT,
    element_type TEXT,
    content TEXT,
    position INTEGER,
    FOREIGN KEY (style_id) REFERENCES agent_prompt_style(id)
);

-- Curriculum stages
CREATE TABLE agent_curriculum_stage (
    id INTEGER PRIMARY KEY,
    stage_name TEXT NOT NULL,
    difficulty REAL,
    prerequisites TEXT,
    completion_criteria TEXT,
    created_at TEXT
);

-- Curriculum metrics
CREATE TABLE agent_curriculum_metric (
    id INTEGER PRIMARY KEY,
    stage_id INTEGER,
    metric_name TEXT,
    metric_value REAL,
    created_at TEXT,
    FOREIGN KEY (stage_id) REFERENCES agent_curriculum_stage(id)
);

-- Policy snapshots
CREATE TABLE agent_policy_snapshot (
    id INTEGER PRIMARY KEY,
    learner_id TEXT,
    snapshot_name TEXT,
    policy_data TEXT,
    performance_score REAL,
    created_at TEXT DEFAULT (datetime('now'))
);

-- ===========================================
-- UI DOMAIN
-- ===========================================

-- TUI sessions
CREATE TABLE ui_session (
    id TEXT PRIMARY KEY,
    name TEXT,
    provider TEXT,
    model TEXT,
    profile TEXT,
    data TEXT,
    created_at TEXT,
    updated_at TEXT
);
CREATE INDEX idx_session_updated ON ui_session(updated_at DESC);

-- Failed tool calls (loop prevention)
CREATE TABLE ui_failed_call (
    id INTEGER PRIMARY KEY,
    tool_name TEXT NOT NULL,
    args_hash TEXT NOT NULL,
    args_json TEXT,
    error_message TEXT,
    failure_count INTEGER DEFAULT 1,
    first_seen REAL,
    last_seen REAL,
    expires_at REAL,
    UNIQUE(tool_name, args_hash)
);
CREATE INDEX idx_failed_lookup ON ui_failed_call(tool_name, args_hash);
CREATE INDEX idx_failed_expires ON ui_failed_call(expires_at);

-- ===========================================
-- SYSTEM DOMAIN
-- ===========================================

-- Database metadata
CREATE TABLE sys_metadata (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TEXT
);

-- Schema version
CREATE TABLE sys_schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT DEFAULT (datetime('now'))
);
```
