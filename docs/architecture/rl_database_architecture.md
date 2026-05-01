# RL Database Architecture

**Status**: ✅ IMPLEMENTED — RL learning data consolidated to global database  
**Date**: 2026-04-21  
**Issue**: "no such column: created_at" error in AdaptiveModeController initialization

---

## Overview

Victor's reinforcement learning (RL) system stores all learning data in a **single global database** (`~/.victor/victor.db`) rather than per-project databases. This architectural decision enables cross-project learning, global prompt optimization (GEPA), and provider-agnostic intelligence.

---

## Design Principles

### 1. Cross-Project Learning

**Rationale**: RL algorithms learn optimal behaviors from ALL projects, not just one.

**Examples**:
- **Tool effectiveness**: Learning that `code_search` works better than `grep` across all codebases
- **Mode transitions**: Learning when to switch from `explore` → `build` mode across different task types
- **Provider selection**: Learning that GPT-4 is better for architecture, Ollama for debugging (across all projects)

**Implementation**:
```python
# All learners share the same global database
from victor.core.database import DatabaseManager

db = DatabaseManager()  # ~/.victor/victor.db
```

### 2. GEPA Integration

**GEPA** (Grounding-Evidence-Prompt-Advice) is Victor's prompt optimization system that evolves system prompts using execution traces. GEPA requires **global RL data** to:

- Learn which prompt sections work best across different providers
- Track prompt effectiveness metrics across all projects
- Perform A/B testing on prompts with cross-project validation

**Data Flow**:
```
All Projects → Execution Traces → GEPA Analysis → Global RL DB
                                                              ↓
                                          Optimized Prompts (Shared)
```

### 3. Provider-Agnostic Learning

**Goal**: Learn provider capabilities and limitations independently of projects.

**Examples**:
- **Anthropic**: Good at complex reasoning, 200K output tokens
- **OpenAI**: Good at code generation, fast responses
- **Ollama/Local**: Good for debugging, privacy, cost-effective

**Storage**:
```sql
-- Provider capability tracking (global)
CREATE TABLE rl_outcome (
    learner_id TEXT NOT NULL,
    provider TEXT,           -- Provider identifier
    model TEXT,              -- Model identifier
    task_type TEXT,          -- Task category
    success INTEGER,
    quality_score REAL,
    created_at TEXT DEFAULT (datetime('now'))
);
```

---

## Database Schema

### Global Database Tables

**Location**: `~/.victor/victor.db`

#### RL Core Tables

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `rl_outcome` | All learner outcomes (central fact table) | `learner_id`, `provider`, `model`, `task_type`, `created_at` |
| `rl_q_value` | Unified Q-values (partitioned by learner) | `learner_id`, `state_key`, `action_key`, `q_value` |
| `rl_transition` | Unified state transitions (partitioned) | `learner_id`, `from_state`, `to_action`, `reward` |
| `rl_param` | Unified parameters (partitioned) | `learner_id`, `param_name`, `param_value` |
| `rl_task_stat` | Unified task statistics (partitioned) | `learner_id`, `task_type`, `success_rate` |
| `rl_metric` | Telemetry and monitoring | `metric_name`, `metric_value`, `timestamp` |

#### Mode Transition Learner Tables

| Table | Purpose |
|-------|---------|
| `rl_mode_q` | Mode transition Q-values |
| `rl_mode_history` | Mode transition history with timestamps |
| `rl_mode_task` | Mode task statistics (optimal budgets, quality scores) |

#### Model Selector Learner Tables

| Table | Purpose |
|-------|---------|
| `rl_model_q` | Model selection Q-values |
| `rl_model_task` | Model task Q-values |
| `rl_model_state` | Model selector state |

#### Tool Selector Learner Tables

| Table | Purpose |
|-------|---------|
| `rl_tool_q` | Tool selection Q-values |
| `rl_tool_task` | Tool task Q-values |
| `rl_tool_outcome` | Tool-specific outcomes |

#### Prompt Optimization Tables (GEPA)

| Table | Purpose |
|-------|---------|
| `agent_prompt_candidate` | Prompt variants for A/B testing |
| `agent_prompt_element` | Prompt section elements |
| `agent_prompt_history` | Prompt evolution history |
| `agent_prompt_pareto_instance` | Pareto-optimal prompt configurations |
| `agent_prompt_style` | Prompt style templates |

#### Agent Learning Tables

| Table | Purpose |
|-------|---------|
| `agent_workflow_q` | Workflow selection Q-values |
| `agent_workflow_run` | Workflow execution history |
| `agent_team_config` | Team composition statistics |
| `agent_team_run` | Team execution history |
| `agent_curriculum_stage` | Learning curriculum stages |
| `agent_curriculum_metrics` | Curriculum progress metrics |

---

## Data Distribution: Global vs Project

### Global Database (`~/.victor/victor.db`)

**Contains**:
- ✅ RL learning data (all tables above)
- ✅ Provider configurations
- ✅ Model families and sizes
- ✅ Context sizes
- ✅ Prompt optimization data (GEPA)
- ✅ User preferences
- ✅ Failed signatures cache
- ✅ Usage analytics

**Rationale**: Cross-project learning, global optimization, provider-agnostic intelligence

### Project Database (`.victor/project.db`)

**Contains**:
- ✅ Sessions and messages (conversations)
- ✅ Context summaries
- ✅ File changes
- ✅ Graph data (nodes, edges)
- ✅ Profile learning (project-specific)
- ✅ Symbols
- ✅ Topic segments
- ✅ Interaction history

**Rationale**: Project-specific data, portability, isolation

---

## Learner Implementations

### Mode Transition Learner

**Purpose**: Learn optimal mode transitions (explore → plan → build → review)

**Storage**:
```python
class QLearningStore:
    def __init__(self, project_path: Optional[Path] = None):
        # Uses GLOBAL database for cross-project learning
        from victor.core.database import DatabaseManager
        self._db = DatabaseManager()
```

**Tables Used**:
- `rl_mode_q`: Q-values for (state, action) pairs
- `rl_mode_history`: Transition history with timestamps
- `rl_mode_task`: Task statistics (optimal tool budgets, quality scores)

### Model Selector Learner

**Purpose**: Learn which models work best for different tasks

**Storage**: Global database via `DatabaseManager()`

**Tables Used**:
- `rl_model_q`: Model selection Q-values
- `rl_model_task`: Per-task model performance
- `rl_model_state`: Current selector state

### Tool Selector Learner

**Purpose**: Learn optimal tool selection strategies

**Storage**: Global database via `DatabaseManager()`

**Tables Used**:
- `rl_tool_q`: Tool selection Q-values
- `rl_tool_task`: Per-task tool performance
- `rl_tool_outcome`: Tool execution outcomes

---

## Migration Path

### Migration History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Initial | Project database with mixed schema |
| 2.0 | 2026-04-21 | **RL data moved to global database** |

### Migration Code

**Before** (Incorrect - project database):
```python
# victor/agent/adaptive_mode_controller.py (OLD)
from victor.core.database import get_project_database

self._db = get_project_database(project_path)  # ✗ Wrong DB
```

**After** (Correct - global database):
```python
# victor/agent/adaptive_mode_controller.py (NEW)
from victor.core.database import DatabaseManager

self._db = DatabaseManager()  # ✓ Correct - global DB
```

### Verification

```bash
# Verify RL tables are in global database
sqlite3 ~/.victor/victor.db ".tables" | grep rl_

# Verify mode transition tables
sqlite3 ~/.victor/victor.db "PRAGMA table_info(rl_mode_q);"

# Verify sample data
sqlite3 ~/.victor/victor.db "SELECT COUNT(*) FROM rl_outcome;"
```

---

## Benefits of Global RL Database

### 1. Cross-Project Knowledge Transfer

**Scenario**: User works on 3 projects (A, B, C)

**Before** (Per-project RL):
- Project A learns: `code_search` is 80% effective
- Project B learns: `code_search` is 75% effective
- Project C starts from scratch

**After** (Global RL):
- Projects A, B, C contribute to shared statistics
- All projects benefit from combined data: `code_search` is 78% effective globally
- New projects start with pre-trained knowledge

### 2. GEPA Prompt Optimization

**GEPA** requires global execution traces to:
- Identify effective prompt patterns across projects
- Perform A/B testing with statistical significance
- Evolve prompts based on cross-project evidence

**Without global RL**: GEPA would only see traces from current project  
**With global RL**: GEPA learns from all projects → better prompts

### 3. Provider-Agnostic Intelligence

**Example**: Learning provider capabilities

```sql
-- Global database aggregates provider performance
SELECT 
    provider,
    model,
    task_type,
    AVG(quality_score) as avg_quality,
    COUNT(*) as sample_count
FROM rl_outcome
GROUP BY provider, model, task_type
ORDER BY avg_quality DESC;
```

**Result**: System learns that:
- GPT-4 → Best for architecture tasks (quality: 0.92)
- Claude 3.5 Sonnet → Best for code reviews (quality: 0.89)
- Qwen-Coder → Best for quick debugging (quality: 0.75, fast)

### 4. Reduced Storage Overhead

**Before** (Per-project):
```
project_a/.victor/project.db (100 MB) - includes RL tables
project_b/.victor/project.db (120 MB) - includes RL tables
project_c/.victor/project.db (90 MB) - includes RL tables
Total: 310 MB with duplicate RL infrastructure
```

**After** (Global):
```
~/.victor/victor.db (150 MB) - all RL data
project_a/.victor/project.db (80 MB) - no RL tables
project_b/.victor/project.db (95 MB) - no RL tables
project_c/.victor/project.db (70 MB) - no RL tables
Total: 315 MB (slightly larger due to indices, but NO duplication)
```

**Benefit**: Single source of truth, efficient queries, no sync issues

---

## Performance Considerations

### Query Optimization

**Indexes** (on global database):
```sql
-- Speed up common queries
CREATE INDEX idx_rl_outcome_learner ON rl_outcome(learner_id, created_at);
CREATE INDEX idx_rl_outcome_context ON rl_outcome(provider, model, task_type);
CREATE INDEX idx_rl_outcome_session ON rl_outcome(session_id, created_at);
CREATE INDEX idx_rl_outcome_repo ON rl_outcome(repo_id, created_at);
```

### Data Retention

**Automatic compaction**:
```python
# victor/core/database.py (DatabaseManager)
- Old Q-values (visit_count > 1000) are archived
- History older than 90 days is compacted
- Failed signatures are pruned weekly
```

### Concurrency

**WAL mode enabled**:
```sql
PRAGMA journal_mode=WAL;  -- Write-Ahead Logging
PRAGMA synchronous=NORMAL;  -- Balanced safety/performance
PRAGMA foreign_keys=ON;     -- Referential integrity
```

---

## API Usage

### For Developers

**Creating a custom learner**:

```python
from victor.core.database import DatabaseManager

class MyCustomLearner:
    def __init__(self):
        # All learners use global DB
        self._db = DatabaseManager()
    
    def record_outcome(self, state, action, reward):
        conn = self._db.get_connection()
        conn.execute("""
            INSERT INTO rl_outcome (
                learner_id, provider, model, task_type, success, quality_score
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, ("my_learner", "ollama", "qwen", "coding", 1, 0.85))
        conn.commit()
```

### For Users

**RL data is transparent**:
```bash
# View your learning progress
victor rl stats --learner mode_transition

# Export RL data for analysis
victor rl export --format csv --output rl_data.csv

# Reset learning (fresh start)
victor rl reset --confirm
```

---

## Troubleshooting

### Issue: "no such column: created_at"

**Cause**: Code using `get_project_database()` instead of `DatabaseManager()`

**Fix**:
```python
# WRONG
from victor.core.database import get_project_database
self._db = get_project_database(project_path)

# RIGHT
from victor.core.database import DatabaseManager
self._db = DatabaseManager()
```

### Issue: "table rl_outcome does not exist"

**Cause**: Global database not initialized

**Fix**:
```bash
# Re-initialize global database
rm -f ~/.victor/victor.db
victor init --quick
```

### Issue: "database is locked"

**Cause**: Multiple processes writing to global DB simultaneously

**Fix**: WAL mode handles concurrency, but for heavy writes:
```python
# Use connection pooling or batch writes
with db.transaction() as conn:
    conn.executemany(sql, params_list)
```

---

## Future Enhancements

### Planned Features

1. **Distributed RL** - Share learning across teams (optional)
2. **RL Model Serving** - Export learned Q-values for external analysis
3. **A/B Testing Framework** - Built-in experiment tracking
4. **Auto-Tuning** - Automatically adjust hyperparameters based on performance

### Migration Path

**Current** (v0.3.0):
- All RL data in global SQLite database

**Future** (v1.0.0):
- Optional PostgreSQL backend for large-scale deployments
- Distributed RL for team learning
- Real-time streaming RL updates

---

## References

- **Design Doc**: `docs/architecture/database_architecture_fix.md`
- **Migration Code**: `victor/agent/conversation/migrations.py`
- **Schema Definitions**: `victor/core/schema.py`
- **Database Manager**: `victor/core/database.py`
- **Learners**:
  - `victor/agent/adaptive_mode_controller.py` (Mode transitions)
  - `victor/framework/rl/coordinator.py` (Unified RL)
  - `victor/framework/rl/mode_transition/` (Mode transition learner)
  - `victor/framework/rl/model_selector/` (Model selection)
  - `victor/framework/rl/tool_selector/` (Tool selection)

---

**Conclusion**: The global RL database architecture enables Victor to learn from ALL projects, optimize prompts globally (GEPA), and provide provider-agnostic intelligence. This is a foundational design choice that scales with usage and improves over time.
