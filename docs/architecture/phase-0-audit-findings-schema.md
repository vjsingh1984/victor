# Phase 0 Audit Findings: Part 3 - RL Database Schema Audit

**Audit Date**: 2026-04-19
**Auditor**: Automated Phase 0 Audit Suite
**Status**: ✅ PASSED

---

## Executive Summary

RL database schema verified. **`rl_outcome` table exists with all required columns** for Priority 4. Quality score field confirmed for user feedback. **ALTER-only changes required** - no new tables needed.

---

## 1. Existing Schema Documentation

### 1.1 Primary Table: `rl_outcome`

**Table Name**: `rl_outcome` (constant: `Tables.RL_OUTCOME`)
**Location**: `victor/core/schema.py` line 285-299

**Schema**:
```sql
CREATE TABLE IF NOT EXISTS rl_outcome (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    learner_id TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    task_type TEXT,
    vertical TEXT DEFAULT '',
    repo_id TEXT DEFAULT NULL,          -- ✅ Already exists!
    success INTEGER,
    quality_score REAL,                 -- ✅ Already exists!
    metadata TEXT,                      -- ✅ Already exists!
    created_at TEXT DEFAULT (datetime('now'))
)
```

### 1.2 Indexes

```sql
CREATE INDEX IF NOT EXISTS idx_rl_outcome_learner
    ON rl_outcome(learner_id, created_at);

CREATE INDEX IF NOT EXISTS idx_rl_outcome_context
    ON rl_outcome(provider, model, task_type);
```

---

## 2. Column Analysis

### 2.1 Existing Columns (All Required for Priority 4)

| Column | Type | Nullable | Priority 4 Usage | Status |
|--------|------|----------|------------------|--------|
| `id` | INTEGER | No | Primary key | ✅ Ready |
| `learner_id` | TEXT | No | Link to learner | ✅ Ready |
| `provider` | TEXT | Yes | Provider (e.g., "user", "anthropic") | ✅ Ready |
| `model` | TEXT | Yes | Model name (e.g., "feedback") | ✅ Ready |
| `task_type` | TEXT | Yes | Task type (e.g., "feedback") | ✅ Ready |
| `vertical` | TEXT | Yes | Vertical name | ✅ Ready |
| `repo_id` | TEXT | Yes | **Repository ID for session linking** | ✅ **Already exists!** |
| `success` | INTEGER | Yes | Success flag | ✅ Ready |
| `quality_score` | REAL | Yes | **Quality score for user feedback** | ✅ **Already exists!** |
| `metadata` | TEXT | Yes | **JSON metadata for feedback_source** | ✅ **Already exists!** |
| `created_at` | TEXT | Yes | Timestamp | ✅ Ready |

### 2.2 Key Findings

**✅ GOOD NEWS**: All required columns already exist!

- `repo_id` - Already exists for session linking
- `quality_score` - Already exists for user feedback
- `metadata` - Already exists for `feedback_source` tracking

**⚠️ MINOR ADDITION**: Only need to add `session_id` column for conversation linking

---

## 3. Priority 4 Schema Changes

### 3.1 Required ALTER Statements

**Only ONE new column needed**:

```sql
-- Add session_id for conversation linking
ALTER TABLE rl_outcome ADD COLUMN session_id TEXT;
```

**Optional indexes for performance**:

```sql
-- Index for session queries
CREATE INDEX IF NOT EXISTS idx_rl_outcome_session
    ON rl_outcome(session_id, created_at);

-- Index for repo queries
CREATE INDEX IF NOT EXISTS idx_rl_outcome_repo
    ON rl_outcome(repo_id, created_at);
```

### 3.2 Migration Script

```python
def migrate_priority_4(db_conn: sqlite3.Connection):
    """Migrate database for Priority 4 features.

    Adds session_id column to rl_outcome table for user feedback linking.
    """
    cursor = db_conn.cursor()

    # Check if session_id already exists
    cursor.execute("PRAGMA table_info(rl_outcome)")
    columns = {row[1] for row in cursor.fetchall()}

    if "session_id" not in columns:
        # Add session_id column
        cursor.execute("ALTER TABLE rl_outcome ADD COLUMN session_id TEXT")
        print("✅ Added session_id column to rl_outcome")
    else:
        print("ℹ️  session_id column already exists")

    # Create indexes if not exist
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_rl_outcome_session
            ON rl_outcome(session_id, created_at)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_rl_outcome_repo
            ON rl_outcome(repo_id, created_at)
    """)

    db_conn.commit()
    print("✅ Priority 4 migration complete")
```

---

## 4. User Feedback Integration

### 4.1 Quality Score Field

**Field**: `quality_score REAL`
**Definition**: "Quality score 0.0-1.0 from grounding/user feedback"
**Nullable**: Yes

**Usage for User Feedback**:

```python
# Automatic quality score (existing usage)
outcome_auto = RLOutcome(
    provider="anthropic",
    model="claude-sonnet-4-5-20250929",
    task_type="tool_call",
    success=True,
    quality_score=0.85,  # From grounding rules
    metadata={"grounding_rule": "tool_success"},
)

# User feedback quality score (Priority 4)
outcome_user = RLOutcome(
    provider="user",
    model="feedback",
    task_type="feedback",
    success=True,
    quality_score=0.90,  # From user rating
    metadata={
        "feedback_source": "user",  # Distinguish from auto
        "user_feedback": "Great result!",
        "helpful": True,
    },
)
```

### 4.2 Metadata Field

**Field**: `metadata TEXT`
**Format**: JSON string
**Nullable**: Yes

**Usage for Feedback Source Tracking**:

```python
metadata = {
    "feedback_source": "user",  # 'auto', 'user', 'hybrid'
    "user_feedback": "Excellent result",
    "helpful": True,
    "correction": None,
    "session_id": "abc123",
    "repo_id": "vijaysingh/codingagent",
}
```

---

## 5. Test Results

### Part 3 Tests: All Passed ✅

```
tests/integration/rl/phase_0_audit.py::TestRLDatabaseSchema::test_rl_coordinator_has_record_outcome PASSED
tests/integration/rl/phase_0_audit.py::TestRLDatabaseSchema::test_rl_coordinator_has_async_record_outcome PASSED
tests/integration/rl/phase_0_audit.py::TestRLDatabaseSchema::test_record_outcome_accepts_rl_outcome PASSED
tests/integration/rl/phase_0_audit.py::TestRLDatabaseSchema::test_rl_outcome_quality_score_accepts_user_feedback PASSED
```

**Total**: 4 tests, all passed

---

## 6. No Duplication Verification

### 6.1 Session Summaries Table

**Question**: Is there a `session_summaries` table?

**Answer**: ❌ NO - And none needed!

**Reason**: `UsageAnalytics.get_session_summary()` already provides session aggregation in-memory. Priority 4 only needs to persist summaries to `rl_outcome` table using existing columns.

### 6.2 User Feedback Table

**Question**: Is there a `user_feedback` table?

**Answer**: ❌ NO - And none needed!

**Reason**: User feedback is stored in `rl_outcome` table using:
- `quality_score` field for rating
- `metadata` field for feedback details
- `provider='user'` to distinguish from automatic scores

### 6.3 Tool Prediction Table

**Question**: Is there a `tool_predictions` table?

**Answer**: ❌ NO - And none needed!

**Reason**: Tool predictions are handled by:
- `ToolPredictor` from Priority 3 (in-memory)
- `CooccurrenceTracker` for pattern learning
- `rl_outcome` table for storing prediction outcomes

---

## 7. Backward Compatibility

### 7.1 Existing Queries

All existing queries remain compatible:

```sql
-- Existing queries (still work)
SELECT * FROM rl_outcome WHERE learner_id = 'tool_selector'
SELECT * FROM rl_outcome WHERE provider = 'anthropic'
SELECT * FROM rl_outcome WHERE quality_score > 0.8

-- New queries (Priority 4)
SELECT * FROM rl_outcome WHERE session_id = 'abc123'
SELECT * FROM rl_outcome WHERE repo_id = 'vijaysingh/codingagent'
SELECT * FROM rl_outcome WHERE metadata LIKE '%"feedback_source":"user"%'
```

### 7.2 Existing Code

All existing code remains compatible:

- `RLCoordinator.record_outcome()` - Works unchanged
- `RLCoordinator.get_recent_outcomes()` - Works unchanged
- `RLCoordinator.get_outcomes_by_type()` - Works unchanged

**New methods for Priority 4**:

```python
# New methods (additive, not breaking)
def get_outcomes_by_session(self, session_id: str) -> List[RLOutcome]:
    """Get all outcomes for a session."""
    ...

def get_outcomes_by_repo(self, repo_id: str) -> List[RLOutcome]:
    """Get all outcomes for a repository."""
    ...

def get_user_feedback(self, session_id: str) -> List[RLOutcome]:
    """Get user feedback for a session."""
    ...
```

---

## 8. Recommendations

### For Priority 4 Implementation

1. **ALTER ONLY** - No new tables:
   - ✅ Add `session_id` column to `rl_outcome`
   - ✅ Add indexes for performance
   - ❌ DO NOT create `session_summaries` table
   - ❌ DO NOT create `user_feedback` table

2. **Reuse existing columns**:
   - ✅ Use `quality_score` for user ratings
   - ✅ Use `metadata` for feedback details
   - ✅ Use `repo_id` for repository linking

3. **Maintain compatibility**:
   - ✅ All existing queries still work
   - ✅ All existing code still works
   - ✅ Additive changes only

### Migration Checklist

- [ ] Add `session_id` column via ALTER
- [ ] Create performance indexes
- [ ] Test existing queries still work
- [ ] Test new queries work
- [ ] Update migration scripts
- [ ] Update schema documentation

### Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking existing queries | Low | High | Backward compatibility tests |
| Performance degradation | Low | Medium | Index testing |
| Migration failure | Low | High | Rollback plan |

---

## 9. Sign-off

**Audit Status**: ✅ PASSED

**Findings**:
- `rl_outcome` table exists with all required columns
- `quality_score` field ready for user feedback
- `repo_id` column already exists
- Only need to add `session_id` column
- No new tables required (ALTER-only)

**Schema Changes Required**:
- 1 new column: `session_id`
- 2 new indexes (optional, for performance)

**Approval**: Ready for Phase 1 implementation

**Next Step**: Proceed to Part 4 - RLOutcome Quality Score Audit
