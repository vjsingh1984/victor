# Priority 4: Phase 0 Infrastructure Audit Plan

**Purpose**: Comprehensive audit of existing RL infrastructure before implementing Priority 4 (Learning from Execution) to prevent duplication and ensure proper integration.

**Timeline**: Week 0 (before any implementation begins)

**Owner**: Engineering Lead + RL Framework Maintainer

**Status**: ⚠️ **REQUIRED** - Must complete and pass all tests before Phase 1 implementation

---

## Executive Summary

This audit plan ensures Priority 4 extends existing Victor infrastructure rather than duplicating it. The audit will:

1. **Document all 14 existing learners** and their capabilities
2. **Verify UsageAnalytics integration points** to avoid rebuilding session aggregation
3. **Validate RL database schema** to ensure ALTER-only changes (no new tables)
4. **Test existing RLOutcome.quality_score field** for user feedback compatibility
5. **Verify ToolPredictor integration** from Priority 3
6. **Create baseline integration tests** for existing components

**Success Criteria**: All 6 audit sections must pass with documented findings before Phase 1 begins.

---

## Part 1: Existing Learners Audit

### Goal

Document all 14 existing learners to prevent accidental duplication.

### Methodology

1. **List all learners** in `victor/framework/rl/learners/`
2. **Document each learner's**:
   - Purpose and capabilities
   - Input/output data structures
   - Learning algorithm
   - Integration points
   - Extension opportunities

3. **Identify learners** that Priority 4 should extend:
   - `ModelSelectorLearner` → extend for hybrid decision improvements
   - `ModeTransitionLearner` → extend for phase-aware context
   - `ToolSelectorLearner` → extend for predictive tool selection

### Audit Checklist

- [ ] **1.1** List all 14 learner files
- [ ] **1.2** Document each learner's purpose
- [ ] **1.3** Document input/output schemas
- [ ] **1.4** Identify extension opportunities for Priority 4
- [ ] **1.5** Create learner capability matrix

### Expected Findings

**14 Existing Learners** (confirmed from codebase scan):

| Learner | File | Purpose | Extension Point |
|---------|------|---------|-----------------|
| CacheEvictionLearner | `cache_eviction.py` | Learn cache eviction policies | - |
| ContextPruningLearner | `context_pruning.py` | Learn context pruning thresholds | - |
| ContinuationPatienceLearner | `continuation_patience.py` | Learn continuation patience | - |
| ContinuationPromptsLearner | `continuation_prompts.py` | Learn continuation prompts | - |
| CrossVerticalLearner | `cross_vertical.py` | Learn cross-vertical patterns | - |
| GroundingThresholdLearner | `grounding_threshold.py` | Learn grounding thresholds | - |
| **ModeTransitionLearner** | `mode_transition.py` | **Learn mode transitions** | **Priority 2 Integration** |
| **ModelSelectorLearner** | `model_selector.py` | **Learn model selection** | **Priority 1 Integration** |
| PromptOptimizerLearner | `prompt_optimizer.py` | Optimize prompts (GEPA/MIPROv2/CoT) | - |
| PromptTemplateLearner | `prompt_template.py` | Learn prompt templates | - |
| QualityWeightsLearner | `quality_weights.py` | Learn quality weights | - |
| SemanticThresholdLearner | `semantic_threshold.py` | Learn semantic thresholds | - |
| **ToolSelectorLearner** | `tool_selector.py` | **Learn tool selection** | **Priority 3 Integration** |
| WorkflowExecutionLearner | `workflow_execution.py` | Learn workflow execution | - |

### Deliverables

- [x] **Learner Inventory** (`learners-audit-findings.md`)
  - Complete list of 14 learners
  - Purpose and capabilities for each
  - Extension opportunities for Priority 4

---

## Part 2: UsageAnalytics Integration Audit

### Goal

Verify UsageAnalytics provides session aggregation to prevent creating duplicate `session_summaries` table.

### Methodology

1. **Audit UsageAnalytics singleton** in `victor/agent/usage_analytics.py`
2. **Document all public methods** and their data structures
3. **Trace data flow** from tool execution to session summary
4. **Identify integration points** for RL database persistence

### Audit Checklist

- [ ] **2.1** Verify `UsageAnalytics.get_instance()` singleton pattern
- [ ] **2.2** Document `get_session_summary()` method
  - Returns: `total_sessions`, `avg_turns_per_session`, `avg_tool_calls_per_session`, `avg_tokens_per_session`, `avg_session_duration_seconds`
- [ ] **2.3** Document `get_tool_insights()` method
  - Input: `tool_name: str`
  - Returns: `success_rate`, `avg_execution_ms`, `error_count`, `execution_count`
- [ ] **2.4** Document `record_tool_execution()` method
  - Tracks tool success, execution time, errors
- [ ] **2.5** Verify in-memory storage vs persistence
  - Current: In-memory only
  - Priority 4: Add RL database persistence

### Expected Findings

**UsageAnalytics Capabilities** (confirmed from codebase):

```python
class UsageAnalytics:
    """Singleton analytics system"""

    def get_session_summary(self) -> Dict[str, Any]:
        """Returns session-level statistics"""
        return {
            "total_sessions": int,
            "avg_turns_per_session": float,
            "avg_tool_calls_per_session": float,
            "avg_tokens_per_session": float,
            "avg_session_duration_seconds": float,
        }

    def get_tool_insights(self, tool_name: str) -> Dict[str, Any]:
        """Returns tool performance metrics"""
        return {
            "success_rate": float,
            "avg_execution_ms": float,
            "error_count": int,
            "execution_count": int,
        }

    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Returns actionable optimization insights"""
        return [
            {
                "type": "tool_usage",
                "recommendation": str,
                "confidence": float,
            }
        ]
```

**Integration Design**:

```python
class MetaLearningCoordinator(RLCoordinator):
    """Extend existing coordinator with UsageAnalytics integration"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use existing analytics, don't rebuild
        self.analytics = UsageAnalytics.get_instance()

    def aggregate_session_metrics(self, repo_id: str, session_window: int = 100):
        """Aggregate metrics across recent sessions using existing analytics"""
        # Get in-memory summary from existing system
        summary = self.analytics.get_session_summary()

        # Persist to RL database for long-term storage
        self._persist_session_summary(summary)

        return summary
```

### Deliverables

- [ ] **UsageAnalytics Integration Report** (`usage-analytics-audit-findings.md`)
  - Complete API documentation
  - Data flow diagrams
  - Integration design for RL persistence
  - Test cases for integration

---

## Part 3: RL Database Schema Audit

### Goal

Verify existing RL database schema to ensure ALTER-only changes (no new tables).

### Methodology

1. **Audit schema definition** in `victor/core/schema.py`
2. **Document all existing tables** and their columns
3. **Identify columns** to add to `rl_outcomes` table
4. **Verify no new tables are needed** for Priority 4

### Audit Checklist

- [ ] **3.1** Document existing `rl_outcomes` table schema
  - Columns: `provider`, `model`, `task_type`, `success`, `quality_score`, `timestamp`, `metadata`, `vertical`
- [ ] **3.2** Verify `quality_score` field exists and is usable for user feedback
  - Definition: "Quality score 0.0-1.0 from grounding/user feedback"
- [ ] **3.3** Identify columns to ADD (not create new tables):
  - `session_id TEXT` - Link to conversation
  - `repo_id TEXT` - Link to repository
  - `session_summary TEXT` - JSON summary from UsageAnalytics
  - `feedback_source TEXT` - Distinguish 'auto', 'user', 'hybrid'
- [ ] **3.4** Verify ALTER statements don't break existing queries
- [ ] **3.5** Test backward compatibility with existing data

### Expected Findings

**Existing Schema** (confirmed from codebase):

```sql
-- Existing table (DO NOT RECREATE)
CREATE TABLE IF NOT EXISTS rl_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    task_type TEXT NOT NULL,
    success BOOLEAN NOT NULL,
    quality_score REAL,  -- Can be used for user feedback!
    timestamp TEXT NOT NULL,
    metadata TEXT,  -- JSON metadata
    vertical TEXT
);

-- Priority 4: ALTER existing table (NOT CREATE new tables)
ALTER TABLE rl_outcomes ADD COLUMN session_id TEXT;
ALTER TABLE rl_outcomes ADD COLUMN repo_id TEXT;
ALTER TABLE rl_outcomes ADD COLUMN session_summary TEXT;
ALTER TABLE rl_outcomes ADD COLUMN feedback_source TEXT;  -- 'auto', 'user', 'hybrid'
```

**User Feedback Integration**:

```python
def create_outcome_with_user_feedback(
    session_id: str,
    rating: float,  # This becomes quality_score
    feedback: Optional[str] = None,
    helpful: Optional[bool] = None,
    correction: Optional[str] = None,
) -> RLOutcome:
    """Create RLOutcome with user feedback"""
    return RLOutcome(
        provider="user",
        model="feedback",
        task_type="feedback",
        success=True,
        quality_score=rating,  # Reuse existing field
        timestamp=datetime.now().isoformat(),
        metadata={
            "session_id": session_id,
            "feedback_source": "user",  # NEW: Track source
            "user_feedback": feedback,
            "helpful": helpful,
            "correction": correction,
        },
        vertical="general"
    )
```

### Deliverables

- [ ] **Schema Audit Report** (`schema-audit-findings.md`)
  - Complete existing schema documentation
  - ALTER statements for Priority 4
  - Backward compatibility verification
  - Migration test cases

---

## Part 4: RLOutcome Quality Score Audit

### Goal

Verify `RLOutcome.quality_score` field can accommodate user feedback without schema changes.

### Methodology

1. **Audit RLOutcome definition** in `victor/framework/rl/base.py`
2. **Document quality_score usage** across existing learners
3. **Verify field type and constraints** support user feedback
4. **Test creating outcomes** with user feedback

### Audit Checklist

- [ ] **4.1** Verify `RLOutcome` dataclass definition
  - Field: `quality_score: Optional[float]`
  - Definition: "Quality score 0.0-1.0 from grounding/user feedback"
- [ ] **4.2** Document existing quality_score sources
  - Automatic: Grounding rules
  - Manual: User feedback (to be added)
- [ ] **4.3** Test creating outcomes with user feedback
- [ ] **4.4** Verify metadata field can store `feedback_source`
- [ ] **4.5** Test querying outcomes by feedback source

### Expected Findings

**RLOutcome Definition** (confirmed from codebase):

```python
@dataclass
class RLOutcome:
    """Outcome from reinforcement learning"""

    provider: str
    model: str
    task_type: str
    success: bool
    quality_score: Optional[float] = None  # Can be used for user feedback!
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Optional[Dict[str, Any]] = None
    vertical: Optional[str] = None
```

**Quality Score Sources**:

```python
# Existing: Automatic quality scores
outcome = RLOutcome(
    provider="anthropic",
    model="claude-sonnet-4-5-20250929",
    task_type="tool_call",
    success=True,
    quality_score=0.85,  # From grounding rules
    metadata={"grounding_rule": "tool_success"}
)

# NEW: User feedback quality scores
outcome = RLOutcome(
    provider="user",
    model="feedback",
    task_type="feedback",
    success=True,
    quality_score=0.90,  # From user rating
    metadata={
        "feedback_source": "user",  # NEW: Track source
        "user_feedback": "Great result!",
        "helpful": True,
    }
)
```

### Deliverables

- [ ] **Quality Score Audit Report** (`quality-score-audit-findings.md`)
  - Field documentation
  - Usage patterns across existing learners
  - User feedback integration design
  - Test cases for feedback outcomes

---

## Part 5: ToolPredictor Integration Audit

### Goal

Verify ToolPredictor from Priority 3 exists and can be integrated with ToolSelectorLearner.

### Methodology

1. **Audit ToolPredictor** in `victor/agent/planning/tool_predictor.py`
2. **Document prediction API** and data structures
3. **Verify integration points** with existing ToolSelectorLearner
4. **Test extending learner** with predictor

### Audit Checklist

- [ ] **5.1** Verify ToolPredictor class exists from Priority 3
- [ ] **5.2** Document `predict_tools()` method
  - Input: `task_description`, `current_step`, `recent_tools`, `task_type`
  - Output: `List[ToolPrediction]` with `tool_name`, `probability`, `confidence_level`
- [ ] **5.3** Document CooccurrenceTracker integration
- [ ] **5.4** Verify ToolSelectorLearner can extend with predictor
- [ ] **5.5** Test integration between learner and predictor

### Expected Findings

**ToolPredictor API** (from Priority 3):

```python
class ToolPredictor:
    """Predictive tool selection from Priority 3"""

    def __init__(self, cooccurrence_tracker: Optional[CooccurrenceTracker] = None):
        self.cooccurrence_tracker = cooccurrence_tracker or CooccurrenceTracker()
        self.weights = {
            "keyword": 0.3,
            "semantic": 0.4,
            "cooccurrence": 0.2,
            "success": 0.1,
        }

    def predict_tools(
        self,
        task_description: str,
        current_step: str,
        recent_tools: List[str],
        task_type: str,
    ) -> List[ToolPrediction]:
        """Predict next tools with ensemble methods"""
        # Returns list of ToolPrediction with confidence scores
```

**ToolSelectorLearner Integration**:

```python
class ToolSelectorLearner(BaseLearner):
    """Extend existing tool_selector learner with Priority 3 predictor"""

    def __init__(self, config: LearnerConfig):
        super().__init__(
            learner_name="tool_selector",
            outcome_type="tool_execution",
            config=config
        )

        # Use existing analytics (don't rebuild tool tracking)
        from victor.agent.usage_analytics import UsageAnalytics
        self.analytics = UsageAnalytics.get_instance()

        # Use existing ToolPredictor from Priority 3 (don't recreate)
        from victor.agent.planning.tool_predictor import ToolPredictor
        self.predictor = ToolPredictor()

    def learn(self, outcomes: List[RLOutcome]) -> List[RLRecommendation]:
        """Learn from tool execution outcomes"""
        # Train predictor with new outcomes
        for outcome in outcomes:
            if outcome.task_type == "tool_execution":
                # Update co-occurrence tracker
                self.predictor.cooccurrence_tracker.record_tool_sequence(
                    tools=outcome.metadata.get("tools_used", []),
                    task_type=outcome.metadata.get("task_type"),
                    success=outcome.success,
                )

        # Generate recommendations using existing analytics
        recommendations = []

        for tool_name in self._get_tool_names(outcomes):
            insights = self.analytics.get_tool_insights(tool_name)

            recommendations.append(RLRecommendation(
                learner_name="tool_selector",
                recommendation_type="tool_usage",
                key=tool_name,
                value="use" if insights["success_rate"] > 0.7 else "avoid",
                confidence=insights["success_rate"],
                metadata={
                    "avg_execution_ms": insights["avg_execution_ms"],
                    "sample_size": insights["execution_count"]
                }
            ))

        return recommendations
```

### Deliverables

- [ ] **ToolPredictor Integration Report** (`tool-predictor-audit-findings.md`)
  - API documentation
  - Integration design with ToolSelectorLearner
  - Test cases for predictor integration
  - Performance benchmarks

---

## Part 6: Baseline Integration Tests

### Goal

Create integration tests for existing components to establish baseline before Priority 4 changes.

### Methodology

1. **Create test suite** for existing UsageAnalytics
2. **Create test suite** for existing RLCoordinator
3. **Create test suite** for existing 14 learners
4. **Establish performance baselines** for all components
5. **Create "No Duplication" guard tests**

### Test Checklist

- [ ] **6.1** UsageAnalytics tests
  - `test_get_session_summary()`
  - `test_get_tool_insights()`
  - `test_record_tool_execution()`
  - `test_get_optimization_recommendations()`

- [ ] **6.2** RLCoordinator tests
  - `test_record_outcome()`
  - `test_get_recent_outcomes()`
  - `test_get_outcomes_by_type()`

- [ ] **6.3** Learner baseline tests (all 14)
  - `test_model_selector_learner_baseline()`
  - `test_tool_selector_learner_baseline()`
  - `test_mode_transition_learner_baseline()`
  - ... (11 more learners)

- [ ] **6.4** "No Duplication" guard tests
  - `test_no_duplicate_session_aggregation()`
  - `test_no_duplicate_user_feedback()`
  - `test_no_duplicate_tool_prediction()`
  - `test_all_learners_extended_not_replaced()`

- [ ] **6.5** Performance baseline tests
  - `test_usage_analytics_performance()`
  - `test_rl_coordinator_performance()`
  - `test_learner_learn_performance()`

### Expected Test Results

**All tests must pass** before Phase 1 implementation begins:

```bash
# Run Phase 0 audit tests
pytest tests/integration/rl/phase_0_audit.py -v

# Expected: All tests pass
# ✅ test_usage_analytics_singleton
# ✅ test_get_session_summary
# ✅ test_get_tool_insights
# ✅ test_rl_outcome_quality_score
# ✅ test_tool_predictor_exists
# ✅ test_no_duplicate_session_aggregation
# ✅ test_no_duplicate_user_feedback
# ✅ test_no_duplicate_tool_prediction
# ✅ test_all_14_learners_exist
# ✅ test_usage_analytics_performance
# ✅ test_rl_coordinator_performance
```

### Deliverables

- [ ] **Baseline Test Suite** (`tests/integration/rl/phase_0_audit.py`)
  - UsageAnalytics integration tests
  - RLCoordinator integration tests
  - Learner baseline tests (all 14)
  - "No Duplication" guard tests
  - Performance baseline tests

- [ ] **Test Results Report** (`phase-0-test-results.md`)
  - All test results (must pass)
  - Performance baselines documented
  - Any issues or warnings

---

## Part 7: Audit Deliverables Summary

### Required Documents (All must be completed)

1. **Learner Inventory** (`learners-audit-findings.md`)
   - All 14 learners documented
   - Extension opportunities identified

2. **UsageAnalytics Integration Report** (`usage-analytics-audit-findings.md`)
   - API documentation
   - Integration design
   - Test cases

3. **Schema Audit Report** (`schema-audit-findings.md`)
   - Existing schema documented
   - ALTER statements for Priority 4
   - Backward compatibility verified

4. **Quality Score Audit Report** (`quality-score-audit-findings.md`)
   - Field documentation
   - User feedback integration design
   - Test cases

5. **ToolPredictor Integration Report** (`tool-predictor-audit-findings.md`)
   - API documentation
   - Integration design
   - Test cases

6. **Baseline Test Suite** (`tests/integration/rl/phase_0_audit.py`)
   - All existing component tests
   - "No Duplication" guard tests
   - Performance baselines

7. **Test Results Report** (`phase-0-test-results.md`)
   - All tests passing
   - Performance baselines
   - Issues and warnings

### Completion Criteria

Phase 0 audit is **COMPLETE** when:

- ✅ All 6 audit sections completed
- ✅ All 7 deliverable documents created
- ✅ All "No Duplication" tests passing
- ✅ Performance baselines established
- ✅ Integration points documented
- ✅ Stakeholder review and approval

### Approval Process

1. **Engineering Lead** reviews all audit findings
2. **RL Framework Maintainer** verifies no duplication
3. **Architecture Review Board** approves integration design
4. **Phase 0 Sign-off** document created
5. **Phase 1 Implementation** authorized

---

## Part 8: Risk Mitigation

### Identified Risks

1. **Incomplete Audit**
   - **Risk**: Missing existing component leads to duplication
   - **Mitigation**: Comprehensive checklist for each audit section
   - **Validation**: All "No Duplication" tests must pass

2. **Schema Compatibility**
   - **Risk**: ALTER operations break existing queries
   - **Mitigation**: Backward compatibility tests
   - **Validation**: All existing tests still pass after ALTER

3. **Performance Regression**
   - **Risk**: Integration adds overhead to existing components
   - **Mitigation**: Performance baseline tests
   - **Validation**: <10% overhead target

4. **Integration Complexity**
   - **Risk**: Extending existing learners is harder than creating new ones
   - **Mitigation**: Detailed integration designs
   - **Validation**: Prototype integration before Phase 1

### Rollback Plan

If Phase 0 audit reveals critical issues:

1. **Stop Phase 1 implementation** immediately
2. **Document blocking issues** in audit findings
3. **Create alternative design** that avoids duplication
4. **Re-run Phase 0 audit** with alternative design
5. **Get re-approval** before proceeding

---

## Part 9: Timeline and Resources

### Timeline

**Week 0 (5 days)**:

- **Day 1**: Part 1 (Learners Audit) + Part 2 (UsageAnalytics Audit)
- **Day 2**: Part 3 (Schema Audit) + Part 4 (Quality Score Audit)
- **Day 3**: Part 5 (ToolPredictor Audit) + Part 6 (Baseline Tests)
- **Day 4**: Complete all audit deliverables
- **Day 5**: Stakeholder review and approval

### Resources

**Required Personnel**:
- Engineering Lead (5 days)
- RL Framework Maintainer (5 days)
- QA Engineer (2 days for tests)

**Required Tools**:
- Access to codebase (`victor/framework/rl/`)
- Database access for schema audit
- Test environment for baseline tests
- Documentation tools (Markdown, diagrams)

---

## Part 10: Sign-off

### Phase 0 Audit Completion

**I certify that**:

- [ ] All 6 audit sections have been completed
- [ ] All 7 deliverable documents have been created
- [ ] All "No Duplication" tests are passing
- [ ] Performance baselines have been established
- [ ] Integration points have been documented
- [ ] Stakeholder review has been completed
- [ ] Phase 1 implementation is authorized

**Engineering Lead**: _________________ **Date**: _________

**RL Framework Maintainer**: _________________ **Date**: _________

**Architecture Review Board**: _________________ **Date**: _________

---

**Phase 0 audit is MANDATORY before beginning Phase 1 implementation.**
**All tests must pass and all deliverables must be approved.**
**Any duplication discovered must be resolved before proceeding.**
