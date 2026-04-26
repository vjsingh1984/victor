# Priority 4: Learning from Execution - Design Document

**Status**: Design Phase
**Created**: 2025-01-18
**Scope**: Extend existing RL infrastructure for meta-learning and predictive execution
**Estimated Timeline**: 6+ months (long-term research)

---

## Executive Summary

This document outlines the design for Priority 4 (Learning from Execution), which **extends** Victor's existing reinforcement learning infrastructure rather than duplicating it. The existing RL framework (`victor/framework/rl/`) already provides:

- ✅ Prompt optimizer with 3 strategies (GEPA, MIPROv2, CoT)
- ✅ 15 specialized learners for different optimization tasks
- ✅ Outcome tracking from JSONL logs + ConversationStore
- ✅ Event-driven learning hooks
- ✅ Metrics exporter with Prometheus support
- ✅ Statistical methods (Thompson Sampling, Pareto optimization)

**Priority 4 will add 3 new capabilities** on top of this foundation:

1. **Meta-Learning System** - Cross-session pattern consolidation
2. **User Feedback Integration** - Ground truth from human ratings
3. **Predictive Execution Strategies** - Build on existing tool selector learner

---

## Part 1: Existing Infrastructure Inventory

### 1.1 RL Framework Components

**Location**: `victor/framework/rl/`

| Component | File | Purpose | Status |
|-----------|------|---------|--------|
| Coordinator | `coordinator.py` | Central RL coordinator with SQLite DB | ✅ Existing |
| Base Classes | `base.py` | BaseLearner, RLOutcome, RLRecommendation | ✅ Existing |
| Hooks | `hooks.py` | Event-driven learner activation (15 events) | ✅ Existing |
| Metrics | `metrics.py` | JSON + Prometheus metrics export | ✅ Existing |
| Schema | `core/schema.py` | Consolidated RL tables with migrations | ✅ Existing |

### 1.2 Existing Learners (14 Total)

| Learner | Purpose | Extension Point |
|---------|---------|-----------------|
| `continuation_patience` | Loop detection thresholds | Can add predictive patterns |
| `continuation_prompts` | Continuation message generation | Can use execution history |
| `semantic_threshold` | Context relevance scoring | ✅ Already optimal |
| `model_selector` | Model selection per task type | **Key extension point** |
| `cache_eviction` | Cache management | ✅ Already optimal |
| `grounding_threshold` | Grounding check thresholds | **Key extension point** |
| `quality_weights` | Multi-objective weighting | **Key extension point** |
| `tool_selector` | Tool selection optimization | **Key extension point** |
| `mode_transition` | Mode transition timing | Can use execution patterns |
| `prompt_template` | Template selection | ✅ Already optimal |
| `cross_vertical` | Cross-vertical knowledge sharing | **Key extension point** |
| `workflow_execution` | Workflow performance | **Key extension point** |
| `context_pruning` | Context window optimization | ✅ Already optimal |
| `team_composition` | Multi-agent optimization | **Key extension point** |
| `prompt_optimizer` | System prompt evolution | ✅ Already comprehensive |

### 1.3 Prompt Optimizer

**Location**: `victor/framework/rl/learners/prompt_optimizer.py`

**Existing Strategies**:
1. **GEPA (GEPAStrategy - Default)**: Trace-reflection prompt evolution
   - Thompson Sampling for candidate serving
   - Provider-scoped Pareto frontier bookkeeping
   - Per-instance frontier evidence now comes from runtime outcomes and only from
     benchmark artifacts that explicitly identify the evaluated prompt candidate
   - Semantic trace zones inspired by PRiME
   - 13 failure categories with corrective hints
   - Session-aligned credit enrichment from runtime tool execution
   - Pareto merge fallback can synthesize a new candidate when reflection/mutation
     produces no novel variant
   - Used for: ASI_TOOL_EFFECTIVENESS_GUIDANCE, GROUNDING_RULES, COMPLETION_GUIDANCE, INIT_SYNTHESIS_RULES

2. **MIPROv2-inspired (MIPROv2Strategy)**: Query-aware few-shot retrieval adaptation
   - KNN-style embedding similarity when embeddings are available
   - Score thresholding, lightweight diversity filtering, and bounded example size
   - Used for: FEW_SHOT_EXAMPLES section only
   - Not a full implementation of MIPROv2 proposal search or Bayesian optimization

3. **CoT Transfer (CoTDistillationStrategy)**: Provider-aware reasoning-template transfer
   - Extracts stepwise execution scaffolds from stronger-provider traces
   - Only transfers when another provider materially outperforms the target provider
   - Layered with GEPA for ASI_TOOL_EFFECTIVENESS_GUIDANCE
   - This is prompt-level transfer, not student-model distillation

**Note**: PRiME is referenced as inspiration for semantic trace zoning and memory organization, but is not implemented here as a standalone strategy.

**Maintainer note**: Keep this section synchronized with the tests around `prompt_optimizer.py`, `miprov2_strategy.py`, and `cot_distillation_strategy.py`. If a paper-inspired feature is only partially adapted, describe the adaptation explicitly instead of naming the full paper method.

**Evolvable Sections** (5):
- `ASI_TOOL_EFFECTIVENESS_GUIDANCE`
- `GROUNDING_RULES`
- `COMPLETION_GUIDANCE`
- `FEW_SHOT_EXAMPLES`
- `INIT_SYNTHESIS_RULES`

### 1.4 Outcome Tracking

**Data Sources**:
1. **JSONL Logs**: Tool execution events, task classifications, results
2. **ConversationStore**: SQLite session history with ML metadata
3. **Event System**: 15 `RLEventType` events mapped to learners

**Database**: `~/.victor/victor.db` (SQLite, async-safe, thread-pooled)

### 1.5 Usage Analytics System

**Location**: `victor/agent/usage_analytics.py`

**Purpose**: Singleton analytics system that tracks tool usage, provider performance, and conversation patterns

**Key Features**:
- `record_tool_execution()` - Tracks tool success, execution time, errors
- `get_tool_insights()` - Returns optimization insights (success rates, performance metrics)
- `get_session_summary()` - Session-level statistics (turns, tools, tokens, duration)
- `get_optimization_recommendations()` - Actionable recommendations
- `export_prometheus_metrics()` - Metrics export for monitoring

**Integration Point**: This is CRITICAL for Priority 4 - must be integrated with RL framework rather than duplicating functionality.

---

## Part 2: Gaps and Opportunities

### 2.1 What's Missing

| Gap | Description | Priority | Existing Foundation |
|-----|-------------|----------|---------------------|
| **Meta-Learning** | Cross-session pattern consolidation, long-term trends | High | UsageAnalytics has in-memory session summaries |
| **User Feedback** | Ground truth from human ratings, preference learning | High | RLOutcome has quality_score field |
| **Analytics Integration** | Bridge UsageAnalytics with RL framework for persistence | High | Both exist but don't integrate |
| **Predictive Execution** | Tool usage prediction, sequence optimization | Medium | Priority 3 has CooccurrenceTracker |
| **Transfer Learning** | Skill transfer between projects/domains | Medium | cross_vertical learner exists |
| **Explainability** | Why recommendations were made | Low | RLRecommendation has metadata field |

### 2.2 Extension Opportunities

**High-Value Extensions** (build on existing learners):

1. **`tool_selector` learner extension**:
   - Add co-occurrence pattern mining (Priority 3 foundation exists)
   - Predict next tool based on sequence history
   - Learn optimal tool chains per task type

2. **`model_selector` learner extension**:
   - Track task success rates per model
   - Learn model performance patterns by task complexity
   - Dynamic model routing based on historical success

3. **`quality_weights` learner extension**:
   - User feedback integration for objective weighting
   - Personalized weights per user/project
   - Multi-stakeholder preference aggregation

4. **`cross_vertical` learner extension**:
   - Transfer learning between verticals
   - Shared pattern recognition
   - Domain adaptation strategies

---

## Part 3: Proposed Design

### 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Existing RL Framework                    │
│  (coordinator.py + 15 learners + hooks + metrics)          │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
    │  Meta-  │    │  User   │    │Predictive│
    │Learning │    │Feedback │    │Execution │
    │ System  │    │Integration│   │Strategies│
    └────┬────┘    └────┬────┘    └────┬────┘
         │              │               │
         └──────────────┼───────────────┘
                        │
              ┌─────────▼─────────┐
              │  Extended Learners│
              │  (build on        │
              │   existing 15)    │
              └───────────────────┘
```

### 3.2 Component 1: Meta-Learning System

**Purpose**: Consolidate learning across sessions, identify long-term patterns

**Design**: Extend existing `UsageAnalytics` and `RLCoordinator` for meta-learning capabilities

**CRITICAL**: Do NOT create new `session_summaries` table. Instead, extend existing `UsageAnalytics.get_session_summary()` to persist to RL database.

**Key Features**:

1. **Session-Level Aggregation** (Using Existing UsageAnalytics):
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

       def detect_long_term_trends(self, repo_id: str, days: int = 30):
           """Detect patterns across long time windows"""
           # Get current state from existing analytics
           current = self.analytics.get_session_summary()

           # Get historical from RL database
           historical = self._query_historical_summaries(repo_id, days)

           # Detect trends
           return self._detect_trends(current, historical)
   ```

2. **Pattern Consolidation**:
   - Consolidate successful patterns from existing `get_optimization_recommendations()`
   - Identify anti-patterns to avoid
   - Build pattern library for reuse

3. **Transfer Learning**:
   - Extend existing `cross_vertical` learner for pattern export/import
   - Import and adapt patterns from one project to another
   - Domain adaptation using existing recommendation system

**Database Extensions**:
```sql
-- DO NOT create session_summaries table
-- INSTEAD: Extend existing rl_outcomes table with session-level fields

ALTER TABLE rl_outcomes ADD COLUMN session_summary TEXT;  -- JSON summary
ALTER TABLE rl_outcomes ADD COLUMN session_id TEXT;      -- Link to conversation
ALTER TABLE rl_outcomes ADD COLUMN repo_id TEXT;         -- Link to repository

-- Index for trend queries
CREATE INDEX idx_rl_outcomes_repo_time
ON rl_outcomes(repo_id, timestamp DESC);

-- Index for session queries
CREATE INDEX idx_rl_outcomes_session
ON rl_outcomes(session_id);
```

**Integration Points**:
- Use existing `UsageAnalytics.get_session_summary()` - don't recreate
- Use existing `get_optimization_recommendations()` - build on it
- Hook into session lifecycle events (already exist)
- Use existing metrics exporter for trend visualization

### 3.3 Component 2: User Feedback Integration

**Purpose**: Incorporate human ground truth to improve learning

**Design**: Extend existing `RLOutcome` and create `user_feedback` learner

**CRITICAL CLARIFICATION**: `RLOutcome.quality_score` already exists and is defined as "from grounding/user feedback". We will extend it to track the source of the score (automatic vs. human).

**Key Features**:

1. **Extended Outcome Schema** (Reuse Existing RLOutcome):
   ```python
   # DON'T create ExtendedRLOutcome - extend existing RLOutcome
   # RLOutcome already has: provider, model, task_type, success, quality_score, timestamp, metadata, vertical

   from victor.framework.rl.base import RLOutcome

   # Just add feedback_source to metadata
   def create_outcome_with_user_feedback(
       session_id: str,
       rating: float,  # This becomes quality_score
       feedback: Optional[str] = None,
       helpful: Optional[bool] = None,
       correction: Optional[str] = None,
   ) -> RLOutcome:
       """Create RLOutcome with user feedback"""
       return RLOutcome(
           provider="user",  # Indicates user feedback
           model="feedback",
           task_type="feedback",
           success=True,  # Feedback collection is always successful
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

2. **User Feedback Learner** (Extends Existing quality_weights Learner):
   ```python
   class UserFeedbackLearner(BaseLearner):
       """Learn from explicit user feedback, extends quality_weights"""

       def __init__(self, config: LearnerConfig):
           super().__init__(
               learner_name="user_feedback",  # New learner
               outcome_type="user_feedback",
               config=config
           )

       async def learn(self, outcomes: List[RLOutcome]) -> List[RLRecommendation]:
           """Learn from user feedback, update quality_weights"""
           # Filter for user feedback outcomes
           user_outcomes = [o for o in outcomes if o.metadata.get("feedback_source") == "user"]

           if not user_outcomes:
               return []  # No user feedback to learn from

           # Aggregate feedback by context
           feedback_by_context = self._aggregate_by_context(user_outcomes)

           # Generate recommendations for quality_weights learner
           # Don't create separate optimization, feed into existing learner
           recommendations = []

           for context, ratings in feedback_by_context.items():
               avg_rating = sum(r.quality_score for r in ratings) / len(ratings)

               # Recommend quality weight adjustments
               recommendations.append(RLRecommendation(
                   learner_name="quality_weights",
                   recommendation_type="weight_adjustment",
                   key=context,
                   value=avg_rating,
                   confidence=min(len(ratings) / 10, 1.0),  # More samples = higher confidence
                   metadata={"sample_size": len(ratings)}
               ))

           return recommendations

       async def record_feedback(self, session_id: str, rating: float,
                                 feedback: Optional[str] = None):
           """Record user feedback for a session"""
           # Create RLOutcome with feedback
           outcome = create_outcome_with_user_feedback(
               session_id=session_id,
               rating=rating,
               feedback=feedback
           )

           # Record through RL coordinator
           await self._coordinator.record_outcome(outcome)

           # Trigger background learning
           await self.learn([outcome])
   ```

3. **Preference Learning**:
   - Learn user preferences over time
   - Personalized recommendations per user
   - Multi-criteria decision analysis

**Database Extensions**:
```sql
-- DO NOT create separate user_feedback table
-- INSTEAD: Extend existing rl_outcomes table with feedback fields

ALTER TABLE rl_outcomes ADD COLUMN feedback_source TEXT;  -- 'auto', 'user', 'hybrid'
ALTER TABLE rl_outcomes ADD COLUMN user_feedback TEXT;
ALTER TABLE rl_outcomes ADD COLUMN helpful BOOL;
ALTER TABLE rl_outcomes ADD COLUMN correction TEXT;

-- Index for feedback queries
CREATE INDEX idx_rl_outcomes_feedback_source
ON rl_outcomes(feedback_source, timestamp DESC);

-- Index for session-based feedback lookup
CREATE INDEX idx_rl_outcomes_session_feedback
ON rl_outcomes(metadata->>'$.session_id') WHERE feedback_source = 'user';
```

**API Extensions**:
```python
# New CLI command for feedback collection
@app.command()
def feedback(session_id: str, rating: float, feedback: Optional[str] = None):
    """Record user feedback for a session"""
    # Use existing RL coordinator
    from victor.core import get_container
    container = get_container()

    if container.has_service("rl_coordinator"):
        coordinator = container.get_service("rl_coordinator")
        learner = coordinator.get_learner("user_feedback")

        if learner:
            import asyncio
            asyncio.run(learner.record_feedback(session_id, rating, feedback))
            typer.echo(f"Feedback recorded for session {session_id}")
            return

    typer.Error("RL coordinator not available")

# New API endpoint (for API server)
@router.post("/feedback")
async def record_feedback(request: FeedbackRequest):
    """REST API for feedback submission"""
    # Implementation similar to CLI command
    pass
```

### 3.4 Component 3: Predictive Execution Strategies

**Purpose**: Build on Priority 3 (predictive tool selection) with learning

**Design**: Extend existing `tool_selector` learner and integrate with `UsageAnalytics`

**CRITICAL**: Reuse existing `ToolPredictor` from Priority 3 instead of recreating prediction logic.

**Key Features**:

1. **Extended Tool Selector Learner** (Integrates Existing Components):
   ```python
   class ToolSelectorLearner(BaseLearner):
       """Extend existing tool_selector learner with UsageAnalytics and Priority 3 predictor"""

       def __init__(self, config: LearnerConfig):
           super().__init__(
               learner_name="tool_selector",
               outcome_type="tool_execution",
               config=config
           )

           # Use existing analytics (don't rebuild tool tracking)
           from victor.agent.usage_analytics import UsageAnalytics
           self.analytics = UsageAnalytics.get_instance()

           # Use existing ToolPredictor from Priority 3
           from victor.agent.planning.tool_predictor import ToolPredictor
           self.predictor = ToolPredictor()

       async def learn(self, outcomes: List[RLOutcome]) -> List[RLRecommendation]:
           """Learn from tool execution outcomes using existing components"""

           # Feed outcomes into existing predictor's tracker
           for outcome in outcomes:
               tools = outcome.metadata.get('tools_used', [])
               task_type = outcome.task_type
               success = outcome.success

               # Use existing tracker from predictor
               self.predictor.cooccurrence_tracker.record_tool_sequence(
                   tools=tools,
                   task_type=task_type,
                   success=success
               )

           # Get predictions from existing predictor
           # Generate recommendations based on learned patterns
           recommendations = []

           # Aggregate tool insights from existing analytics
           for tool_name in self._get_tool_names(outcomes):
               insights = self.analytics.get_tool_insights(tool_name)

               # Create recommendation for tool usage
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

       def get_next_tool_prediction(self, context: Dict[str, Any]) -> Optional[str]:
           """Get prediction for next tool using existing predictor"""
           predictions = self.predictor.predict_tools(
               task_description=context.get("task_description"),
               current_step=context.get("current_step"),
               recent_tools=context.get("recent_tools", []),
               task_type=context.get("task_type")
           )

           return predictions[0].tool_name if predictions else None
   ```

2. **Execution Pattern Mining** (Using Existing Analytics):
   - Use existing `get_optimization_recommendations()` from UsageAnalytics
   - Mine patterns from existing tool insights
   - Identify successful tool chains from historical data

3. **Predictive Caching** (Extend Existing ToolPreloader):
   ```python
   # Extend existing ToolPreloader with RL feedback
   from victor.agent.planning.tool_preloader import ToolPreloader

   class LearningToolPreloader(ToolPreloader):
       """Extend existing preloader with learned patterns"""

       def __init__(self, *args, tool_selector_learner=None, **kwargs):
           super().__init__(*args, **kwargs)
           self.tool_learner = tool_selector_learner

       async def preload_for_next_step(self, *args, **kwargs):
           """Enhanced preloading using learned patterns"""
           # Get prediction from tool selector learner
           prediction = self.tool_learner.get_next_tool_prediction({
               "task_description": kwargs.get("task_description"),
               "current_step": kwargs.get("current_step"),
               "recent_tools": kwargs.get("recent_tools"),
               "task_type": kwargs.get("task_type")
           })

           if prediction:
               # Add to preload queue
               self._preload_queue.add(prediction)

           # Call existing preloading logic
           await super().preload_for_next_step(*args, **kwargs)
   ```

**Integration with Existing Components**:
- **UsageAnalytics**: Get tool insights and optimization recommendations
- **Priority 3 ToolPredictor**: Reuse existing prediction logic and CooccurrenceTracker
- **Priority 3 ToolPreloader**: Extend with learned patterns, don't recreate
- **Existing tool_selector learner**: Extend rather than replace

**Learning Loop**:
```
Tool Execution → UsageAnalytics.record() → RLOutcome →
ToolSelectorLearner.learn() → ToolPredictor (Priority 3) →
ToolPreloader (Priority 3) → (Repeat)
```

**No Duplication Checklist**:
- ✅ Uses existing UsageAnalytics (doesn't recreate tool tracking)
- ✅ Uses existing ToolPredictor (doesn't recreate prediction logic)
- ✅ Extends existing ToolPreloader (doesn't recreate preloading)
- ✅ Extends existing tool_selector learner (doesn't create new learner)

---

## Part 4: Implementation Roadmap

### Phase 0: Infrastructure Audit (Week 0 - BEFORE Phase 1) **NEW**

**CRITICAL**: Must complete this phase before any implementation to avoid duplication.

**Goals**:
1. Audit existing RL schema in `victor/core/schema.py`
2. Document UsageAnalytics data flow and storage
3. Create integration test for existing components
4. Document all 14 existing learners and their capabilities
5. Create "No Duplication" test suite

**Week 1: Schema & Components Audit**
- [ ] Review existing RL schema (document all tables, columns, indexes)
- [ ] Document all 14 existing learners (purpose, methods, dependencies)
- [ ] Map UsageAnalytics data flow (in-memory → persistence)
- [ ] Document existing outcome tracking (JSONL + ConversationStore)
- [ ] Identify all extension points (what can be extended, what must be replaced)

**Week 2: Integration Testing**
- [ ] Create baseline integration tests for existing components
- [ ] Test UsageAnalytics → RL database flow (verify no data loss)
- [ ] Test existing learner learn() methods
- [ ] Test RLRecommendation generation
- [ ] Create "No Duplication" test (verifies new code extends, not replaces)

**Deliverables**:
- Infrastructure Audit Report (detailed component inventory)
- Data Flow Diagrams (UsageAnalytics → RL, JSONL → RL, etc.)
- Baseline Integration Test Suite
- "No Duplication" Test Suite
- Extension Point Documentation (what can be extended, how)

**Success Criteria**:
- All 14 learners documented with extension points
- UsageAnalytics integration plan documented
- Baseline tests passing
- No duplication risks identified

---

### Phase 1: Foundation (Months 1-2) - REVISED

**Week 1-2: Database Schema Extensions** (REVISED)
- [ ] Audit existing RL schema for conflicts (from Phase 0)
- [ ] Extend existing `rl_outcomes` table (NOT create new tables):
  - ADD COLUMN `feedback_source` TEXT
  - ADD COLUMN `user_feedback` TEXT
  - ADD COLUMN `helpful` BOOL
  - ADD COLUMN `correction` TEXT
  - ADD COLUMN `session_summary` TEXT
  - ADD COLUMN `session_id` TEXT
  - ADD COLUMN `repo_id` TEXT
- [ ] Create migration scripts (ALTER TABLE, not CREATE TABLE)
- [ ] Update schema version
- [ ] Create indexes for trend and feedback queries

**Week 3-4: User Feedback Infrastructure** (REVISED)
- [ ] Create `create_outcome_with_user_feedback()` helper function
- [ ] Create `UserFeedbackLearner` (extends quality_weights, doesn't replace)
- [ ] Add feedback CLI command (follow existing patterns in `victor/ui/slash/commands/`)
- [ ] Add feedback API endpoint (follow existing patterns in `victor/ui/api/`)
- [ ] Integrate with existing RL coordinator (don't create new coordinator)

**Week 5-6: Meta-Learning Integration** (REVISED)
- [ ] Extend UsageAnalytics with `persist_to_rl_database()` method
- [ ] Implement session aggregation using existing `get_session_summary()`
- [ ] Add trend detection using historical data from RL database
- [ ] Create pattern consolidation using existing `get_optimization_recommendations()`
- [ ] Integrate with existing `cross_vertical` learner for pattern export/import

**Week 7-8: Integration & Testing** (REVISED)
- [ ] Integration tests for UsageAnalytics → RL database flow
- [ ] Unit tests for extensions (verify they extend, not replace)
- [ ] "No Duplication" tests (verify new code extends existing components)
- [ ] Backward compatibility tests (ensure existing learners still work)
- [ ] Documentation updates

**Deliverables**:
- Extended database schema (no new tables, only ALTER TABLE on existing)
- User feedback collection system (extends existing outcomes)
- Meta-learning capabilities (uses existing UsageAnalytics)
- Test coverage >85%
- No duplication verified

---

### Phase 2: Predictive Execution (Months 3-4) - REVISED

**Week 9-10: Extend Tool Selector Learner** (REVISED)
- [ ] Integrate with existing `UsageAnalytics` (get_tool_insights)
- [ ] Use existing `ToolPredictor` from Priority 3 (don't recreate)
- [ ] Implement execution pattern mining using existing analytics
- [ ] Add next-tool prediction using existing predictor
- [ ] Extend existing `tool_selector` learner (don't create new)

**Week 11-12: Predictive Caching** (REVISED)
- [ ] Extend existing `ToolPreloader` with RL feedback loop
- [ ] Add prediction metadata to cache (extend existing cache)
- [ ] Implement dynamic preloading using existing `get_next_tool_prediction()`
- [ ] Add cache size adjustment based on prediction accuracy
- [ ] Test with existing ToolPredictor integration

**Week 13-14: Model Selector Enhancement**
- [ ] Extend existing `ModelSelectorLearner` with learned thresholds
- [ ] Track model success rates using existing outcome tracking
- [ ] Implement dynamic model routing using existing recommendation system
- [ ] Integrate with Priority 1's HybridDecisionService (extend, don't replace)

**Week 15-16: Integration & Testing** (REVISED)
- [ ] Verify all components extend existing (not duplicate)
- [ ] Performance testing (compare to baseline, ensure improvements)
- [ ] A/B testing vs baseline (existing learners vs. extended)
- [ ] "No Duplication" regression tests
- [ ] Documentation

**Deliverables**:
- Extended tool_selector learner (uses existing ToolPredictor + UsageAnalytics)
- Extended model_selector learner (uses existing recommendation system)
- Extended cache (uses existing ToolPreloader)
- Performance improvements (15-25% tool latency reduction)
- Zero duplication verified

### Phase 3: Advanced Features (Months 5-6)

**Week 17-18: Transfer Learning**
- Implement pattern export/import
- Add domain adaptation
- Create cross-project learning

**Week 19-20: Preference Learning**
- Personalized quality weights
- Multi-stakeholder preference aggregation
- User-specific recommendations

**Week 21-22: Explainability**
- Recommendation explanation system
- Visualization of learned patterns
- Debugging tools for recommendations

**Week 23-24: Optimization & Production Readiness**
- Performance optimization
- Production monitoring setup
- Gradual rollout with feature flags
- Documentation and training materials

**Deliverables**:
- Transfer learning system
- Preference learning engine
- Explainability tools
- Production-ready system

### Phase 4: Long-Term Research (Months 7+)

**Research Directions**:
1. **Multi-Agent Learning**: Team composition optimization
2. **Hierarchical Learning**: Multi-level decision optimization
3. **Online Learning**: Real-time adaptation without offline training
4. **Unsupervised Learning**: Pattern discovery without explicit feedback
5. **Reinforcement Learning**: Deep RL for complex decision sequences

---

## Part 5: Success Metrics

### 5.1 Learning Quality Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Feedback incorporation rate | >50% of sessions | % sessions with user feedback |
| Pattern prediction accuracy | >75% | Top-1 prediction accuracy |
| Transfer learning success | >60% | Success rate in new projects |
| Meta-learning improvement | >20% | Improvement over baseline learners |

### 5.2 Performance Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Tool selection accuracy | >85% | Correct tool predictions |
| Model routing accuracy | >80% | Optimal model selection |
| Cache hit rate (predictive) | >70% | Predictive cache effectiveness |
| Recommendation latency | <100ms | Time to generate recommendations |

### 5.3 User Satisfaction Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| User feedback rating | >0.75 | Average user rating |
| Helpfulness rate | >80% | % recommendations marked helpful |
| Preference alignment | >70% | Alignment with user preferences |

---

## Part 6: Risk Mitigation

### 6.1 Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Overfitting to feedback** | High | Regularization, diversity injection, exploration bonuses |
| **Slow learning convergence** | Medium | Warm start with existing patterns, adaptive learning rates |
| **Database performance** | Medium | Indexing, batched writes, async operations |
| **Memory consumption** | Medium | Pattern pruning, size limits, LRU eviction |

### 6.2 Operational Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Poor quality feedback** | High | Feedback validation, spam filtering, reputation scores |
| **Privacy concerns** | High | Data anonymization, user consent, retention policies |
| **System complexity** | Medium | Gradual rollout, feature flags, comprehensive testing |
| **Maintenance burden** | Low | Automated monitoring, self-healing, clear documentation |

---

## Part 7: Integration with Existing Features

### 7.1 Priority 1 (Hybrid Decision Service) - CORRECTED

**Integration Points**:
- Extend existing `ModelSelectorLearner` with learned thresholds
- Learn optimal confidence thresholds per decision type
- Adaptive LLM fallback based on historical success

**CRITICAL**: Do NOT create `AdaptiveHybridDecisionService`. Extend existing `ModelSelectorLearner` instead.

**Corrected Enhancement**:
```python
# EXTEND existing ModelSelectorLearner, don't create new service
class ModelSelectorLearner(BaseLearner):
    """Extend existing ModelSelectorLearner with learned thresholds"""

    async def learn(self, outcomes: List[RLOutcome]) -> List[RLRecommendation]:
        """Learn optimal confidence thresholds per decision type"""
        # Aggregate outcomes by decision_type
        thresholds = {}

        for outcome in outcomes:
            decision_type = outcome.metadata.get("decision_type")
            confidence = outcome.metadata.get("heuristic_confidence")
            success = outcome.success

            if decision_type not in thresholds:
                thresholds[decision_type] = []

            thresholds[decision_type].append((confidence, success))

        # Calculate optimal threshold for each decision type
        recommendations = []
        for dt, values in thresholds.items():
            # Find threshold that maximizes success rate
            optimal_threshold = self._find_optimal_threshold(values)

            recommendations.append(RLRecommendation(
                learner_name="model_selector",
                recommendation_type="confidence_threshold",
                key=dt,
                value=optimal_threshold,
                confidence=0.8,
                metadata={"sample_size": len(values)}
            ))

        return recommendations

# Existing HybridDecisionService uses recommendations from learner
class HybridDecisionService:
    def _should_use_llm(self, decision_type, heuristic_confidence):
        """Check if should use LLM based on learned threshold"""
        # Check for learned threshold from model_selector learner
        learned_threshold = self._get_learned_threshold(decision_type)

        if learned_threshold is not None:
            return heuristic_confidence < learned_threshold

        # Fall back to static threshold
        return heuristic_confidence < self._static_thresholds.get(decision_type, 0.5)

    def _get_learned_threshold(self, decision_type):
        """Get learned threshold from model_selector learner"""
        # Query RL coordinator for latest recommendation
        rec = self._rl_coordinator.get_latest_recommendation(
            learner_name="model_selector",
            key=decision_type
        )

        return rec.value if rec else None
```

### 7.2 Priority 2 (Phase-Based Context Management) - CORRECTED

**Integration Points**:
- Extend existing `ModeTransitionLearner` with phase transition patterns
- Learn optimal phase transition timing per task type
- Adaptive phase-aware scoring weights

**CRITICAL**: Do NOT create `LearningPhaseDetector`. Extend existing `ModeTransitionLearner` instead.

**Corrected Enhancement**:
```python
# EXTEND existing ModeTransitionLearner, don't create new detector
class ModeTransitionLearner(BaseLearner):
    """Extend existing ModeTransitionLearner with phase transition patterns"""

    async def learn(self, outcomes: List[RLOutcome]) -> List[RLRecommendation]:
        """Learn optimal phase transition timing"""
        # Aggregate phase transitions by outcome
        transitions_by_phase = {}

        for outcome in outcomes:
            phase_sequence = outcome.metadata.get("phase_sequence", [])
            success = outcome.success

            for i in range(len(phase_sequence) - 1):
                transition = (phase_sequence[i], phase_sequence[i+1])

                if transition not in transitions_by_phase:
                    transitions_by_phase[transition] = []

                transitions_by_phase[transition].append(success)

        # Generate recommendations for optimal transitions
        recommendations = []
        for (from_phase, to_phase), outcomes_list in transitions_by_phase.items():
            success_rate = sum(outcomes_list) / len(outcomes_list)

            if success_rate > 0.7:  # Good transition
                recommendations.append(RLRecommendation(
                    learner_name="mode_transition",
                    recommendation_type="phase_transition",
                    key=f"{from_phase}->{to_phase}",
                    value="encourage",
                    confidence=success_rate
                ))

        return recommendations

# Existing PhaseDetector uses recommendations
class PhaseDetector:
    def should_transition(self, new_phase):
        """Check if transition should be allowed"""
        # Get learned recommendation
        rec = self._rl_coordinator.get_latest_recommendation(
            learner_name="mode_transition",
            key=f"{self._current_phase}->{new_phase}"
        )

        if rec and rec.value == "discourage":
            return False

        # Use existing logic (cooldown, validity checks)
        return self._check_cooldown_and_validity(new_phase)
```

### 7.3 Priority 3 (Predictive Tool Selection) - CORRECTED

**Integration Points**:
- Extend existing `ToolSelectorLearner` with RL feedback loop
- Use existing `ToolPredictor` from Priority 3 (don't recreate)
- Integrate with existing `UsageAnalytics`

**CRITICAL**: Do NOT create `LearningToolPredictor`. Extend existing `ToolSelectorLearner` instead.

**Corrected Enhancement**:
```python
# EXTEND existing ToolSelectorLearner, don't create new predictor
class ToolSelectorLearner(BaseLearner):
    """Extend existing ToolSelectorLearner with UsageAnalytics and Priority 3 predictor"""

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

    async def learn(self, outcomes: List[RLOutcome]) -> List[RLRecommendation]:
        """Learn from tool execution outcomes using existing components"""

        # Feed outcomes into existing predictor's tracker
        for outcome in outcomes:
            tools = outcome.metadata.get('tools_used', [])
            task_type = outcome.task_type
            success = outcome.success

            # Use existing tracker from predictor
            self.predictor.cooccurrence_tracker.record_tool_sequence(
                tools=tools,
                task_type=task_type,
                success=success
            )

        # Get predictions from existing predictor
        # Generate recommendations based on learned patterns
        recommendations = []

        # Aggregate tool insights from existing analytics
        for tool_name in self._get_tool_names(outcomes):
            insights = self.analytics.get_tool_insights(tool_name)

            # Create recommendation for tool usage
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

    def get_next_tool_prediction(self, context):
        """Get prediction for next tool using existing predictor"""
        predictions = self.predictor.predict_tools(
            task_description=context.get("task_description"),
            current_step=context.get("current_step"),
            recent_tools=context.get("recent_tools", []),
            task_type=context.get("task_type")
        )

        return predictions[0].tool_name if predictions else None
```

---

## Part 8: Testing Strategy

### 8.1 Phase 0: "No Duplication" Tests (Critical)

**CRITICAL**: These tests must pass before any implementation begins.

**Test Areas**:
- Verify no duplicate session aggregation (UsageAnalytics integration)
- Verify no duplicate user feedback tracking (RLOutcome.quality_score reuse)
- Verify no duplicate tool prediction (ToolPredictor integration)
- Verify all 14 existing learners are extended, not replaced

**Example Tests**:
```python
def test_no_duplicate_session_aggregation():
    """Verify we use existing UsageAnalytics, not create new aggregation"""
    # Should use existing singleton
    analytics = UsageAnalytics.get_instance()

    # Should NOT create separate session_summaries table
    schema = get_rl_database_schema()
    assert "session_summaries" not in schema.tables

    # Should extend existing rl_outcomes table
    assert "session_id" in schema.tables["rl_outcomes"].columns
    assert "session_summary" in schema.tables["rl_outcomes"].columns

def test_no_duplicate_user_feedback():
    """Verify we reuse existing quality_score field"""
    # Should reuse existing RLOutcome.quality_score
    outcome = RLOutcome(
        provider="user",
        model="feedback",
        task_type="feedback",
        success=True,
        quality_score=0.8,  # Reuse existing field
        metadata={"feedback_source": "user"}  # Add source tracking
    )

    # Should NOT create separate user_feedback table
    schema = get_rl_database_schema()
    assert "user_feedback" not in schema.tables

def test_no_duplicate_tool_prediction():
    """Verify we extend existing ToolPredictor, not recreate"""
    # Should use existing ToolPredictor from Priority 3
    from victor.agent.planning.tool_predictor import ToolPredictor

    predictor = ToolPredictor()
    assert predictor is not None

    # ToolSelectorLearner should integrate with existing predictor
    learner = ToolSelectorLearner(config=test_config)
    assert hasattr(learner, "predictor")
    assert isinstance(learner.predictor, ToolPredictor)

def test_all_learners_extended_not_replaced():
    """Verify all 14 existing learners are extended, not replaced"""
    existing_learners = [
        "cache_eviction",
        "context_pruning",
        "continuation_patience",
        "continuation_prompts",
        "cross_vertical",
        "grounding_threshold",
        "mode_transition",
        "model_selector",
        "prompt_optimizer",
        "prompt_template",
        "quality_weights",
        "semantic_threshold",
        "tool_selector",
        "workflow_execution",
    ]

    for learner_name in existing_learners:
        # Verify learner still exists
        learner_module = import_module(f"victor.framework.rl.learners.{learner_name}")
        assert hasattr(learner_module, "BaseLearner")

        # Verify we're extending, not replacing
        learner_class = getattr(learner_module, f"{to_camel_case(learner_name)}")
        assert issubclass(learner_class, BaseLearner)
```

### 8.2 Unit Tests

**Target Coverage**: >85%

**Test Areas** (After Phase 0 passes):
- Meta-learning aggregation logic (using existing UsageAnalytics)
- User feedback processing (reusing RLOutcome.quality_score)
- Predictive model integration (with existing ToolPredictor)
- Pattern mining algorithms (extending existing learners)
- Database schema migrations (ALTER, not CREATE)

**Example Tests**:
```python
def test_usage_analytics_integration():
    """Test integration with existing UsageAnalytics"""
    coordinator = MetaLearningCoordinator()

    # Create test sessions using existing analytics
    analytics = UsageAnalytics.get_instance()

    # Record tool usage using existing system
    for i in range(100):
        analytics.record_tool_execution(
            tool_name="read",
            success=True,
            execution_ms=50,
            error=None
        )

    # Aggregate using existing get_session_summary()
    summary = coordinator.aggregate_session_metrics(repo_id="test")

    # Verify integration with existing system
    assert summary.total_sessions == analytics.get_session_summary()["total_sessions"]
    assert summary.avg_quality_score > 0.0
    assert summary.success_rate > 0.0

def test_feedback_integration():
    """Test user feedback integration with existing RLOutcome"""
    learner = UserFeedbackLearner(config=test_config)

    # Record feedback using existing RLOutcome
    await learner.record_feedback(
        session_id="test123",
        rating=0.8,
        feedback="Good result"
    )

    # Verify feedback was stored in existing rl_outcomes table
    feedback = await learner.get_feedback(session_id="test123")
    assert feedback.quality_score == 0.8  # Reused field
    assert feedback.metadata["feedback_source"] == "user"  # New metadata

    # Verify learning occurred
    recommendations = await learner.learn([feedback_outcome])
    assert len(recommendations) > 0

def test_tool_predictor_integration():
    """Test predictive tool selection with existing ToolPredictor"""
    learner = ToolSelectorLearner(config=test_config)

    # Should integrate with existing predictor
    assert hasattr(learner, "predictor")
    assert isinstance(learner.predictor, ToolPredictor)

    # Should integrate with existing analytics
    assert hasattr(learner, "analytics")
    assert isinstance(learner.analytics, UsageAnalytics)

    # Train with known patterns
    outcomes = create_test_outcomes(pattern="search->read->edit")
    await learner.learn(outcomes)

    # Test prediction using existing predictor
    predictions = learner.predictor.predict_tools(
        task_description="Fix bug in login.py",
        current_step="exploration",
        recent_tools=["search"],
        task_type="bugfix"
    )

    # Should predict "read" with high confidence
    assert predictions[0].tool_name == "read"
    assert predictions[0].confidence > 0.7
```

### 8.3 Integration Tests

**Test Scenarios**:

1. **Full Learning Loop** (with existing components):
   - Execute task → Record outcome via existing RLCoordinator → Learn → Generate recommendations → Verify improvement

2. **Cross-Session Learning** (with UsageAnalytics):
   - Execute 100 sessions → Aggregate using existing get_session_summary() → Persist to RL database → Test prediction accuracy

3. **User Feedback Integration** (with RLOutcome):
   - Collect feedback → Store in existing rl_outcomes table → Learn from quality_score → Check recommendation quality

4. **Transfer Learning** (with existing learners):
   - Learn in Project A → Export patterns from existing learners → Import to Project B → Verify success

5. **No Duplication Verification**:
   - Verify no new tables created (only ALTER existing)
   - Verify all new code extends existing classes
   - Verify no duplicate functionality across components

### 8.4 Performance Tests

**Benchmarks**:

1. **Learning Latency**:
   - Target: <1s for 100 outcomes
   - Measure: Time to process outcomes via existing RLCoordinator
   - Verify: No degradation compared to baseline (existing learners)

2. **Prediction Accuracy**:
   - Target: >75% top-1 accuracy
   - Measure: Correct predictions / total predictions
   - Verify: Improvement over existing ToolSelectorLearner baseline

3. **Memory Usage**:
   - Target: <500MB for 10K patterns
   - Measure: Memory consumption with large pattern libraries
   - Verify: No significant increase from existing UsageAnalytics memory footprint

4. **Database Performance**:
   - Target: <100ms for 1K outcome writes
   - Measure: Batch insert performance to existing rl_outcomes table
   - Verify: ALTER operations don't degrade existing write performance

5. **Integration Overhead**:
   - Target: <10% overhead from UsageAnalytics integration
   - Measure: Time to aggregate sessions via get_session_summary()
   - Verify: No duplicate aggregation logic

---

## Part 9: Documentation

### 9.1 User Documentation

**Documents to Create**:

1. **User Feedback Guide**:
   - How to provide feedback via CLI
   - Feedback API reference (using existing RLOutcome.quality_score)
   - CLI commands: `victor feedback add|list|analyze`
   - Best practices for effective feedback

2. **Meta-Learning Guide**:
   - How to interpret patterns from existing UsageAnalytics
   - Trend analysis using get_session_summary()
   - Transfer learning procedures between projects
   - Understanding pattern consolidation

3. **Predictive Execution Guide**:
   - How to use predictive features with existing ToolPredictor
   - Configuration options for tool prediction
   - Performance tuning for tool selection
   - Integration with Priority 3 enhancements

### 9.2 Developer Documentation

**Documents to Create**:

1. **Architecture Documentation**:
   - System design (extending existing RL framework)
   - Component interactions (UsageAnalytics, RLOutcome, ToolPredictor)
   - Data flow diagrams (showing integration points)
   - **CRITICAL**: "No Duplication" design principles

2. **API Documentation**:
   - Learner extension API (how to extend existing 14 learners)
   - Database schema reference (ALTER operations only)
   - Configuration options (integrating with existing settings)
   - UsageAnalytics integration guide

3. **Contribution Guide**:
   - How to extend existing learners (not create new ones)
   - Testing guidelines (including Phase 0 "no duplication" tests)
   - Code review checklist (verify no duplicate functionality)
   - Integration test templates for existing components

---

## Part 10: Summary

### 10.1 Key Principles

1. **Build on Existing Infrastructure**: Extend, don't duplicate (CRITICAL)
2. **Phase 0 Audit First**: Verify existing components before implementation
3. **Gradual Rollout**: Feature flags, A/B testing, monitoring
4. **Data-Driven**: Use existing outcome tracking and UsageAnalytics
5. **User-Centric**: Incorporate human feedback via RLOutcome.quality_score
6. **Performance-Aware**: Maintain <100ms recommendation latency
7. **Test-Driven**: Phase 0 "no duplication" tests must pass first

### 10.2 Success Criteria

- ✅ **Phase 0 audit completed** (existing infrastructure documented)
- ✅ All components **extend** existing RL framework (no duplication)
- ✅ No new tables created (only ALTER existing rl_outcomes)
- ✅ All 14 existing learners preserved and extended
- ✅ UsageAnalytics integrated (not replaced)
- ✅ RLOutcome.quality_score reused for feedback (not new table)
- ✅ ToolPredictor integrated (not recreated)
- ✅ Test coverage >85% (including Phase 0 tests)
- ✅ Performance targets met (see Part 5)
- ✅ Comprehensive documentation (including integration guides)
- ✅ Production-ready with monitoring and rollback

### 10.3 Next Steps

1. **Complete Phase 0 Audit** (Week 0):
   - Document all 14 existing learners
   - Audit UsageAnalytics data flow and storage
   - Verify RL database schema in `victor/core/schema.py`
   - Create integration tests for baseline verification

2. **Review and Approve Design**:
   - Get stakeholder feedback on corrected approach
   - Verify no duplication in design document
   - Approve Phase 0 audit findings

3. **Set Up Success Metrics**:
   - Establish baseline measurements for existing learners
   - Document current UsageAnalytics performance
   - Create monitoring dashboards for integration points

4. **Begin Phase 1 Implementation** (after Phase 0 approval):
   - Database schema extensions (ALTER only, no CREATE)
   - Extend existing learners (not replace)
   - Integrate with UsageAnalytics and ToolPredictor

5. **Iterate Based on Data**:
   - Monitor production metrics
   - Adjust based on performance data
   - Verify no regressions in existing functionality

---

**This design document provides a roadmap for extending Victor's existing RL infrastructure with meta-learning, user feedback, and predictive execution capabilities while building on what already exists rather than duplicating it.**

**CRITICAL REMINDER**: All implementation must pass Phase 0 "no duplication" tests before proceeding to Phase 1.
