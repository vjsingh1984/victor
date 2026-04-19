# Priority 4 Phase 2: Component Integration Status

**Status**: ✅ **COMPONENTS ALREADY IMPLEMENTED - Integration Required**

---

## Overview

Phase 2 involves extending 3 RL learners to integrate with components from Priorities 1-3. The components themselves are **already implemented** - Phase 2 is about **wiring them together**.

---

## Existing Components (From Priorities 1-3)

### ✅ Priority 1: Hybrid Decision Service

**Location**: `victor/agent/services/hybrid_decision_service.py`

**Status**: Implemented and tested

**Capabilities**:
- Deterministic decision rules (lookup tables, pattern matching)
- Confidence calibrator (adaptive thresholds)
- Multi-level decision cache (L1/L2)
- LLM fallback for complex cases

**Integration Point**: `ModelSelectorLearner`

### ✅ Priority 2: Phase-Based Context Management

**Location**: `victor/agent/context_phase_detector.py`

**Status**: Implemented and tested

**Capabilities**:
- `PhaseDetector` - Detects task phase (EXPLORATION, PLANNING, EXECUTION, REVIEW)
- `PhaseTransitionDetector` - Manages phase transitions with cooldown
- Phase-aware scoring weights

**Integration Point**: `ModeTransitionLearner`

### ✅ Priority 3: Predictive Tool Selection

**Location**: `victor/agent/planning/tool_predictor.py`

**Status**: Implemented and tested

**Capabilities**:
- `ToolPredictor` - Ensemble prediction (keyword, semantic, co-occurrence)
- `CooccurrenceTracker` - Tracks tool usage patterns
- `ToolPreloader` - Preloads tool schemas

**Integration Point**: `ToolSelectorLearner`

---

## Phase 2 Work: Extend RL Learners

### Integration 1: Extend ModelSelectorLearner

**File to Create**: `victor/framework/rl/learners/model_selector_extended.py`

**Purpose**: Extend `ModelSelectorLearner` to use `HybridDecisionService`

**Implementation**:
```python
from victor.framework.rl.learners.model_selector import ModelSelectorLearner
from victor.agent.services.hybrid_decision_service import HybridDecisionService
from victor.framework.rl.base import LearnerConfig, RLOutcome, RLRecommendation

class ExtendedModelSelectorLearner(ModelSelectorLearner):
    """Extend ModelSelectorLearner with HybridDecisionService integration.

    Priority 1 Feature: Hybrid Decision Service
    - Uses deterministic rules for fast decisions
    - Falls back to LLM for complex cases
    - Learns from decision outcomes
    """

    def __init__(self, config: LearnerConfig):
        super().__init__(
            learner_name="model_selector",
            outcome_type="model_selection",
            config=config
        )

        # Integrate HybridDecisionService from Priority 1
        self.decision_service = HybridDecisionService()

    def learn(self, outcomes: List[RLOutcome]) -> List[RLRecommendation]:
        """Learn from model selection outcomes using hybrid decisions."""
        recommendations = []

        for outcome in outcomes:
            # Track which decision path was used
            used_llm = outcome.metadata.get("used_llm", False)
            decision_latency_ms = outcome.metadata.get("decision_latency_ms", 0)

            # Learn optimal decision thresholds
            if not used_llm and decision_latency_ms < 100:
                # Fast path worked well - reinforce
                recommendations.append(RLRecommendation(
                    learner_name="model_selector",
                    recommendation_type="decision_threshold",
                    key="fast_path_confidence",
                    value="increase",
                    confidence=0.8,
                    metadata={
                        "decision_latency_ms": decision_latency_ms,
                        "used_llm": used_llm,
                    }
                ))

        return recommendations

    def select_model(
        self,
        task_type: str,
        context: Dict[str, Any]
    ) -> str:
        """Select model using hybrid decision service."""
        from victor.agent.decisions.schemas import DecisionType

        # Use hybrid decision service
        decision = self.decision_service.decide_sync(
            decision_type=DecisionType.MODEL_SELECTION,
            context={
                "task_type": task_type,
                **context
            }
        )

        return decision.result.model_name
```

### Integration 2: Extend ModeTransitionLearner

**File to Create**: `victor/framework/rl/learners/mode_transition_extended.py`

**Purpose**: Extend `ModeTransitionLearner` to use `PhaseDetector`

**Implementation**:
```python
from victor.framework.rl.learners.mode_transition import ModeTransitionLearner
from victor.agent.context_phase_detector import PhaseDetector, PhaseTransitionDetector
from victor.core.shared_types import TaskPhase
from victor.framework.rl.base import LearnerConfig, RLOutcome, RLRecommendation
from victor.agent.conversation.state_machine import ConversationStage

class ExtendedModeTransitionLearner(ModeTransitionLearner):
    """Extend ModeTransitionLearner with PhaseDetector integration.

    Priority 2 Feature: Phase-Based Context Management
    - Detects task phase (EXPLORATION, PLANNING, EXECUTION, REVIEW)
    - Manages phase transitions with cooldown
    - Phase-aware context scoring
    """

    def __init__(self, config: LearnerConfig):
        super().__init__(
            learner_name="mode_transition",
            outcome_type="mode_transition",
            config=config
        )

        # Integrate PhaseDetector from Priority 2
        self.phase_detector = PhaseDetector()
        self.transition_detector = PhaseTransitionDetector()

    def learn(self, outcomes: List[RLOutcome]) -> List[RLRecommendation]:
        """Learn from mode transitions using phase detection."""
        recommendations = []

        for outcome in outcomes:
            # Get phase information
            detected_phase = outcome.metadata.get("detected_phase")
            transition_successful = outcome.metadata.get("transition_successful", True)

            if detected_phase and transition_successful:
                # Learn phase transition patterns
                recommendations.append(RLRecommendation(
                    learner_name="mode_transition",
                    recommendation_type="phase_transition",
                    key=f"to_{detected_phase}",
                    value="allow",
                    confidence=0.9,
                    metadata={
                        "detected_phase": detected_phase,
                        "transition_successful": transition_successful,
                    }
                ))

        return recommendations

    def detect_phase(
        self,
        current_stage: ConversationStage,
        recent_tools: List[str],
        message_content: str
    ) -> TaskPhase:
        """Detect current task phase using PhaseDetector."""
        return self.phase_detector.detect_phase(
            current_stage=current_stage,
            recent_tools=recent_tools,
            message_content=message_content
        )

    def should_transition(
        self,
        current_phase: TaskPhase,
        new_phase: TaskPhase
    ) -> bool:
        """Check if phase transition should be allowed."""
        return self.transition_detector.should_transition(
            old_phase=current_phase,
            new_phase=new_phase
        )
```

### Integration 3: Extend ToolSelectorLearner

**File to Create**: `victor/framework/rl/learners/tool_selector_extended.py`

**Purpose**: Extend `ToolSelectorLearner` to use `ToolPredictor` and `UsageAnalytics`

**Implementation**:
```python
from victor.framework.rl.learners.tool_selector import ToolSelectorLearner
from victor.agent.planning.tool_predictor import ToolPredictor
from victor.agent.usage_analytics import UsageAnalytics
from victor.framework.rl.base import LearnerConfig, RLOutcome, RLRecommendation

class ExtendedToolSelectorLearner(ToolSelectorLearner):
    """Extend ToolSelectorLearner with ToolPredictor and UsageAnalytics integration.

    Priority 3 Feature: Predictive Tool Selection
    - Ensemble prediction (keyword, semantic, co-occurrence)
    - Usage analytics for tool insights
    - Preloads tool schemas for performance
    """

    def __init__(self, config: LearnerConfig):
        super().__init__(
            learner_name="tool_selector",
            outcome_type="tool_execution",
            config=config
        )

        # Integrate ToolPredictor from Priority 3
        self.predictor = ToolPredictor()

        # Integrate UsageAnalytics
        self.analytics = UsageAnalytics.get_instance()

    def learn(self, outcomes: List[RLOutcome]) -> List[RLRecommendation]:
        """Learn from tool execution outcomes using predictor and analytics."""
        recommendations = []

        # Train predictor with new outcomes
        for outcome in outcomes:
            if outcome.task_type == "tool_execution":
                # Update co-occurrence tracker
                self.predictor.cooccurrence_tracker.record_tool_sequence(
                    tools=outcome.metadata.get("tools_used", []),
                    task_type=outcome.metadata.get("task_type"),
                    success=outcome.success,
                )

        # Generate recommendations using analytics
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

    def predict_next_tool(
        self,
        task_description: str,
        current_step: str,
        recent_tools: List[str],
        task_type: str
    ) -> Optional[str]:
        """Predict next tool using ToolPredictor."""
        predictions = self.predictor.predict_tools(
            task_description=task_description,
            current_step=current_step,
            recent_tools=recent_tools,
            task_type=task_type,
        )

        return predictions[0].tool_name if predictions else None
```

---

## Phase 2 Completion Checklist

### Already Done (From Priorities 1-3)

- [x] HybridDecisionService implemented
- [x] PhaseDetector implemented
- [x] PhaseTransitionDetector implemented
- [x] ToolPredictor implemented
- [x] CooccurrenceTracker implemented
- [x] UsageAnalytics implemented

### Phase 2 Work Required

- [ ] Create ExtendedModelSelectorLearner
- [ ] Create ExtendedModeTransitionLearner
- [ ] Create ExtendedToolSelectorLearner
- [ ] Add integration tests
- [ ] Update RLCoordinator to use extended learners
- [ ] Document integration patterns

---

## Summary

**Phase 2 Status**: Components exist, integration required

**What's Done**:
- ✅ All Priority 1-3 components implemented
- ✅ Components tested and working
- ✅ Integration points identified

**What's Left**:
- [ ] Extend 3 RL learners with integration
- [ ] Add integration tests
- [ ] Update coordinator configuration

**Estimated Effort**: 2-3 days

---

*End of Phase 2 Status Documentation*
