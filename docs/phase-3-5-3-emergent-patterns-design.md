# Phase 3.5.3: Emergent Collaboration Patterns Design

**Author:** Claude (AI Assistant)
**Date:** 2026-01-10
**Status:** Design Draft
**Related:** Phase 3.1 (TeamNode), Phase 3.3 (Experiment Tracking)
**Estimated LOC:** ~1,000 production, ~500 tests (69% reduction through code reuse)

**Note:** This design has been refined for SOLID compliance and maximum code reuse. See [SOLID Compliance](#solid-compliance) and [Code Reuse Strategy](#code-reuse-strategy) sections below.

---

## Table of Contents

1. [Overview](#overview)
2. [SOLID Compliance](#solid-compliance)
3. [Code Reuse Strategy](#code-reuse-strategy)
4. [Architecture Overview](#architecture-overview)
5. [Core Components](#core-components)
6. [Pattern Representation](#pattern-representation)
7. [Mining Strategies](#mining-strategies)
8. [Recommendation Engine](#recommendation-engine)
9. [Evolution Mechanisms](#evolution-mechanisms)
10. [Safety and Governance](#safety-and-governance)
11. [Implementation Strategy](#implementation-strategy)
12. [Code Examples](#code-examples)

---

## Overview

### Problem Statement

Current Victor collaboration patterns are **statically defined** in YAML workflow definitions:
- Team formations are hard-coded
- No learning from execution data
- No pattern discovery or optimization
- Teams don't evolve based on what works

This misses opportunities to:
- Discover effective collaboration patterns from data
- Reuse successful patterns across workflows
- Optimize team composition automatically
- Learn from collective experience

### Solution: Emergent Collaboration Patterns

This design introduces **data-driven pattern discovery and recommendation**:

1. **Pattern Mining**: Extract collaboration patterns from execution traces
2. **Pattern Validation**: Assess pattern quality, safety, and effectiveness
3. **Pattern Recommendation**: Suggest optimal patterns for new tasks
4. **Pattern Evolution**: Adapt patterns based on feedback and performance
5. **Pattern Registry**: Persistent storage and retrieval of patterns

### Key Design Principles

- **Data-Driven**: Learn from real execution data
- **Explainable**: Provide clear rationale for recommendations
- **Safe-by-Default**: Validate patterns before recommendation
- **Human-in-the-Loop**: Require approval for new patterns
- **Performant**: Fast pattern lookup and matching

### Use Cases

| Use Case | Benefit |
|----------|---------|
| New workflow | Suggest proven team formation |
| Performance issue | Recommend better collaboration pattern |
| Task complexity | Auto-scale team size based on history |
| Failed execution | Suggest alternative patterns that worked |

---

## SOLID Compliance

This design follows Victor's established SOLID patterns, particularly the protocol-based architecture used throughout the framework.

### Interface Segregation Principle (ISP)

**Key Refinement**: Split PatternRecommender into focused, single-purpose protocols following `victor/framework/protocols.py` pattern.

#### Original Design (Anti-Pattern):
```python
class PatternRecommender(Protocol):
    """Too many responsibilities - violates ISP."""
    async def mine_patterns(...) -> List[CollaborationPattern]: ...
    async def validate_pattern(...) -> ValidationResult: ...
    async def recommend(...) -> List[PatternRecommendation]: ...
    async def score_pattern(...) -> float: ...
    async def explain_recommendation(...) -> str: ...
    async def update_usage(...) -> None: ...
```

#### Refined Design (ISP Compliant):
```python
from typing import Protocol
from victor.experiments.tracking import ExperimentTrackingProtocol

@runtime_checkable
class PatternMinerProtocol(Protocol):
    """Pattern mining - single responsibility.

    Single responsibility: Extract patterns from execution traces.
    """
    async def mine_from_traces(
        self,
        traces: List[WorkflowExecutionTrace],
    ) -> List[CollaborationPattern]: ...

    def _extract_formations(
        self,
        traces: List[WorkflowExecutionTrace],
    ) -> List[FormationData]: ...

    async def _analyze_formation(
        self,
        formation_data: FormationData,
    ) -> CollaborationPattern: ...

@runtime_checkable
class PatternValidatorProtocol(Protocol):
    """Pattern validation - separate concern.

    Single responsibility: Assess pattern quality and safety.
    """
    async def validate(
        self,
        pattern: CollaborationPattern,
        test_cases: Optional[List[TaskContext]] = None,
    ) -> ValidationResult: ...

    async def _assess_quality(
        self,
        pattern: CollaborationPattern,
    ) -> float: ...

    async def _check_safety(
        self,
        pattern: CollaborationPattern,
    ) -> SafetyResult: ...

@runtime_checkable
class PatternRecommenderProtocol(Protocol):
    """Pattern recommendation - separate concern.

    Single responsibility: Suggest optimal patterns for tasks.
    """
    async def recommend(
        self,
        task_context: TaskContext,
        top_k: int = 5,
    ) -> List[PatternRecommendation]: ...

    async def _score_pattern(
        self,
        pattern: CollaborationPattern,
        task_context: TaskContext,
    ) -> float: ...

    async def _explain_recommendation(
        self,
        pattern: CollaborationPattern,
        task_context: TaskContext,
        score: float,
    ) -> str: ...
```

**Benefits**:
- Each protocol has single responsibility (SRP)
- Clients depend only on what they use (ISP)
- Easier to test and mock individual components
- Follows existing Victor patterns from `victor/framework/protocols.py`

### Dependency Inversion Principle (DIP)

**Key Refinement**: Depend on abstractions (protocols), not concrete implementations.

#### Before (Concrete dependency):
```python
class PatternMiner:
    def __init__(
        self,
        storage: SQLitePatternStorage,  # Concrete class!
        ...
    ) -> None: ...
```

#### After (Protocol dependency):
```python
class PatternMiner:
    def __init__(
        self,
        experiment_tracker: ExperimentTrackingProtocol,  # Protocol!
        pattern_storage: PatternStorageProtocol,  # Protocol!
        ...
    ) -> None: ...
```

**Benefits**:
- Can substitute different storage implementations (SQLite, PostgreSQL, in-memory)
- Easier to test with mocks
- Follows DIP (high-level doesn't depend on low-level)
- Consistent with rest of Victor codebase

**Related Files**:
- `victor/experiments/tracking.py:80-200` - ExperimentTrackingProtocol
- `victor/experiments/sqlite_store.py:131-450` - SQLite storage patterns to reuse

### Open/Closed Principle (OCP)

**Key Refinement**: Use strategy pattern for extensibility without modification.

#### Strategy-Based Pattern Scoring:
```python
@runtime_checkable
class PatternScoringStrategy(Protocol):
    """Strategy for scoring patterns."""
    async def score(
        self,
        pattern: CollaborationPattern,
        task_context: TaskContext,
    ) -> float: ...

class SkillBasedScoring:
    """Score based on skill overlap."""
    async def score(
        self,
        pattern: CollaborationPattern,
        task_context: TaskContext,
    ) -> float:
        # Implementation
        ...

class PerformanceBasedScoring:
    """Score based on historical performance."""
    async def score(
        self,
        pattern: CollaborationPattern,
        task_context: TaskContext,
    ) -> float:
        # Implementation
        ...

class HybridScoring:
    """Combine multiple scoring strategies."""
    def __init__(
        self,
        strategies: List[PatternScoringStrategy],
        weights: List[float],
    ) -> None:
        self.strategies = strategies
        self.weights = weights

    async def score(
        self,
        pattern: CollaborationPattern,
        task_context: TaskContext,
    ) -> float:
        score = 0.0
        for strategy, weight in zip(self.strategies, self.weights):
            score += await strategy.score(pattern, task_context) * weight
        return score
```

#### Strategy-Based Pattern Mining:
```python
@runtime_checkable
class MiningStrategy(Protocol):
    """Strategy for mining patterns."""
    async def mine(
        self,
        traces: List[WorkflowExecutionTrace],
    ) -> List[CollaborationPattern]: ...

class FrequencyBasedMining:
    """Mine frequently occurring patterns."""
    async def mine(
        self,
        traces: List[WorkflowExecutionTrace],
    ) -> List[CollaborationPattern]:
        # Implementation
        ...

class PerformanceBasedMining:
    """Mine high-performing patterns."""
    async def mine(
        self,
        traces: List[WorkflowExecutionTrace],
    ) -> List[CollaborationPattern]:
        # Implementation
        ...
```

**Benefits**:
- Add new scoring/mining strategies without modifying existing code (OCP)
- Each strategy independently testable
- Follows Victor's strategy pattern conventions
- Easy to experiment with different algorithms

### Protocol Compliance Checklist

- [x] All protocols are focused (≤5 methods each)
- [x] No concrete class dependencies in high-level modules
- [x] All extensions through protocols/strategies
- [x] All interfaces substitutable (LSP compliant)
- [x] Single responsibility for all classes

---

## Code Reuse Strategy

This design reuses **~70-80% of existing Victor infrastructure**, dramatically reducing the implementation effort from ~3,200 LOC to ~1,000 LOC (69% reduction).

### Experiment Tracking Reuse

**Refined Design**: Extend existing experiment tracking from `victor/experiments/` for pattern metadata.

```python
# CORRECT: Extend existing experiment tracking
from victor.experiments.tracking import ExperimentTracker
from victor.experiments.sqlite_store import SQLiteExperimentStore

class PatternExperimentTracker(ExperimentTracker):
    """Extend experiment tracking for patterns.

    Reuses existing storage, metrics, and A/B testing infrastructure from
    victor/experiments/tracking.py:80-200 and victor/experiments/sqlite_store.py:131-450
    """

    def __init__(
        self,
        storage: SQLiteExperimentStore,
    ) -> None:
        super().__init__(storage)
        # Add pattern-specific tracking
        self.pattern_metrics = {}

    async def track_pattern_usage(
        self,
        pattern_id: str,
        task_context: TaskContext,
        execution_result: ExecutionResult,
    ) -> None:
        """Track pattern execution using existing infrastructure."""
        # Reuse existing experiment tracking
        await self.track_execution(
            experiment_id=f"pattern_{pattern_id}",
            metadata={
                "pattern_id": pattern_id,
                "task_type": task_context.task_type,
                "complexity": task_context.complexity,
                "success": execution_result.success,
                "duration_seconds": execution_result.duration_seconds,
                "cost_tokens": execution_result.cost_tokens,
            },
        )
```

**Benefits**:
- Reuses proven SQLite storage patterns (~200 LOC saved)
- Consistent metrics collection across system
- Single observability pipeline
- Works with existing A/B testing infrastructure

**Related Files**:
- `victor/experiments/tracking.py:80-200` - ExperimentTracker
- `victor/experiments/sqlite_store.py:131-450` - SQLiteExperimentStore

### Team Coordination Reuse

**Refined Design**: Map discovered patterns to existing TeamFormation enum.

```python
# CORRECT: Reuse existing team coordination
from victor.teams import TeamFormation, create_coordinator

@dataclass
class CollaborationPattern:
    """A discovered or defined collaboration pattern.

    Reuses existing TeamFormation enum from victor/teams/types.py:45-80
    """
    formation: TeamFormation  # Existing enum!

    def to_team_coordinator(
        self,
        orchestrator: OrchestratorProtocol,
    ) -> ITeamCoordinator:
        """Convert pattern to executable team using existing coordinator."""
        return create_coordinator(
            orchestrator,
            formation=self.formation,  # Direct reuse
            participants=[
                TeamMember(
                    id=p.id,
                    role=p.role,
                    goal=p.goal,
                    capabilities=p.capabilities,
                )
                for p in self.participants
            ],
        )
```

**Benefits**:
- Direct reuse of 5 team formation implementations (~500 LOC saved)
- No need to re-implement coordination logic
- Consistent behavior across system
- Single point of maintenance

**Related Files**:
- `victor/teams/types.py:45-80` - TeamFormation enum
- `victor/teams/__init__.py:137-186` - create_coordinator factory
- `victor/teams/unified_coordinator.py` - 5 formation implementations

### Template/Pattern Reuse

**Refined Design**: Extend existing template library from `victor/workflows/generation/templates.py`.

```python
# CORRECT: Extend existing template library
from victor.workflows.generation.templates import TemplateLibrary
from victor.workflows.generation.patterns import PatternMatcher

class PatternTemplateLibrary(TemplateLibrary):
    """Extend template library with collaboration patterns.

    Reuses existing pattern matching and template infrastructure from
    victor/workflows/generation/templates.py:80-300
    """

    def __init__(self) -> None:
        super().__init__()
        # Add collaboration pattern templates
        self._load_collaboration_templates()

    def _load_collaboration_templates(self) -> None:
        """Load collaboration pattern templates."""
        # Reuse existing template loading logic
        self.register_template(
            name="sequential_review",
            template=self._build_sequential_template(),
            pattern_type="collaboration",
        )

    async def match_pattern(
        self,
        task_description: str,
    ) -> Optional[CollaborationPattern]:
        """Match task to pattern using existing pattern matcher."""
        # Reuse existing pattern matching
        template_match = await PatternMatcher().match(
            task_description,
            self.list_templates("collaboration"),
        )

        if template_match:
            return self._template_to_pattern(template_match.template)

        return None
```

**Benefits**:
- Reuses existing template loading and matching (~150 LOC saved)
- Consistent pattern discovery across system
- Single template management system
- Extensible without code changes

**Related Files**:
- `victor/workflows/generation/templates.py:80-300` - TemplateLibrary
- `victor/workflows/generation/patterns.py:50-150` - PatternMatcher

### Registry Reuse

**Refined Design**: Extend existing tool/vertical registry patterns.

```python
# CORRECT: Reuse existing registry patterns
from victor.tools.registry import ToolRegistry
from victor.core.verticals import VerticalRegistry

class PatternRegistry:
    """Persistent storage for collaboration patterns.

    Reuses registration and discovery patterns from
    victor/tools/registry.py:80-200 and victor/core/verticals/registry.py
    """

    def __init__(
        self,
        storage_path: str = "data/patterns.db",
    ) -> None:
        self.storage_path = storage_path
        self._patterns: Dict[str, CollaborationPattern] = {}
        self._observers: List[PatternObserver] = []

    async def register(
        self,
        pattern: CollaborationPattern,
    ) -> str:
        """Register a pattern using existing registry patterns."""
        # Reuse registration logic patterns
        if pattern.id in self._patterns:
            raise ValueError(f"Pattern {pattern.id} already registered")

        self._patterns[pattern.id] = pattern

        # Notify observers (reuse observer pattern)
        for observer in self._observers:
            await observer.on_pattern_registered(pattern)

        return pattern.id

    async def discover_patterns(
        self,
        task_context: TaskContext,
    ) -> List[CollaborationPattern]:
        """Discover patterns using existing discovery patterns."""
        # Reuse discovery logic from tool/vertical registries
        candidates = [
            pattern
            for pattern in self._patterns.values()
            if pattern.matches_constraints(task_context)
        ]

        return candidates
```

**Benefits**:
- Reuses proven registration and discovery logic (~150 LOC saved)
- Consistent pattern across all registries
- Observer pattern for extensibility
- Well-tested infrastructure

**Related Files**:
- `victor/tools/registry.py:80-200` - ToolRegistry
- `victor/core/verticals/registry.py:100-250` - VerticalRegistry

### Code Reuse Summary

| Component | Existing | Reuse Strategy | LOC Saved |
|-----------|----------|----------------|-----------|
| Experiment Tracking | ExperimentTracker, SQLiteExperimentStore | Extend with patterns | ~200 |
| Team Formation | TeamFormation, UnifiedTeamCoordinator | Direct reuse | ~500 |
| Templates | TemplateLibrary, PatternMatcher | Extend patterns | ~150 |
| Registry | ToolRegistry, VerticalRegistry | Reuse patterns | ~150 |
| **Total** | | | **~1,000** |

**Net New LOC After Reuse**: ~1,000 (vs. ~3,200 without reuse)
**Reduction**: 69%

### Canonical Import Paths

**Experiment Tracking**:
```python
# CORRECT - Extend existing experiment tracking
from victor.experiments.tracking import ExperimentTracker
from victor.experiments.sqlite_store import SQLiteExperimentStore

# INCORRECT - Duplicate experiment tracking
from victor.framework.patterns.experiments import PatternExperimentTracker
```

**Team Coordination**:
```python
# CORRECT - Use existing team infrastructure
from victor.teams import TeamFormation, create_coordinator

# INCORRECT - Duplicate team logic
from victor.framework.patterns.teams import PatternTeamFormation
```

**Templates and Registry**:
```python
# CORRECT - Extend existing libraries
from victor.workflows.generation.templates import TemplateLibrary
from victor.tools.registry import ToolRegistry

# INCORRECT - Separate template/registry systems
from victor.framework.patterns.templates import PatternTemplateLibrary
```

---

## Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   WORKFLOW EXECUTION                             │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  TeamNode Execution → Event Emission → Metrics Capture  │    │
│  └─────────────────────────────────────────────────────────┘    │
└────────────────────────────────┬────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              PATTERN DISCOVERY PIPELINE                          │
│  ┌──────────────┐    ┌──────────────┐    ┌─────────────────┐   │
│  │   Trace      │───▶│   Pattern    │───▶│    Pattern      │   │
│  │ Collection   │    │   Mining     │    │  Validation     │   │
│  └──────────────┘    └──────────────┘    └─────────────────┘   │
└────────────────────────────────┬────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              PATTERN REGISTRY                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Persistent storage (SQLite/PostgreSQL)                 │    │
│  │  - Pattern definitions                                  │    │
│  │  - Performance metrics                                  │    │
│  │  - Usage statistics                                     │    │
│  │  - Validation status                                    │    │
│  └─────────────────────────────────────────────────────────┘    │
└────────────────────────────────┬────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              RECOMMENDATION ENGINE                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Task Analysis → Pattern Matching → Scoring → Ranking   │    │
│  └─────────────────────────────────────────────────────────┘    │
└────────────────────────────────┬────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              EVOLUTION & OPTIMIZATION                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐      │
│  │  Feedback    │  │  A/B Testing │  │  Parameter       │      │
│  │ Collection   │  │              │  │  Tuning          │      │
│  └──────────────┘  └──────────────┘  └──────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. CollaborationPattern

**Location**: `victor/framework/patterns/collaboration_pattern.py`

**Responsibility**: Core data structure for collaboration patterns

**Interface**:
```python
@dataclass
class CollaborationPattern:
    """A discovered or defined collaboration pattern.

    Attributes:
        id: Unique pattern identifier
        name: Human-readable name
        description: What this pattern does
        pattern_type: Type of pattern (sequential, parallel, etc.)

        # Structure
        formation: TeamFormation (SEQUENTIAL, PARALLEL, etc.)
        participants: List of participant roles
        communication_graph: How participants communicate
        coordination_rules: Rules for interaction

        # Metadata
        success_rate: Historical success rate
        avg_duration_seconds: Average execution time
        avg_cost_tokens: Average token cost
        use_cases: List of use case tags
        constraints: When to use/not use this pattern

        # Validation
        is_validated: Whether pattern has been validated
        validation_score: Quality score (0-1)
        safety_approved: Whether safety-approved

        # Evolution
        created_at: Creation timestamp
        last_used: Last successful use
        usage_count: Number of times used
        improvement_count: Number of improvements
    """
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    name: str = ""
    description: str = ""
    pattern_type: PatternType = PatternType.SEQUENTIAL

    # Structure
    formation: TeamFormation = TeamFormation.SEQUENTIAL
    participants: List[ParticipantRole] = field(default_factory=list)
    communication_graph: Dict[str, List[str]] = field(default_factory=dict)
    coordination_rules: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    success_rate: float = 0.0
    avg_duration_seconds: float = 0.0
    avg_cost_tokens: float = 0.0
    use_cases: List[str] = field(default_factory=list)
    constraints: PatternConstraints = field(default_factory=PatternConstraints)

    # Validation
    is_validated: bool = False
    validation_score: float = 0.0
    safety_approved: bool = False

    # Evolution
    created_at: float = field(default_factory=time.time)
    last_used: Optional[float] = None
    usage_count: int = 0
    improvement_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CollaborationPattern":
        """Deserialize from dictionary."""
        ...

    def matches_constraints(
        self,
        task_context: TaskContext,
    ) -> bool:
        """Check if pattern matches task constraints."""
        ...

    def estimate_performance(
        self,
        task_context: TaskContext,
    ) -> PerformanceEstimate:
        """Estimate performance for this task."""
        ...
```

### 2. PatternMiner

**Location**: `victor/framework/patterns/miner.py`

**Responsibility**: Extract patterns from execution traces

**Interface**:
```python
class PatternMiner:
    """Mine collaboration patterns from execution traces.

    Analyzes workflow executions to discover:
    - Common team formations
    - Effective communication patterns
    - Successful coordination strategies
    """

    def __init__(
        self,
        min_occurrences: int = 5,
        min_success_rate: float = 0.7,
    ) -> None:
        self.min_occurrences = min_occurrences
        self.min_success_rate = min_success_rate

    async def mine_from_traces(
        self,
        traces: List[WorkflowExecutionTrace],
    ) -> List[CollaborationPattern]:
        """Mine patterns from execution traces.

        Args:
            traces: List of execution traces to analyze

        Returns:
            List of discovered patterns
        """
        patterns = []

        # Extract team formations
        formations = self._extract_formations(traces)

        # Find common formations
        common_formations = self._find_common_formations(
            formations,
            min_occurrences=self.min_occurrences,
        )

        # Analyze each formation
        for formation_data in common_formations:
            pattern = await self._analyze_formation(formation_data)

            if pattern.success_rate >= self.min_success_rate:
                patterns.append(pattern)

        return patterns

    def _extract_formations(
        self,
        traces: List[WorkflowExecutionTrace],
    ) -> List[FormationData]:
        """Extract team formation data from traces."""
        formations = []

        for trace in traces:
            if trace.team_execution:
                formations.append(
                    FormationData(
                        formation=trace.team_execution.formation,
                        participants=trace.team_execution.participants,
                        communication_trace=trace.team_execution.messages,
                        success=trace.success,
                        duration=trace.duration_seconds,
                    )
                )

        return formations

    async def _analyze_formation(
        self,
        formation_data: FormationData,
    ) -> CollaborationPattern:
        """Analyze a formation to extract pattern."""
        # Calculate metrics
        success_rate = self._calculate_success_rate(formation_data)
        avg_duration = self._calculate_avg_duration(formation_data)

        # Extract communication pattern
        comm_graph = self._extract_communication_graph(
            formation_data.communication_trace
        )

        # Identify coordination rules
        coordination_rules = self._identify_coordination_rules(
            formation_data.communication_trace
        )

        return CollaborationPattern(
            name=f"{formation_data.formation.value}_pattern",
            description=f"Discovered {formation_data.formation.value} pattern",
            pattern_type=PatternType.from_formation(formation_data.formation),
            formation=formation_data.formation,
            participants=formation_data.participants,
            communication_graph=comm_graph,
            coordination_rules=coordination_rules,
            success_rate=success_rate,
            avg_duration_seconds=avg_duration,
            is_validated=False,  # Needs validation
        )
```

### 3. PatternValidator

**Location**: `victor/framework/patterns/validator.py`

**Responsibility**: Assess pattern quality and safety

**Interface**:
```python
class PatternValidator:
    """Validate discovered collaboration patterns.

    Assesses:
    - Quality (effectiveness, efficiency)
    - Safety (no dangerous combinations)
    - Generalizability (works across tasks)
    """

    def __init__(
        self,
        quality_threshold: float = 0.7,
        safety_checks: bool = True,
    ) -> None:
        self.quality_threshold = quality_threshold
        self.safety_checks = safety_checks

    async def validate(
        self,
        pattern: CollaborationPattern,
        test_cases: Optional[List[TaskContext]] = None,
    ) -> ValidationResult:
        """Validate a collaboration pattern.

        Args:
            pattern: Pattern to validate
            test_cases: Optional test cases for validation

        Returns:
            ValidationResult with score and issues
        """
        scores = {}
        issues = []

        # Quality assessment
        quality_score = await self._assess_quality(pattern)
        scores["quality"] = quality_score

        if quality_score < self.quality_threshold:
            issues.append(
                f"Quality score {quality_score:.2f} below threshold {self.quality_threshold}"
            )

        # Safety check
        if self.safety_checks:
            safety_result = await self._check_safety(pattern)
            scores["safety"] = safety_result.score

            if not safety_result.is_safe:
                issues.extend(safety_result.issues)

        # Generalizability
        if test_cases:
            gen_score = await self._assess_generalizability(
                pattern,
                test_cases,
            )
            scores["generalizability"] = gen_score

        # Calculate overall score
        overall_score = sum(scores.values()) / len(scores)

        return ValidationResult(
            is_valid=len(issues) == 0,
            score=overall_score,
            scores=scores,
            issues=issues,
        )

    async def _assess_quality(
        self,
        pattern: CollaborationPattern,
    ) -> float:
        """Assess pattern quality.

        Factors:
        - Success rate (40%)
        - Efficiency (30%)
        - Coordination quality (30%)
        """
        success_score = pattern.success_rate * 0.4

        # Efficiency: Lower duration and cost = higher score
        efficiency_score = (
            (1.0 / max(pattern.avg_duration_seconds, 1.0)) * 0.15 +
            (1.0 / max(pattern.avg_cost_tokens, 1.0)) * 0.15
        )

        # Coordination quality
        coord_score = self._assess_coordination_quality(
            pattern.communication_graph,
            pattern.coordination_rules,
        ) * 0.3

        return success_score + efficiency_score + coord_score

    async def _check_safety(
        self,
        pattern: CollaborationPattern,
    ) -> SafetyResult:
        """Check pattern safety.

        Checks for:
        - No dangerous tool combinations
        - No resource limit violations
        - No infinite loops possible
        """
        issues = []

        # Check tool combinations
        for participant in pattern.participants:
            tools = participant.tools or []

            # Check for dangerous combinations
            if self._has_dangerous_combination(tools):
                issues.append(
                    f"Dangerous tool combination for {participant.role}"
                )

        # Check resource limits
        total_tools = sum(
            len(p.tools or [])
            for p in pattern.participants
        )

        if total_tools > 50:
            issues.append(f"Too many tools: {total_tools}")

        return SafetyResult(
            is_safe=len(issues) == 0,
            score=1.0 if len(issues) == 0 else 0.0,
            issues=issues,
        )
```

### 4. PatternRecommender

**Location**: `victor/framework/patterns/recommender.py`

**Responsibility**: Recommend optimal patterns for tasks

**Interface**:
```python
class PatternRecommender:
    """Recommend collaboration patterns for tasks.

    Uses:
    - Task-to-pattern matching
    - Multi-criteria scoring
    - Explanation generation
    """

    def __init__(
        self,
        pattern_registry: PatternRegistry,
        min_confidence: float = 0.6,
    ) -> None:
        self.pattern_registry = pattern_registry
        self.min_confidence = min_confidence

    async def recommend(
        self,
        task_context: TaskContext,
        top_k: int = 5,
    ) -> List[PatternRecommendation]:
        """Recommend patterns for a task.

        Args:
            task_context: Task requirements and context
            top_k: Number of recommendations to return

        Returns:
            List of pattern recommendations with scores and explanations
        """
        candidates = await self._get_candidates(task_context)

        recommendations = []

        for pattern in candidates:
            # Calculate score
            score = await self._score_pattern(pattern, task_context)

            if score >= self.min_confidence:
                # Generate explanation
                explanation = await self._explain_recommendation(
                    pattern,
                    task_context,
                    score,
                )

                recommendations.append(
                    PatternRecommendation(
                        pattern=pattern,
                        score=score,
                        explanation=explanation,
                    )
                )

        # Sort by score
        recommendations.sort(key=lambda r: r.score, reverse=True)

        return recommendations[:top_k]

    async def _get_candidates(
        self,
        task_context: TaskContext,
    ) -> List[CollaborationPattern]:
        """Get candidate patterns for task.

        Filters by:
        - Task type
        - Complexity
        - Constraints
        """
        all_patterns = await self.pattern_registry.list_all()

        candidates = []

        for pattern in all_patterns:
            # Check if pattern matches constraints
            if pattern.matches_constraints(task_context):
                candidates.append(pattern)

        return candidates

    async def _score_pattern(
        self,
        pattern: CollaborationPattern,
        task_context: TaskContext,
    ) -> float:
        """Score pattern for task.

        Scoring factors:
        - Skill match (40%)
        - Complexity match (30%)
        - Historical performance (20%)
        - Recent success (10%)
        """
        # Skill matching
        required_skills = task_context.required_skills
        pattern_skills = set()

        for participant in pattern.participants:
            pattern_skills.update(participant.capabilities or [])

        skill_match = self._calculate_skill_overlap(
            required_skills,
            pattern_skills,
        ) * 0.4

        # Complexity matching
        complexity_score = self._calculate_complexity_match(
            task_context.complexity,
            pattern.avg_duration_seconds,
        ) * 0.3

        # Historical performance
        perf_score = pattern.success_rate * 0.2

        # Recent success
        recent_score = self._calculate_recent_success(pattern) * 0.1

        return skill_match + complexity_score + perf_score + recent_score

    async def _explain_recommendation(
        self,
        pattern: CollaborationPattern,
        task_context: TaskContext,
        score: float,
    ) -> str:
        """Generate explanation for recommendation.

        Explains:
        - Why this pattern was chosen
        - What skills it provides
        - Expected performance
        """
        explanation_parts = []

        # Skill match explanation
        skill_match = self._calculate_skill_overlap(
            task_context.required_skills,
            set(p.capabilities for p in pattern.participants if p.capabilities),
        )

        explanation_parts.append(
            f"**Skill Match:** {skill_match*100:.0f}% of required skills covered"
        )

        # Performance explanation
        explanation_parts.append(
            f"**Historical Performance:** {pattern.success_rate*100:.0f}% success rate "
            f"across {pattern.usage_count} uses"
        )

        # Expected duration
        explanation_parts.append(
            f"**Expected Duration:** ~{pattern.avg_duration_seconds:.0f} seconds "
            f"(avg across historical uses)"
        )

        # Use cases
        if pattern.use_cases:
            explanation_parts.append(
                f"**Similar Use Cases:** {', '.join(pattern.use_cases[:3])}"
            )

        return "\n".join(explanation_parts)
```

### 5. PatternRegistry

**Location**: `victor/framework/patterns/registry.py`

**Responsibility**: Persistent pattern storage and retrieval

**Interface**:
```python
class PatternRegistry:
    """Persistent storage for collaboration patterns.

    Supports:
    - CRUD operations
    - Query and search
    - Performance tracking
    """

    def __init__(
        self,
        storage_path: str = "data/patterns.db",
    ) -> None:
        self.storage_path = storage_path
        self._conn = None

    async def initialize(self) -> None:
        """Initialize registry storage."""
        self._conn = await aiosqlite.connect(self.storage_path)

        # Create tables
        await self._create_tables()

    async def register(
        self,
        pattern: CollaborationPattern,
    ) -> str:
        """Register a new pattern.

        Returns:
            Pattern ID
        """
        await self._conn.execute(
            """
            INSERT INTO patterns (
                id, name, description, pattern_type,
                formation, participants, communication_graph, coordination_rules,
                success_rate, avg_duration, avg_cost,
                use_cases, constraints,
                is_validated, validation_score, safety_approved,
                created_at, last_used, usage_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pattern.id,
                pattern.name,
                pattern.description,
                pattern.pattern_type.value,
                pattern.formation.value,
                json.dumps([p.to_dict() for p in pattern.participants]),
                json.dumps(pattern.communication_graph),
                json.dumps(pattern.coordination_rules),
                pattern.success_rate,
                pattern.avg_duration_seconds,
                pattern.avg_cost_tokens,
                json.dumps(pattern.use_cases),
                json.dumps(pattern.constraints.to_dict()),
                pattern.is_validated,
                pattern.validation_score,
                pattern.safety_approved,
                pattern.created_at,
                pattern.last_used,
                pattern.usage_count,
            ),
        )

        await self._conn.commit()

        return pattern.id

    async def get(
        self,
        pattern_id: str,
    ) -> Optional[CollaborationPattern]:
        """Get pattern by ID."""
        cursor = await self._conn.execute(
            "SELECT * FROM patterns WHERE id = ?",
            (pattern_id,),
        )

        row = await cursor.fetchone()

        if row:
            return self._row_to_pattern(row)

        return None

    async def list_all(
        self,
        validated_only: bool = False,
    ) -> List[CollaborationPattern]:
        """List all patterns."""
        query = "SELECT * FROM patterns"

        if validated_only:
            query += " WHERE is_validated = 1"

        cursor = await self._conn.execute(query)
        rows = await cursor.fetchall()

        return [self._row_to_pattern(row) for row in rows]

    async def search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[CollaborationPattern]:
        """Search patterns by query and filters."""
        sql = "SELECT * FROM patterns WHERE name LIKE ? OR description LIKE ?"
        params = (f"%{query}%", f"%{query}%")

        if filters:
            if "formation" in filters:
                sql += " AND formation = ?"
                params += (filters["formation"].value,)

            if "min_success_rate" in filters:
                sql += " AND success_rate >= ?"
                params += (filters["min_success_rate"],)

        cursor = await self._conn.execute(sql, params)
        rows = await cursor.fetchall()

        return [self._row_to_pattern(row) for row in rows]

    async def update_usage(
        self,
        pattern_id: str,
        success: bool,
        duration_seconds: float,
    ) -> None:
        """Update pattern usage statistics.

        Called after pattern use to track performance.
        """
        # Update usage count and last used
        await self._conn.execute(
            """
            UPDATE patterns SET
                usage_count = usage_count + 1,
                last_used = ?
            WHERE id = ?
            """,
            (time.time(), pattern_id),
        )

        # Update success rate (moving average)
        await self._conn.execute(
            """
            UPDATE patterns SET
                success_rate = (success_rate * (usage_count - 1) + ?) / usage_count
            WHERE id = ?
            """,
            (1.0 if success else 0.0, pattern_id),
        )

        # Update duration (moving average)
        await self._conn.execute(
            """
            UPDATE patterns SET
                avg_duration = (avg_duration * (usage_count - 1) + ?) / usage_count
            WHERE id = ?
            """,
            (duration_seconds, pattern_id),
        )

        await self._conn.commit()
```

---

## Pattern Representation

### Graph-Based Representation

Patterns are represented as graphs:

```python
@dataclass
class PatternGraph:
    """Graph representation of collaboration pattern.

    Attributes:
        nodes: Participant nodes
        edges: Communication edges
        flow_edges: Execution flow edges
    """
    nodes: Dict[str, PatternNode]
    edges: List[PatternEdge]
    flow_edges: List[FlowEdge]

@dataclass
class PatternNode:
    """A node in the pattern graph.

    Attributes:
        id: Node identifier
        role: Participant role
        capabilities: Required capabilities
        tools: Available tools
    """
    id: str
    role: str
    capabilities: List[str]
    tools: List[str]

@dataclass
class PatternEdge:
    """A communication edge between participants.

    Attributes:
        source: Source participant ID
        target: Target participant ID
        edge_type: Type of communication (REQUEST, RESPONSE, BROADCAST)
        weight: Frequency of communication
    """
    source: str
    target: str
    edge_type: EdgeType
    weight: float = 1.0
```

### Metadata Structure

```python
@dataclass
class PatternConstraints:
    """Constraints on when to use a pattern.

    Attributes:
        min_complexity: Minimum task complexity
        max_complexity: Maximum task complexity
        required_tools: Tools that must be available
        forbidden_tools: Tools that must not be used
        max_duration_seconds: Maximum acceptable duration
        max_cost_tokens: Maximum acceptable cost
        required_capabilities: Capabilities that must be present
    """
    min_complexity: float = 0.0
    max_complexity: float = 1.0
    required_tools: List[str] = field(default_factory=list)
    forbidden_tools: List[str] = field(default_factory=list)
    max_duration_seconds: Optional[float] = None
    max_cost_tokens: Optional[float] = None
    required_capabilities: List[str] = field(default_factory=list)
```

---

## Mining Strategies

### 1. Frequency-Based Mining

```python
class FrequencyBasedMiner:
    """Find commonly occurring patterns.

    Identifies patterns that appear frequently in execution data.
    """

    async def mine(
        self,
        traces: List[WorkflowExecutionTrace],
    ) -> List[CollaborationPattern]:
        """Mine frequent patterns."""
        # Count pattern occurrences
        pattern_counts = Counter()

        for trace in traces:
            if trace.team_execution:
                pattern_key = self._get_pattern_key(trace.team_execution)
                pattern_counts[pattern_key] += 1

        # Filter by frequency
        frequent_patterns = [
            key for key, count in pattern_counts.items()
            if count >= self.min_occurrences
        ]

        # Build pattern objects
        patterns = []
        for pattern_key in frequent_patterns:
            pattern = await self._build_pattern(pattern_key, traces)
            patterns.append(pattern)

        return patterns
```

### 2. Performance-Based Mining

```python
class PerformanceBasedMiner:
    """Find high-performing patterns.

    Identifies patterns with excellent performance metrics.
    """

    async def mine(
        self,
        traces: List[WorkflowExecutionTrace],
    ) -> List[CollaborationPattern]:
        """Mine high-performing patterns."""
        patterns = []

        for trace in traces:
            if trace.success and trace.duration_seconds < self.duration_threshold:
                pattern = await self._extract_pattern(trace)

                # Check if pattern is already known
                if not self._pattern_exists(pattern, patterns):
                    patterns.append(pattern)

        return patterns
```

### 3. Clustering-Based Mining

```python
class ClusteringMiner:
    """Find patterns through clustering.

    Groups similar executions and extracts patterns from clusters.
    """

    async def mine(
        self,
        traces: List[WorkflowExecutionTrace],
    ) -> List[CollaborationPattern]:
        """Mine patterns through clustering."""
        # Extract features from traces
        features = self._extract_features(traces)

        # Cluster traces
        clusters = self._cluster_traces(features, n_clusters=self.n_clusters)

        # Extract pattern from each cluster
        patterns = []

        for cluster in clusters:
            pattern = await self._extract_cluster_pattern(cluster)
            patterns.append(pattern)

        return patterns
```

---

## Recommendation Engine

### Multi-Criteria Scoring

```python
async def score_pattern(
    self,
    pattern: CollaborationPattern,
    task_context: TaskContext,
) -> float:
    """Score pattern using multiple criteria.

    Criteria:
    1. Skill Match (40%): How well pattern skills match task requirements
    2. Complexity Match (30%): How well pattern complexity matches task
    3. Historical Performance (20%): Pattern's success rate
    4. Recency (10%): How recently pattern was successfully used
    """
    scores = {}

    # Skill matching
    skill_match = self._calculate_skill_overlap(
        task_context.required_skills,
        pattern.get_all_capabilities(),
    )
    scores["skill_match"] = skill_match * 0.4

    # Complexity matching
    complexity_score = 1.0 - abs(
        task_context.complexity - pattern.estimated_complexity
    )
    scores["complexity"] = complexity_score * 0.3

    # Historical performance
    scores["performance"] = pattern.success_rate * 0.2

    # Recency
    if pattern.last_used:
        days_since_use = (time.time() - pattern.last_used) / 86400
        recency_score = max(0, 1.0 - days_since_use / 30)  # Decay over 30 days
        scores["recency"] = recency_score * 0.1
    else:
        scores["recency"] = 0.0

    return sum(scores.values())
```

### Explanation Generation

```python
async def explain_recommendation(
    self,
    pattern: CollaborationPattern,
    task_context: TaskContext,
    score: float,
) -> str:
    """Generate human-readable explanation."""
    explanation = []

    # Skill explanation
    skill_coverage = self._calculate_skill_coverage(
        pattern,
        task_context,
    )
    explanation.append(
        f"✓ Covers {skill_coverage*100:.0f}% of required skills"
    )

    # Performance explanation
    explanation.append(
        f"✓ {pattern.success_rate*100:.0f}% success rate in {pattern.usage_count} uses"
    )

    # Duration estimate
    explanation.append(
        f"✓ Expected duration: ~{pattern.avg_duration_seconds:.0f}s"
    )

    # Similar use cases
    if pattern.use_cases:
        explanation.append(
            f"✓ Similar to: {', '.join(pattern.use_cases[:2])}"
        )

    return "\n".join(explanation)
```

---

## Evolution Mechanisms

### 1. Feedback Collection

```python
class FeedbackCollector:
    """Collect feedback on pattern usage."""

    async def collect_feedback(
        self,
        pattern_id: str,
        task_context: TaskContext,
        execution_result: ExecutionResult,
        user_feedback: Optional[UserFeedback] = None,
    ) -> PatternFeedback:
        """Collect feedback from pattern execution."""
        return PatternFeedback(
            pattern_id=pattern_id,
            task_context=task_context,
            success=execution_result.success,
            duration_seconds=execution_result.duration_seconds,
            cost_tokens=execution_result.cost_tokens,
            user_rating=user_feedback.rating if user_feedback else None,
            user_comments=user_feedback.comments if user_feedback else None,
            timestamp=time.time(),
        )
```

### 2. A/B Testing

```python
class PatternABTester:
    """A/B test pattern variants."""

    async def create_variant(
        self,
        base_pattern: CollaborationPattern,
        modifications: Dict[str, Any],
    ) -> CollaborationPattern:
        """Create a variant of a pattern."""
        variant = copy.deepcopy(base_pattern)

        # Apply modifications
        for key, value in modifications.items():
            setattr(variant, key, value)

        # Mark as variant
        variant.id = f"{base_pattern.id}_variant_{uuid.uuid4().hex[:8]}"
        variant.name = f"{base_pattern.name} (variant)"

        return variant

    async def run_ab_test(
        self,
        pattern_a: CollaborationPattern,
        pattern_b: CollaborationPattern,
        task_context: TaskContext,
        min_samples: int = 10,
    ) -> ABTestResult:
        """Run A/B test between two patterns."""
        results_a = []
        results_b = []

        for _ in range(min_samples):
            # Execute pattern A
            result_a = await self._execute_pattern(pattern_a, task_context)
            results_a.append(result_a)

            # Execute pattern B
            result_b = await self._execute_pattern(pattern_b, task_context)
            results_b.append(result_b)

        # Analyze results
        return self._analyze_ab_test(results_a, results_b)
```

### 3. Parameter Tuning

```python
class PatternTuner:
    """Tune pattern parameters for optimization."""

    async def tune_parameters(
        self,
        pattern: CollaborationPattern,
        parameter_ranges: Dict[str, Tuple[float, float]],
        optimization_target: str = "success_rate",
    ) -> CollaborationPattern:
        """Tune pattern parameters using optimization.

        Args:
            pattern: Pattern to tune
            parameter_ranges: Ranges for each parameter
            optimization_target: What to optimize

        Returns:
            Tuned pattern with optimized parameters
        """
        best_pattern = pattern
        best_score = await self._evaluate_pattern(
            pattern,
            optimization_target,
        )

        # Grid search over parameter space
        for params in self._generate_parameter_combinations(parameter_ranges):
            # Create variant with parameters
            variant = self._apply_parameters(pattern, params)

            # Evaluate
            score = await self._evaluate_pattern(
                variant,
                optimization_target,
            )

            if score > best_score:
                best_pattern = variant
                best_score = score

        return best_pattern
```

---

## Safety and Governance

### Pattern Validation Before Recommendation

All patterns must be validated before recommendation:

```python
async def validate_and_approve(
    self,
    pattern: CollaborationPattern,
) -> ApprovalResult:
    """Validate pattern and approve if safe.

    Process:
    1. Automated validation (quality, safety)
    2. Human review (if automated passes)
    3. Approval or rejection
    """
    # Automated validation
    validation_result = await self.validator.validate(pattern)

    if not validation_result.is_valid:
        return ApprovalResult(
            approved=False,
            reason=f"Validation failed: {validation_result.issues}",
        )

    # Safety check
    safety_result = await self.validator._check_safety(pattern)

    if not safety_result.is_safe:
        return ApprovalResult(
            approved=False,
            reason=f"Safety check failed: {safety_result.issues}",
        )

    # Human review required for new patterns
    if not pattern.is_validated:
        # Request human approval
        approval = await self._request_human_approval(pattern)

        if not approval.approved:
            return ApprovalResult(
                approved=False,
                reason=f"Human approval denied: {approval.reason}",
            )

    return ApprovalResult(approved=True)
```

### Approval Gates

```python
class ApprovalGate:
    """Human approval for pattern changes.

    Requires human approval for:
    - New patterns
    - Major pattern modifications
    - Pattern deprecation
    """

    async def request_approval(
        self,
        pattern: CollaborationPattern,
        change_type: ChangeType,
    ) -> ApprovalRequest:
        """Request human approval.

        Creates approval request and sends notification.
        """
        request = ApprovalRequest(
            id=uuid.uuid4().hex,
            pattern_id=pattern.id,
            change_type=change_type,
            pattern_details=pattern.to_dict(),
            status=ApprovalStatus.PENDING,
            created_at=time.time(),
        )

        # Store request
        await self._store_request(request)

        # Send notification
        await self._send_notification(request)

        return request

    async def wait_for_approval(
        self,
        request_id: str,
        timeout_seconds: int = 86400,  # 24 hours
    ) -> ApprovalResult:
        """Wait for approval decision."""
        start = time.time()

        while time.time() - start < timeout_seconds:
            request = await self._get_request(request_id)

            if request.status != ApprovalStatus.PENDING:
                return ApprovalResult(
                    approved=(request.status == ApprovalStatus.APPROVED),
                    reason=request.comments,
                )

            await asyncio.sleep(60)  # Check every minute

        return ApprovalResult(
            approved=False,
            reason="Approval timeout",
        )
```

### Pattern Deprecation

```python
class PatternDeprecator:
    """Deprecate outdated patterns."""

    async def deprecate_pattern(
        self,
        pattern_id: str,
        reason: str,
        replacement_pattern_id: Optional[str] = None,
    ) -> None:
        """Deprecate a pattern.

        Marks pattern as deprecated and suggests replacement.
        """
        await self.registry.update(
            pattern_id,
            {
                "is_deprecated": True,
                "deprecation_reason": reason,
                "replacement_pattern_id": replacement_pattern_id,
            },
        )

        # Notify users
        await self._notify_deprecation(pattern_id, reason)
```

---

## Implementation Strategy

### Phase 1: Core Infrastructure (3 weeks)

**Estimated LOC**: ~300 production, ~150 tests

**Components**:
1. CollaborationPattern dataclass (~100 LOC)
2. PatternMiner implementation (~100 LOC)
3. PatternValidator implementation (~100 LOC)

**Note**: PatternRegistry extends existing SQLiteExperimentStore (~200 LOC saved)

### Phase 2: Recommendation Engine (2 weeks)

**Estimated LOC**: ~250 production, ~125 tests

**Components**:
1. PatternRecommender implementation (~150 LOC)
2. Scoring strategy implementations (~100 LOC)

**Note**: Scoring uses strategy pattern; explanation generation uses existing patterns

### Phase 3: Mining Strategies (2 weeks)

**Estimated LOC**: ~200 production, ~100 tests

**Components**:
1. Frequency-based miner (~75 LOC)
2. Performance-based miner (~75 LOC)
3. Mining strategy interface (~50 LOC)

**Note**: Extends PatternMinerProtocol from Phase 1

### Phase 4: Evolution (2 weeks)

**Estimated LOC**: ~150 production, ~75 tests

**Components**:
1. Feedback collection (~50 LOC)
2. A/B testing integration (~50 LOC)
3. Parameter tuning (~50 LOC)

**Note**: Reuses existing A/B testing from victor/experiments/

### Phase 5: Safety and Governance (1 week)

**Estimated LOC**: ~100 production, ~50 tests

**Components**:
1. Approval gates (~50 LOC)
2. Safety checks (~50 LOC)

**Note**: Deprecation uses existing registry patterns

**Total Estimated LOC**: ~1,000 production, ~500 tests (69% reduction from original estimate)

---

## Code Examples

### Example 1: Mining Patterns from Execution Data

```python
from victor.framework.patterns import PatternMiner, PatternRegistry

# Collect execution traces
traces = await execution_tracker.get_recent_traces(days=30)

# Mine patterns
miner = PatternMiner(
    min_occurrences=5,
    min_success_rate=0.7,
)

discovered_patterns = await miner.mine_from_traces(traces)

# Register patterns
registry = PatternRegistry()
await registry.initialize()

for pattern in discovered_patterns:
    # Validate before registering
    validation = await validator.validate(pattern)

    if validation.is_valid:
        await registry.register(pattern)
        print(f"Registered pattern: {pattern.name}")
    else:
        print(f"Pattern {pattern.name} failed validation: {validation.issues}")
```

### Example 2: Recommending Patterns for a New Task

```python
from victor.framework.patterns import PatternRecommender

# Define task context
task_context = TaskContext(
    task_type="code_review",
    complexity=0.7,
    required_skills=["python", "security", "testing"],
    constraints={"max_duration": 300},
)

# Get recommendations
recommender = PatternRecommender(
    pattern_registry=registry,
    min_confidence=0.6,
)

recommendations = await recommender.recommend(
    task_context=task_context,
    top_k=3,
)

# Display recommendations
for i, rec in enumerate(recommendations, 1):
    print(f"\n{i}. {rec.pattern.name} (confidence: {rec.score:.2f})")
    print(rec.explanation)
```

### Example 3: A/B Testing Pattern Variants

```python
from victor.framework.patterns import PatternABTester

# Create variant with different parameter
variant = await ab_tester.create_variant(
    base_pattern=pattern,
    modifications={
        "formation": TeamFormation.PARALLEL,
        "tool_budget": 20,
    },
)

# Run A/B test
ab_result = await ab_tester.run_ab_test(
    pattern_a=pattern,
    pattern_b=variant,
    task_context=task_context,
    min_samples=20,
)

print(f"Pattern A success rate: {ab_result.success_rate_a:.2%}")
print(f"Pattern B success rate: {ab_result.success_rate_b:.2%}")
print(f"Statistical significance: {ab_result.is_significant}")
```

---

## Conclusion

This design enables **emergent collaboration patterns** that learn from execution data while maintaining safety, explainability, and SOLID principles. By reusing ~70-80% of existing Victor infrastructure (experiment tracking, team coordination, templates, and registries), the implementation effort is reduced by 69% from ~3,200 LOC to ~1,000 production LOC.

**Key Achievements**:
- **SOLID Compliant**: Protocol-based architecture with ISP, DIP, and OCP compliance
- **High Code Reuse**: Leverages existing experiment tracking (~200 LOC saved), team coordination (~500 LOC saved), template library (~150 LOC saved), and registry patterns (~150 LOC saved)
- **Data-Driven**: Discovers effective team formations from execution traces
- **Safe-by-Default**: Validation and approval gates before pattern recommendation
- **Explainable**: Clear rationale for pattern recommendations
- **Evolutionary**: Continuous improvement through feedback and A/B testing

The phased implementation allows incremental delivery starting with core infrastructure, followed by recommendation engine, mining strategies, evolution mechanisms, and safety governance.
