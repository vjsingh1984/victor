"""Shared domain types used across architectural layers.

This module is the canonical location for types that are needed by both the
framework and agent layers. Moving these here eliminates framework→agent
import violations while maintaining backward compatibility via re-exports
in the original modules.

Types here MUST:
- Be pure data (enums, dataclasses, protocols)
- Have no runtime dependencies on victor.agent.*
- Be importable without side effects

Types provided:
- ConversationStage: Enum for conversation lifecycle stages
- SubAgentRole: Enum for sub-agent role specialization
- QualityResult, DimensionScore, ResponseQualityDimension: Quality scoring types
- VerticalContextProtocol: Protocol for vertical context access
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    runtime_checkable,
)

if TYPE_CHECKING:
    from victor.core.verticals.protocols import (
        MiddlewareProtocol,
        ModeConfig,
        SafetyPattern,
        TaskTypeHint,
    )


# =============================================================================
# ConversationStage — canonical location (was victor.agent.conversation_state)
# =============================================================================


class ConversationStage(str, Enum):
    """Stages in a typical coding assistant conversation.

    This is the canonical source for conversation stages.
    Uses string values for serialization compatibility.
    """

    INITIAL = "initial"
    PLANNING = "planning"
    READING = "reading"
    ANALYSIS = "analysis"
    EXECUTION = "execution"
    VERIFICATION = "verification"
    COMPLETION = "completion"


class TaskPhase(str, Enum):
    """High-level phases of task execution for context management.

    Maps 7 ConversationStages to 4 broader TaskPhases for more efficient
    context management. Each phase has different context needs:

    EXPLORATION: Keep diverse file coverage (40-60% of conversation)
    PLANNING: Focus on task-relevant messages (10-20% of conversation)
    EXECUTION: Prioritize recent context with tool results (20-30% of conversation)
    REVIEW: Full context with comprehensive history (10-15% of conversation)

    This is the canonical source for task phases.
    Uses string values for serialization compatibility.
    """

    EXPLORATION = "exploration"
    PLANNING = "planning"
    EXECUTION = "execution"
    REVIEW = "review"


# Stage-to-phase mapping (7 stages → 4 phases)
STAGE_TO_PHASE_MAP: Dict[ConversationStage, TaskPhase] = {
    # EXPLORATION: INITIAL, READING, ANALYSIS (40-60% of conversation)
    ConversationStage.INITIAL: TaskPhase.EXPLORATION,
    ConversationStage.READING: TaskPhase.EXPLORATION,
    ConversationStage.ANALYSIS: TaskPhase.EXPLORATION,
    # PLANNING: PLANNING (10-20% of conversation)
    ConversationStage.PLANNING: TaskPhase.PLANNING,
    # EXECUTION: EXECUTION (20-30% of conversation)
    ConversationStage.EXECUTION: TaskPhase.EXECUTION,
    # REVIEW: VERIFICATION, COMPLETION (10-15% of conversation)
    ConversationStage.VERIFICATION: TaskPhase.REVIEW,
    ConversationStage.COMPLETION: TaskPhase.REVIEW,
}


# =============================================================================
# SubAgentRole — canonical location (was victor.agent.subagents.base)
# =============================================================================


class SubAgentRole(Enum):
    """Role specialization for sub-agents.

    Each role has specific capabilities and constraints:
    - RESEARCHER: Read-only exploration
    - PLANNER: Task breakdown and planning
    - EXECUTOR: Code changes and execution
    - REVIEWER: Quality checks and testing
    - TESTER: Test writing and running
    """

    RESEARCHER = "researcher"
    PLANNER = "planner"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"
    TESTER = "tester"


# =============================================================================
# Quality scoring types — canonical location (was victor.agent.response_quality)
# =============================================================================


class ResponseQualityDimension(str, Enum):
    """Dimensions for scoring LLM response quality."""

    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONCISENESS = "conciseness"
    ACTIONABILITY = "actionability"
    COHERENCE = "coherence"
    CODE_QUALITY = "code_quality"


@dataclass
class DimensionScore:
    """Score for a single quality dimension."""

    dimension: ResponseQualityDimension
    score: float
    weight: float = 1.0
    feedback: str = ""
    evidence: List[str] = field(default_factory=list)


@dataclass
class QualityResult:
    """Result of quality scoring."""

    overall_score: float
    dimension_scores: List[DimensionScore]
    passes_threshold: bool
    improvement_suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_dimension_score(self, dimension: ResponseQualityDimension) -> Optional[float]:
        """Get score for a specific dimension."""
        for ds in self.dimension_scores:
            if ds.dimension == dimension:
                return ds.score
        return None

    def get_weakest_dimensions(self, n: int = 3) -> List[DimensionScore]:
        """Get the n weakest scoring dimensions."""
        sorted_dims = sorted(self.dimension_scores, key=lambda x: x.score)
        return sorted_dims[:n]


# =============================================================================
# VerticalContextProtocol — canonical location (was victor.agent.vertical_context)
# =============================================================================


@runtime_checkable
class VerticalContextProtocol(Protocol):
    """Protocol for accessing vertical context (read-only interface)."""

    @property
    def vertical_name(self) -> Optional[str]: ...

    @property
    def has_vertical(self) -> bool: ...

    @property
    def middleware(self) -> List["MiddlewareProtocol"]: ...

    @property
    def safety_patterns(self) -> List["SafetyPattern"]: ...

    @property
    def task_hints(self) -> Dict[str, "TaskTypeHint"]: ...

    @property
    def mode_configs(self) -> Dict[str, "ModeConfig"]: ...

    @property
    def enabled_tools(self) -> Set[str]: ...


@runtime_checkable
class MutableVerticalContextProtocol(VerticalContextProtocol, Protocol):
    """Protocol for applying vertical configuration to context state.

    This is the framework-facing contract for vertical integration handlers.
    The concrete ``VerticalContext`` implementation remains a runtime state
    container, but setup code should depend on this protocol.
    """

    def apply_vertical(self, name: str, config: Optional[Any] = None) -> None: ...

    def apply_stages(self, stages: Dict[str, Any]) -> None: ...

    def apply_middleware(self, middleware: List[Any]) -> None: ...

    def apply_safety_patterns(self, patterns: List[Any]) -> None: ...

    def apply_task_hints(self, hints: Dict[str, Any]) -> None: ...

    def apply_mode_configs(
        self,
        configs: Dict[str, Any],
        default_mode: str = "default",
        default_budget: int = 10,
    ) -> None: ...

    def apply_tool_dependencies(
        self,
        dependencies: List[Any],
        sequences: List[List[str]],
    ) -> None: ...

    def apply_system_prompt(self, prompt: str) -> None: ...

    def apply_enabled_tools(self, tools: Set[str]) -> None: ...

    def apply_tiered_config(self, config: Any) -> None: ...

    def add_prompt_section(self, section: str) -> None: ...

    def apply_rl_config(self, config: Any) -> None: ...

    def apply_rl_hooks(self, hooks: List[Any]) -> None: ...

    def apply_team_specs(self, specs: List[Any]) -> None: ...

    def apply_workflows(self, workflows: List[Any]) -> None: ...

    def apply_tool_selection_strategy(self, strategy: Any) -> None: ...

    def apply_enrichment_strategy(self, strategy: Any) -> None: ...

    def set_capability_config(self, name: str, config: Any) -> None: ...

    def get_capability_config(self, name: str, default: Any = None) -> Any: ...

    def apply_capability_configs(self, configs: Dict[str, Any]) -> None: ...


# =============================================================================
# SubAgentContext — canonical location (was victor.agent.subagents.protocols)
# =============================================================================


@runtime_checkable
class SubAgentContext(Protocol):
    """Minimal context required by SubAgent (ISP-compliant interface)."""

    @property
    def settings(self) -> Any: ...

    @property
    def provider(self) -> Any: ...

    @property
    def provider_name(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def tool_registry(self) -> Any: ...

    @property
    def temperature(self) -> float: ...

    @property
    def vertical_context(self) -> Any: ...
