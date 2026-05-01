# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AgenticLoopState - State model for StateGraph-based agentic loop.

This module provides the Pydantic state model for the agentic loop execution
in StateGraph. The state tracks all information needed for the PERCEIVE-PLAN-
ACT-EVALUATE-DECIDE loop.

Design Principles:
- Pydantic v2 for runtime validation and serialization
- Dict-like interface for StateGraph compatibility
- Immutable via model_copy() for functional updates
- Checkpointable via model_dump() / model_validate()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict, Union

from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator

from victor.framework.evaluation_nodes import EvaluationDecision


class AgenticLoopState(TypedDict, total=False):
    """TypedDict type alias for agentic loop state.

    This provides backward compatibility with code expecting dict-based state.
    For new code, prefer AgenticLoopStateModel for better validation.

    Attributes:
        query: User's input query or task
        iteration: Current loop iteration number (0-indexed)
        max_iterations: Maximum iterations before forced termination
        stage: Current stage (perceive, plan, act, evaluate, decide)
        perception: Result from PERCEIVE stage (intent, complexity, etc.)
        task_type: Classification of task type
        complexity: Task complexity level
        plan: Execution plan from PLAN stage
        action_result: Result from ACT stage execution
        tool_results: Individual tool execution results
        evaluation: Result from EVALUATE stage (decision, score)
        progress_scores: History of progress scores across iterations
        fulfillment: Optional fulfillment check results
        context: Additional execution context
        conversation_history: Conversation messages
        _execution_context: Internal ExecutionContext for service injection
    """

    # Core loop state
    query: str
    iteration: int
    max_iterations: int

    # Stage tracking
    stage: str

    # Perception results
    perception: Optional[Dict[str, Any]]
    task_type: str
    complexity: str

    # Planning
    plan: Optional[Dict[str, Any]]

    # Execution results
    action_result: Optional[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]

    # Evaluation
    evaluation: Optional[Dict[str, Any]]
    progress_scores: List[float]

    # Fulfillment
    fulfillment: Optional[Dict[str, Any]]

    # Metadata
    context: Optional[Dict[str, Any]]
    conversation_history: Optional[List[Dict[str, Any]]]

    # Internal ExecutionContext reference (not serialized)
    _execution_context: Optional[Any]


class AgenticLoopStateModel(BaseModel):
    """Pydantic model for agentic loop state.

    This is the RECOMMENDED way to manage state in the agentic loop graph.
    Provides validation, serialization, and dict-like interface for StateGraph
    compatibility.

    Example:
        state = AgenticLoopStateModel(query="Fix the bug")
        state = state.model_copy(update={"iteration": 1, "stage": "perceive"})
        assert state["query"] == "Fix the bug"  # Dict-like access

    Attributes:
        query: User's input query or task (required)
        iteration: Current loop iteration (default: 0)
        max_iterations: Maximum iterations (default: 10)
        stage: Current stage name (optional)
        perception: Perception result dict (optional)
        task_type: Task type classification (optional)
        complexity: Complexity level (optional)
        plan: Execution plan dict (optional)
        action_result: Action execution result (optional)
        tool_results: List of tool results (default: [])
        evaluation: Evaluation result dict (optional)
        progress_scores: Progress score history (default: [])
        fulfillment: Fulfillment check result (optional)
        context: Additional context dict (optional)
        conversation_history: Conversation messages (optional)
        _execution_context: Internal ExecutionContext (excluded from serialization)
    """

    # Core loop state
    query: str
    iteration: int = Field(default=0, ge=0)
    max_iterations: int = Field(default=10, ge=1, le=100)

    # Stage tracking
    stage: Optional[str] = Field(default=None)

    # Perception results
    perception: Optional[Dict[str, Any]] = Field(default=None)
    task_type: Optional[str] = Field(default=None)
    complexity: Optional[str] = Field(default=None)

    # Planning
    plan: Optional[Dict[str, Any]] = Field(default=None)

    # Execution results
    action_result: Optional[Dict[str, Any]] = Field(default=None)
    tool_results: List[Dict[str, Any]] = Field(default_factory=list)

    # Evaluation
    evaluation: Optional[Dict[str, Any]] = Field(default=None)
    progress_scores: List[float] = Field(default_factory=list)

    # Fulfillment
    fulfillment: Optional[Dict[str, Any]] = Field(default=None)

    # Metadata
    context: Optional[Dict[str, Any]] = Field(default=None)
    conversation_history: Optional[List[Dict[str, Any]]] = Field(default=None)

    # Internal ExecutionContext (private attribute, excluded from serialization)
    _execution_context: Optional[Any] = PrivateAttr(default=None)

    model_config = {"arbitrary_types_allowed": True, "validate_assignment": True}

    @field_validator("max_iterations")
    @classmethod
    def validate_max_iterations(cls, v: int) -> int:
        """Validate max_iterations is positive."""
        if v < 1:
            raise ValueError("max_iterations must be at least 1")
        if v > 100:
            raise ValueError("max_iterations cannot exceed 100")
        return v

    @field_validator("progress_scores")
    @classmethod
    def validate_progress_scores(cls, v: List[float]) -> List[float]:
        """Validate progress scores are in valid range."""
        for score in v:
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"progress score {score} not in [0.0, 1.0]")
        return v

    # ==========================================================================
    # Dict-like interface for StateGraph compatibility
    # ==========================================================================

    def __getitem__(self, key: str) -> Any:
        """Get item by key (dict-like access)."""
        return getattr(self, key, None)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set item by key (dict-like mutation)."""
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """Check if key exists (dict-like membership)."""
        return hasattr(self, key) and getattr(self, key) is not None

    def get(self, key: str, default: Any = None) -> Any:
        """Get item with default (dict-like)."""
        return getattr(self, key, default)

    def keys(self) -> List[str]:
        """Return list of keys (dict-like)."""
        return [
            "query",
            "iteration",
            "max_iterations",
            "stage",
            "perception",
            "task_type",
            "complexity",
            "plan",
            "action_result",
            "tool_results",
            "evaluation",
            "progress_scores",
            "fulfillment",
            "context",
            "conversation_history",
        ]

    def values(self) -> List[Any]:
        """Return list of values (dict-like)."""
        return [getattr(self, k) for k in self.keys()]

    def items(self) -> List[tuple[str, Any]]:
        """Return list of key-value tuples (dict-like)."""
        return [(k, getattr(self, k)) for k in self.keys()]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dict (excluding internal fields)."""
        return {
            "query": self.query,
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "stage": self.stage,
            "perception": self.perception,
            "task_type": self.task_type,
            "complexity": self.complexity,
            "plan": self.plan,
            "action_result": self.action_result,
            "tool_results": self.tool_results,
            "evaluation": self.evaluation,
            "progress_scores": self.progress_scores,
            "fulfillment": self.fulfillment,
            "context": self.context,
            "conversation_history": self.conversation_history,
        }


# =============================================================================
# Helper Functions
# =============================================================================


def create_initial_state(
    query: str,
    context: Optional[Dict[str, Any]] = None,
    max_iterations: int = 10,
) -> AgenticLoopStateModel:
    """Create initial state for agentic loop execution.

    Args:
        query: User's input query or task
        context: Optional additional context
        max_iterations: Maximum iterations (default: 10)

    Returns:
        Initial AgenticLoopStateModel ready for execution
    """
    return AgenticLoopStateModel(
        query=query,
        iteration=0,
        max_iterations=max_iterations,
        context=context or {},
        progress_scores=[],
        tool_results=[],
    )


def should_continue_loop(state: AgenticLoopStateModel) -> bool:
    """Check if agentic loop should continue iterating.

    Args:
        state: Current agentic loop state

    Returns:
        True if loop should continue, False otherwise
    """
    # Check max iterations
    if state.iteration >= state.max_iterations:
        return False

    # Check evaluation decision (handle both string and enum)
    if state.evaluation:
        decision = state.evaluation.get("decision")
        # Normalize to string for comparison
        if isinstance(decision, EvaluationDecision):
            decision_str = decision.value
        else:
            decision_str = str(decision).lower()

        if decision_str == EvaluationDecision.COMPLETE.value or decision_str == "complete":
            return False
        if decision_str == EvaluationDecision.FAIL.value or decision_str == "fail":
            return False

    # Default: continue
    return True


def get_evaluation_decision(state: AgenticLoopStateModel) -> str:
    """Get the evaluation decision from state.

    Args:
        state: Current agentic loop state

    Returns:
        Evaluation decision string (default: "continue")
    """
    if state.evaluation:
        return state.evaluation.get("decision", "continue")
    return "continue"
