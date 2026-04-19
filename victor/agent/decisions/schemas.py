"""Pydantic models for structured LLM decision output.

Each schema is designed for minimal token usage (5-20 tokens) to keep
LLM decision calls fast and cheap. Fields use constrained types to
enable strict validation of LLM responses.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class DecisionType(str, Enum):
    """Types of decisions the LLM decision service can make."""

    TASK_COMPLETION = "task_completion"
    INTENT_CLASSIFICATION = "intent_classification"
    TASK_TYPE_CLASSIFICATION = "task_type_classification"
    QUESTION_CLASSIFICATION = "question_classification"
    LOOP_DETECTION = "loop_detection"
    ERROR_CLASSIFICATION = "error_classification"
    CONTINUATION_ACTION = "continuation_action"
    TOOL_SELECTION = "tool_selection"
    PROMPT_FOCUS = "prompt_focus"
    STAGE_DETECTION = "stage_detection"
    SKILL_SELECTION = "skill_selection"
    MULTI_SKILL_DECOMPOSITION = "multi_skill_decomposition"
    TOOL_NECESSITY = "tool_necessity"
    COMPACTION = "compaction"


class TaskCompletionDecision(BaseModel):
    """Is the task done?"""

    is_complete: bool = Field(description="Whether the task appears complete")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the assessment")
    phase: Literal["working", "finalizing", "done", "stuck"] = Field(
        description="Current phase of task execution"
    )


class IntentDecision(BaseModel):
    """What is the model doing?"""

    intent: Literal["continuation", "completion", "asking_input", "stuck_loop"] = Field(
        description="Classified intent of the model's response"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the classification")


class TaskTypeDecision(BaseModel):
    """What kind of task is this, and what deliverables are expected?"""

    task_type: Literal["analysis", "action", "generation", "search", "edit"] = Field(
        description="Classified type of the task"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the classification")
    deliverables: list[
        Literal[
            "file_created",
            "file_modified",
            "analysis_provided",
            "answer_provided",
            "plan_provided",
            "code_executed",
        ]
    ] = Field(
        default_factory=list,
        description="Expected deliverable types (empty = infer from task_type)",
    )


class QuestionTypeDecision(BaseModel):
    """Should we auto-continue past this question?"""

    question_type: Literal["rhetorical", "continuation", "clarification", "info"] = Field(
        description="Type of question being asked"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the classification")


class LoopDetection(BaseModel):
    """Is the model stuck in a loop?"""

    is_loop: bool = Field(description="Whether a loop is detected")
    loop_type: Literal["stalling", "circular", "repetition", "none"] = Field(
        description="Type of loop detected"
    )


class ErrorClassDecision(BaseModel):
    """Can we retry this error?"""

    error_type: Literal["permanent", "transient", "retryable"] = Field(
        description="Classification of the error"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the classification")


class ToolSelectionDecision(BaseModel):
    """Which tools are most relevant for this task?"""

    tools: list[str] = Field(description="Selected tool names")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in selection")


class PromptFocusDecision(BaseModel):
    """Which system prompt sections to include?"""

    sections: list[str] = Field(description="Selected section names")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in selection")


class StageDetectionDecision(BaseModel):
    """What conversation stage is this?"""

    stage: Literal["initial", "planning", "reading", "analysis", "execution", "verification"] = (
        Field(description="Detected conversation stage")
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in detection")


class ToolNecessityDecision(BaseModel):
    """Does this request require tools?"""

    requires_tools: bool = Field(description="Whether tools/file operations are needed")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in assessment")


class ContinuationDecision(BaseModel):
    """What action should we take next?"""

    action: Literal["finish", "prompt_tool_call", "request_summary", "return_to_user"] = Field(
        description="Recommended next action"
    )
    reason: str = Field(max_length=100, description="Brief reason for the recommendation")


class SkillSelectionDecision(BaseModel):
    """Which skill best matches this user request?"""

    skill: str = Field(description="Selected skill name, or empty string if none match")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in selection")


class MultiSkillDecision(BaseModel):
    """How to decompose a complex request into multiple skills?"""

    skills: list[str] = Field(description="Ordered list of skill names to execute")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in decomposition")


class CompactionDecision(BaseModel):
    """Decision for context compaction routing.

    Determines which tier should handle compaction based on complexity
    and estimated token count. Simple compaction (≤8 messages) can use
    edge tier for speed, while complex compaction (>8 messages) should
    use performance tier for quality.

    Attributes:
        complexity: simple (≤8 messages) or complex (>8 messages)
        recommended_tier: Which tier should handle compaction (edge, balanced, performance)
        estimated_tokens: Estimated size of content to compact
        confidence: Decision confidence (0.0-1.0)
        reason: Brief explanation of the decision
    """

    complexity: Literal["simple", "complex"] = Field(
        description="simple (≤8 messages) or complex (>8 messages)"
    )
    recommended_tier: Literal["edge", "balanced", "performance"] = Field(
        description="Recommended tier for compaction"
    )
    estimated_tokens: int = Field(description="Estimated token count of content to compact")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the decision")
    reason: str = Field(max_length=200, description="Brief explanation of the decision")


class SystemPromptOptimizationDecision(BaseModel):
    """Decision for system prompt optimization during compaction.

    When system_prompt_strategy is 'dynamic', this decision determines
    which prompt sections to include based on current context and task state.

    Attributes:
        include_sections: Which prompt sections to include (subset of all sections)
        add_context_reminder: Whether to add compaction summary reminder
        add_failure_hints: Whether to include recent failure patterns
        adjust_for_complexity: Tailor prompt for task complexity
        confidence: Decision confidence (0.0-1.0)
        reason: Brief explanation of the decision
    """

    include_sections: list[str] = Field(
        description="List of section names to include in the prompt"
    )
    add_context_reminder: bool = Field(
        description="Whether to add a reminder about recent compaction"
    )
    add_failure_hints: bool = Field(description="Whether to include hints based on recent failures")
    adjust_for_complexity: bool = Field(
        description="Whether to tailor prompt based on task complexity"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the decision")
    reason: str = Field(max_length=200, description="Brief explanation of the decision")
