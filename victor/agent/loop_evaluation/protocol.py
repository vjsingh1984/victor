"""Shared protocol types for the hybrid loop termination evaluator.

The loop evaluator decides, after each model response without tool calls,
whether to finish the turn, prompt for more tool usage, request a summary,
or return to the user.  Two backends implement this protocol:

- ``LegacyEvaluator``: thin wrapper around ``ContinuationStrategy`` (current).
- ``AgenticLoopEvaluator``: PERCEIVE→EVALUATE→DECIDE path using
  ``FulfillmentDetector`` + ``progress_tracking_evaluator`` (future default).

``HybridLoopEvaluator`` routes between them based on the
``USE_STATEGRAPH_AGENTIC_LOOP`` feature flag so either path can be activated
at runtime without code changes.

Migration path
--------------
1. Today: flag OFF → ``LegacyEvaluator`` (zero behavior change).
2. Near-term: flag ON → ``AgenticLoopEvaluator`` (opt-in validation).
3. Eventually: remove flag, only ``AgenticLoopEvaluator`` remains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from victor.agent.continuation_contract import (
        ContinuationActionType,
        ContinuationStatePatch,
    )
    from victor.agent.tool_call_extractor import ExtractedToolCall


@dataclass
class LoopContext:
    """Unified input for loop termination evaluation.

    Contains everything both the legacy ``ContinuationStrategy`` backend and
    the ``AgenticLoopEvaluator`` backend need to make a termination decision.
    """

    # ---- Task characterization ------------------------------------------------
    user_message: str
    task_type: str              # "general" | "analysis" | "action"
    is_analysis_task: bool
    is_action_task: bool
    is_direct_response: bool    # from classify_direct_response_prompt

    # ---- This turn's response -------------------------------------------------
    full_content: str
    content_length: int
    mentioned_tools: List[str]
    intent_result: Any          # IntentClassificationResult from IntentClassifier

    # ---- Loop / budget state --------------------------------------------------
    iteration: int
    continuation_prompts: int
    asking_input_prompts: int
    max_prompts_summary_requested: bool
    force_tool_execution_attempts: int
    synthesis_nudge_count: int

    # ---- Quality / progress signals (AgenticLoopEvaluator) --------------------
    quality_score: float = 0.0          # 0.0-1.0 from runtime intelligence

    # ---- Task completion signals (already computed by IntentClassification) ---
    task_completion_signals: Dict[str, Any] = field(default_factory=dict)

    # ---- Config / runtime -----------------------------------------------------
    one_shot_mode: bool = False
    compaction_occurred: bool = False
    compaction_messages_removed: int = 0
    degraded_resume_state: bool = False
    resume_summary: str = ""
    settings: Any = None
    rl_coordinator: Any = None
    provider_name: str = ""
    model: str = ""
    tool_budget: int = 20
    unified_tracker_config: Dict[str, Any] = field(default_factory=dict)
    plan_step_count: Optional[int] = None
    query_classification: Any = None
    runtime_intelligence: Optional[Any] = None


@dataclass
class LoopDecision:
    """Unified output from loop termination evaluation.

    Downstream callers convert this to a ``ContinuationDirective`` via
    ``LoopDecision.to_directive()``.
    """

    action: "ContinuationActionType"
    reason: str
    confidence: float = 1.0
    message: Optional[str] = None

    # Source label for logging / observability.
    source: str = "legacy"          # "legacy" | "agentic_loop"

    # State updates to apply to the orchestrator.
    state_patch: Optional["ContinuationStatePatch"] = None

    # Passthrough fields for legacy handlers.
    extracted_call: Optional["ExtractedToolCall"] = None
    mentioned_tools_override: Optional[List[str]] = None
    set_final_summary_requested: bool = False
    set_max_prompts_summary_requested: bool = False

    def to_directive(self) -> "ContinuationDirective":
        """Convert to the ``ContinuationDirective`` expected by downstream handlers."""

        from victor.agent.continuation_contract import (
            ContinuationDirective,
            ContinuationStatePatch,
        )

        patch = self.state_patch or ContinuationStatePatch()
        # Fold flag fields into the patch.
        if self.set_final_summary_requested:
            patch = ContinuationStatePatch(
                continuation_prompts=patch.continuation_prompts,
                asking_input_prompts=patch.asking_input_prompts,
                synthesis_nudge_count=patch.synthesis_nudge_count,
                cumulative_prompt_interventions=patch.cumulative_prompt_interventions,
                final_summary_requested=True,
                max_prompts_summary_requested=patch.max_prompts_summary_requested,
            )
        if self.set_max_prompts_summary_requested:
            patch = ContinuationStatePatch(
                continuation_prompts=patch.continuation_prompts,
                asking_input_prompts=patch.asking_input_prompts,
                synthesis_nudge_count=patch.synthesis_nudge_count,
                cumulative_prompt_interventions=patch.cumulative_prompt_interventions,
                final_summary_requested=patch.final_summary_requested,
                max_prompts_summary_requested=True,
            )

        return ContinuationDirective(
            action=self.action,
            reason=self.reason,
            message=self.message,
            state_patch=patch,
            extracted_call=self.extracted_call,
            mentioned_tools=self.mentioned_tools_override,
        )


class LoopEvaluator:
    """Base class / informal protocol for loop termination evaluators.

    Subclasses implement ``evaluate()`` and return a ``LoopDecision``.
    Using a concrete base class (not ``typing.Protocol``) so that
    ``isinstance`` checks work without import-time overhead.
    """

    def evaluate(self, ctx: LoopContext) -> LoopDecision:
        raise NotImplementedError
