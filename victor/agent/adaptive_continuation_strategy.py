# Copyright 2025 Vijaykumar Singh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Adaptive continuation strategy selection (P2-1).

Addresses the DeepSeek Chat issue where simple "continue" prompts after
compaction cause the model to stop. Different models need different
continuation strategies.

Key insights:
- DeepSeek Chat: Needs task re-statement (TASK_SUMMARY)
- Claude/GPT: Work fine with simple "continue" (DIRECT_CONTINUE)
- High compaction: Needs context reminder (CONTEXT_REMINDER)
- Late turns: Fresh start is better (FRESH_START)

Based on empirical testing and compaction analysis.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ContinuationStrategy(Enum):
    """Continuation strategies for post-compaction prompts.

    Different strategies work better for different models and situations.
    """

    DIRECT_CONTINUE = "direct_continue"  # Simple "continue" prompt
    TASK_SUMMARY = "task_summary"  # Re-state task before continuing
    CONTEXT_REMINDER = "context_reminder"  # Remind what was compacted
    FRESH_START = "fresh_start"  # Start new turn without continuation hint
    SILENT = "silent"  # No continuation prompt


@dataclass
class StrategySelectionContext:
    """Context for strategy selection.

    Attributes:
        provider: LLM provider name (anthropic, openai, deepseek, etc.)
        model: Model name
        compaction_occurred: Whether context was compacted
        messages_removed: Number of messages removed during compaction
        current_turn: Current turn number
        task_description: Optional task description for context
    """

    provider: str
    model: str
    compaction_occurred: bool
    messages_removed: int
    current_turn: int
    task_description: str = ""


class StrategySelector:
    """Selects appropriate continuation strategy based on context.

    Uses model-specific knowledge and context state to choose the best
    continuation strategy.

    Example:
        selector = StrategySelector()
        strategy = selector.select_strategy(
            provider="deepseek",
            model="deepseek-chat",
            compaction_occurred=True,
            messages_removed=20,
            current_turn=5,
        )
        prompt = get_strategy_prompt(strategy, task_description="Implement feature X")
    """

    # Models that need special handling
    TASK_SUMMARY_MODELS = {
        "deepseek-chat",
        "deepseek-coder",
        "deepseek",
    }

    # Thresholds for strategy selection
    HIGH_COMPACTION_THRESHOLD = 50  # messages
    LATE_TURN_THRESHOLD = 20  # turn number

    def __init__(
        self,
        default_strategy: ContinuationStrategy = ContinuationStrategy.DIRECT_CONTINUE,
        force_default: bool = False,
    ):
        """Initialize the strategy selector.

        Args:
            default_strategy: Default strategy if no specific rule applies
            force_default: If True, always use default_strategy (overrides all logic)
        """
        self.default_strategy = default_strategy
        self.force_default = force_default

    def select_strategy(
        self,
        provider: Optional[str],
        model: Optional[str],
        compaction_occurred: bool,
        messages_removed: int = 0,
        current_turn: int = 0,
    ) -> ContinuationStrategy:
        """Select the appropriate continuation strategy.

        Args:
            provider: LLM provider name
            model: Model name
            compaction_occurred: Whether context was compacted
            messages_removed: Number of messages removed during compaction
            current_turn: Current turn number

        Returns:
            Selected continuation strategy
        """
        # If force_default is set, always use the default strategy
        if self.force_default:
            return self.default_strategy

        # No compaction = no continuation needed
        if not compaction_occurred:
            return ContinuationStrategy.SILENT

        # Late turns = fresh start is better
        if current_turn >= self.LATE_TURN_THRESHOLD:
            return ContinuationStrategy.FRESH_START

        # Normalize inputs
        provider_name = (provider or "").lower()
        model_name = (model or "").lower()

        # DeepSeek models need task summary strategy
        if model_name in self.TASK_SUMMARY_MODELS or provider_name == "deepseek":
            if messages_removed >= self.HIGH_COMPACTION_THRESHOLD:
                return ContinuationStrategy.CONTEXT_REMINDER
            return ContinuationStrategy.TASK_SUMMARY

        # Cloud models (Claude, GPT) work fine with direct continue
        if provider_name in {"anthropic", "openai", "google"}:
            return ContinuationStrategy.DIRECT_CONTINUE

        # Local models also work with direct continue
        if provider_name in {"ollama", "lmstudio", "llama.cpp", "mlx", "vllm"}:
            return ContinuationStrategy.DIRECT_CONTINUE

        # Default fallback
        return self.default_strategy

    def get_strategy_for_context(self, context: StrategySelectionContext) -> ContinuationStrategy:
        """Get strategy for a selection context.

        Args:
            context: Strategy selection context

        Returns:
            Selected continuation strategy
        """
        return self.select_strategy(
            provider=context.provider,
            model=context.model,
            compaction_occurred=context.compaction_occurred,
            messages_removed=context.messages_removed,
            current_turn=context.current_turn,
        )


def get_strategy_prompt(
    strategy: ContinuationStrategy,
    task_description: str = "",
    compaction_summary: str = "",
) -> str:
    """Get the prompt text for a continuation strategy.

    Args:
        strategy: The continuation strategy
        task_description: Optional task description for context
        compaction_summary: Optional summary of what was compacted

    Returns:
        Prompt text to inject (empty for SILENT strategy)
    """
    if strategy == ContinuationStrategy.SILENT:
        return ""

    if strategy == ContinuationStrategy.DIRECT_CONTINUE:
        return "Continue."

    if strategy == ContinuationStrategy.TASK_SUMMARY:
        if task_description:
            return f"Continue with: {task_description}"
        return "Continue with what you were doing."

    if strategy == ContinuationStrategy.CONTEXT_REMINDER:
        parts = ["Context was compacted to continue."]
        if task_description:
            parts.append(f"You were working on: {task_description}")
        if compaction_summary:
            parts.append(f"Previous: {compaction_summary[:100]}...")
        return " ".join(parts) + " Continue."

    if strategy == ContinuationStrategy.FRESH_START:
        # Fresh start - minimal prompt, let model drive
        if task_description:
            return f"Continuing session. Task: {task_description}"
        return "Continuing session."

    # Fallback
    return "Continue."


def select_strategy_for_context(
    provider: Optional[str],
    model: Optional[str],
    compaction_occurred: bool,
    messages_removed: int = 0,
    current_turn: int = 0,
) -> ContinuationStrategy:
    """Convenience function for strategy selection.

    Args:
        provider: LLM provider name
        model: Model name
        compaction_occurred: Whether context was compacted
        messages_removed: Number of messages removed during compaction
        current_turn: Current turn number

    Returns:
        Selected continuation strategy
    """
    selector = StrategySelector()
    return selector.select_strategy(
        provider=provider,
        model=model,
        compaction_occurred=compaction_occurred,
        messages_removed=messages_removed,
        current_turn=current_turn,
    )


def get_continuation_prompt(
    provider: Optional[str],
    model: Optional[str],
    compaction_occurred: bool,
    messages_removed: int = 0,
    current_turn: int = 0,
    task_description: str = "",
    compaction_summary: str = "",
) -> str:
    """Get the continuation prompt for a given context.

    This is the main entry point for getting adaptive continuation prompts.

    Args:
        provider: LLM provider name
        model: Model name
        compaction_occurred: Whether context was compacted
        messages_removed: Number of messages removed during compaction
        current_turn: Current turn number
        task_description: Optional task description for context
        compaction_summary: Optional summary of what was compacted

    Returns:
        Continuation prompt to inject (may be empty)
    """
    strategy = select_strategy_for_context(
        provider=provider,
        model=model,
        compaction_occurred=compaction_occurred,
        messages_removed=messages_removed,
        current_turn=current_turn,
    )

    return get_strategy_prompt(
        strategy=strategy,
        task_description=task_description,
        compaction_summary=compaction_summary,
    )


# Singleton instance for convenience
_default_selector: Optional[StrategySelector] = None


def get_strategy_selector() -> StrategySelector:
    """Get the singleton strategy selector instance.

    Returns:
        The default StrategySelector instance
    """
    global _default_selector
    if _default_selector is None:
        _default_selector = StrategySelector()
    return _default_selector
