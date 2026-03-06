"""Prompt templates for LLM decision calls.

Each decision type maps to a focused system prompt and user template
designed to elicit a minimal JSON response (5-20 tokens).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Type

from pydantic import BaseModel

from victor.agent.decisions.schemas import (
    ContinuationDecision,
    DecisionType,
    ErrorClassDecision,
    IntentDecision,
    LoopDetection,
    QuestionTypeDecision,
    TaskCompletionDecision,
    TaskTypeDecision,
)


@dataclass(frozen=True)
class DecisionPrompt:
    """Configuration for a single decision type's LLM call."""

    system: str
    user_template: str
    schema: Type[BaseModel]
    max_tokens: int


DECISION_PROMPTS: Dict[DecisionType, DecisionPrompt] = {
    DecisionType.TASK_COMPLETION: DecisionPrompt(
        system=(
            "You are a task completion classifier. "
            "Respond ONLY with a JSON object, no other text."
        ),
        user_template=(
            "Given this AI assistant response tail and context, is the task complete?\n\n"
            "Response tail:\n{response_tail}\n\n"
            "Deliverables found: {deliverable_count}\n"
            "Completion signals: {signal_count}\n\n"
            'Respond with JSON: {{"is_complete": bool, "confidence": 0.0-1.0, '
            '"phase": "working"|"finalizing"|"done"|"stuck"}}'
        ),
        schema=TaskCompletionDecision,
        max_tokens=30,
    ),
    DecisionType.INTENT_CLASSIFICATION: DecisionPrompt(
        system=(
            "You are an intent classifier for AI assistant responses. "
            "Respond ONLY with a JSON object, no other text."
        ),
        user_template=(
            "Classify the intent of this AI response:\n\n"
            "Response tail:\n{text_tail}\n\n"
            "Has tool calls: {has_tool_calls}\n\n"
            'Respond with JSON: {{"intent": '
            '"continuation"|"completion"|"asking_input"|"stuck_loop", '
            '"confidence": 0.0-1.0}}'
        ),
        schema=IntentDecision,
        max_tokens=20,
    ),
    DecisionType.TASK_TYPE_CLASSIFICATION: DecisionPrompt(
        system=(
            "You are a task type classifier. " "Respond ONLY with a JSON object, no other text."
        ),
        user_template=(
            "Classify this task message:\n\n"
            "{message_excerpt}\n\n"
            'Respond with JSON: {{"task_type": '
            '"analysis"|"action"|"generation"|"search"|"edit", '
            '"confidence": 0.0-1.0}}'
        ),
        schema=TaskTypeDecision,
        max_tokens=20,
    ),
    DecisionType.QUESTION_CLASSIFICATION: DecisionPrompt(
        system=(
            "You are a question type classifier. " "Respond ONLY with a JSON object, no other text."
        ),
        user_template=(
            "Classify this question from an AI assistant:\n\n"
            "{question_text}\n\n"
            'Respond with JSON: {{"question_type": '
            '"rhetorical"|"continuation"|"clarification"|"info", '
            '"confidence": 0.0-1.0}}'
        ),
        schema=QuestionTypeDecision,
        max_tokens=20,
    ),
    DecisionType.LOOP_DETECTION: DecisionPrompt(
        system=(
            "You are a loop detector for AI assistant behavior. "
            "Respond ONLY with a JSON object, no other text."
        ),
        user_template=(
            "Is this AI assistant stuck in a loop?\n\n"
            "Recent content:\n{content_excerpt}\n\n"
            "Recent thinking blocks: {recent_blocks}\n\n"
            'Respond with JSON: {{"is_loop": bool, '
            '"loop_type": "stalling"|"circular"|"repetition"|"none"}}'
        ),
        schema=LoopDetection,
        max_tokens=20,
    ),
    DecisionType.ERROR_CLASSIFICATION: DecisionPrompt(
        system=("You are an error classifier. " "Respond ONLY with a JSON object, no other text."),
        user_template=(
            "Classify this error:\n\n"
            "{error_message}\n\n"
            'Respond with JSON: {{"error_type": '
            '"permanent"|"transient"|"retryable", '
            '"confidence": 0.0-1.0}}'
        ),
        schema=ErrorClassDecision,
        max_tokens=20,
    ),
    DecisionType.CONTINUATION_ACTION: DecisionPrompt(
        system=(
            "You are a continuation action selector for an AI assistant. "
            "Respond ONLY with a JSON object, no other text."
        ),
        user_template=(
            "What should the AI assistant do next?\n\n"
            "Response excerpt:\n{response_excerpt}\n\n"
            "Continuation prompts sent: {continuation_prompts}\n"
            "Task type: {task_type}\n\n"
            'Respond with JSON: {{"action": '
            '"finish"|"prompt_tool_call"|"request_summary"|"return_to_user", '
            '"reason": "brief reason"}}'
        ),
        schema=ContinuationDecision,
        max_tokens=40,
    ),
}
