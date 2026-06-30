"""Centralized LLM-assisted decision schemas and prompts.

This package provides structured schemas for LLM-based decision making
used as a fallback when heuristic confidence is low. Each schema defines
a focused classifier output (5-20 tokens) for a specific decision type.

Schemas:
    - TaskCompletionDecision: Is the task done?
    - IntentDecision: What is the model doing?
    - TaskTypeDecision: What kind of task is this?
    - QuestionTypeDecision: Should we auto-continue?
    - LoopDetection: Is the model stuck?
    - ErrorClassDecision: Can we retry this error?
    - ContinuationDecision: What action should we take next?
"""

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

# FEP-0012 Phase 6: stamp the decision → outcome reward junction.
from victor.agent.decisions.outcome import record_session_outcome

__all__ = [
    "ContinuationDecision",
    "DecisionType",
    "ErrorClassDecision",
    "IntentDecision",
    "LoopDetection",
    "QuestionTypeDecision",
    "TaskCompletionDecision",
    "TaskTypeDecision",
    "record_session_outcome",
]
