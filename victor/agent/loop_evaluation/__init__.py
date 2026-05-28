"""Hybrid loop termination evaluator package.

Public surface
--------------
- ``LoopContext`` — unified input for loop termination evaluation.
- ``LoopDecision`` — unified output (action + reason + optional message).
- ``LoopEvaluator`` — base class / informal protocol.
- ``HybridLoopEvaluator`` — routes between legacy and agentic backends.
- ``LegacyEvaluator`` — wraps ``ContinuationStrategy`` (current default).
- ``AgenticLoopEvaluator`` — PERCEIVE→EVALUATE→DECIDE (opt-in via flag).
"""

from victor.agent.loop_evaluation.protocol import (
    LoopContext,
    LoopDecision,
    LoopEvaluator,
)
from victor.agent.loop_evaluation.hybrid import HybridLoopEvaluator
from victor.agent.loop_evaluation.legacy import LegacyEvaluator
from victor.agent.loop_evaluation.agentic import AgenticLoopEvaluator

__all__ = [
    "LoopContext",
    "LoopDecision",
    "LoopEvaluator",
    "HybridLoopEvaluator",
    "LegacyEvaluator",
    "AgenticLoopEvaluator",
]
