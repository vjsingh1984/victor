"""HybridLoopEvaluator — routes between legacy and agentic backends.

The active backend is selected by the ``USE_STATEGRAPH_AGENTIC_LOOP``
feature flag:

* **OFF** (default today): ``LegacyEvaluator`` — wraps ``ContinuationStrategy``,
  zero behavior change.
* **ON** (opt-in): ``AgenticLoopEvaluator`` — PERCEIVE→EVALUATE→DECIDE path.

The ``AgenticLoopEvaluator`` has one special escape hatch: when it returns a
``LoopDecision`` with ``confidence=0.0``, the hybrid falls back to the legacy
evaluator for that turn.  This covers cases the agentic evaluator delegates
(e.g., hallucinated tool call recovery) without requiring the caller to know
which backend was used.

Migration path
--------------
1. Ship this module with the flag OFF (legacy path, no change).
2. Enable the flag on a subset of sessions (canary).
3. Once validated, flip the flag default in FeatureFlag definition.
4. Delete LegacyEvaluator and this routing logic.
"""

from __future__ import annotations

import logging
from typing import Optional

from victor.agent.loop_evaluation.protocol import (
    LoopContext,
    LoopDecision,
    LoopEvaluator,
)

logger = logging.getLogger(__name__)


class HybridLoopEvaluator(LoopEvaluator):
    """Routes termination decisions between legacy and agentic backends.

    A single instance is created per ``IntentClassificationHandler`` so the
    ``AgenticLoopEvaluator``'s score history persists across the turns of one
    streaming session.
    """

    def __init__(self) -> None:
        self._legacy: Optional[LoopEvaluator] = None
        self._agentic: Optional[LoopEvaluator] = None
        self._use_agentic: Optional[bool] = None  # Cached flag value.

    def evaluate(self, ctx: LoopContext) -> LoopDecision:
        use_agentic = self._resolve_flag()

        if not use_agentic:
            return self._get_legacy().evaluate(ctx)

        # Agentic path with legacy fallback for confidence=0.0 decisions.
        decision = self._get_agentic().evaluate(ctx)
        if decision.confidence == 0.0:
            logger.debug(
                "[hybrid-eval] agentic delegated (reason=%s) → falling back to legacy",
                decision.reason,
            )
            return self._get_legacy().evaluate(ctx)

        logger.debug(
            "[hybrid-eval] agentic decision: action=%s reason=%s confidence=%.2f",
            decision.action,
            decision.reason,
            decision.confidence,
        )
        return decision

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_flag(self) -> bool:
        if self._use_agentic is None:
            try:
                from victor.framework.agentic_loop_executor import (
                    use_stategraph_executor,
                )

                self._use_agentic = use_stategraph_executor()
            except Exception:
                self._use_agentic = False
        return self._use_agentic

    def _get_legacy(self) -> LoopEvaluator:
        if self._legacy is None:
            from victor.agent.loop_evaluation.legacy import LegacyEvaluator

            self._legacy = LegacyEvaluator()
        return self._legacy

    def _get_agentic(self) -> LoopEvaluator:
        if self._agentic is None:
            from victor.agent.loop_evaluation.agentic import AgenticLoopEvaluator

            self._agentic = AgenticLoopEvaluator()
        return self._agentic
