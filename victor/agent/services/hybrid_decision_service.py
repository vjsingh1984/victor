"""Hybrid decision service integrating deterministic rules with LLM fallback.

Implements a 4-tier decision pipeline:
1. Lookup Tables (O(1)) → 95% accuracy, 0ms
2. Pattern Matcher (regex) → 85% accuracy, 1-5ms
3. Ensemble Voting → 90% accuracy, 5-10ms
4. LLM Fallback → 98% accuracy, 500-2000ms

This service combines the speed of deterministic rules with the accuracy
of LLM-based decisions, using adaptive confidence thresholds and
multi-level caching for optimal performance.

Usage:
    from victor.agent.services.hybrid_decision_service import HybridDecisionService

    service = HybridDecisionService(provider, model)
    result = await service.decide(
        DecisionType.TASK_COMPLETION,
        {"message": "I'm done with the task"},
    )
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

from victor.agent.decisions.schemas import DecisionType
from victor.agent.services.confidence_calibrator import (
    ConfidenceCalibrator,
    create_confidence_calibrator,
)
from victor.agent.services.decision_cache import (
    DecisionCache,
    create_decision_cache,
)
from victor.agent.services.deterministic_decision_rules import (
    EnsembleVoter,
    LookupTables,
    LookupResult,
    PatternMatcher,
)
from victor.agent.services.protocols.decision_service import (
    DecisionMetrics,
    DecisionResult,
)

if TYPE_CHECKING:
    from victor.providers.base import BaseProvider

logger = logging.getLogger(__name__)


@dataclass
class HybridDecisionServiceConfig:
    """Configuration for the hybrid decision service."""

    # Deterministic rules
    enable_lookup_tables: bool = True
    enable_pattern_matcher: bool = True
    enable_ensemble_voting: bool = True

    # Confidence calibration
    enable_calibration: bool = True
    base_threshold: float = 0.70
    target_accuracy: float = 0.95

    # Caching
    enable_cache: bool = True
    l1_cache_size: int = 100
    l1_cache_ttl: int = 300
    enable_l2_cache: bool = False

    # LLM fallback
    enable_llm_fallback: bool = True
    llm_timeout_ms: int = 2000
    llm_temperature: float = 0.0

    # Budget
    micro_budget: int = 10


class HybridDecisionService:
    """Hybrid decision service with deterministic rules and LLM fallback.

    Implements a fast decision pipeline using lookup tables, pattern matching,
    ensemble voting, and LLM fallback. Uses adaptive confidence thresholds
    and multi-level caching for optimal performance.
    """

    def __init__(
        self,
        provider: Optional[BaseProvider] = None,
        model: str = "default",
        config: Optional[HybridDecisionServiceConfig] = None,
    ) -> None:
        """Initialize the hybrid decision service.

        Args:
            provider: LLM provider for fallback (can be None for rules-only mode)
            model: Model name for LLM calls
            config: Service configuration
        """
        self._provider = provider
        self._model = model
        self._config = config or HybridDecisionServiceConfig()

        # Initialize components
        self._ensemble_voter = EnsembleVoter() if self._config.enable_ensemble_voting else None

        # Confidence calibrator
        if self._config.enable_calibration:
            self._calibrator = create_confidence_calibrator(
                strategy="adaptive",
                base_threshold=self._config.base_threshold,
                target_accuracy=self._config.target_accuracy,
            )
        else:
            self._calibrator = None

        # Decision cache
        if self._config.enable_cache:
            self._cache = create_decision_cache(
                l1_size=self._config.l1_cache_size,
                l1_ttl=self._config.l1_cache_ttl,
                l2_enabled=self._config.enable_l2_cache,
            )
        else:
            self._cache = None

        # Budget tracking
        self._budget_used: int = 0

        # Metrics
        self._metrics = DecisionMetrics()
        self._hybrid_metrics = HybridMetrics()

        logger.info(
            "HybridDecisionService initialized: lookup=%s, pattern=%s, ensemble=%s, calibration=%s, cache=%s, llm=%s",
            self._config.enable_lookup_tables,
            self._config.enable_pattern_matcher,
            self._config.enable_ensemble_voting,
            self._config.enable_calibration,
            self._config.enable_cache,
            self._config.enable_llm_fallback,
        )

    async def decide(
        self,
        decision_type: DecisionType,
        context: Dict[str, Any],
        *,
        heuristic_result: Any = None,
        heuristic_confidence: float = 0.0,
    ) -> DecisionResult:
        """Make a decision using the hybrid pipeline.

        Pipeline:
        1. Check cache
        2. Try lookup tables (O(1), 95% accuracy)
        3. Try pattern matcher (regex, 85% accuracy)
        4. Try ensemble voting (5-10ms, 90% accuracy)
        5. Fall back to LLM (500-2000ms, 98% accuracy)

        Args:
            decision_type: Type of decision to make
            context: Decision context
            heuristic_result: Optional heuristic result (used as fallback)
            heuristic_confidence: Confidence of heuristic result

        Returns:
            Decision result with source, confidence, and latency
        """
        start = time.monotonic()
        self._metrics.total_calls += 1

        # Step 1: Check cache
        if self._cache:
            cached_result = self._cache.get(decision_type, context)
            if cached_result is not None:
                self._metrics.cache_hits += 1
                self._hybrid_metrics.cache_hits += 1
                cached_result.latency_ms = _elapsed_ms(start)
                return cached_result

        # Step 2: Try lookup tables
        if self._config.enable_lookup_tables:
            lookup_result = LookupTables.lookup(decision_type, context)
            if lookup_result is not None:
                self._hybrid_metrics.lookup_hits += 1

                # Check if confidence is sufficient
                should_use_llm = self._should_use_llm(
                    decision_type,
                    lookup_result.confidence,
                )

                if not should_use_llm:
                    self._record_outcome(
                        decision_type,
                        lookup_result.confidence,
                        used_llm=False,
                        was_correct=None,  # Unknown until later
                        source="lookup",
                    )

                    result = DecisionResult(
                        decision_type=decision_type,
                        result=lookup_result.decision,
                        source="lookup",
                        confidence=lookup_result.confidence,
                        latency_ms=_elapsed_ms(start),
                    )

                    # Cache the result
                    if self._cache:
                        self._cache.put(decision_type, context, result)

                    return result

        # Step 3: Try pattern matcher
        if self._config.enable_pattern_matcher:
            pattern_result = PatternMatcher.match(decision_type, context)
            if pattern_result is not None:
                self._hybrid_metrics.pattern_hits += 1

                # Check if confidence is sufficient
                should_use_llm = self._should_use_llm(
                    decision_type,
                    pattern_result.confidence,
                )

                if not should_use_llm:
                    self._record_outcome(
                        decision_type,
                        pattern_result.confidence,
                        used_llm=False,
                        was_correct=None,
                        source="pattern",
                    )

                    result = DecisionResult(
                        decision_type=decision_type,
                        result=pattern_result.decision,
                        source="pattern",
                        confidence=pattern_result.confidence,
                        latency_ms=_elapsed_ms(start),
                    )

                    # Cache the result
                    if self._cache:
                        self._cache.put(decision_type, context, result)

                    return result

        # Step 4: Try ensemble voting
        if self._config.enable_ensemble_voting and self._ensemble_voter:
            # Collect signals from lookup and pattern
            lookup_result = LookupTables.lookup(decision_type, context)
            pattern_result = PatternMatcher.match(decision_type, context)

            ensemble_result = self._ensemble_voter.vote(
                decision_type,
                context,
                keyword_result=lookup_result or pattern_result,
                semantic_result=None,  # Could add semantic similarity here
                heuristic_result=heuristic_result,
                heuristic_confidence=heuristic_confidence,
            )

            if ensemble_result is not None:
                self._hybrid_metrics.ensemble_hits += 1

                # Check if confidence is sufficient
                should_use_llm = self._should_use_llm(
                    decision_type,
                    ensemble_result.confidence,
                )

                if not should_use_llm:
                    self._record_outcome(
                        decision_type,
                        ensemble_result.confidence,
                        used_llm=False,
                        was_correct=None,
                        source="ensemble",
                    )

                    result = DecisionResult(
                        decision_type=decision_type,
                        result=ensemble_result.decision,
                        source="ensemble",
                        confidence=ensemble_result.confidence,
                        latency_ms=_elapsed_ms(start),
                    )

                    # Cache the result
                    if self._cache:
                        self._cache.put(decision_type, context, result)

                    return result

        # Step 5: Fall back to LLM
        if self._config.enable_llm_fallback and self._provider:
            return await self._llm_fallback(
                decision_type,
                context,
                heuristic_result,
                heuristic_confidence,
                start,
            )

        # No LLM available, use heuristic fallback
        self._hybrid_metrics.heuristic_fallbacks += 1
        self._metrics.total_calls += 1

        return DecisionResult(
            decision_type=decision_type,
            result=heuristic_result,
            source="heuristic_fallback",
            confidence=heuristic_confidence,
            latency_ms=_elapsed_ms(start),
        )

    def decide_sync(
        self,
        decision_type: DecisionType,
        context: Dict[str, Any],
        *,
        heuristic_result: Any = None,
        heuristic_confidence: float = 0.0,
    ) -> DecisionResult:
        """Synchronous version of decide().

        For compatibility with existing code, this provides a synchronous
        interface. The deterministic pipeline is fully synchronous, so
        this works efficiently. Only LLM fallback requires async.

        Args:
            decision_type: Type of decision to make
            context: Decision context
            heuristic_result: Optional heuristic result
            heuristic_confidence: Confidence of heuristic result

        Returns:
            Decision result
        """
        start = time.monotonic()

        # Try deterministic pipeline first (all sync)
        if self._cache:
            cached_result = self._cache.get(decision_type, context)
            if cached_result is not None:
                self._metrics.cache_hits += 1
                self._hybrid_metrics.cache_hits += 1
                cached_result.latency_ms = _elapsed_ms(start)
                return cached_result

        # Try lookup tables
        if self._config.enable_lookup_tables:
            lookup_result = LookupTables.lookup(decision_type, context)
            if lookup_result is not None:
                should_use_llm = self._should_use_llm(
                    decision_type,
                    lookup_result.confidence,
                )

                if not should_use_llm:
                    self._hybrid_metrics.lookup_hits += 1

                    result = DecisionResult(
                        decision_type=decision_type,
                        result=lookup_result.decision,
                        source="lookup",
                        confidence=lookup_result.confidence,
                        latency_ms=_elapsed_ms(start),
                    )

                    if self._cache:
                        self._cache.put(decision_type, context, result)

                    return result

        # Try pattern matcher
        if self._config.enable_pattern_matcher:
            pattern_result = PatternMatcher.match(decision_type, context)
            if pattern_result is not None:
                should_use_llm = self._should_use_llm(
                    decision_type,
                    pattern_result.confidence,
                )

                if not should_use_llm:
                    self._hybrid_metrics.pattern_hits += 1

                    result = DecisionResult(
                        decision_type=decision_type,
                        result=pattern_result.decision,
                        source="pattern",
                        confidence=pattern_result.confidence,
                        latency_ms=_elapsed_ms(start),
                    )

                    if self._cache:
                        self._cache.put(decision_type, context, result)

                    return result

        # Try ensemble voting
        if self._config.enable_ensemble_voting and self._ensemble_voter:
            lookup_result = LookupTables.lookup(decision_type, context)
            pattern_result = PatternMatcher.match(decision_type, context)

            ensemble_result = self._ensemble_voter.vote(
                decision_type,
                context,
                keyword_result=lookup_result or pattern_result,
                semantic_result=None,
                heuristic_result=heuristic_result,
                heuristic_confidence=heuristic_confidence,
            )

            if ensemble_result is not None:
                should_use_llm = self._should_use_llm(
                    decision_type,
                    ensemble_result.confidence,
                )

                if not should_use_llm:
                    self._hybrid_metrics.ensemble_hits += 1

                    result = DecisionResult(
                        decision_type=decision_type,
                        result=ensemble_result.decision,
                        source="ensemble",
                        confidence=ensemble_result.confidence,
                        latency_ms=_elapsed_ms(start),
                    )

                    if self._cache:
                        self._cache.put(decision_type, context, result)

                    return result

        # Fall back to heuristic
        self._hybrid_metrics.heuristic_fallbacks += 1

        return DecisionResult(
            decision_type=decision_type,
            result=heuristic_result,
            source="heuristic_fallback",
            confidence=heuristic_confidence,
            latency_ms=_elapsed_ms(start),
        )

    def record_outcome(
        self,
        decision_type: DecisionType,
        heuristic_confidence: float,
        used_llm: bool,
        was_correct: bool,
        source: str = "unknown",
    ) -> None:
        """Record the outcome of a decision for calibration.

        This should be called after the actual outcome is known to
        improve future decisions through confidence calibration.

        Args:
            decision_type: Type of decision that was made
            heuristic_confidence: Confidence of the heuristic prediction
            used_llm: Whether LLM was used
            was_correct: Whether the decision was correct
            source: Source of the decision ("lookup", "pattern", "ensemble", "llm")
        """
        self._record_outcome(
            decision_type,
            heuristic_confidence,
            used_llm,
            was_correct,
            source,
        )

    def _record_outcome(
        self,
        decision_type: DecisionType,
        heuristic_confidence: float,
        used_llm: bool,
        was_correct: bool,
        source: str,
    ) -> None:
        """Internal method to record decision outcome."""
        if self._calibrator:
            self._calibrator.record_outcome(
                decision_type=decision_type,
                heuristic_confidence=heuristic_confidence,
                used_llm=used_llm,
                was_correct=was_correct,
            )

        logger.debug(
            "Recorded outcome: type=%s source=%s used_llm=%s correct=%s confidence=%.2f",
            decision_type.value,
            source,
            used_llm,
            was_correct,
            heuristic_confidence,
        )

    def _should_use_llm(
        self,
        decision_type: DecisionType,
        heuristic_confidence: float,
    ) -> bool:
        """Determine if LLM should be used based on calibrated threshold.

        Args:
            decision_type: Type of decision
            heuristic_confidence: Confidence of heuristic prediction

        Returns:
            True if LLM should be used
        """
        if self._calibrator:
            return self._calibrator.should_use_llm(
                decision_type,
                heuristic_confidence,
            )

        # Fallback to static threshold
        return heuristic_confidence < self._config.base_threshold

    async def _llm_fallback(
        self,
        decision_type: DecisionType,
        context: Dict[str, Any],
        heuristic_result: Any,
        heuristic_confidence: float,
        start: float,
    ) -> DecisionResult:
        """Fall back to LLM for decision.

        Args:
            decision_type: Type of decision
            context: Decision context
            heuristic_result: Heuristic result to use if LLM fails
            heuristic_confidence: Confidence of heuristic result
            start: Start time for latency calculation

        Returns:
            Decision result from LLM or heuristic fallback
        """
        import asyncio

        self._hybrid_metrics.llm_calls += 1

        # Check budget
        if self._budget_used >= self._config.micro_budget:
            self._metrics.budget_exhaustions += 1
            self._hybrid_metrics.budget_exhaustions += 1

            logger.debug(
                "LLM budget exhausted (%d/%d), using heuristic fallback",
                self._budget_used,
                self._config.micro_budget,
            )

            return DecisionResult(
                decision_type=decision_type,
                result=heuristic_result,
                source="budget_exhausted",
                confidence=heuristic_confidence,
                latency_ms=_elapsed_ms(start),
            )

        # Call LLM with timeout
        try:
            timeout_s = self._config.llm_timeout_ms / 1000.0

            # Import here to avoid circular dependency
            from victor.agent.decisions.prompts import DECISION_PROMPTS

            prompt_config = DECISION_PROMPTS[decision_type]

            # Build user message
            try:
                user_message = prompt_config.user_template.format(**context)
            except KeyError as e:
                logger.warning("Missing context key for %s: %s", decision_type.value, e)
                raise

            messages = [
                {"role": "system", "content": prompt_config.system},
                {"role": "user", "content": user_message},
            ]

            response = await asyncio.wait_for(
                self._provider.chat(
                    messages=messages,
                    model=self._model,
                    temperature=self._config.llm_temperature,
                    max_tokens=prompt_config.max_tokens,
                    tools=None,
                    think=False,
                ),
                timeout=timeout_s,
            )

            self._budget_used += 1
            self._metrics.llm_calls += 1

            # Parse response
            raw_text = self._extract_response_text(response)
            parsed = self._parse_response(raw_text, prompt_config.schema)

            confidence = getattr(parsed, "confidence", 0.8)

            result = DecisionResult(
                decision_type=decision_type,
                result=parsed,
                source="llm",
                confidence=confidence,
                latency_ms=_elapsed_ms(start),
            )

            # Cache the result
            if self._cache:
                self._cache.put(decision_type, context, result)

            return result

        except asyncio.TimeoutError:
            self._metrics.timeouts += 1
            self._hybrid_metrics.timeouts += 1

            logger.debug(
                "LLM decision timed out after %dms, using heuristic fallback",
                self._config.llm_timeout_ms,
            )

            return DecisionResult(
                decision_type=decision_type,
                result=heuristic_result,
                source="timeout_fallback",
                confidence=heuristic_confidence,
                latency_ms=_elapsed_ms(start),
            )

        except Exception as e:
            logger.debug("LLM decision call failed: %s", e)

            return DecisionResult(
                decision_type=decision_type,
                result=heuristic_result,
                source="llm_error",
                confidence=heuristic_confidence,
                latency_ms=_elapsed_ms(start),
            )

    def _extract_response_text(self, response: Any) -> str:
        """Extract text content from a provider response."""
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        return block["text"]
                    if hasattr(block, "text"):
                        return block.text
        if isinstance(response, dict):
            if "content" in response:
                return str(response["content"])
            if "text" in response:
                return str(response["text"])
        return str(response)

    def _parse_response(self, raw_text: str, schema: type) -> Any:
        """Parse raw LLM text into a Pydantic model."""
        import json

        text = raw_text.strip()

        if text.startswith("```"):
            lines = text.split("\n")
            lines = [line for line in lines[1:] if not line.startswith("```")]
            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
            return schema.model_validate(data)
        except json.JSONDecodeError:
            pass

        brace_pos = text.find("{")
        if brace_pos > 0:
            last_brace = text.rfind("}")
            if last_brace > brace_pos:
                try:
                    data = json.loads(text[brace_pos : last_brace + 1])
                    return schema.model_validate(data)
                except (json.JSONDecodeError, Exception):
                    pass

        raise ValueError(f"Could not extract JSON from response: {raw_text[:100]}")

    def reset_budget(self) -> None:
        """Reset the per-turn LLM call budget."""
        self._budget_used = 0
        logger.debug("LLM decision budget reset")

    @property
    def budget_remaining(self) -> int:
        """Number of LLM decision calls remaining in the current turn budget."""
        return max(0, self._config.micro_budget - self._budget_used)

    def get_metrics(self) -> DecisionMetrics:
        """Get standard decision metrics."""
        return self._metrics

    def get_hybrid_metrics(self) -> HybridMetrics:
        """Get hybrid-specific metrics."""
        return self._hybrid_metrics

    def get_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of service state."""
        summary = {
            "config": {
                "lookup_enabled": self._config.enable_lookup_tables,
                "pattern_enabled": self._config.enable_pattern_matcher,
                "ensemble_enabled": self._config.enable_ensemble_voting,
                "calibration_enabled": self._config.enable_calibration,
                "cache_enabled": self._config.enable_cache,
                "llm_enabled": self._config.enable_llm_fallback,
            },
            "metrics": {
                "total_calls": self._metrics.total_calls,
                "cache_hits": self._metrics.cache_hits,
                "llm_calls": self._metrics.llm_calls,
            },
            "hybrid_metrics": {
                "cache_hits": self._hybrid_metrics.cache_hits,
                "lookup_hits": self._hybrid_metrics.lookup_hits,
                "pattern_hits": self._hybrid_metrics.pattern_hits,
                "ensemble_hits": self._hybrid_metrics.ensemble_hits,
                "llm_calls": self._hybrid_metrics.llm_calls,
                "heuristic_fallbacks": self._hybrid_metrics.heuristic_fallbacks,
            },
        }

        if self._cache:
            summary["cache_stats"] = self._cache.get_stats()

        if self._calibrator:
            summary["calibration"] = self._calibrator.get_summary()

        return summary


@dataclass
class HybridMetrics:
    """Metrics specific to hybrid decision service."""

    cache_hits: int = 0
    lookup_hits: int = 0
    pattern_hits: int = 0
    ensemble_hits: int = 0
    llm_calls: int = 0
    heuristic_fallbacks: int = 0
    budget_exhaustions: int = 0
    timeouts: int = 0

    def get_hit_rate(self) -> float:
        """Calculate overall hit rate (cache + lookup + pattern + ensemble)."""
        total_hits = (
            self.cache_hits +
            self.lookup_hits +
            self.pattern_hits +
            self.ensemble_hits
        )
        total_requests = total_hits + self.llm_calls + self.heuristic_fallbacks
        return total_hits / total_requests if total_requests > 0 else 0.0


def _elapsed_ms(start: float) -> float:
    """Calculate elapsed milliseconds since start."""
    return (time.monotonic() - start) * 1000
