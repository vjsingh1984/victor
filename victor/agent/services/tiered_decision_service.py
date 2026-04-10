"""Tiered decision service — routes DecisionTypes to different model tiers.

Follows the GEPATierManager pattern: edge (fast local), balanced (mid-tier cloud),
performance (frontier). Each decision type is mapped to a tier via settings.
Fallback chain: performance → balanced → edge → heuristic.

Usage:
    from victor.agent.services.tiered_decision_service import TieredDecisionService
    from victor.config.decision_settings import DecisionServiceSettings

    config = DecisionServiceSettings()
    service = TieredDecisionService(config)
    result = service.decide_sync(DecisionType.TOOL_SELECTION, {"message": "fix test"})
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Set

from victor.agent.decisions.schemas import DecisionType
from victor.agent.services.protocols.decision_service import DecisionResult
from victor.config.decision_settings import DecisionServiceSettings

logger = logging.getLogger(__name__)

FALLBACK_CHAIN = {
    "performance": ["balanced", "edge"],
    "balanced": ["edge"],
    "edge": [],
}


class TieredDecisionService:
    """Decision service with per-DecisionType tier routing.

    Maintains a pool of LLMDecisionService instances (one per tier).
    Routes each decision type to the appropriate tier via settings.
    Falls back through the chain when a tier is unavailable.
    """

    def __init__(self, config: DecisionServiceSettings) -> None:
        self._config = config
        self._services: Dict[str, Any] = {}  # tier name → LLMDecisionService
        self._tier_routing = dict(config.tier_routing)
        self._failed_tiers: Set[str] = set()

    def decide_sync(
        self,
        decision_type: DecisionType,
        context: Dict[str, Any],
        *,
        heuristic_result: Any = None,
        heuristic_confidence: float = 0.0,
    ) -> DecisionResult:
        """Route to the correct tier's decision service.

        Args:
            decision_type: What kind of decision to make
            context: Context dict for the decision
            heuristic_result: Fallback result if no LLM available
            heuristic_confidence: Confidence of the heuristic result
        """
        tier = self._tier_routing.get(decision_type.value, "edge")

        # Try primary tier
        service = self._get_service(tier)
        if service is not None:
            return service.decide_sync(
                decision_type,
                context,
                heuristic_result=heuristic_result,
                heuristic_confidence=heuristic_confidence,
            )

        # Fallback chain
        for fallback_tier in FALLBACK_CHAIN.get(tier, []):
            if fallback_tier in self._failed_tiers:
                continue
            service = self._get_service(fallback_tier)
            if service is not None:
                logger.debug(
                    "Decision %s: tier '%s' unavailable, using '%s' fallback",
                    decision_type.value,
                    tier,
                    fallback_tier,
                )
                return service.decide_sync(
                    decision_type,
                    context,
                    heuristic_result=heuristic_result,
                    heuristic_confidence=heuristic_confidence,
                )

        # All tiers exhausted — return heuristic
        logger.debug("Decision %s: all tiers unavailable, using heuristic", decision_type.value)
        return DecisionResult(
            decision_type=decision_type,
            result=heuristic_result,
            source="heuristic",
            confidence=heuristic_confidence,
        )

    def _get_service(self, tier: str) -> Optional[Any]:
        """Get or create LLMDecisionService for a tier. Cached."""
        if tier in self._failed_tiers:
            return None
        if tier in self._services:
            return self._services[tier]
        return self._create_service(tier)

    def _create_service(self, tier: str) -> Optional[Any]:
        """Create an LLMDecisionService for the given tier."""
        spec = getattr(self._config, tier, None)
        if spec is None:
            self._failed_tiers.add(tier)
            return None

        try:
            from victor.agent.services.decision_service import (
                LLMDecisionService,
                LLMDecisionServiceConfig,
            )
            from victor.providers.registry import ProviderRegistry

            provider_kwargs: Dict[str, Any] = {
                "timeout": spec.timeout_ms // 1000,
            }

            if spec.provider == "ollama":
                provider_kwargs["base_url"] = "http://localhost:11434"
            else:
                try:
                    from victor.config.api_keys import get_api_key

                    api_key = get_api_key(spec.provider)
                    if api_key:
                        provider_kwargs["api_key"] = api_key
                except Exception:
                    pass

            provider = ProviderRegistry.create(spec.provider, **provider_kwargs)

            svc_config = LLMDecisionServiceConfig(
                confidence_threshold=0.7,
                micro_budget=20 if tier == "edge" else 10,
                timeout_ms=spec.timeout_ms,
                cache_ttl=120 if tier == "edge" else 60,
                temperature=0.0,
                max_tokens_override=spec.max_tokens,
            )

            service = LLMDecisionService(provider=provider, model=spec.model, config=svc_config)
            self._services[tier] = service
            logger.info(
                "Created %s decision service: %s/%s (timeout=%dms)",
                tier,
                spec.provider,
                spec.model,
                spec.timeout_ms,
            )
            return service

        except Exception as e:
            logger.warning(
                "Failed to create %s decision service (%s/%s): %s",
                tier,
                spec.provider,
                spec.model,
                e,
            )
            self._failed_tiers.add(tier)
            return None

    async def decide(
        self,
        decision_type: DecisionType,
        context: Dict[str, Any],
        *,
        heuristic_result: Any = None,
        heuristic_confidence: float = 0.0,
    ) -> DecisionResult:
        """Async version — delegates to decide_sync for compatibility."""
        return self.decide_sync(
            decision_type,
            context,
            heuristic_result=heuristic_result,
            heuristic_confidence=heuristic_confidence,
        )

    async def decide_async(
        self,
        decision_type: DecisionType,
        context: Dict[str, Any],
        *,
        heuristic_result: Any = None,
        heuristic_confidence: float = 0.0,
    ) -> DecisionResult:
        """Async alias for decide()."""
        return await self.decide(
            decision_type,
            context,
            heuristic_result=heuristic_result,
            heuristic_confidence=heuristic_confidence,
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Export metrics from all active tier services."""
        return {
            "active_tiers": list(self._services.keys()),
            "failed_tiers": list(self._failed_tiers),
            "routing": dict(self._tier_routing),
        }


def create_tiered_decision_service(
    config: Optional[DecisionServiceSettings] = None,
) -> Optional["TieredDecisionService"]:
    """Factory function to create a TieredDecisionService.

    Returns None if disabled in settings.
    """
    if config is None:
        config = DecisionServiceSettings()

    if not config.enabled:
        return None

    return TieredDecisionService(config)
