"""Tiered decision service — routes DecisionTypes to different model tiers.

Provider-Agnostic Design:
- Auto-detects active provider from orchestrator
- Resolves model from provider_model_tiers mapping
- Ensures decision service uses same provider as main LLM
- Supports explicit overrides via tier_overrides

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
import os
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

    Provider-Agnostic Behavior:
    - Auto-detects active provider from orchestrator/container
    - Resolves model from provider_model_tiers mapping
    - Falls back gracefully when provider or model unavailable
    - Supports tier_overrides for explicit provider/model selection

    Maintains a pool of LLMDecisionService instances (one per tier).
    Routes each decision type to the appropriate tier via settings.
    Falls back through the chain when a tier is unavailable.
    """

    def __init__(self, config: DecisionServiceSettings) -> None:
        self._config = config
        self._services: Dict[str, Any] = {}  # tier name → LLMDecisionService
        self._tier_routing = dict(config.tier_routing)
        self._failed_tiers: Set[str] = set()
        self._detected_provider: Optional[str] = None  # Cached provider detection

    def _detect_active_provider(self) -> Optional[str]:
        """Detect the active provider from the orchestrator.

        Checks multiple sources in order:
        1. Service container (ProviderManager/ProviderService)
        2. Settings.default_provider
        3. Environment variables (VICTOR_PROVIDER)

        Returns:
            Provider name (e.g., "anthropic", "openai") or None
        """
        # Return cached provider if available
        if self._detected_provider:
            return self._detected_provider

        # Try to get from service container
        try:
            from victor.core import get_container

            container = get_container()

            # Try ProviderService first (new service layer)
            try:
                from victor.agent.services.protocols.provider_service import (
                    ProviderService,
                )

                provider_service = container.get(ProviderService)
                if provider_service:
                    provider_name = getattr(provider_service, "provider_name", None)
                    if provider_name:
                        self._detected_provider = provider_name
                        logger.debug(
                            f"Auto-detected provider from ProviderService: {provider_name}"
                        )
                        return provider_name
            except Exception:
                pass

            # Try ProviderManager (legacy)
            try:
                from victor.agent.provider_manager import ProviderManager

                provider_mgr = container.get(ProviderManager)
                if provider_mgr:
                    provider_name = getattr(provider_mgr, "provider_name", None)
                    if provider_name:
                        self._detected_provider = provider_name
                        logger.debug(
                            f"Auto-detected provider from ProviderManager: {provider_name}"
                        )
                        return provider_name
            except Exception:
                pass

        except Exception:
            pass

        # Try settings
        try:
            from victor.config.settings import Settings

            settings = Settings()
            if hasattr(settings, "default_provider") and settings.provider.default_provider:
                self._detected_provider = settings.provider.default_provider
                logger.debug(
                    f"Auto-detected provider from settings: {settings.provider.default_provider}"
                )
                return settings.provider.default_provider
        except Exception:
            pass

        # Try environment variable
        try:
            provider = os.getenv("VICTOR_PROVIDER")
            if provider:
                self._detected_provider = provider.lower()
                logger.debug(f"Auto-detected provider from env var: {provider}")
                return self._detected_provider
        except Exception:
            pass

        logger.debug("Could not auto-detect active provider (normal at bootstrap)")
        return None

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

        # Handle "auto" routing for COMPACTION decision type
        # Auto-route based on complexity: simple→edge, complex→performance
        if tier == "auto" and decision_type == DecisionType.COMPACTION:
            complexity = context.get("complexity", "simple")
            if complexity == "complex":
                tier = "performance"
            else:
                tier = "edge"
            logger.debug(
                "Decision %s: auto-routed to '%s' tier (complexity=%s)",
                decision_type.value,
                tier,
                complexity,
            )

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
        """Create an LLMDecisionService for the given tier.

        Auto-detects provider from active orchestrator if provider="auto".
        Resolves model from provider_model_tiers mapping if model="auto".
        """
        # Check for tier overrides first
        if tier in self._config.tier_overrides:
            override = self._config.tier_overrides[tier]
            override_provider = override.get("provider", "auto")
            override_model = override.get("model", "auto")

            # Use override values to create a temporary spec
            from victor.config.decision_settings import DecisionModelSpec

            spec = DecisionModelSpec(
                provider=override_provider,
                model=override_model,
                timeout_ms=getattr(self._config, tier).timeout_ms,
                max_tokens=getattr(self._config, tier).max_tokens,
            )

            logger.debug(f"Using override for tier '{tier}': {override_provider}/{override_model}")
        else:
            spec = getattr(self._config, tier, None)

        if spec is None:
            self._failed_tiers.add(tier)
            return None

        # Resolve provider="auto" to actual provider
        provider = spec.provider
        model = spec.model

        if provider == "auto" or model == "auto":
            # Detect active provider from orchestrator
            active_provider = self._detect_active_provider()
            if active_provider is None:
                # Fallbacks for local providers to ensure stability when auto-detection fails
                if tier == "edge":
                    logger.info(
                        "Using hardcoded fallback for 'edge' tier: ollama/qwen2.5-coder:1.5b"
                    )
                    provider = "ollama"
                    model = "qwen2.5-coder:1.5b"
                elif tier == "balanced":
                    logger.info(
                        "Using hardcoded fallback for 'balanced' tier: ollama/qwen2.5-coder:1.5b"
                    )
                    provider = "ollama"
                    model = "qwen2.5-coder:1.5b"
                else:
                    logger.warning(f"Could not detect active provider for tier '{tier}'")
                    self._failed_tiers.add(tier)
                    return None
            else:
                provider = active_provider

        # Resolve model="auto" from provider's tier mapping
        if model == "auto":
            provider_tiers = self._config.provider_model_tiers.get(provider, {})
            model = provider_tiers.get(tier)
            if model is None:
                logger.warning(
                    f"No {tier} model defined for provider '{provider}'. "
                    f"Available tiers: {list(provider_tiers.keys())}"
                )
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

            if provider == "ollama":
                provider_kwargs["base_url"] = "http://localhost:11434"
            else:
                try:
                    from victor.config.api_keys import get_api_key

                    api_key = get_api_key(provider)
                    if api_key:
                        provider_kwargs["api_key"] = api_key
                except Exception:
                    pass

            provider_instance = ProviderRegistry.create(provider, **provider_kwargs)

            svc_config = LLMDecisionServiceConfig(
                confidence_threshold=0.7,
                micro_budget=20 if tier == "edge" else 10,
                timeout_ms=spec.timeout_ms,
                cache_ttl=120 if tier == "edge" else 60,
                temperature=0.0,
                max_tokens_override=spec.max_tokens,
            )

            service = LLMDecisionService(provider=provider_instance, model=model, config=svc_config)
            self._services[tier] = service
            logger.info(
                "Created %s decision service: %s/%s (timeout=%dms)",
                tier,
                provider,
                model,
                spec.timeout_ms,
            )
            return service

        except Exception as e:
            logger.warning(
                "Failed to create %s decision service (%s/%s): %s",
                tier,
                provider,
                model,
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

    def escalate_tier(
        self,
        decision_type: DecisionType,
        reason: str = "low_confidence",
    ) -> str:
        """Escalate a decision type to a higher tier.

        Called when repeated low-confidence results suggest the current
        tier isn't powerful enough for this decision. Follows the
        GEPATierManager pattern: edge → balanced → performance.

        Args:
            decision_type: Decision type to escalate
            reason: Why escalating (for logging)

        Returns:
            New tier name after escalation
        """
        current = self._tier_routing.get(decision_type.value, "edge")
        escalation = {"edge": "balanced", "balanced": "performance"}
        new_tier = escalation.get(current, current)

        if new_tier != current:
            self._tier_routing[decision_type.value] = new_tier
            logger.info(
                "Decision %s: tier escalated %s → %s (reason: %s)",
                decision_type.value,
                current,
                new_tier,
                reason,
            )

        return new_tier

    def deescalate_tier(
        self,
        decision_type: DecisionType,
        reason: str = "high_confidence",
    ) -> str:
        """De-escalate a decision type to a lower tier.

        Called when consistent high-confidence results suggest a cheaper
        tier is sufficient. performance → balanced → edge.

        Args:
            decision_type: Decision type to de-escalate
            reason: Why de-escalating (for logging)

        Returns:
            New tier name after de-escalation
        """
        current = self._tier_routing.get(decision_type.value, "edge")
        deescalation = {"performance": "balanced", "balanced": "edge"}
        new_tier = deescalation.get(current, current)

        if new_tier != current:
            self._tier_routing[decision_type.value] = new_tier
            logger.info(
                "Decision %s: tier de-escalated %s → %s (reason: %s)",
                decision_type.value,
                current,
                new_tier,
                reason,
            )

        return new_tier

    def get_tier(self, decision_type: DecisionType) -> str:
        """Get current tier for a decision type."""
        return self._tier_routing.get(decision_type.value, "edge")

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
