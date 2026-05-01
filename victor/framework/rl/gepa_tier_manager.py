"""GEPA v2 tiered model selection and auto-switching.

Three tiers for GEPA reflection/mutation:
- economic: Local model (Ollama qwen3:8b) — post-convergence maintenance
- balanced: Mid-tier cloud (GPT-4.1-mini) — default
- performance: Frontier (Sonnet) — initial convergence

Auto-switches based on windowed convergence metrics:
- Improvement delta < threshold for N consecutive rounds → downgrade tier
- Significant regression → upgrade to performance tier
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, Optional

from victor.framework.rl.gepa_service import GEPAService

logger = logging.getLogger(__name__)

VALID_TIERS = ("economic", "balanced", "performance")
TIER_ORDER = {"economic": 0, "balanced": 1, "performance": 2}
TIER_DOWNGRADE = {"performance": "balanced", "balanced": "economic", "economic": "economic"}
TIER_UPGRADE = {"economic": "balanced", "balanced": "performance", "performance": "performance"}


class GEPATierManager:
    """Manages GEPA model tier selection and auto-switching."""

    def __init__(self, config: Any):
        """Initialize with GEPASettings config.

        Args:
            config: GEPASettings instance with tier model specs
        """
        self._config = config
        self._current_tier: str = getattr(config, "default_tier", "balanced")
        if self._current_tier not in VALID_TIERS:
            self._current_tier = "balanced"

        self._auto_switch: bool = getattr(config, "auto_tier_switch", True)
        self._convergence_window: int = getattr(config, "convergence_window", 10)
        self._convergence_threshold: float = getattr(config, "convergence_threshold", 0.02)
        self._regression_threshold: float = -0.1

        # Per-section convergence tracking
        self._deltas: Dict[str, deque] = {}

        # Cached services per tier
        self._services: Dict[str, GEPAService] = {}

        # Metrics
        self._tier_switches: int = 0
        self._total_evolutions: int = 0

    def get_current_tier(self) -> str:
        """Return current active tier name."""
        return self._current_tier

    def get_service(self) -> GEPAService:
        """Get or create GEPAService for current tier. Cached per tier."""
        tier = self._current_tier
        if tier in self._services:
            return self._services[tier]

        service = self._create_service(tier)
        self._services[tier] = service
        return service

    def record_evolution_delta(self, section_name: str, delta: float) -> None:
        """Record improvement delta from an evolution cycle.

        Triggers automatic tier switching when convergence is detected
        or significant regression occurs.

        Args:
            section_name: Which prompt section was evolved
            delta: Improvement delta (positive = improved, negative = regressed)
        """
        self._total_evolutions += 1

        if section_name not in self._deltas:
            self._deltas[section_name] = deque(maxlen=self._convergence_window)
        self._deltas[section_name].append(delta)

        if not self._auto_switch:
            return

        # Check for regression → upgrade
        if delta < self._regression_threshold:
            old = self._current_tier
            self._current_tier = TIER_UPGRADE[self._current_tier]
            if old != self._current_tier:
                self._tier_switches += 1
                logger.info(
                    "GEPA tier upgraded %s → %s (regression %.3f on %s)",
                    old,
                    self._current_tier,
                    delta,
                    section_name,
                )
            return

        # Check for convergence → downgrade
        window = self._deltas[section_name]
        if len(window) >= self._convergence_window:
            if all(abs(d) < self._convergence_threshold for d in window):
                old = self._current_tier
                self._current_tier = TIER_DOWNGRADE[self._current_tier]
                if old != self._current_tier:
                    self._tier_switches += 1
                    logger.info(
                        "GEPA tier downgraded %s → %s (converged on %s, "
                        "window=%d, max_delta=%.4f)",
                        old,
                        self._current_tier,
                        section_name,
                        len(window),
                        max(abs(d) for d in window),
                    )

    def force_tier(self, tier: str) -> None:
        """Manual tier override. Disables auto-switch for this session."""
        if tier not in VALID_TIERS:
            raise ValueError(f"Invalid tier '{tier}'. Must be one of: {VALID_TIERS}")
        old = self._current_tier
        self._current_tier = tier
        self._auto_switch = False
        logger.info("GEPA tier forced %s → %s (auto-switch disabled)", old, tier)

    def get_metrics(self) -> Dict[str, Any]:
        """Export tier manager metrics."""
        return {
            "current_tier": self._current_tier,
            "auto_switch": self._auto_switch,
            "tier_switches": self._tier_switches,
            "total_evolutions": self._total_evolutions,
            "convergence_windows": {name: list(window) for name, window in self._deltas.items()},
        }

    def _create_service(self, tier: str) -> GEPAService:
        """Create a GEPAService for the given tier using ProviderRegistry."""
        spec = self._get_tier_spec(tier)
        provider_name = spec.provider
        model = spec.model

        try:
            from victor.providers.registry import ProviderRegistry

            provider_cls = ProviderRegistry.get(provider_name)
            provider = provider_cls()
        except Exception as e:
            logger.warning(
                "Failed to create %s provider for GEPA %s tier: %s. " "Falling back to Ollama.",
                provider_name,
                tier,
                e,
            )
            try:
                from victor.providers.ollama_provider import OllamaProvider

                provider = OllamaProvider()
                model = "qwen3:8b"
            except Exception:
                raise RuntimeError(f"Cannot create any provider for GEPA {tier} tier") from e

        return GEPAService(
            provider=provider,
            model=model,
            tier=tier,
            max_prompt_chars=getattr(self._config, "max_prompt_chars", 1500),
            timeout_s=spec.timeout_s,
            max_tokens=spec.max_tokens,
        )

    def _get_tier_spec(self, tier: str) -> Any:
        """Get the GEPAModelSpec for a tier from config."""
        attr_map = {
            "economic": "economic_model",
            "balanced": "balanced_model",
            "performance": "performance_model",
        }
        attr_name = attr_map.get(tier, "balanced_model")
        return getattr(self._config, attr_name)
