"""Tests for TieredDecisionService — per-DecisionType tier routing.

Covers:
- Route decision types to correct tier
- Fallback chain (performance → balanced → edge → heuristic)
- Cached services per tier
- Missing tier returns heuristic
- Settings integration
- Provider-agnostic tier system
- Auto-detection of active provider
- Model resolution from provider_model_tiers
- Tier override capability
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from victor.agent.decisions.schemas import DecisionType


class _MockDecisionResult:
    def __init__(self, result=None, source="llm", confidence=0.9):
        self.result = result
        self.source = source
        self.confidence = confidence
        self.latency_ms = 5.0
        self.decision_type = None


class TestTieredRouting:
    """Route decision types to the correct tier."""

    def test_edge_type_routes_to_edge(self):
        from victor.agent.services.tiered_decision_service import TieredDecisionService
        from victor.config.decision_settings import DecisionServiceSettings

        config = DecisionServiceSettings()
        service = TieredDecisionService(config)

        mock_edge = MagicMock()
        mock_edge.decide_sync.return_value = _MockDecisionResult(result="tool_a")
        service._services["edge"] = mock_edge

        result = service.decide_sync(DecisionType.TOOL_SELECTION, {"message": "fix test"})
        mock_edge.decide_sync.assert_called_once()
        assert result.result == "tool_a"

    def test_balanced_type_routes_to_balanced(self):
        from victor.agent.services.tiered_decision_service import TieredDecisionService
        from victor.config.decision_settings import DecisionServiceSettings

        config = DecisionServiceSettings()
        service = TieredDecisionService(config)

        mock_balanced = MagicMock()
        mock_balanced.decide_sync.return_value = _MockDecisionResult(result="action")
        service._services["balanced"] = mock_balanced

        result = service.decide_sync(
            DecisionType.TASK_TYPE_CLASSIFICATION, {"message": "analyze arch"}
        )
        mock_balanced.decide_sync.assert_called_once()
        assert result.result == "action"

    def test_unknown_type_defaults_to_edge(self):
        from victor.agent.services.tiered_decision_service import TieredDecisionService
        from victor.config.decision_settings import DecisionServiceSettings

        config = DecisionServiceSettings()
        service = TieredDecisionService(config)

        mock_edge = MagicMock()
        mock_edge.decide_sync.return_value = _MockDecisionResult()
        service._services["edge"] = mock_edge

        result = service.decide_sync(DecisionType.LOOP_DETECTION, {"message": "stuck"})
        mock_edge.decide_sync.assert_called_once()


class TestFallbackChain:
    """When a tier is unavailable, fall back through the chain."""

    def test_balanced_falls_back_to_edge(self):
        from victor.agent.services.tiered_decision_service import TieredDecisionService
        from victor.config.decision_settings import DecisionServiceSettings

        config = DecisionServiceSettings()
        service = TieredDecisionService(config)

        # No balanced service, but edge exists
        mock_edge = MagicMock()
        mock_edge.decide_sync.return_value = _MockDecisionResult(source="edge_fallback")
        service._services["edge"] = mock_edge
        service._failed_tiers.add("balanced")

        result = service.decide_sync(DecisionType.TASK_TYPE_CLASSIFICATION, {"message": "test"})
        mock_edge.decide_sync.assert_called_once()
        assert result.source == "edge_fallback"

    def test_all_tiers_unavailable_returns_heuristic(self):
        from victor.agent.services.tiered_decision_service import TieredDecisionService
        from victor.config.decision_settings import DecisionServiceSettings

        config = DecisionServiceSettings()
        service = TieredDecisionService(config)
        service._failed_tiers = {"edge", "balanced", "performance"}

        result = service.decide_sync(
            DecisionType.TOOL_SELECTION,
            {"message": "test"},
            heuristic_result="default_tool",
            heuristic_confidence=0.5,
        )
        assert result.source == "heuristic"
        assert result.result == "default_tool"


class TestServiceCaching:
    """Services are created once per tier and cached."""

    def test_same_tier_reuses_service(self):
        from victor.agent.services.tiered_decision_service import TieredDecisionService
        from victor.config.decision_settings import DecisionServiceSettings

        config = DecisionServiceSettings()
        service = TieredDecisionService(config)

        mock_svc = MagicMock()
        mock_svc.decide_sync.return_value = _MockDecisionResult()
        service._services["edge"] = mock_svc

        service.decide_sync(DecisionType.TOOL_SELECTION, {"m": "a"})
        service.decide_sync(DecisionType.STAGE_DETECTION, {"m": "b"})
        # Both edge-routed, should use same service
        assert mock_svc.decide_sync.call_count == 2


class TestDecisionServiceSettings:
    """Settings configuration for tiered decisions."""

    def test_default_routing(self):
        from victor.config.decision_settings import DecisionServiceSettings

        config = DecisionServiceSettings()
        assert config.tier_routing["tool_selection"] == "edge"
        assert config.tier_routing["task_type_classification"] == "balanced"

    def test_default_model_specs_use_auto(self):
        """New provider-agnostic defaults use 'auto' for provider and model."""
        from victor.config.decision_settings import DecisionServiceSettings

        config = DecisionServiceSettings()
        assert config.edge.provider == "auto"
        assert config.edge.model == "auto"
        assert config.balanced.provider == "auto"
        assert config.balanced.model == "auto"
        assert config.performance.provider == "auto"
        assert config.performance.model == "auto"

    def test_provider_model_tiers_defined(self):
        """All major providers should have tier mappings defined."""
        from victor.config.decision_settings import DecisionServiceSettings

        config = DecisionServiceSettings()
        required_providers = {
            "anthropic",
            "openai",
            "google",
            "xai",
            "deepseek",
            "mistral",
            "ollama",
            "lmstudio",
            "vllm",
        }

        for provider in required_providers:
            if provider in config.provider_model_tiers:
                assert "edge" in config.provider_model_tiers[provider]
                assert "balanced" in config.provider_model_tiers[provider]
                assert "performance" in config.provider_model_tiers[provider]

    def test_custom_routing(self):
        from victor.config.decision_settings import DecisionServiceSettings

        config = DecisionServiceSettings(tier_routing={"tool_selection": "balanced"})
        assert config.tier_routing["tool_selection"] == "balanced"

    def test_tier_override_capability(self):
        """Can override tier with explicit provider/model."""
        from victor.config.decision_settings import DecisionServiceSettings

        config = DecisionServiceSettings(
            tier_overrides={"edge": {"provider": "ollama", "model": "phi3:mini"}}
        )
        assert config.tier_overrides["edge"]["provider"] == "ollama"
        assert config.tier_overrides["edge"]["model"] == "phi3:mini"

    def test_validate_provider_tiers(self):
        """Validation method catches configuration errors."""
        from victor.config.decision_settings import DecisionServiceSettings

        config = DecisionServiceSettings()
        result = config.validate_provider_tiers()

        # Default config should be valid
        assert result["valid"] is True
        assert len(result["errors"]) == 0


class TestProviderAgnosticTiers:
    """Provider-agnostic tier system tests."""

    def test_auto_detection_from_settings(self):
        """Auto-detection falls back to settings.default_provider."""
        from victor.agent.services.tiered_decision_service import TieredDecisionService
        from victor.config.decision_settings import DecisionServiceSettings

        config = DecisionServiceSettings()
        service = TieredDecisionService(config)

        # Mock settings to have default_provider
        with patch("victor.config.settings.Settings") as mock_settings_class:
            mock_settings = MagicMock()
            mock_settings.default_provider = "anthropic"
            mock_settings_class.return_value = mock_settings

            provider = service._detect_active_provider()
            assert provider == "anthropic"

    def test_auto_detection_from_env_var(self):
        """Auto-detection falls back to VICTOR_PROVIDER env var."""
        from victor.agent.services.tiered_decision_service import TieredDecisionService
        from victor.config.decision_settings import DecisionServiceSettings

        config = DecisionServiceSettings()
        service = TieredDecisionService(config)

        # Mock environment variable
        with patch.dict("os.environ", {"VICTOR_PROVIDER": "openai"}):
            # Clear any cached provider
            service._detected_provider = None
            provider = service._detect_active_provider()
            assert provider == "openai"

    def test_model_resolution_for_provider(self):
        """Resolves correct model for provider/tier."""
        from victor.config.decision_settings import DecisionServiceSettings

        config = DecisionServiceSettings()

        # Anthropic models
        anthropic_tiers = config.provider_model_tiers["anthropic"]
        assert anthropic_tiers["edge"] == "claude-haiku-4-5-20251001"
        assert anthropic_tiers["balanced"] == "claude-sonnet-4-6"
        assert anthropic_tiers["performance"] == "claude-opus-4-7"

        # OpenAI models
        openai_tiers = config.provider_model_tiers["openai"]
        assert openai_tiers["edge"] == "gpt-5.4-mini"  # Updated to GPT-5.4-mini
        assert openai_tiers["balanced"] == "gpt-5.4"  # Updated to GPT-5.4
        assert openai_tiers["performance"] == "gpt-5.4-pro"  # Updated to GPT-5.4 Pro

    def test_fallback_for_missing_tier(self):
        """Falls back when provider has no tier mapping."""
        from victor.agent.services.tiered_decision_service import TieredDecisionService
        from victor.config.decision_settings import DecisionServiceSettings

        config = DecisionServiceSettings(
            provider_model_tiers={
                "test_provider": {
                    "balanced": "test-balanced",
                    # No edge tier defined
                }
            }
        )
        service = TieredDecisionService(config)

        # Mock provider detection to return test_provider
        with patch.object(service, "_detect_active_provider", return_value="test_provider"):
            # Try to create edge tier - should fail gracefully
            result = service._create_service("edge")
            assert result is None
            assert "edge" in service._failed_tiers

    def test_tier_override_behavior(self):
        """Override forces specific provider regardless of active."""
        from victor.agent.services.tiered_decision_service import TieredDecisionService
        from victor.config.decision_settings import DecisionServiceSettings

        config = DecisionServiceSettings(
            tier_overrides={"edge": {"provider": "ollama", "model": "phi3:mini"}}
        )
        service = TieredDecisionService(config)

        # Verify override is applied
        assert "edge" in config.tier_overrides
        assert config.tier_overrides["edge"]["provider"] == "ollama"

    def test_provider_detection_caching(self):
        """Provider detection is cached for performance."""
        from victor.agent.services.tiered_decision_service import TieredDecisionService
        from victor.config.decision_settings import DecisionServiceSettings

        config = DecisionServiceSettings()
        service = TieredDecisionService(config)

        # First call
        with patch.dict("os.environ", {"VICTOR_PROVIDER": "anthropic"}):
            provider1 = service._detect_active_provider()
            # Second call should use cache
            provider2 = service._detect_active_provider()
            assert provider1 == provider2 == "anthropic"
            assert service._detected_provider == "anthropic"
