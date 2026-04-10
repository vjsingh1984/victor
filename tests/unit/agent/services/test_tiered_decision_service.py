"""Tests for TieredDecisionService — per-DecisionType tier routing.

Covers:
- Route decision types to correct tier
- Fallback chain (performance → balanced → edge → heuristic)
- Cached services per tier
- Missing tier returns heuristic
- Settings integration
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

    def test_default_model_specs(self):
        from victor.config.decision_settings import DecisionServiceSettings

        config = DecisionServiceSettings()
        assert config.edge.provider == "ollama"
        assert config.balanced.provider == "deepseek"
        assert config.performance.provider == "anthropic"

    def test_custom_routing(self):
        from victor.config.decision_settings import DecisionServiceSettings

        config = DecisionServiceSettings(tier_routing={"tool_selection": "balanced"})
        assert config.tier_routing["tool_selection"] == "balanced"
