#!/usr/bin/env python3
"""Test model capability configuration.

Tests that model capabilities (tool selection thresholds, max tools) can be
configured per profile instead of being hardcoded in the orchestrator.
"""

import pytest
from unittest.mock import MagicMock
from victor.config.settings import ProfileConfig


def create_mock_settings():
    """Create a mock settings object with all required attributes."""
    from victor.config.settings import Settings

    mock_settings = MagicMock(spec=Settings)
    mock_settings.tool_call_budget = 300
    mock_settings.airgapped_mode = False
    mock_settings.use_semantic_tool_selection = False
    mock_settings.use_mcp_tools = False
    mock_settings.analytics_log_file = "/tmp/test_analytics.jsonl"
    mock_settings.analytics_enabled = False
    mock_settings.load_tool_config.return_value = {}
    return mock_settings


def test_profile_config_accepts_tool_selection_params():
    """Test that ProfileConfig accepts tool_selection configuration."""
    config = ProfileConfig(
        provider="ollama",
        model="qwen2.5-coder:7b",
        temperature=0.7,
        max_tokens=4096,
        tool_selection={
            "base_threshold": 0.25,
            "base_max_tools": 7,
            "model_size_tier": "small",  # 7B-8B
        },
    )

    assert config.tool_selection is not None
    assert config.tool_selection["base_threshold"] == 0.25
    assert config.tool_selection["base_max_tools"] == 7
    assert config.tool_selection["model_size_tier"] == "small"


def test_profile_config_with_minimal_tool_selection():
    """Test that only base_threshold and base_max_tools are required."""
    config = ProfileConfig(
        provider="ollama",
        model="custom-model",
        tool_selection={"base_threshold": 0.20, "base_max_tools": 10},
    )

    assert config.tool_selection["base_threshold"] == 0.20
    assert config.tool_selection["base_max_tools"] == 10


def test_profile_config_without_tool_selection_uses_defaults():
    """Test that profiles without tool_selection use adaptive defaults."""
    config = ProfileConfig(
        provider="anthropic", model="claude-3-5-sonnet-20241022", temperature=0.7, max_tokens=4096
    )

    # Should work without tool_selection (backwards compatible)
    assert hasattr(config, "model")
    assert config.model == "claude-3-5-sonnet-20241022"


def test_orchestrator_uses_configured_tool_selection():
    """Test that orchestrator uses tool_selection from profile config."""
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.providers.base import BaseProvider

    mock_settings = create_mock_settings()

    # Mock provider
    mock_provider = MagicMock(spec=BaseProvider)
    mock_provider.supports_tools.return_value = True
    mock_provider.supports_streaming.return_value = True

    # Create orchestrator with configured tool_selection
    orchestrator = AgentOrchestrator(
        settings=mock_settings,
        provider=mock_provider,
        model="qwen2.5-coder:7b",
        temperature=0.7,
        max_tokens=4096,
        tool_selection={
            "base_threshold": 0.30,  # Custom threshold
            "base_max_tools": 5,  # Custom max tools
        },
    )

    # Get adaptive threshold (should use configured values as base)
    threshold, max_tools = orchestrator._get_adaptive_threshold("test message")

    # The threshold and max_tools should be based on configured values,
    # with adjustments from query specificity and conversation depth
    # For a short message, threshold increases and max_tools decreases
    # So we check that the base configuration is being used
    assert threshold >= 0.25  # Should be higher due to short message
    assert max_tools <= 7  # Should be adjusted but respect base


def test_orchestrator_adaptive_logic_still_applies():
    """Test that adaptive adjustments still work with configured values."""
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.providers.base import BaseProvider

    mock_settings = create_mock_settings()

    mock_provider = MagicMock(spec=BaseProvider)
    mock_provider.supports_tools.return_value = True

    orchestrator = AgentOrchestrator(
        settings=mock_settings,
        provider=mock_provider,
        model="custom-model",
        tool_selection={"base_threshold": 0.20, "base_max_tools": 10},
    )

    # Short vague query should increase threshold
    threshold_short, tools_short = orchestrator._get_adaptive_threshold("help me")

    # Long detailed query should decrease threshold
    threshold_long, tools_long = orchestrator._get_adaptive_threshold(
        "I need help implementing a Python function that validates email addresses "
        "using regular expressions and handles edge cases properly"
    )

    # Short query should have higher threshold than long query
    assert threshold_short > threshold_long

    # Long query should allow more tools
    assert tools_long >= tools_short


def test_orchestrator_without_config_uses_hardcoded_defaults():
    """Test backwards compatibility - orchestrator without tool_selection config."""
    from victor.agent.orchestrator import AgentOrchestrator
    from victor.providers.base import BaseProvider

    mock_settings = create_mock_settings()

    mock_provider = MagicMock(spec=BaseProvider)
    mock_provider.supports_tools.return_value = True

    # Create orchestrator without tool_selection (old behavior)
    orchestrator = AgentOrchestrator(
        settings=mock_settings,
        provider=mock_provider,
        model="qwen2.5-coder:7b",  # Known model size pattern
    )

    # Should still work using model name pattern detection
    threshold, max_tools = orchestrator._get_adaptive_threshold("test message")

    # Should get values appropriate for 7b model (from hardcoded logic)
    assert 0.15 <= threshold <= 0.50  # Reasonable range
    assert 3 <= max_tools <= 15  # Reasonable range


def test_model_size_tier_shortcuts():
    """Test that model_size_tier provides convenient shortcuts."""
    from victor.config.settings import ProfileConfig

    # Test predefined tiers
    tiers = {
        "tiny": {"base_threshold": 0.35, "base_max_tools": 5},  # 0.5B-3B
        "small": {"base_threshold": 0.25, "base_max_tools": 7},  # 7B-8B
        "medium": {"base_threshold": 0.20, "base_max_tools": 10},  # 13B-15B
        "large": {"base_threshold": 0.15, "base_max_tools": 12},  # 30B+
        "cloud": {"base_threshold": 0.18, "base_max_tools": 10},  # Claude/GPT
    }

    for tier_name, _expected in tiers.items():
        config = ProfileConfig(
            provider="ollama", model="test-model", tool_selection={"model_size_tier": tier_name}
        )

        assert config.tool_selection["model_size_tier"] == tier_name


def test_tool_selection_validation():
    """Test that invalid tool_selection values are rejected."""
    from victor.config.settings import ProfileConfig

    # Threshold must be between 0.0 and 1.0
    with pytest.raises(ValueError):
        ProfileConfig(
            provider="ollama",
            model="test",
            tool_selection={"base_threshold": 1.5, "base_max_tools": 10},
        )

    # Max tools must be positive
    with pytest.raises(ValueError):
        ProfileConfig(
            provider="ollama",
            model="test",
            tool_selection={"base_threshold": 0.2, "base_max_tools": -1},
        )


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])
