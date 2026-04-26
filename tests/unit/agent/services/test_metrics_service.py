# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import SimpleNamespace
from unittest.mock import MagicMock

from victor.agent.services.metrics_service import MetricsCoordinator


def _make_metrics_coordinator() -> MetricsCoordinator:
    return MetricsCoordinator(
        metrics_collector=MagicMock(),
        session_cost_tracker=MagicMock(),
        cumulative_token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )


def test_emit_tool_strategy_event_uses_provider_capability_for_metrics_labels(caplog):
    coordinator = _make_metrics_coordinator()
    provider = MagicMock()
    provider.name = "openai"
    provider.supports_prompt_caching.return_value = True
    tools = [SimpleNamespace(name="read"), SimpleNamespace(name="git_diff")]

    with caplog.at_level("DEBUG"):
        coordinator.emit_tool_strategy_event(
            strategy="session_lock",
            tool_count=2,
            tool_tokens=512,
            context_window=32768,
            provider=provider,
            model="test-model",
            reason="cache_discount_or_large_context",
            tools=tools,
            v2_enabled=True,
        )

    assert "category=caching" in caplog.text
    assert 'victor_tool_strategy_v2_enabled{provider="openai"} 1' in caplog.text
    assert 'victor_tool_tier_count{provider="openai",tier="FULL"} 1' in caplog.text


def test_emit_tool_strategy_event_falls_back_to_provider_name_for_strings(caplog):
    coordinator = _make_metrics_coordinator()

    with caplog.at_level("INFO"):
        coordinator.emit_tool_strategy_event(
            strategy="semantic_selection",
            tool_count=1,
            tool_tokens=64,
            context_window=8192,
            provider="ollama",
            model="test-model",
            reason="small_context_window",
            tools=[SimpleNamespace(name="read")],
            v2_enabled=False,
        )

    assert "provider=ollama, category=local" in caplog.text
