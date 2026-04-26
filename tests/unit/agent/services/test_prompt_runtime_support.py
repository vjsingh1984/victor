# Copyright 2026 Vijaykumar Singh <singhvijay@users.noreply.github.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

from unittest.mock import MagicMock, patch

from victor.agent.services.prompt_runtime_support import PromptRuntimeSupport


def test_build_system_prompt_large_context_includes_parallel_budget():
    builder = MagicMock()
    builder.build.return_value = "You are a helpful assistant."

    runtime = PromptRuntimeSupport(
        prompt_builder=builder,
        get_context_window=lambda: 65536,
        provider_name="anthropic",
        model_name="claude-3-sonnet",
        get_tools=lambda: MagicMock(),
        get_mode_controller=lambda: None,
        task_analyzer=MagicMock(),
        session_id="test-session",
    )

    result = runtime.build_system_prompt()

    assert "PARALLEL READ BUDGET" in result
    assert "You are a helpful assistant." in result


def test_resolve_shell_variant_and_task_classification_delegate_to_runtime_dependencies():
    task_analyzer = MagicMock()
    task_analyzer.classify_task_keywords.return_value = {
        "task_type": "general",
        "confidence": 0.5,
    }
    task_analyzer.classify_task_with_context.return_value = {
        "task_type": "coding",
        "confidence": 0.8,
    }
    tools = MagicMock()
    runtime = PromptRuntimeSupport(
        prompt_builder=MagicMock(),
        get_context_window=lambda: 8192,
        provider_name="anthropic",
        model_name="claude-3-sonnet",
        get_tools=lambda: tools,
        get_mode_controller=lambda: None,
        task_analyzer=task_analyzer,
    )

    with patch("victor.agent.shell_resolver.resolve_shell_variant", return_value="shell"):
        assert runtime.resolve_shell_variant("bash") == "shell"

    assert runtime.classify_task_keywords("fix the bug")["task_type"] == "general"
    assert runtime.classify_task_with_context("write a function", [])["task_type"] == "coding"
