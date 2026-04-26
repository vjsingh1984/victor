# Copyright 2026 Vijaykumar Singh <singhvijay@users.noreply.github.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from unittest.mock import MagicMock

from victor.agent.service_provider import OrchestratorServiceProvider
from victor.agent.services.protocols import PromptRuntimeProtocol
from victor.core.container import ServiceContainer


def _mock_settings():
    settings = MagicMock()
    settings.search.unified_embedding_model = "test-model"
    settings.enable_observability = True
    settings.max_conversation_history = 50
    settings.grounding_mode = "minimal"
    settings.enable_task_hints = True
    settings.max_context_tokens = 2000
    settings.base_identity = "You are Victor, an AI coding assistant."
    return settings


def test_prompt_runtime_adapter_satisfies_protocol_and_builds_prompt():
    from victor.agent.services.prompt_runtime import (
        PromptRuntimeAdapter,
        PromptRuntimeConfig,
        PromptRuntimeContext,
    )

    adapter = PromptRuntimeAdapter(
        config=PromptRuntimeConfig(default_grounding_mode="minimal"),
        base_identity="You are Victor, an AI coding assistant.",
    )

    assert isinstance(adapter, PromptRuntimeProtocol)

    adapter.add_task_hint("edit", "Read the target file before editing")
    adapter.add_section("custom", "Custom section content")
    adapter.set_grounding_mode("extended")

    prompt = adapter.build_system_prompt(
        PromptRuntimeContext(
            message="Fix the bug",
            task_type="edit",
            additional_context={"project": "victor"},
        )
    )

    assert "Victor" in prompt
    assert "Read the target file before editing" in prompt
    assert "Custom section content" in prompt
    assert "project: victor" in prompt.lower()
    assert adapter._config.default_grounding_mode == "extended"


def test_prompt_runtime_protocol_resolves_to_canonical_adapter():
    from victor.agent.services.prompt_runtime import PromptRuntimeAdapter

    container = ServiceContainer()
    provider = OrchestratorServiceProvider(_mock_settings())
    provider.register_singleton_services(container)

    with container.create_scope() as scope:
        runtime = scope.get(PromptRuntimeProtocol)

    assert isinstance(runtime, PromptRuntimeAdapter)
