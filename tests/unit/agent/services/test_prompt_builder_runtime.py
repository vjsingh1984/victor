from types import SimpleNamespace
from unittest.mock import MagicMock

from victor.agent.services.orchestrator_protocol_adapter import OrchestratorProtocolAdapter
from victor.agent.services.prompt_builder_runtime import PromptBuilderRuntime


def test_sync_prompt_builder_runtime_state_updates_builder_and_invalidates_cache():
    builder = SimpleNamespace(
        available_tools=["read"],
        mode_prompt_addition="old mode",
        invalidate_cache=MagicMock(),
    )
    host = SimpleNamespace(
        prompt_builder=builder,
        get_enabled_tools=MagicMock(return_value={"write", "read"}),
        get_mode_system_prompt=MagicMock(return_value="new mode"),
    )
    runtime = PromptBuilderRuntime(OrchestratorProtocolAdapter(host))

    runtime.sync_prompt_builder_runtime_state()

    assert builder.available_tools == ["read", "write"]
    assert builder.mode_prompt_addition == "new mode"
    builder.invalidate_cache.assert_called_once_with()


def test_sync_prompt_builder_runtime_state_clears_stale_values_on_errors():
    builder = SimpleNamespace(
        available_tools=["read"],
        mode_prompt_addition="old mode",
        invalidate_cache=MagicMock(),
    )
    host = SimpleNamespace(
        prompt_builder=builder,
        get_enabled_tools=MagicMock(side_effect=RuntimeError("tools unavailable")),
        get_mode_system_prompt=MagicMock(side_effect=RuntimeError("mode unavailable")),
    )
    runtime = PromptBuilderRuntime(OrchestratorProtocolAdapter(host))

    runtime.sync_prompt_builder_runtime_state()

    assert builder.available_tools == []
    assert builder.mode_prompt_addition == ""
    builder.invalidate_cache.assert_called_once_with()


def test_build_system_prompt_fallback_prefers_runtime_support():
    runtime_support = MagicMock()
    prompt_orchestrator = MagicMock()
    prompt_orchestrator.build_system_prompt.return_value = "runtime prompt"
    host = SimpleNamespace(
        prompt_builder=SimpleNamespace(
            available_tools=[],
            mode_prompt_addition="",
            invalidate_cache=MagicMock(),
            task_type="edit",
        ),
        get_enabled_tools=MagicMock(return_value=set()),
        get_mode_system_prompt=MagicMock(return_value=""),
        _prompt_runtime_support=runtime_support,
        _system_prompt_coordinator=None,
        _prompt_orchestrator=prompt_orchestrator,
        provider_name="anthropic",
        model="claude",
        _get_model_context_window=lambda: 65536,
    )
    runtime = PromptBuilderRuntime(OrchestratorProtocolAdapter(host))

    result = runtime.build_system_prompt_fallback()

    assert result == "runtime prompt"
    prompt_orchestrator.build_system_prompt.assert_called_once_with(
        builder_type="legacy",
        provider="anthropic",
        model="claude",
        task_type="edit",
        builder=host.prompt_builder,
        get_context_window=host._get_model_context_window,
        on_prompt_built=runtime_support._emit_prompt_used_event,
    )


def test_compose_system_prompt_appends_project_context_when_present():
    host = SimpleNamespace(
        project_context=SimpleNamespace(
            content="repo context",
            get_system_prompt_addition=MagicMock(return_value="project addition"),
        )
    )
    runtime = PromptBuilderRuntime(OrchestratorProtocolAdapter(host))

    result = runtime.compose_system_prompt("base prompt")

    assert result == "base prompt\n\nproject addition"


def test_update_system_prompt_for_query_updates_runtime_and_conversation():
    builder = SimpleNamespace(
        query_classification=None,
        invalidate_cache=MagicMock(),
    )
    conversation = SimpleNamespace(
        system_prompt="old prompt",
        _system_added=True,
        _messages=[SimpleNamespace(role="system", content="old prompt")],
    )
    project_context = SimpleNamespace(
        content="repo context",
        get_system_prompt_addition=MagicMock(return_value="project addition"),
    )
    classification = object()
    host = SimpleNamespace(
        prompt_builder=builder,
        build_system_prompt=MagicMock(return_value="base prompt"),
        project_context=project_context,
        conversation=conversation,
        _prompt_pipeline=None,
        _system_prompt_frozen=False,
        _kv_optimization_enabled=True,
        _system_prompt="",
    )
    runtime = PromptBuilderRuntime(OrchestratorProtocolAdapter(host))

    runtime.update_system_prompt_for_query(query_classification=classification)

    assert builder.query_classification is classification
    builder.invalidate_cache.assert_called_once_with()
    host.build_system_prompt.assert_called_once_with()
    assert host._system_prompt == "base prompt\n\nproject addition"
    assert host._system_prompt_frozen is True
    assert conversation.system_prompt == host._system_prompt
    assert conversation._messages[0].role == "system"
    assert conversation._messages[0].content == host._system_prompt


def test_update_system_prompt_for_query_skips_when_prompt_is_frozen():
    builder = SimpleNamespace(invalidate_cache=MagicMock())
    host = SimpleNamespace(
        prompt_builder=builder,
        build_system_prompt=MagicMock(),
        _prompt_pipeline=SimpleNamespace(is_frozen=True),
        _kv_optimization_enabled=True,
    )
    runtime = PromptBuilderRuntime(OrchestratorProtocolAdapter(host))

    runtime.update_system_prompt_for_query(query_classification="coding")

    builder.invalidate_cache.assert_not_called()
    host.build_system_prompt.assert_not_called()


def test_refresh_system_prompt_preserves_existing_classification():
    builder = SimpleNamespace(
        query_classification="existing",
        invalidate_cache=MagicMock(),
    )
    host = SimpleNamespace(
        prompt_builder=builder,
        _prompt_pipeline=SimpleNamespace(unfreeze=MagicMock()),
        _system_prompt_frozen=True,
        _session_tools=["read"],
    )
    runtime = PromptBuilderRuntime(OrchestratorProtocolAdapter(host))
    runtime.update_system_prompt_for_query = MagicMock()

    runtime.refresh_system_prompt()

    host._prompt_pipeline.unfreeze.assert_called_once_with()
    builder.invalidate_cache.assert_called_once_with()
    assert host._system_prompt_frozen is False
    assert host._session_tools is None
    runtime.update_system_prompt_for_query.assert_called_once_with(query_classification="existing")


def test_get_set_and_append_system_prompt_delegate_to_builder():
    builder = SimpleNamespace(
        build=MagicMock(return_value="base prompt"),
        set_custom_prompt=MagicMock(),
    )
    host = SimpleNamespace(prompt_builder=builder)
    runtime = PromptBuilderRuntime(OrchestratorProtocolAdapter(host))

    assert runtime.get_system_prompt() == "base prompt"

    runtime.set_system_prompt("custom prompt")
    runtime.append_to_system_prompt("extra guidance")

    builder.set_custom_prompt.assert_any_call("custom prompt")
    builder.set_custom_prompt.assert_any_call("base prompt\n\nextra guidance")


def test_sync_conversation_system_prompt_updates_live_system_message():
    conversation = SimpleNamespace(
        system_prompt="old prompt",
        _system_added=True,
        _messages=[SimpleNamespace(role="system", content="old prompt")],
    )
    host = SimpleNamespace(
        conversation=conversation,
        _system_prompt="new prompt",
    )
    runtime = PromptBuilderRuntime(OrchestratorProtocolAdapter(host))

    runtime.sync_conversation_system_prompt()

    assert conversation.system_prompt == "new prompt"
    assert conversation._messages[0].role == "system"
    assert conversation._messages[0].content == "new prompt"
