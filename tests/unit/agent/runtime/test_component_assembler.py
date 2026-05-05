from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from victor.agent.runtime.component_assembler import ComponentAssembler


def test_assemble_tools_prefers_canonical_registrar_registration_surface():
    registrar = MagicMock()
    registrar.register_default_tools = MagicMock()
    registrar._register_tool_dependencies = MagicMock()
    registrar._load_tool_configurations = MagicMock()

    tools = MagicMock()
    tools.register_before_hook = MagicMock()

    factory = MagicMock()
    factory.create_tool_cache.return_value = MagicMock()
    factory.create_tool_dependency_graph.return_value = MagicMock()
    factory.create_code_execution_manager.return_value = MagicMock()
    factory.create_tool_registry.return_value = tools
    factory.create_tool_registrar.return_value = registrar
    factory.initialize_plugin_system.return_value = None
    factory.create_argument_normalizer.return_value = MagicMock()
    factory.create_middleware_chain.return_value = (MagicMock(), MagicMock())
    factory.create_safety_checker.return_value = MagicMock()
    factory.create_auto_committer.return_value = MagicMock()
    factory.create_tool_executor.return_value = MagicMock()
    factory.create_parallel_executor.return_value = MagicMock()
    factory.create_response_completer.return_value = MagicMock()
    factory.setup_semantic_selection.return_value = (False, None)
    factory.create_semantic_selector.return_value = MagicMock()
    factory.create_unified_tracker.return_value = SimpleNamespace(unique_resources=set())
    factory.create_tool_selector.return_value = MagicMock()
    factory.create_tool_access_controller.return_value = MagicMock()
    factory.create_budget_manager.return_value = MagicMock()

    orchestrator = SimpleNamespace(
        _factory=factory,
        _create_background_task=MagicMock(),
        _log_tool_call=MagicMock(),
        _record_tool_selection=MagicMock(),
        tool_calling_caps=MagicMock(),
        conversation_state=MagicMock(),
        model="test-model",
        provider_name="test-provider",
        tool_selection={},
    )

    ComponentAssembler.assemble_tools(orchestrator, provider=MagicMock(), model="test-model")

    registrar.register_default_tools.assert_called_once_with()


def test_assemble_conversation_does_not_wire_removed_prompt_compatibility_into_runtime():
    factory = SimpleNamespace(
        create_conversation_controller=MagicMock(return_value=MagicMock()),
        create_hierarchical_compaction_manager=MagicMock(return_value=MagicMock()),
        create_session_ledger=MagicMock(return_value=MagicMock()),
        create_lifecycle_manager=MagicMock(return_value=MagicMock()),
        create_tool_deduplication_tracker=MagicMock(return_value=MagicMock()),
        create_tool_pipeline=MagicMock(return_value=MagicMock()),
        create_streaming_controller=MagicMock(return_value=MagicMock()),
        create_streaming_coordinator=MagicMock(return_value=MagicMock()),
        create_streaming_chat_handler=MagicMock(return_value=MagicMock()),
    )

    container = MagicMock()
    container.register_or_replace = MagicMock()

    orchestrator = SimpleNamespace(
        _factory=factory,
        conversation=MagicMock(),
        conversation_state=MagicMock(),
        memory_manager=MagicMock(),
        _memory_session_id="session-1",
        _system_prompt="base prompt",
        reminder_manager=MagicMock(),
        _metrics_coordinator=SimpleNamespace(metrics_collector=MagicMock()),
        _context_compactor=MagicMock(),
        _sequence_tracker=MagicMock(),
        _usage_analytics=MagicMock(),
        _reminder_manager=MagicMock(),
        _session_service=MagicMock(),
        tools=MagicMock(),
        tool_executor=MagicMock(),
        tool_budget=MagicMock(),
        tool_cache=MagicMock(),
        argument_normalizer=MagicMock(),
        _on_tool_start_callback=MagicMock(),
        _on_tool_complete_callback=MagicMock(),
        _middleware_chain=MagicMock(),
        search_router=MagicMock(),
        _container=container,
        _pending_semantic_cache=None,
        streaming_metrics_collector=MagicMock(),
        _on_streaming_session_complete=MagicMock(),
        prompt_builder=MagicMock(),
        _get_model_context_window=lambda: 65536,
        provider_name="anthropic",
        model="claude-3-sonnet",
        mode_controller=MagicMock(),
        _session_id="test-session",
    )

    analyzer = MagicMock()

    with patch("victor.agent.task_analyzer.get_task_analyzer", return_value=analyzer):
        ComponentAssembler.assemble_conversation(
            orchestrator,
            provider=MagicMock(),
            model="claude-3-sonnet",
        )

    assert orchestrator._task_analyzer is analyzer
    analyzer.set_runtime_subject.assert_called_once_with(orchestrator)
    assert not hasattr(factory, "create_system_prompt_coordinator")
    assert not hasattr(factory, "create_prompt_runtime_support")
    assert not hasattr(orchestrator, "_system_prompt_coordinator")
    assert not hasattr(orchestrator, "_prompt_runtime_support")
