from types import SimpleNamespace
from unittest.mock import MagicMock

from victor.agent.services.orchestrator_protocol_adapter import OrchestratorProtocolAdapter
from victor.agent.services.session_runtime import SessionRuntime


def test_session_runtime_records_prompt_optimization_metadata_and_updates_session():
    metadata = {
        "entries": [{"provider": "anthropic", "section_name": "GROUNDING_RULES"}],
        "by_section": {"GROUNDING_RULES": {"provider": "anthropic"}},
        "prompt_overlays": [
            {
                "name": "planner.research_step",
                "placement": "turn_prefix",
                "content": "Do not persist prompt content in metadata.",
            }
        ],
    }
    expected = {
        "entries": [{"provider": "anthropic", "section_name": "GROUNDING_RULES"}],
        "by_section": {"GROUNDING_RULES": {"provider": "anthropic"}},
        "prompt_overlays": [
            {
                "name": "planner.research_step",
                "placement": "turn_prefix",
            }
        ],
    }
    session_service = MagicMock()
    session_service._current_session = MagicMock()
    host = SimpleNamespace(
        _session_service=session_service,
        _active_prompt_optimization_metadata=None,
        usage_logger=MagicMock(),
    )
    runtime = SessionRuntime(OrchestratorProtocolAdapter(host))

    runtime.record_prompt_optimization_metadata(
        SimpleNamespace(prompt_optimization_metadata=metadata)
    )

    assert host._active_prompt_optimization_metadata == expected
    session_service.update_session_metadata.assert_called_once_with(
        {"prompt_optimization": expected}
    )
    host.usage_logger.log_event.assert_called_once_with(
        "prompt_overlays.active",
        {
            "prompt_overlays": expected["prompt_overlays"],
            "prompt_overlay_count": 1,
        },
    )


def test_session_runtime_does_not_emit_overlay_event_without_overlays():
    metadata = {
        "entries": [],
        "by_section": {},
    }
    session_service = MagicMock()
    session_service._current_session = MagicMock()
    host = SimpleNamespace(
        _session_service=session_service,
        _active_prompt_optimization_metadata=None,
        usage_logger=MagicMock(),
    )
    runtime = SessionRuntime(OrchestratorProtocolAdapter(host))

    runtime.record_prompt_optimization_metadata(
        SimpleNamespace(prompt_optimization_metadata=metadata)
    )

    host.usage_logger.log_event.assert_not_called()


def test_session_runtime_skips_session_update_without_active_session():
    session_service = MagicMock()
    session_service._current_session = None
    host = SimpleNamespace(
        _session_service=session_service,
        _active_prompt_optimization_metadata=None,
    )
    runtime = SessionRuntime(OrchestratorProtocolAdapter(host))

    runtime.record_prompt_optimization_metadata(
        SimpleNamespace(prompt_optimization_metadata={"entries": [], "by_section": {}})
    )

    session_service.update_session_metadata.assert_not_called()


def test_session_runtime_get_active_prompt_optimization_metadata_returns_default():
    host = SimpleNamespace(_active_prompt_optimization_metadata=None)
    runtime = SessionRuntime(OrchestratorProtocolAdapter(host))

    assert runtime.get_active_prompt_optimization_metadata() == {
        "entries": [],
        "by_section": {},
        "prompt_overlays": [],
    }


def test_session_runtime_get_active_prompt_optimization_metadata_preserves_overlay_names():
    host = SimpleNamespace(
        _active_prompt_optimization_metadata={
            "entries": [],
            "by_section": {},
            "prompt_overlays": [
                {
                    "name": "planner.plan_generation",
                    "placement": "turn_prefix",
                    "content": "not returned",
                },
                {
                    "name": "",
                    "placement": "turn_prefix",
                },
            ],
        }
    )
    runtime = SessionRuntime(OrchestratorProtocolAdapter(host))

    assert runtime.get_active_prompt_optimization_metadata() == {
        "entries": [],
        "by_section": {},
        "prompt_overlays": [
            {
                "name": "planner.plan_generation",
                "placement": "turn_prefix",
            }
        ],
    }
