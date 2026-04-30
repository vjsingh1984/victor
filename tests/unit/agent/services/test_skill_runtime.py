from types import SimpleNamespace
from unittest.mock import MagicMock

from victor.agent.services.orchestrator_protocol_adapter import OrchestratorProtocolAdapter
from victor.agent.services.skill_runtime import SkillRuntime


def test_skill_runtime_applies_single_match_and_records_analytics():
    skill = SimpleNamespace(name="debug")
    matcher = MagicMock()
    matcher._initialized = True
    matcher.match_multiple_sync.return_value = [(skill, 0.912)]
    analytics = MagicMock()
    host = SimpleNamespace(
        clear_active_skills=MagicMock(),
        inject_skill=MagicMock(),
        inject_skills=MagicMock(),
        _skill_matcher=matcher,
        _skill_auto_disabled=False,
        _manual_skill_active=False,
        _skill_analytics=analytics,
        _last_skill_match_info=None,
    )
    runtime = SkillRuntime(OrchestratorProtocolAdapter(host))

    runtime.apply_skill_for_turn("hello")

    host.clear_active_skills.assert_called_once_with()
    host.inject_skill.assert_called_once_with(skill)
    assert host._last_skill_match_info == {
        "auto_skill": "debug",
        "auto_skill_score": 0.91,
    }
    analytics.record_selection.assert_called_once_with("debug", 0.912)
    host.inject_skills.assert_not_called()


def test_skill_runtime_applies_multiple_matches_and_records_analytics():
    matches = [
        (SimpleNamespace(name="debug"), 0.9),
        (SimpleNamespace(name="refactor"), 0.876),
    ]
    matcher = MagicMock()
    matcher._initialized = True
    matcher.match_multiple_sync.return_value = matches
    analytics = MagicMock()
    host = SimpleNamespace(
        clear_active_skills=MagicMock(),
        inject_skill=MagicMock(),
        inject_skills=MagicMock(),
        _skill_matcher=matcher,
        _skill_auto_disabled=False,
        _manual_skill_active=False,
        _skill_analytics=analytics,
        _last_skill_match_info=None,
    )
    runtime = SkillRuntime(OrchestratorProtocolAdapter(host))

    runtime.apply_skill_for_turn("hello")

    host.inject_skills.assert_called_once_with(matches)
    assert host._last_skill_match_info == {
        "auto_skills": [
            {"name": "debug", "score": 0.9},
            {"name": "refactor", "score": 0.88},
        ],
    }
    analytics.record_multi_selection.assert_called_once_with([("debug", 0.9), ("refactor", 0.876)])
    host.inject_skill.assert_not_called()


def test_skill_runtime_clears_last_match_info_on_miss():
    matcher = MagicMock()
    matcher._initialized = True
    matcher.match_multiple_sync.return_value = []
    analytics = MagicMock()
    host = SimpleNamespace(
        clear_active_skills=MagicMock(),
        inject_skill=MagicMock(),
        inject_skills=MagicMock(),
        _skill_matcher=matcher,
        _skill_auto_disabled=False,
        _manual_skill_active=False,
        _skill_analytics=analytics,
        _last_skill_match_info={"auto_skill": "stale"},
    )
    runtime = SkillRuntime(OrchestratorProtocolAdapter(host))

    runtime.apply_skill_for_turn("hello")

    assert host._last_skill_match_info is None
    analytics.record_miss.assert_called_once_with()
