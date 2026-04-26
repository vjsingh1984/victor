from victor.framework.request_scope_heuristics import (
    contains_keyword_marker,
    conversation_history_has_explicit_target,
    has_ambiguous_target_reference,
)


def test_contains_keyword_marker_uses_whole_word_matching():
    assert contains_keyword_marker("please implement this change", ("implement", "change")) is True
    assert (
        contains_keyword_marker("clean code implementations and changes", ("implement", "change"))
        is False
    )


def test_has_ambiguous_target_reference_ignores_relative_clause_that():
    assert (
        has_ambiguous_target_reference(
            "identify deprecated call sites that should be modified first"
        )
        is False
    )


def test_has_ambiguous_target_reference_detects_true_follow_up_reference():
    assert has_ambiguous_target_reference("fix that first and add tests") is True


def test_conversation_history_has_explicit_target_detects_recent_file_reference():
    history = [
        {"role": "assistant", "content": "Start with victor/agent/services/tool_compat.py first."}
    ]

    assert conversation_history_has_explicit_target(history) is True


def test_conversation_history_has_explicit_target_returns_false_without_scope():
    history = [{"role": "assistant", "content": "Start with the biggest migration first."}]

    assert conversation_history_has_explicit_target(history) is False
