from victor.framework.task.direct_response import (
    DirectResponseOutputState,
    classify_direct_response_prompt,
    extract_exact_response_literal,
    has_codebase_context,
    normalize_direct_response_output,
)


def test_exact_direct_response_prompt_is_detected():
    result = classify_direct_response_prompt("Reply with exactly READY")

    assert result.is_direct_response is True
    assert result.is_exact_response is True
    assert result.reason == "exact_response"


def test_general_knowledge_prompt_is_detected():
    result = classify_direct_response_prompt("What is Python?")

    assert result.is_direct_response is True
    assert result.is_exact_response is False
    assert result.reason == "short_question"


def test_codebase_question_is_not_treated_as_direct_response():
    result = classify_direct_response_prompt("What does the auth module do?")

    assert result.is_direct_response is False
    assert has_codebase_context("What does the auth module do?") is True


def test_action_request_is_not_treated_as_direct_response():
    result = classify_direct_response_prompt("Write a Python function that returns 4")

    assert result.is_direct_response is False


def test_extracts_exact_response_literal():
    literal = extract_exact_response_literal('Reply with exactly "READY"')

    assert literal == "READY"


def test_normalizes_exact_response_output_when_literal_is_present():
    normalized = normalize_direct_response_output(
        "Reply with exactly READY",
        "The user wants the exact response. READY",
    )

    assert normalized == "READY"


def test_output_state_buffers_exact_response_streams():
    state = DirectResponseOutputState("Reply with exactly READY")

    assert state.consume_stream_content("The answer is ") == ""
    assert state.consume_stream_content("READY") == ""

    content, metadata = state.flush_stream_content()

    assert content == "READY"
    assert metadata == {}


def test_output_state_passthrough_for_non_exact_responses():
    state = DirectResponseOutputState("What is Python?")

    assert state.consume_stream_content("Python is a language.") == "Python is a language."

    content, metadata = state.flush_stream_content()

    assert content == ""
    assert metadata == {}
