from victor.tools.semantic_selector import SemanticToolSelector


def test_pending_actions_pr_not_triggered_by_prioritized():
    selector = SemanticToolSelector()
    history = [
        {"role": "user", "content": "Provide a prioritized list of improvements"},
        {"role": "assistant", "content": ""},
    ]
    pending = selector._extract_pending_actions(history)
    assert "pr" not in pending


def test_pending_actions_pr_word_detected():
    selector = SemanticToolSelector()
    history = [
        {"role": "user", "content": "Raise a PR for the changes"},
        {"role": "assistant", "content": ""},
    ]
    pending = selector._extract_pending_actions(history)
    assert "pr" in pending
