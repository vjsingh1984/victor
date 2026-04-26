from victor.agent.conversation.state_machine import ConversationStage


def test_cache_key_builder_uses_stage_value_and_semantic_mode():
    from victor.agent.tool_selection_cache_key import SemanticToolSelectionCacheKeyBuilder

    builder = SemanticToolSelectionCacheKeyBuilder()

    key = builder.build(
        user_message="find the bug",
        conversation_history=[{"role": "user", "content": "hi"}],
        conversation_depth=3,
        stage=ConversationStage.ANALYSIS,
    )

    assert isinstance(key, str)
    assert key


def test_cache_key_builder_accepts_missing_stage_and_history():
    from victor.agent.tool_selection_cache_key import SemanticToolSelectionCacheKeyBuilder

    builder = SemanticToolSelectionCacheKeyBuilder()

    key = builder.build(
        user_message="read file",
        conversation_history=None,
        conversation_depth=0,
        stage=None,
    )

    assert isinstance(key, str)
    assert key
