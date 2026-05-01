from victor.agent.continuation_contract import (
    ContinuationActionType,
    ContinuationDirective,
    ContinuationStatePatch,
)


def test_continuation_directive_preserves_legacy_mapping_shape():
    directive = ContinuationDirective.from_legacy(
        action=ContinuationActionType.PROMPT_TOOL_CALL,
        reason="Need tool execution",
        message="Continue with tools.",
        updates={"continuation_prompts": 2},
        mentioned_tools=["read", "code_search"],
    )

    assert directive.action is ContinuationActionType.PROMPT_TOOL_CALL
    assert directive["action"] is ContinuationActionType.PROMPT_TOOL_CALL
    assert directive.get("message") == "Continue with tools."
    assert directive["updates"] == {"continuation_prompts": 2}
    assert directive["mentioned_tools"] == ["read", "code_search"]


def test_continuation_state_patch_exposes_legacy_fields():
    patch = ContinuationStatePatch(
        continuation_prompts=3,
        final_summary_requested=True,
        max_prompts_summary_requested=True,
    )

    assert patch.to_updates_dict() == {
        "continuation_prompts": 3,
        "final_summary_requested": True,
        "max_prompts_summary_requested": True,
    }
    assert patch.to_legacy_fields() == {
        "updates": {
            "continuation_prompts": 3,
            "final_summary_requested": True,
            "max_prompts_summary_requested": True,
        },
        "set_final_summary_requested": True,
        "set_max_prompts_summary_requested": True,
    }
