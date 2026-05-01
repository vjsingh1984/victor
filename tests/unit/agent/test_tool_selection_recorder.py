from victor.agent.tool_selection import ToolSelectionStats


def test_selection_recorder_records_semantic_and_invokes_callback():
    from victor.agent.tool_selection_recorder import ToolSelectionRecorder

    stats = ToolSelectionStats()
    seen: list[tuple[str, int]] = []
    recorder = ToolSelectionRecorder(
        stats=stats,
        on_selection_recorded=lambda method, count: seen.append((method, count)),
    )

    recorder.record_result(is_fallback=False, num_tools=4)

    assert stats.semantic_selections == 1
    assert stats.fallback_selections == 0
    assert stats.total_tools_selected == 4
    assert seen == [("semantic", 4)]


def test_selection_recorder_records_fallback_without_callback():
    from victor.agent.tool_selection_recorder import ToolSelectionRecorder

    stats = ToolSelectionStats()
    recorder = ToolSelectionRecorder(stats=stats)

    recorder.record_result(is_fallback=True, num_tools=2)

    assert stats.semantic_selections == 0
    assert stats.fallback_selections == 1
    assert stats.total_tools_selected == 2
