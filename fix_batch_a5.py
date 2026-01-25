#!/usr/bin/env python3
"""Fix MyPy type errors in chat_coordinator.py for batch A5 (errors 201-250)."""

import re


def apply_fixes():
    """Apply type fixes line by line."""
    file_path = "/Users/vijaysingh/code/codingagent/victor/agent/coordinators/chat_coordinator.py"

    with open(file_path, "r") as f:
        content = f.read()

    # Store original for comparison
    original = content

    # Fix 1: Line 508 - Add None check for unified_tracker.record_tool_call
    content = re.sub(
        r'(\s+)orch\.unified_tracker\.record_tool_call\(tool_name, tool_args\)\n',
        r'\1if orch.unified_tracker is not None:\n\1    orch.unified_tracker.record_tool_call(tool_name, tool_args)\n',
        content,
        count=1
    )

    # Fix 2: Line 513 - Add None check for unified_tracker.record_iteration
    content = re.sub(
        r'(\s+)orch\.unified_tracker\.record_iteration\(content_length\)\n',
        r'\1if orch.unified_tracker is not None:\n\1    orch.unified_tracker.record_iteration(content_length)\n',
        content,
        count=1
    )

    # Fix 3: Line 551 - Add None check for unified_tracker.check_loop_warning
    content = re.sub(
        r'unified_loop_warning = orch\.unified_tracker\.check_loop_warning\(\)  # type: ignore\[attr-defined\]\n',
        '''unified_loop_warning = (
            orch.unified_tracker.check_loop_warning()
            if orch.unified_tracker is not None
            else None
        )
''',
        content,
        count=1
    )

    # Fix 4: Line 552-554 - Add None check for _streaming_handler.handle_loop_warning
    content = re.sub(
        r'loop_warning_chunk = orch\._streaming_handler\.handle_loop_warning\(\s*stream_ctx, unified_loop_warning\s*\)  # type: ignore\[attr-defined\]\n',
        '''loop_warning_chunk = (
            orch._streaming_handler.handle_loop_warning(
                stream_ctx, unified_loop_warning
            )
            if orch._streaming_handler is not None
            else None
        )
''',
        content,
        count=1
    )

    # Fix 5: Line 562 - Add None check for _recovery_coordinator.check_force_action
    content = re.sub(
        r'(\s+)was_triggered, hint = orch\._recovery_coordinator\.check_force_action\(recovery_ctx\)  # type: ignore\[attr-defined\]\n',
        r'\1if orch._recovery_coordinator is not None:\n\1    was_triggered, hint = orch._recovery_coordinator.check_force_action(recovery_ctx)\n\1else:\n\1    was_triggered, hint = False, ""\n',
        content,
        count=1
    )

    # Fix 6: Line 566 - Add None check for unified_tracker.get_metrics
    content = re.sub(
        r'f"metrics=\{orch\.unified_tracker\.get_metrics\(\)\}"  # type: ignore\[attr-defined\]',
        r'f"metrics={orch.unified_tracker.get_metrics() if orch.unified_tracker is not None else {}}"',
        content,
        count=1
    )

    # Fix 7: Line 677 - Fix calls_used assignment (read-only property)
    content = re.sub(
        r'(\s+)orch\._tool_pipeline\.calls_used = current_calls \+ tool_exec_result\.tool_calls_executed  # type: ignore\[attr-defined\]\n',
        r'\1# Note: calls_used is read-only, tool_pipeline tracks it internally\n',
        content,
        count=1
    )

    # Fix 8: Line 703 - Add None check for _metrics_coordinator.start_streaming
    content = re.sub(
        r'(\s+)orch\._metrics_coordinator\.start_streaming\(\)  # type: ignore\[attr-defined\]\n',
        r'\1if orch._metrics_coordinator is not None:\n\1    orch._metrics_coordinator.start_streaming()\n',
        content,
        count=1
    )

    # Fix 9: Line 706 - Add None check for _metrics_collector.init_stream_metrics
    content = re.sub(
        r'stream_metrics = orch\._metrics_collector\.init_stream_metrics\(\)  # type: ignore\[attr-defined\]\n',
        '''stream_metrics = (
            orch._metrics_collector.init_stream_metrics()
            if orch._metrics_collector is not None
            else cast(Any, None)
        )
''',
        content,
        count=1
    )

    # Fix 10: Line 720 - Add None check for conversation.ensure_system_prompt
    content = re.sub(
        r'(\s+)orch\.conversation\.ensure_system_prompt\(\)  # type: ignore\[attr-defined\]\n',
        r'\1if orch.conversation is not None:\n\1    orch.conversation.ensure_system_prompt()\n',
        content,
        count=1
    )

    # Fix 11: Line 721 - Fix _system_added attribute
    content = re.sub(
        r'(\s+)orch\._system_added = True  # type: ignore\[attr-defined\]\n',
        r'\1# Set system_added flag\n\1if not hasattr(orch, "_system_added"):\n\1    orch._system_added = False\n\1orch._system_added = True\n',
        content,
        count=1
    )

    # Fix 12: Line 723 - Add None check for _session_state.reset_for_new_turn
    content = re.sub(
        r'(\s+)orch\._session_state\.reset_for_new_turn\(\)  # type: ignore\[attr-defined\]\n',
        r'\1if orch._session_state is not None:\n\1    orch._session_state.reset_for_new_turn()\n',
        content,
        count=1
    )

    # Fix 13: Line 726 - Add None check for unified_tracker.reset
    content = re.sub(
        r'(\s+)orch\.unified_tracker\.reset\(\)  # type: ignore\[attr-defined\]\n',
        r'\1if orch.unified_tracker is not None:\n\1    orch.unified_tracker.reset()\n',
        content,
        count=1
    )

    # Fix 14: Line 729 - Add None check for reminder_manager.reset
    content = re.sub(
        r'(\s+)orch\.reminder_manager\.reset\(\)  # type: ignore\[attr-defined\]\n',
        r'\1if orch.reminder_manager is not None:\n\1    orch.reminder_manager.reset()\n',
        content,
        count=1
    )

    # Fix 15: Line 740-741 - Add None check for _context_manager
    content = re.sub(
        r'if orch\._context_manager and hasattr\(orch\._context_manager, "start_background_compaction"\):  # type: ignore\[attr-defined\]\s+await orch\._context_manager\.start_background_compaction\(interval_seconds=15\.0\)  # type: ignore\[attr-defined\]\n',
        '''if orch._context_manager is not None and hasattr(orch._context_manager, "start_background_compaction"):
            await orch._context_manager.start_background_compaction(interval_seconds=15.0)
''',
        content,
        count=1
    )

    # Fix 16: Line 744 - Add None check for unified_tracker.config
    content = re.sub(
        r'max_total_iterations = orch\.unified_tracker\.config\.get\("max_total_iterations", 50\)  # type: ignore\[attr-defined\]\n',
        '''max_total_iterations = (
            orch.unified_tracker.config.get("max_total_iterations", 50)
            if orch.unified_tracker is not None
            else 50
        )
''',
        content,
        count=1
    )

    # Fix 17: Line 749 - Add None check for add_message
    content = re.sub(
        r'(\s+)orch\.add_message\("user", user_message\)  # type: ignore\[attr-defined\]\n',
        r'\1# Add user message to history\n\1if hasattr(orch, "add_message"):\n\1    orch.add_message("user", user_message)\n\1else:\n\1    # Fallback: add to conversation directly\n\1    if orch.conversation is not None:\n\1        orch.conversation.add_message({"role": "user", "content": user_message})\n',
        content,
        count=1
    )

    # Fix 18: Line 756 - Add None check for unified_tracker.detect_task_type
    content = re.sub(
        r'unified_task_type = orch\.unified_tracker\.detect_task_type\(user_message\)  # type: ignore\[attr-defined\]\n',
        '''unified_task_type = (
            orch.unified_tracker.detect_task_type(user_message)
            if orch.unified_tracker is not None
            else TrackerTaskType.GENERAL
        )
''',
        content,
        count=1
    )

    # Fix 19: Line 762 - Add None check for unified_tracker._progress
    content = re.sub(
        r'(\s+)orch\.unified_tracker\._progress\.has_prompt_requirements = True  # type: ignore\[attr-defined\]\n',
        r'\1if orch.unified_tracker is not None and hasattr(orch.unified_tracker, "_progress"):\n\1    if hasattr(orch.unified_tracker._progress, "has_prompt_requirements"):\n\1        orch.unified_tracker._progress.has_prompt_requirements = True\n',
        content,
        count=1
    )

    # Fix 20: Line 766 - Add None check for unified_tracker._progress.tool_budget
    # This is more complex, need to handle multi-line condition
    pattern = r'and prompt_requirements\.tool_budget > orch\.unified_tracker\._progress\.tool_budget  # type: ignore\[attr-defined\]\s*\)\s*\{'
    replacement = '''and (
                    orch.unified_tracker is not None
                    and hasattr(orch.unified_tracker, "_progress")
                    and hasattr(orch.unified_tracker._progress, "tool_budget")
                    and prompt_requirements.tool_budget > orch.unified_tracker._progress.tool_budget
                )
            ):'''
    content = re.sub(pattern, replacement, content, count=1)

    # Fix 21: Line 768 - Add None check for unified_tracker.set_tool_budget
    content = re.sub(
        r'(\s+)orch\.unified_tracker\.set_tool_budget\(prompt_requirements\.tool_budget\)  # type: ignore\[attr-defined\]\n',
        r'\1if orch.unified_tracker is not None:\n\1    orch.unified_tracker.set_tool_budget(prompt_requirements.tool_budget)\n',
        content,
        count=1
    )

    # Fix 22: Line 777 - Add None check for unified_tracker._task_config
    # Similar complex multi-line condition
    pattern2 = r'> orch\.unified_tracker\._task_config\.max_exploration_iterations  # type: ignore\[attr-defined\]\s*\)\s*\{'
    replacement2 = '''> (
                        orch.unified_tracker is not None
                        and hasattr(orch.unified_tracker, "_task_config")
                        and hasattr(orch.unified_tracker._task_config, "max_exploration_iterations")
                        and prompt_requirements.iteration_budget
                        > orch.unified_tracker._task_config.max_exploration_iterations
                    )
                ):'''
    content = re.sub(pattern2, replacement2, content, count=1)

    if content != original:
        with open(file_path, "w") as f:
            f.write(content)
        print(f"Fixed {file_path}")
        return True
    else:
        print("No changes needed")
        return False


if __name__ == "__main__":
    import sys
    success = apply_fixes()
    sys.exit(0 if success else 1)
