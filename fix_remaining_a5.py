#!/usr/bin/env python3
"""Fix remaining MyPy errors in chat_coordinator.py batch A5."""

import re

def fix_file():
    file_path = "/Users/vijaysingh/code/codingagent/victor/agent/coordinators/chat_coordinator.py"

    with open(file_path, "r") as f:
        lines = f.readlines()

    # Fixes to apply (line numbers are 1-indexed in the error log, 0-indexed in list)
    fixes = {
        # Line 714: _metrics_coordinator.start_streaming()
        713: ('        orch._metrics_coordinator.start_streaming()  # type: ignore[attr-defined]\n',
              '        if orch._metrics_coordinator is not None:\n            orch._metrics_coordinator.start_streaming()\n'),

        # Line 717: _metrics_collector.init_stream_metrics()
        716: ('        stream_metrics = orch._metrics_collector.init_stream_metrics()  # type: ignore[attr-defined]\n',
              '        stream_metrics = (\n            orch._metrics_collector.init_stream_metrics()\n            if orch._metrics_collector is not None\n            else cast(Any, None)\n        )\n'),

        # Line 718: start_time assignment
        717: ('        start_time = stream_metrics.start_time\n',
              '        start_time = stream_metrics.start_time if stream_metrics else 0.0\n'),

        # Line 731: conversation.ensure_system_prompt()
        730: ('        orch.conversation.ensure_system_prompt()  # type: ignore[attr-defined]\n',
              '        if orch.conversation is not None:\n            orch.conversation.ensure_system_prompt()\n'),

        # Line 732: _system_added
        731: ('        orch._system_added = True  # type: ignore[attr-defined]\n',
              '        # Set system_added flag\n        if not hasattr(orch, "_system_added"):\n            orch._system_added = False\n        orch._system_added = True\n'),

        # Line 734: _session_state.reset_for_new_turn()
        733: ('        orch._session_state.reset_for_new_turn()  # type: ignore[attr-defined]\n',
              '        if orch._session_state is not None:\n            orch._session_state.reset_for_new_turn()\n'),

        # Line 737: unified_tracker.reset()
        736: ('        orch.unified_tracker.reset()  # type: ignore[attr-defined]\n',
              '        if orch.unified_tracker is not None:\n            orch.unified_tracker.reset()\n'),

        # Line 740: reminder_manager.reset()
        739: ('        orch.reminder_manager.reset()  # type: ignore[attr-defined]\n',
              '        if orch.reminder_manager is not None:\n            orch.reminder_manager.reset()\n'),

        # Line 751-752: _context_manager
        750: ('        if orch._context_manager and hasattr(orch._context_manager, "start_background_compaction"):  # type: ignore[attr-defined]\n',
              '        if orch._context_manager is not None and hasattr(orch._context_manager, "start_background_compaction"):\n'),

        # Line 755: unified_tracker.config.get
        754: ('        max_total_iterations = orch.unified_tracker.config.get("max_total_iterations", 50)  # type: ignore[attr-defined]\n',
              '        max_total_iterations = (\n            orch.unified_tracker.config.get("max_total_iterations", 50)\n            if orch.unified_tracker is not None\n            else 50\n        )\n'),

        # Line 760: add_message
        759: ('        orch.add_message("user", user_message)  # type: ignore[attr-defined]\n',
              '        # Add user message to history\n        if hasattr(orch, "add_message"):\n            orch.add_message("user", user_message)\n        else:\n            # Fallback: add to conversation directly\n            if orch.conversation is not None:\n                orch.conversation.add_message({"role": "user", "content": user_message})\n'),

        # Line 767: unified_tracker.detect_task_type
        766: ('        unified_task_type = orch.unified_tracker.detect_task_type(user_message)  # type: ignore[attr-defined]\n',
              '        unified_task_type = (\n            orch.unified_tracker.detect_task_type(user_message)\n            if orch.unified_tracker is not None\n            else TrackerTaskType.GENERAL\n        )\n'),
    }

    modified = False

    for line_num, (old, new) in fixes.items():
        if line_num < len(lines):
            if old in lines[line_num]:
                lines[line_num] = new
                modified = True
                print(f"Fixed line {line_num + 1}")

    # Also need to fix lines around 772, 776, 778
    # These are more complex multi-line fixes
    for i in range(len(lines) - 5):
        # Fix line 772: unified_tracker._progress.has_prompt_requirements
        if i == 771 and "orch.unified_tracker._progress.has_prompt_requirements = True" in lines[i]:
            lines[i] = ('        if orch.unified_tracker is not None and hasattr(orch.unified_tracker, "_progress"):\n'
                       '            if hasattr(orch.unified_tracker._progress, "has_prompt_requirements"):\n'
                       '                orch.unified_tracker._progress.has_prompt_requirements = True\n')
            modified = True
            print(f"Fixed line 772")

        # Fix line 776: unified_tracker._progress.tool_budget
        if i == 775 and "and prompt_requirements.tool_budget > orch.unified_tracker._progress.tool_budget" in lines[i]:
            # This is part of a multi-line if statement
            lines[i] = ('                and (\n'
                       '                    orch.unified_tracker is not None\n'
                       '                    and hasattr(orch.unified_tracker, "_progress")\n'
                       '                    and hasattr(orch.unified_tracker._progress, "tool_budget")\n'
                       '                    and prompt_requirements.tool_budget > orch.unified_tracker._progress.tool_budget\n'
                       '                )\n')
            modified = True
            print(f"Fixed line 776")

        # Fix line 778: unified_tracker.set_tool_budget
        if i == 777 and "orch.unified_tracker.set_tool_budget(prompt_requirements.tool_budget)" in lines[i]:
            lines[i] = ('            if orch.unified_tracker is not None:\n'
                       '                orch.unified_tracker.set_tool_budget(prompt_requirements.tool_budget)\n')
            modified = True
            print(f"Fixed line 778")

    if modified:
        with open(file_path, "w") as f:
            f.writelines(lines)
        print("\nFile updated successfully")
        return True
    else:
        print("\nNo changes made")
        return False

if __name__ == "__main__":
    import sys
    success = fix_file()
    sys.exit(0 if success else 1)
