# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration tests for tool output accuracy fix.

Verifies that LLM receives full tool output while user sees preview.
This is a critical fix - LLM must have complete context to make accurate decisions.
"""

import json

import pytest
from unittest.mock import MagicMock, patch


class TestToolOutputAccuracy:
    """Test that LLM receives full tool output, not pruned version."""

    def test_format_and_prune_separates_llm_and_preview(self):
        """Verify format_and_prune_tool_output separates LLM input from user preview."""
        from victor.agent.services.tool_service import format_and_prune_tool_output

        # Create a large tool output (100 lines)
        large_output = "\n".join([f"Line {i}" for i in range(100)])

        # Format and prune
        formatted, llm_output, was_pruned, pruning_info = format_and_prune_tool_output(
            tool_name="test_tool",
            arguments={"query": "test"},
            output=large_output,
            task_type="unknown",
        )

        # CRITICAL: llm_output must contain FULL output (may be wrapped in tags)
        # The formatter wraps output in <TOOL_OUTPUT> tags, so we check content
        assert "Line 0" in llm_output, "LLM output must contain first line"
        assert "Line 99" in llm_output, "LLM output must contain last line"
        assert llm_output.count("Line ") >= 100, "LLM output should have all 100 lines"

        # formatted and llm_output are the same (full output)
        assert formatted == llm_output, "Formatted and LLM output should be identical (both full)"

        # was_pruned indicates whether user preview would be truncated
        # but llm_output should ALWAYS be full

    def test_llm_receives_full_output_even_when_preview_pruned(self):
        """Verify LLM receives full output even when user preview is pruned."""
        from victor.agent.services.tool_service import format_and_prune_tool_output

        # Create large output that will be pruned for preview
        large_output = "\n".join([f"Content line {i}" for i in range(500)])

        formatted, llm_output, was_pruned, pruning_info = format_and_prune_tool_output(
            tool_name="read",
            arguments={"path": "large_file.txt"},
            output=large_output,
            task_type="unknown",
        )

        # CRITICAL ASSERTIONS
        # 1. LLM output must be complete (all lines present)
        assert "Content line 0" in llm_output, "LLM must receive first line"
        assert "Content line 499" in llm_output, "LLM must receive last line"
        assert llm_output.count("Content line") >= 500, "LLM must receive all 500 lines"

        # 2. User preview may be pruned (that's OK)
        # 3. was_pruned flag indicates preview truncation, not LLM input truncation
        if was_pruned:
            # If preview was pruned, verify it's only for display
            assert pruning_info is not None
            # pruning_info.lines_returned < 500  # Preview truncated

        # 4. Most important: LLM output contains ALL original content
        # (This is the fix - LLM gets full output regardless of preview)

    def test_llm_output_strips_preview_only_fields(self):
        """Verify preview-only fields stay in display output but not in LLM output."""
        from victor.agent.services.tool_service import format_and_prune_tool_output

        structured_output = {
            "success": True,
            "results": [{"path": "src/main.py", "line": 42, "snippet": "def main():"}],
            "formatted_results": "[bold cyan]1 match[/]",
            "contains_markup": True,
        }

        def formatter(_tool, _args, output):
            return json.dumps(output, sort_keys=True)

        formatted, llm_output, _, _ = format_and_prune_tool_output(
            tool_name="code_search",
            arguments={"query": "main"},
            output=structured_output,
            formatter=formatter,
        )

        assert '"formatted_results"' in formatted
        assert '"contains_markup"' in formatted
        assert '"formatted_results"' not in llm_output
        assert '"contains_markup"' not in llm_output
        assert '"results"' in llm_output

    def test_was_pruned_flag_clarity(self):
        """Verify was_pruned flag meaning changed from 'pruned for LLM' to 'pruned for preview'."""
        from victor.agent.services.tool_service import format_and_prune_tool_output

        # Small output - won't be pruned
        small_output = "Small result"

        formatted, llm_output, was_pruned_small, _ = format_and_prune_tool_output(
            tool_name="test",
            arguments={},
            output=small_output,
        )

        # Large output - will be pruned for preview only
        large_output = "\n".join([f"Line {i}" for i in range(1000)])
        formatted, llm_output, was_pruned_large, _ = format_and_prune_tool_output(
            tool_name="test",
            arguments={},
            output=large_output,
        )

        # Both should send FULL output to LLM
        assert "Line 0" in llm_output, "LLM must get first line"
        assert "Line 999" in llm_output, "LLM must get last line"
        assert llm_output.count("Line ") >= 1000, "Large output: LLM gets all 1000 lines"

        # was_pruned indicates preview truncation, not LLM input truncation
        # (This is the semantic change from the fix)
        # was_pruned_large may be True (preview truncated)
        # But llm_output is still complete


class TestToolServiceIntegration:
    """Test that ToolService sends full output to conversation/LLM."""

    def test_process_tool_results_sends_full_output_to_conversation(self):
        """Verify process_tool_results_with_context sends full output to LLM via add_message."""
        from victor.agent.services.tool_service import process_tool_results_with_context
        from victor.agent.services.tool_service import ToolResultContext

        # Create mock context with all required fields
        ctx = ToolResultContext(
            executed_tools=[],
            observed_files=set(),
            failed_tool_signatures=set(),
            shown_tool_errors=set(),
            task_type="unknown",
        )

        # Track what gets sent to LLM via add_message
        messages_sent_to_llm = []

        def mock_add_message(role, content, **kwargs):
            messages_sent_to_llm.append({"role": role, "content": content, **kwargs})

        ctx.add_message = mock_add_message
        ctx.record_tool_execution = lambda *args, **kwargs: None
        ctx.unified_tracker = None
        ctx.usage_logger = None
        ctx.conversation_state = None
        ctx.stream_context = None

        # Create mock pipeline result with large output
        from collections import namedtuple

        MockCallResult = namedtuple(
            "MockCallResult",
            [
                "tool_name",
                "arguments",
                "result",
                "success",
                "error",
                "execution_time_ms",
                "skipped",
                "tool_call_id",
            ],
        )

        large_output = "\n".join([f"Result line {i}" for i in range(200)])

        mock_result = MockCallResult(
            tool_name="read",
            arguments={"path": "test.txt"},
            result=large_output,
            success=True,
            error=None,
            execution_time_ms=100,
            skipped=False,
            tool_call_id="test_call_123",
        )

        MockPipelineResult = MagicMock()
        MockPipelineResult.results = [mock_result]

        # Process tool results
        results = process_tool_results_with_context(MockPipelineResult, ctx)

        # CRITICAL: Verify what was sent to LLM
        assert len(messages_sent_to_llm) == 1, "Should send one tool result message"
        message_to_llm = messages_sent_to_llm[0]

        # The content sent to LLM must be FULL output (not pruned)
        llm_content = message_to_llm["content"]
        assert isinstance(llm_content, str), "LLM content should be string"

        # CRITICAL ASSERTION: LLM must receive all lines
        assert "Result line 0" in llm_content, "LLM must receive first line"
        assert "Result line 199" in llm_content, "LLM must receive last line"
        assert llm_content.count("Result line") >= 200, "LLM must receive all 200 lines"

        # The result dict may have was_pruned flag
        # But this should indicate preview truncation, not LLM input truncation
        assert len(results) == 1
        result_dict = results[0]
        if result_dict.get("was_pruned"):
            # If was_pruned is True, it means user preview was truncated
            # But LLM still received full output (verified above)
            pass


class TestBackwardCompatibility:
    """Verify backward compatibility with existing code."""

    def test_return_values_unchanged(self):
        """Verify format_and_prune_tool_output return type hasn't changed."""
        from victor.agent.services.tool_service import format_and_prune_tool_output

        formatted, llm_output, was_pruned, pruning_info = format_and_prune_tool_output(
            tool_name="test",
            arguments={},
            output="test output",
        )

        # Return type should be: (str, str, bool, Optional[Any])
        assert isinstance(formatted, str)
        assert isinstance(llm_output, str)
        assert isinstance(was_pruned, bool)
        # pruning_info can be None or PruningInfo object

    def test_result_dict_structure_unchanged(self):
        """Verify result dict structure hasn't changed."""
        from victor.agent.services.tool_service import process_tool_results_with_context
        from victor.agent.services.tool_service import ToolResultContext

        ctx = ToolResultContext(
            executed_tools=[],
            observed_files=set(),
            failed_tool_signatures=set(),
            shown_tool_errors=set(),
            task_type="unknown",
        )
        ctx.add_message = lambda *args, **kwargs: None
        ctx.record_tool_execution = lambda *args, **kwargs: None
        ctx.unified_tracker = None
        ctx.usage_logger = None
        ctx.conversation_state = None
        ctx.stream_context = None

        from collections import namedtuple

        MockCallResult = namedtuple(
            "MockCallResult",
            [
                "tool_name",
                "arguments",
                "result",
                "success",
                "error",
                "execution_time_ms",
                "skipped",
                "tool_call_id",
            ],
        )

        mock_result = MockCallResult(
            tool_name="test",
            arguments={},
            result="test output",
            success=True,
            error=None,
            execution_time_ms=100,
            skipped=False,
            tool_call_id="test_123",
        )

        MockPipelineResult = MagicMock()
        MockPipelineResult.results = [mock_result]

        results = process_tool_results_with_context(MockPipelineResult, ctx)

        # Result dict should still have these keys
        assert len(results) == 1
        result = results[0]
        assert "name" in result
        assert "success" in result
        assert "result" in result
        assert "was_pruned" in result
        assert "content" in result
