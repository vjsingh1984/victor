"""Test tool output preview functionality."""

import pytest

from victor.config.tool_settings import ToolSettings
from victor.ui.output_formatter import OutputFormatter, OutputConfig
from victor.ui.rendering.live_renderer import LiveDisplayRenderer
from rich.console import Console


def test_preview_generation_output_formatter():
    """Test preview generation from long output in OutputFormatter."""
    formatter = OutputFormatter()
    long_output = "\n".join([f"Line {i}" for i in range(100)])
    preview = formatter._generate_preview(long_output, num_lines=3)

    assert "Line 0" in preview
    assert "Line 1" in preview
    assert "Line 2" in preview
    assert "Line 3" not in preview
    assert "..." in preview


def test_preview_generation_live_renderer():
    """Test preview generation from long output in LiveDisplayRenderer."""
    console = Console()
    renderer = LiveDisplayRenderer(console)
    long_output = "\n".join([f"Line {i}" for i in range(100)])
    preview = renderer._generate_preview(long_output, num_lines=3)

    assert "Line 0" in preview
    assert "Line 1" in preview
    assert "Line 2" in preview
    assert "Line 3" not in preview
    assert "..." in preview


def test_preview_truncates_long_lines():
    """Test that preview truncates long lines."""
    formatter = OutputFormatter()
    long_line = "a" * 200
    output = f"{long_line}\nLine 2\nLine 3"
    preview = formatter._generate_preview(output, num_lines=3)

    # Long line should be truncated to 120 chars + "..."
    assert len(preview.split("\n")[0]) == 123  # 120 + "..."
    assert "..." in preview


def test_preview_empty_text():
    """Test preview generation with empty text."""
    formatter = OutputFormatter()
    assert formatter._generate_preview("") == ""
    assert formatter._generate_preview(None) == ""


def test_expand_functionality_output_formatter():
    """Test expanding collapsed output in OutputFormatter."""
    formatter = OutputFormatter()
    formatter._last_tool_result = {
        "tool_name": "read",
        "success": True,
        "result": "Full content here\n" * 10,
        "pruned_result": "Truncated...",
        "was_pruned": False,
    }
    # Should not raise exception
    formatter.expand_last_tool_output()


def test_expand_functionality_live_renderer():
    """Test expanding collapsed output in LiveDisplayRenderer."""
    console = Console()
    renderer = LiveDisplayRenderer(console)
    renderer._last_tool_result = {
        "name": "grep",
        "success": True,
        "result": "Full content here\n" * 10,
        "arguments": {},
        "elapsed": 1.5,
    }
    # Should not raise exception
    renderer.expand_last_output()


def test_expand_no_result_available():
    """Test expand when no tool result is available."""
    formatter = OutputFormatter()
    # No _last_tool_result set
    formatter.expand_last_tool_output()  # Should handle gracefully

    console = Console()
    renderer = LiveDisplayRenderer(console)
    # No _last_tool_result set
    renderer.expand_last_output()  # Should handle gracefully


def test_expand_failed_tool():
    """Test that expand doesn't show failed tools."""
    console = Console()
    renderer = LiveDisplayRenderer(console)
    renderer._last_tool_result = {
        "name": "read",
        "success": False,  # Failed tool
        "result": "Error output",
        "arguments": {},
        "elapsed": 1.0,
    }
    # Should not display anything for failed tools
    renderer.expand_last_output()


def test_settings_defaults():
    """Test that new settings have correct safe-default pruning behavior."""
    settings = ToolSettings()
    assert settings.tool_output_preview_enabled is True  # Show preview by default
    assert settings.tool_output_pruning_enabled is True  # Safe-default pruning enabled
    assert settings.tool_output_pruning_safe_only is True  # Limited to safe verbose tools
    assert settings.tool_output_show_transparency is True
    assert settings.tool_output_preview_lines == 3
    assert settings.tool_output_expand_hotkey == "^O"


def test_backward_compatibility_settings():
    """Test that broader pruning can be enabled explicitly via settings."""
    settings = ToolSettings(
        tool_output_pruning_enabled=True,
        tool_output_pruning_safe_only=False,
        tool_output_preview_enabled=False,  # No preview
    )
    assert settings.tool_output_pruning_enabled is True
    assert settings.tool_output_pruning_safe_only is False
    assert settings.tool_output_preview_enabled is False


def test_preview_lines_boundary():
    """Test preview_lines respects min/max boundaries."""
    # Test minimum boundary
    settings_min = ToolSettings(tool_output_preview_lines=1)
    assert settings_min.tool_output_preview_lines == 1

    # Test maximum boundary
    settings_max = ToolSettings(tool_output_preview_lines=10)
    assert settings_max.tool_output_preview_lines == 10

    # Test default
    settings_default = ToolSettings()
    assert settings_default.tool_output_preview_lines == 3


def test_preview_with_exact_line_count():
    """Test preview when output has exact number of lines."""
    formatter = OutputFormatter()
    output = "Line 1\nLine 2\nLine 3"
    preview = formatter._generate_preview(output, num_lines=3)

    assert "Line 1" in preview
    assert "Line 2" in preview
    assert "Line 3" in preview
    assert "..." not in preview  # No ellipsis when exact match


def test_preview_with_fewer_lines():
    """Test preview when output has fewer lines than requested."""
    formatter = OutputFormatter()
    output = "Line 1\nLine 2"
    preview = formatter._generate_preview(output, num_lines=5)

    assert "Line 1" in preview
    assert "Line 2" in preview
    assert "..." not in preview  # No ellipsis when output is shorter


def test_output_formatter_tool_result_with_preview_params():
    """Test that tool_result accepts preview parameters."""
    formatter = OutputFormatter()
    # Should accept new parameters without error
    formatter.tool_result(
        tool_name="read",
        success=True,
        original_result="Full output\n" * 10,
        preview_lines=3,
        was_pruned=False,
        show_preview=True,
    )
    # Verify state is stored
    assert formatter._last_tool_result is not None
    assert formatter._last_tool_result["tool_name"] == "read"
