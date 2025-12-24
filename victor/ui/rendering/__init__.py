"""Stream rendering package for CLI output.

This package implements the Strategy pattern to unify streaming response handling
across different CLI modes (oneshot vs interactive) while allowing pluggable
rendering strategies.

Design Pattern: Strategy + Protocol
- StreamRenderer protocol defines the interface
- FormatterRenderer uses OutputFormatter (for oneshot mode)
- LiveDisplayRenderer uses Rich Live display (for interactive mode)
- stream_response() is the unified handler that works with any renderer

Benefits:
- Single streaming loop eliminates code duplication
- Easy to add new renderers (JSON, plain text, TUI, etc.)
- Consistent behavior across all modes
- Testable - can mock the renderer for unit tests

Package Structure:
- protocol.py: StreamRenderer protocol definition
- utils.py: Shared rendering utilities
- formatter_renderer.py: OutputFormatter-based renderer
- live_renderer.py: Rich Live display-based renderer
- handler.py: stream_response() unified handler

Thinking Content Handling (dual-mode):
- **API-based reasoning**: DeepSeek API sends reasoning via metadata field
  (`chunk.metadata["reasoning_content"]`), rendered as dim/italic text
- **Inline markers**: Qwen3/Ollama local models use inline markers
  (`<think>...</think>`, `<|begin_of_thinking|>`), processed by
  StreamingContentFilter from response_sanitizer
- Automatic state transitions when switching between reasoning and normal output
- `suppress_thinking` option to completely hide thinking content
"""

from victor.ui.rendering.formatter_renderer import FormatterRenderer
from victor.ui.rendering.handler import stream_response
from victor.ui.rendering.live_renderer import LiveDisplayRenderer
from victor.ui.rendering.protocol import StreamRenderer
from victor.ui.rendering.utils import (
    format_tool_args,
    render_edit_preview,
    render_file_preview,
    render_thinking_indicator,
    render_thinking_text,
)

__all__ = [
    # Protocol
    "StreamRenderer",
    # Renderers
    "FormatterRenderer",
    "LiveDisplayRenderer",
    # Handler
    "stream_response",
    # Utilities
    "format_tool_args",
    "render_file_preview",
    "render_edit_preview",
    "render_thinking_indicator",
    "render_thinking_text",
]
