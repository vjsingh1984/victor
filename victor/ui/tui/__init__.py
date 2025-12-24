"""Victor TUI - Modern terminal user interface.

A Textual-based TUI providing a modern chat interface similar to
Claude Code, Codex, and Gemini CLI with:
- Input box at the bottom
- Scrollable conversation history in the middle
- Status bar at the top
- Beautiful spacing and aesthetics
"""

from victor.ui.tui.app import VictorTUI
from victor.ui.tui.widgets import MessageWidget, InputWidget, StatusBar

__all__ = ["VictorTUI", "MessageWidget", "InputWidget", "StatusBar"]
