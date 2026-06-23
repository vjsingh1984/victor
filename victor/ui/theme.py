"""Professional UI theme for Victor Console."""

from rich.theme import Theme

# A professional, subdued color palette inspired by modern IDEs
victor_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "green",
        # Tool specific styling
        "tool.name": "bold blue",
        "tool.args": "dim",
        "tool.time": "dim italic",
        # Reasoning/Thinking
        "thinking.border": "dim",
        "thinking.text": "dim italic",
        "thinking.indicator": "cyan",
        # Layout and Chrome
        "chrome.border": "dim",
        "chrome.title": "bold",
    }
)
