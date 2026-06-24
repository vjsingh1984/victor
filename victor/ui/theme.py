"""Professional UI theme for Victor Console with terminal capability detection."""

from __future__ import annotations

import os
import sys
from typing import Literal

from rich.theme import Theme


# Terminal capability levels
class TerminalCapability:
    """Terminal capability detection for adaptive styling."""

    # Capability levels
    BASIC = "basic"  # 16 colors, no Unicode support
    STANDARD = "standard"  # 256 colors, Unicode support
    TRUECOLOR = "truecolor"  # 24-bit color, full Unicode

    @staticmethod
    def detect() -> str:
        """Detect terminal capability level.

        Checks environment variables and terminal type to determine
        the appropriate styling level.

        Returns:
            Capability level: basic, standard, or truecolor
        """
        # Check for truecolor support
        truecolor_terms = {
            "iterm",
            "nv",
            "kitty",
            "wezterm",
            "alacritty",
            "contour",
            "foot",
            "rio",
            "mintty",
            "windows-terminal",
        }
        term = os.getenv("TERM", "").lower()
        term_program = os.getenv("TERM_PROGRAM", "").lower()
        # Handle common TERM_PROGRAM values like "iTerm.app" -> "iterm"
        term_program_base = term_program.split(".")[0]

        if (
            os.getenv("COLORTERM") in {"truecolor", "24bit"}
            or term_program_base in truecolor_terms
            or term_program in truecolor_terms
            or "truecolor" in term
            or "24bit" in term
        ):
            return TerminalCapability.TRUECOLOR

        # Check for 256-color support
        if "256color" in term or term in {"xterm-256color", "screen-256color"}:
            return TerminalCapability.STANDARD

        # Check for basic terminal
        if term in {"xterm", "xterm-color", "vt100", "ansi"}:
            return TerminalCapability.BASIC

        # Default to standard for unknown terminals
        return TerminalCapability.STANDARD


# Color schemes for different capability levels
_BASIC_COLORS = {
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "green",
    "tool.name": "bold blue",
    "tool.args": "dim",
    "tool.time": "dim",
    "thinking.border": "dim",
    "thinking.text": "dim",
    "thinking.indicator": "cyan",
    "chrome.border": "dim",
    "chrome.title": "bold",
}

_STANDARD_COLORS = {
    "info": "bright_cyan",
    "warning": "bright_yellow",
    "error": "bold bright_red",
    "success": "bright_green",
    "tool.name": "bold bright_blue",
    "tool.args": "dim",
    "tool.time": "dim italic",
    "thinking.border": "dim",
    "thinking.text": "dim italic",
    "thinking.indicator": "bright_cyan",
    "chrome.border": "dim",
    "chrome.title": "bold",
}

_TRUECOLOR_COLORS = {
    "info": "#00d7ff",  # Cyan-ish
    "warning": "#ffd700",  # Gold
    "error": "#ff4444",  # Soft red
    "success": "#50fa7b",  # Pastel green
    "tool.name": "#8be9fd",  # Light blue
    "tool.args": "#6272a4",  # Dim blue
    "tool.time": "#50fa7b italic",  # Green italic
    "thinking.border": "#44475a",  # Dark purple
    "thinking.text": "#f8f8f2 italic",  # Off-white italic
    "thinking.indicator": "#bd93f9",  # Purple
    "chrome.border": "#44475a",  # Dark purple
    "chrome.title": "#f8f8f2 bold",  # Off-white bold
}

# Per-tool-category color scheme (Phase 5)
_TOOL_CATEGORY_COLORS = {
    "filesystem": "#8be9fd",  # Light blue
    "git": "#50fa7b",  # Green
    "web": "#ffb86c",  # Orange
    "code": "#bd93f9",  # Purple
    "system": "#ff79c6",  # Pink
    "database": "#ff5555",  # Red
    "security": "#f1fa8c",  # Yellow
    "testing": "#6272a4",  # Dim blue
    "network": "#8be9fd",  # Cyan
    "default": "#f8f8f2",  # Off-white
}


def get_adaptive_theme(
    capability: str | None = None,
    theme_mode: Literal["dark", "light", "high_contrast"] = "dark",
) -> Theme:
    """Get an adaptive theme based on terminal capability.

    Args:
        capability: Terminal capability level (auto-detected if None)
        theme_mode: Theme mode (dark, light, high_contrast)

    Returns:
        Rich Theme configured for the detected capability
    """
    if capability is None:
        capability = TerminalCapability.detect()

    # Select base colors based on capability
    if capability == TerminalCapability.TRUECOLOR:
        base_colors = _TRUECOLOR_COLORS
    elif capability == TerminalCapability.STANDARD:
        base_colors = _STANDARD_COLORS
    else:
        base_colors = _BASIC_COLORS

    # Adjust for theme mode
    if theme_mode == "high_contrast":
        # Brighter colors, stronger borders
        adjusted_colors = {}
        for key, value in base_colors.items():
            if "dim" in value:
                adjusted_colors[key] = value.replace("dim", "bold")
            elif "italic" in value:
                adjusted_colors[key] = value.replace("italic", "bold italic")
            else:
                adjusted_colors[key] = value
        adjusted_colors["chrome.border"] = "white"
        return Theme(adjusted_colors)

    elif theme_mode == "light":
        # Invert for light background
        inverted_colors = {}
        for key, value in base_colors.items():
            # Swap dark/bright for light theme
            if "bright_" in str(value):
                inverted_colors[key] = value.replace("bright_", "")
            elif value == "dim":
                inverted_colors[key] = "bold"
            else:
                inverted_colors[key] = value
        inverted_colors["chrome.border"] = "black"
        inverted_colors["thinking.text"] = "black italic"
        return Theme(inverted_colors)

    return Theme(base_colors)


def get_tool_category_color(category: str) -> str:
    """Get color for a tool category.

    Args:
        category: Tool category name

    Returns:
        Color string for the category
    """
    return _TOOL_CATEGORY_COLORS.get(category.lower(), _TOOL_CATEGORY_COLORS["default"])


# Legacy default theme (for backward compatibility)
victor_theme = get_adaptive_theme()

# Pre-configured theme variants
dark_theme = get_adaptive_theme(theme_mode="dark")
light_theme = get_adaptive_theme(theme_mode="light")
high_contrast_theme = get_adaptive_theme(theme_mode="high_contrast")
