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

"""Emoji utility module for Victor UI.

Provides centralized emoji/icon handling with toggle support based on settings.
When emojis are disabled, text alternatives are used for accessibility and
compatibility with terminals that don't support Unicode emojis.

Usage:
    from victor.ui.emoji import Icons, get_icon

    # Get an icon (respects settings.use_emojis)
    success_icon = get_icon("success")  # Returns "+" or "[green]+[/]" when disabled

    # Or use the Icons class directly with Rich markup
    icon = Icons.success()  # Returns "[green]✓[/]" or "[green]+[/]"

    # Force emoji/text regardless of settings
    icon = Icons.success(force_emoji=True)
    icon = Icons.success(force_text=True)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

# Global settings reference (lazy import to avoid circular deps)
_settings_cache: Optional[object] = None  # Type: Settings (lazy import)


def _get_settings():
    """Get settings with lazy import to avoid circular dependencies."""
    global _settings_cache
    if _settings_cache is None:
        try:
            from victor.config.settings import get_settings

            _settings_cache = get_settings()
        except ImportError:
            return None
    return _settings_cache


def is_emoji_enabled() -> bool:
    """Check if emojis are enabled in settings.

    Returns:
        True if emojis should be used, False for text alternatives.
    """
    settings = _get_settings()
    if settings is None:
        return True  # Default to enabled if settings unavailable
    return settings.ui.use_emojis


@dataclass(frozen=True)
class IconSet:
    """Defines emoji and text alternatives for an icon."""

    emoji: str
    text: str
    rich_color: Optional[str] = None  # Rich markup color

    def get(self, use_emoji: bool = True, with_color: bool = True) -> str:
        """Get the icon with optional Rich color markup.

        Args:
            use_emoji: Whether to use emoji (True) or text (False)
            with_color: Whether to wrap in Rich color markup

        Returns:
            The icon string, optionally with Rich color markup
        """
        icon = self.emoji if use_emoji else self.text
        if with_color and self.rich_color:
            return f"[{self.rich_color}]{icon}[/]"
        return icon


# Icon definitions: emoji version and text alternative
ICONS = {
    # Status indicators
    "success": IconSet("✓", "+", "green"),
    "error": IconSet("✗", "x", "red"),
    "warning": IconSet("⚠", "!", "yellow"),
    "info": IconSet("ℹ", "i", "blue"),
    # Tool execution
    "running": IconSet("🔧", "*", "cyan"),
    "pending": IconSet("⏳", "...", "dim"),
    "done": IconSet("✔", "OK", "green"),
    # Navigation/UI
    "arrow_right": IconSet("→", "->", None),
    "arrow_left": IconSet("←", "<-", None),
    "bullet": IconSet("•", "-", None),
    # Files/folders
    "file": IconSet("📄", "F", None),
    "folder": IconSet("📁", "D", None),
    # Misc
    "sparkle": IconSet("✨", "*", "yellow"),
    "search": IconSet("🔍", "?", None),
    "chart": IconSet("📊", "#", None),
    "target": IconSet("🎯", "@", None),
    "rocket": IconSet("🚀", "^", None),
    "bolt": IconSet("⚡", "!", "yellow"),
    "bulb": IconSet("💡", "*", "yellow"),
    "note": IconSet("📝", "#", None),
    "refresh": IconSet("🔄", "~", None),
    # Safety/risk level icons
    "risk_high": IconSet("🔴", "!!", "red"),
    "risk_critical": IconSet("⛔", "!!!", "red"),
    "unknown": IconSet("❓", "?", None),
    # Step status icons
    "skipped": IconSet("⏭️", ">>", "dim"),
    "blocked": IconSet("🔒", "[X]", "yellow"),
    # Additional UI icons
    "thinking": IconSet("💭", "...", "dim"),
    "gear": IconSet("⚙️", "[*]", "cyan"),
    "clipboard": IconSet("📋", "[=]", None),
    "stop": IconSet("⛔", "[!]", "red"),
    "clock": IconSet("⏰", "[T]", "yellow"),
    "stop_sign": IconSet("🛑", "[X]", "red"),
    # Severity/complexity indicators
    "level_low": IconSet("🟢", "[L]", "green"),
    "level_medium": IconSet("🟡", "[M]", "yellow"),
    "level_high": IconSet("🟠", "[H]", "yellow"),
    "level_critical": IconSet("🔴", "[!]", "red"),
    "level_info": IconSet("🔵", "[I]", "blue"),
    "level_unknown": IconSet("⚪", "[?]", None),
    "person": IconSet("👤", "[P]", None),
    # Platform/technology icons
    "terraform": IconSet("🏗️", "[TF]", None),
    "docker": IconSet("🐳", "[DK]", None),
    "kubernetes": IconSet("☸️", "[K8]", None),
    # Miscellaneous
    "hint": IconSet("💡", "*", "yellow"),
    "alert": IconSet("🚨", "(!)", "red"),
    "trend_up": IconSet("📈", "^", "green"),
    "trend_down": IconSet("📉", "v", "red"),
}


def get_icon(
    name: str,
    *,
    force_emoji: bool = False,
    force_text: bool = False,
    with_color: bool = True,
) -> str:
    """Get an icon by name, respecting settings.

    Args:
        name: Icon name (e.g., "success", "error", "warning")
        force_emoji: Force emoji regardless of settings
        force_text: Force text regardless of settings
        with_color: Include Rich color markup

    Returns:
        The icon string

    Raises:
        KeyError: If icon name is not found
    """
    if name not in ICONS:
        raise KeyError(f"Unknown icon: {name}. Available: {list(ICONS.keys())}")

    icon_set = ICONS[name]

    if force_emoji:
        use_emoji = True
    elif force_text:
        use_emoji = False
    else:
        use_emoji = is_emoji_enabled()

    return icon_set.get(use_emoji=use_emoji, with_color=with_color)


class Icons:
    """Class-based interface for icons with Rich color support.

    All methods return strings with Rich markup for terminal coloring.

    Example:
        print(Icons.success())  # "[green]✓[/]" or "[green]+[/]"
    """

    @staticmethod
    def _get(name: str, force_emoji: bool = False, force_text: bool = False) -> str:
        return get_icon(name, force_emoji=force_emoji, force_text=force_text)

    @classmethod
    def success(cls, force_emoji: bool = False, force_text: bool = False) -> str:
        """Green checkmark for success."""
        return cls._get("success", force_emoji, force_text)

    @classmethod
    def error(cls, force_emoji: bool = False, force_text: bool = False) -> str:
        """Red X for errors."""
        return cls._get("error", force_emoji, force_text)

    @classmethod
    def warning(cls, force_emoji: bool = False, force_text: bool = False) -> str:
        """Yellow warning sign."""
        return cls._get("warning", force_emoji, force_text)

    @classmethod
    def info(cls, force_emoji: bool = False, force_text: bool = False) -> str:
        """Blue info indicator."""
        return cls._get("info", force_emoji, force_text)

    @classmethod
    def running(cls, force_emoji: bool = False, force_text: bool = False) -> str:
        """Wrench for running operations."""
        return cls._get("running", force_emoji, force_text)

    @classmethod
    def pending(cls, force_emoji: bool = False, force_text: bool = False) -> str:
        """Hourglass for pending operations."""
        return cls._get("pending", force_emoji, force_text)

    @classmethod
    def done(cls, force_emoji: bool = False, force_text: bool = False) -> str:
        """Checkmark for completed operations."""
        return cls._get("done", force_emoji, force_text)


def reset_settings_cache() -> None:
    """Reset the settings cache (for testing)."""
    global _settings_cache
    _settings_cache = None


__all__ = [
    "Icons",
    "IconSet",
    "ICONS",
    "get_icon",
    "is_emoji_enabled",
    "reset_settings_cache",
]
