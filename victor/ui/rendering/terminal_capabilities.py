"""Terminal capabilities detection and adaptation.

Detects terminal features to adapt output rendering for optimal display
across different terminal emulators and environments. Supports graceful
degradation for basic terminals, CI/CD, and headless environments.

Features:
- **Emoji support**: Check if terminal can display emoji
- **Unicode support**: Check if terminal supports Unicode characters
- **Color depth**: Detect 8/256/16M color support
- **Terminal width**: Get usable terminal width in characters
- **Adaptation**: Provide adapted renderables for current terminal
"""

from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class TerminalCapability(str, Enum):
    """Capability levels for terminal features."""

    FULL = "full"  # All features enabled
    BASIC = "basic"  # ASCII-only, limited colors
    RESTRICTED = "restricted"  # Plain text, no formatting
    UNKNOWN = "unknown"  # Capability not yet detected


class ColorDepth(int, Enum):
    """Color depth levels."""

    MONOCHROME = 0  # No color support
    ANSI_8 = 8  # 8 colors (ANSI)
    ANSI_256 = 256  # 256 colors
    TRUECOLOR = 16777216  # 16.7M colors (TrueColor)


@dataclass
class TerminalProfile:
    """Detected terminal capabilities profile.

    Attributes:
        terminal_emulator: Name of the terminal emulator (e.g., "iTerm2", "xterm")
        color_depth: Detected color depth
        supports_emoji: Whether emoji display is supported
        supports_unicode: Whether Unicode is supported
        width: Terminal width in characters
        height: Terminal height in characters (0 if unknown)
        is_interactive: Whether running in an interactive terminal
        is_ci: Whether running in a CI/CD environment
        os_type: Operating system type
        capabilities: Overall capability level
    """

    terminal_emulator: str = "unknown"
    color_depth: ColorDepth = ColorDepth.TRUECOLOR
    supports_emoji: bool = True
    supports_unicode: bool = True
    width: int = 80
    height: int = 24
    is_interactive: bool = True
    is_ci: bool = False
    os_type: str = ""
    capabilities: TerminalCapability = TerminalCapability.FULL


class TerminalCapabilities:
    """Detect and adapt to terminal capabilities.

    Singleton-style class that detects terminal capabilities once and
    provides adaptation methods for rendering decisions.

    Usage:
        caps = TerminalCapabilities()
        if caps.supports_emoji():
            icon = "🔍"
        else:
            icon = "[?]"

        color_depth = caps.get_color_depth()
        width = caps.get_terminal_width()
    """

    _instance: Optional[TerminalCapabilities] = None
    _profile: Optional[TerminalProfile] = None

    def __new__(cls) -> TerminalCapabilities:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._profile is None:
            self._profile = self._detect_profile()

    # ── Public API ────────────────────────────────────────────────

    def supports_emoji(self) -> bool:
        """Check if terminal supports emoji display.

        Returns:
            True if emoji display is supported
        """
        return self._profile.supports_emoji

    def supports_unicode(self) -> bool:
        """Check if terminal supports Unicode characters.

        Returns:
            True if Unicode is supported
        """
        return self._profile.supports_unicode

    def get_color_depth(self) -> ColorDepth:
        """Return detected color depth.

        Returns:
            ColorDepth enum value
        """
        return self._profile.color_depth

    def supports_color(self) -> int:
        """Return color depth as integer.

        Returns:
            0 for monochrome, 8 for 8 colors, 256 for 256 colors,
            16777216 for TrueColor
        """
        return self._profile.color_depth.value

    def get_terminal_width(self) -> int:
        """Return terminal width in characters.

        Returns:
            Terminal width (defaults to 80 if undetectable)
        """
        return self._profile.width

    def get_terminal_height(self) -> int:
        """Return terminal height in characters.

        Returns:
            Terminal height (defaults to 24 if undetectable)
        """
        return self._profile.height

    def get_profile(self) -> TerminalProfile:
        """Return the full terminal profile.

        Returns:
            TerminalProfile with all detected capabilities
        """
        return self._profile

    def is_interactive(self) -> bool:
        """Check if running in an interactive terminal.

        Returns:
            True if interactive
        """
        return self._profile.is_interactive

    def is_ci_environment(self) -> bool:
        """Check if running in a CI/CD environment.

        Returns:
            True if running in CI/CD
        """
        return self._profile.is_ci

    def get_capability_level(self) -> TerminalCapability:
        """Return the overall capability level.

        Returns:
            TerminalCapability enum value
        """
        return self._profile.capabilities

    def get_os_type(self) -> str:
        """Return the operating system type.

        Returns:
            OS type string (e.g., "darwin", "linux", "windows")
        """
        return self._profile.os_type

    # ── Adaptation Helpers ────────────────────────────────────────

    def emoji_or_text(self, emoji: str, text: str) -> str:
        """Return emoji or text fallback based on terminal support.

        Args:
            emoji: Emoji character to use if supported
            text: Text fallback if emoji not supported

        Returns:
            Emoji or text string based on terminal capabilities
        """
        return emoji if self.supports_emoji() else text

    def status_icon(self, status: str) -> str:
        """Return a status icon adapted to terminal capabilities.

        Args:
            status: Status name ("success", "failure", "warning", "pending",
                   "running", "cached", "pruned")

        Returns:
            Appropriate icon or text representation
        """
        if self.supports_emoji():
            icons = {
                "success": "✅",
                "failure": "❌",
                "warning": "⚠️",
                "pending": "⏳",
                "running": "🔄",
                "cached": "⚡",
                "pruned": "📎",
            }
            return icons.get(status, "•")
        else:
            icons = {
                "success": "[OK]",
                "failure": "[FAIL]",
                "warning": "[WARN]",
                "pending": "[...]",
                "running": "[RUN]",
                "cached": "[CACHE]",
                "pruned": "[TRIM]",
            }
            return icons.get(status, "[*]")

    def section_header(self, title: str) -> str:
        """Return a section header adapted to terminal capabilities.

        Args:
            title: Section title

        Returns:
            Formatted section header
        """
        width = min(self.get_terminal_width(), 80)
        if self.supports_unicode():
            return f"\n{'─' * (width // 2)} {title} {'─' * (width // 2)}\n"
        else:
            return f"\n--- {title} ---\n"

    # ── Detection ─────────────────────────────────────────────────

    def _detect_profile(self) -> TerminalProfile:
        """Detect all terminal capabilities.

        Returns:
            TerminalProfile with detected capabilities
        """
        os_type = platform.system().lower()
        term = os.environ.get("TERM", "").lower()
        colorterm = os.environ.get("COLORTERM", "").lower()
        is_ci = self._detect_ci()
        is_interactive = self._detect_interactive()

        # Detect terminal emulator
        emulator = self._detect_emulator(term, os_type)

        # Detect color depth
        color_depth = self._detect_color_depth(term, colorterm, os_type)

        # Detect Unicode/emoji support
        supports_unicode = self._detect_unicode_support(term, os_type, emulator)
        supports_emoji = self._detect_emoji_support(os_type, emulator, is_ci)

        # Get terminal size
        width, height = self._get_terminal_size()

        # Determine overall capability level
        capabilities = self._determine_capability_level(
            color_depth, supports_unicode, is_ci, is_interactive
        )

        return TerminalProfile(
            terminal_emulator=emulator,
            color_depth=color_depth,
            supports_emoji=supports_emoji,
            supports_unicode=supports_unicode,
            width=width,
            height=height,
            is_interactive=is_interactive,
            is_ci=is_ci,
            os_type=os_type,
            capabilities=capabilities,
        )

    @staticmethod
    def _detect_emulator(term: str, os_type: str) -> str:
        """Detect the terminal emulator."""
        # Check common environment variables
        emulator = os.environ.get("TERM_PROGRAM", "").lower()
        if emulator:
            return emulator

        # Check TERM
        if "iterm" in term:
            return "iterm2"
        if "kitty" in term:
            return "kitty"
        if "alacritty" in term:
            return "alacritty"
        if "konsole" in term:
            return "konsole"
        if "gnome" in term or "vte" in term:
            return "gnome-terminal"
        if "xterm" in term:
            return "xterm"
        if "screen" in term:
            return "screen"
        if "tmux" in term:
            return "tmux"

        # Check OS-specific
        if os_type == "darwin":
            return "terminal.app"
        if os_type == "windows":
            return "windows-terminal"

        return "unknown"

    @staticmethod
    def _detect_color_depth(term: str, colorterm: str, os_type: str) -> ColorDepth:
        """Detect terminal color depth."""
        # COLORTERM is the most reliable indicator
        if colorterm in ("truecolor", "24bit"):
            return ColorDepth.TRUECOLOR

        # Check TERM
        if "truecolor" in term or "24bit" in term:
            return ColorDepth.TRUECOLOR
        if term.endswith("-256color"):
            return ColorDepth.ANSI_256

        # Check environment variables
        color_depth_env = os.environ.get("VICTOR_COLOR_DEPTH", "")
        if color_depth_env == "truecolor":
            return ColorDepth.TRUECOLOR
        if color_depth_env == "256":
            return ColorDepth.ANSI_256
        if color_depth_env == "8":
            return ColorDepth.ANSI_8
        if color_depth_env == "0" or color_depth_env == "mono":
            return ColorDepth.MONOCHROME

        # macOS Terminal.app and iTerm2 support TrueColor
        if os_type == "darwin":
            return ColorDepth.TRUECOLOR

        # Windows Terminal supports TrueColor
        if os_type == "windows" and "WT_SESSION" in os.environ:
            return ColorDepth.TRUECOLOR

        # Check if we can detect via tput
        try:
            result = subprocess.run(
                ["tput", "colors"],
                capture_output=True,
                text=True,
                timeout=1,
            )
            if result.returncode == 0:
                colors = int(result.stdout.strip())
                if colors >= 16777216:
                    return ColorDepth.TRUECOLOR
                elif colors >= 256:
                    return ColorDepth.ANSI_256
                elif colors >= 8:
                    return ColorDepth.ANSI_8
        except (FileNotFoundError, ValueError, subprocess.TimeoutExpired):
            pass

        # Default based on TERM
        if "256" in term:
            return ColorDepth.ANSI_256
        if "color" in term:
            return ColorDepth.ANSI_8

        return ColorDepth.ANSI_8  # Safe default

    @staticmethod
    def _detect_unicode_support(term: str, os_type: str, emulator: str) -> bool:
        """Detect Unicode support."""
        # Modern terminals almost always support Unicode
        if os_type == "darwin":
            return True  # macOS always supports Unicode

        if os_type == "windows":
            # Windows Terminal supports Unicode
            if "WT_SESSION" in os.environ:
                return True
            # Old CMD.exe has limited Unicode support
            return False

        # Linux: most modern terminals support Unicode
        unicode_terminals = {"kitty", "alacritty", "gnome-terminal", "konsole", "tmux", "screen"}
        if emulator in unicode_terminals:
            return True
        if "utf" in term or "unicode" in term:
            return True
        if "linux" in term:
            return False  # Linux console has limited Unicode

        # Check locale
        locale = os.environ.get("LC_ALL", os.environ.get("LC_CTYPE", os.environ.get("LANG", "")))
        if "UTF-8" in locale or "utf-8" in locale:
            return True

        return True  # Assume Unicode support by default

    @staticmethod
    def _detect_emoji_support(os_type: str, emulator: str, is_ci: bool) -> bool:
        """Detect emoji support."""
        if is_ci:
            return False  # CI/CD rarely supports emoji

        if os_type == "darwin":
            return True  # macOS supports emoji in all terminals

        if os_type == "windows":
            # Windows Terminal supports emoji
            if "WT_SESSION" in os.environ:
                return True
            return False

        # Linux: check terminal emulator
        emoji_terminals = {"kitty", "alacritty", "gnome-terminal", "konsole"}
        return emulator in emoji_terminals

    @staticmethod
    def _detect_ci() -> bool:
        """Detect CI/CD environment."""
        ci_vars = {
            "CI",
            "GITHUB_ACTIONS",
            "GITLAB_CI",
            "JENKINS_URL",
            "CIRCLECI",
            "TRAVIS",
            "BUILDKITE",
            "TEAMCITY_VERSION",
            "TF_BUILD",
            "CODEBUILD_BUILD_ID",
        }
        return any(os.environ.get(var) for var in ci_vars)

    @staticmethod
    def _detect_interactive() -> bool:
        """Detect if running in an interactive terminal."""
        # Check if stdout is a terminal
        if not os.isatty(1):  # stdout
            return False

        # Check for common non-interactive indicators
        if os.environ.get("VICTOR_NON_INTERACTIVE"):
            return False

        return True

    @staticmethod
    def _get_terminal_size() -> tuple:
        """Get terminal size in characters.

        Returns:
            Tuple of (width, height), defaulting to (80, 24)
        """
        try:
            size = shutil.get_terminal_size()
            return size.columns, size.lines
        except Exception:
            return 80, 24

    @staticmethod
    def _determine_capability_level(
        color_depth: ColorDepth,
        supports_unicode: bool,
        is_ci: bool,
        is_interactive: bool,
    ) -> TerminalCapability:
        """Determine overall capability level."""
        if not is_interactive or is_ci:
            return TerminalCapability.RESTRICTED

        if color_depth == ColorDepth.MONOCHROME or not supports_unicode:
            return TerminalCapability.BASIC

        return TerminalCapability.FULL
