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

"""Emoji presentation adapter for agent layer.

This adapter implements PresentationProtocol by delegating to victor.ui.emoji,
providing settings-aware emoji/icon rendering for the agent layer.

Usage:
    from victor.agent.presentation import EmojiPresentationAdapter

    adapter = EmojiPresentationAdapter()
    icon = adapter.icon("success")  # Respects use_emojis setting
    status = adapter.format_status("Done", "success")  # "[green]✓[/] Done"
"""

from __future__ import annotations

from victor.ui.emoji import get_icon, is_emoji_enabled


class EmojiPresentationAdapter:
    """Presentation adapter that delegates to victor.ui.emoji module.

    This adapter provides a clean abstraction for the agent layer to use
    icons/emojis without directly depending on the UI module.

    The adapter respects the `use_emojis` setting from configuration,
    automatically switching between emoji and text alternatives.

    Example:
        adapter = EmojiPresentationAdapter()

        # Get icon (respects settings)
        icon = adapter.icon("success")

        # Force emoji regardless of settings
        icon = adapter.icon("error", force_emoji=True)

        # Format status message
        msg = adapter.format_status("Operation complete", "success")
        # Returns: "[green]✓[/] Operation complete"
    """

    @property
    def emojis_enabled(self) -> bool:
        """Whether emoji display is enabled.

        Returns:
            True if emojis should be displayed, False for text alternatives.
        """
        return is_emoji_enabled()

    def icon(
        self,
        name: str,
        *,
        force_emoji: bool = False,
        force_text: bool = False,
        with_color: bool = True,
    ) -> str:
        """Get an icon by name.

        Delegates to victor.ui.emoji.get_icon().

        Args:
            name: Icon name (e.g., "success", "error", "warning", "info",
                  "running", "pending", "done", "arrow_right", "bullet", etc.)
            force_emoji: Force emoji regardless of settings
            force_text: Force text regardless of settings
            with_color: Include Rich color markup

        Returns:
            The icon string, optionally with Rich color markup.

        Raises:
            KeyError: If icon name is not found.
        """
        return get_icon(
            name,
            force_emoji=force_emoji,
            force_text=force_text,
            with_color=with_color,
        )

    def format_status(self, message: str, icon_name: str) -> str:
        """Format a status message with an icon prefix.

        Args:
            message: The status message text
            icon_name: Name of the icon to prepend

        Returns:
            Formatted message like "[icon] message"
        """
        icon = self.icon(icon_name)
        return f"{icon} {message}"

    def format_tool_name(self, name: str) -> str:
        """Format a tool name for display.

        Applies consistent styling to tool names for display in the agent output.

        Args:
            name: The tool name (e.g., "read_file", "shell_exec")

        Returns:
            Formatted tool name with Rich markup styling
        """
        # Use cyan color for tool names, consistent with existing UI
        return f"[cyan]{name}[/]"


__all__ = ["EmojiPresentationAdapter"]
