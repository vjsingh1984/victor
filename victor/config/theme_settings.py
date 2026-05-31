"""Theme and accessibility settings for Victor CLI.

This module provides configuration for visual appearance and accessibility
features to make the CLI usable for all users.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ThemeSettings(BaseModel):
    """Theme and accessibility configuration for CLI rendering."""

    # High contrast mode for better visibility
    high_contrast: bool = Field(
        default=False,
        description="Enable high contrast mode with brighter colors and stronger borders (default: disabled)",
    )

    # Colorblind mode support
    colorblind_mode: Literal["none", "protanopia", "deuteranopia", "tritanopia"] = Field(
        default="none",
        description="Colorblind-friendly palette mode (default: none)",
    )

    # Font size adjustment (for terminals that support it)
    font_size: Literal["small", "medium", "large"] = Field(
        default="medium",
        description="Font size preference for terminal output (default: medium)",
    )

    # Use symbols/icons alongside colors (not just color differentiation)
    use_symbol_indicators: bool = Field(
        default=True,
        description="Show symbols/icons alongside colors for better accessibility (default: enabled)",
    )

    # Reduce motion for users sensitive to animations
    reduce_motion: bool = Field(
        default=False,
        description="Reduce animations and motion effects (default: disabled)",
    )

    # Maximum line width for content wrapping
    max_line_width: int = Field(
        default=120,
        ge=80,
        le=200,
        description="Maximum line width before wrapping (default: 120)",
    )


def get_theme_settings() -> ThemeSettings:
    """Get the current theme settings from the global settings singleton.

    Returns:
        ThemeSettings instance with current configuration

    Example:
        >>> from victor.config.theme_settings import get_theme_settings
        >>> settings = get_theme_settings()
        >>> if settings.high_contrast:
        ...     use_high_contrast_colors()
    """
    from victor.config.settings import get_settings

    global_settings = get_settings()
    if global_settings.theme_settings is None:
        # Return default ThemeSettings if not configured
        return ThemeSettings()
    return global_settings.theme_settings
