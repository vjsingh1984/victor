"""TUI Theme System for Victor.

Provides predefined themes and theme customization for the TUI.
Supports light, dark, high-contrast, and custom themes.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Theme:
    """TUI color theme configuration.

    Attributes:
        name: Theme identifier
        display_name: Human-readable theme name
        primary: Primary color (branding, links)
        secondary: Secondary color (accents)
        background: Background color
        surface: Surface color (cards, panels)
        panel: Panel color (sidebar, headers)
        panel_alt: Alternate panel color
        foreground: Text color
        text: Primary text color (alias for foreground)
        text_muted: Muted text color (metadata)
        success: Success color (positive feedback)
        warning: Warning color (cautions)
        error: Error color (errors, failures)
        error_bg: Error background color
        border_muted: Muted border color
        border_strong: Strong border color
        surface_darken_1: Darkened surface color
        primary_darken_2: Darkened primary color
    """

    name: str
    display_name: str
    primary: str
    secondary: str
    background: str
    surface: str
    panel: str
    panel_alt: str
    foreground: str
    text: str
    text_muted: str
    success: str
    warning: str
    error: str
    error_bg: str
    border_muted: str
    border_strong: str
    surface_darken_1: str
    primary_darken_2: str

    def to_css_vars(self) -> str:
        """Convert theme to Textual CSS variables."""
        return f"""
$text: {self.text};
$text-muted: {self.text_muted};
$background: {self.background};
$surface: {self.surface};
$surface-darken-1: {self.surface_darken_1};
$panel: {self.panel};
$panel-alt: {self.panel_alt};
$primary: {self.primary};
$primary-darken-2: {self.primary_darken_2};
$secondary: {self.secondary};
$success: {self.success};
$warning: {self.warning};
$error: {self.error};
$error-bg: {self.error_bg};
$border-muted: {self.border_muted};
$border-strong: {self.border_strong};
"""

    def to_dict(self) -> Dict[str, str]:
        """Convert theme to dictionary."""
        return {
            "primary": self.primary,
            "secondary": self.secondary,
            "background": self.background,
            "surface": self.surface,
            "panel": self.panel,
            "panel_alt": self.panel_alt,
            "foreground": self.foreground,
            "text": self.text,
            "text_muted": self.text_muted,
            "success": self.success,
            "warning": self.warning,
            "error": self.error,
            "error_bg": self.error_bg,
            "border_muted": self.border_muted,
            "border_strong": self.border_strong,
            "surface_darken_1": self.surface_darken_1,
            "primary_darken_2": self.primary_darken_2,
        }


# =============================================================================
# PREDEFINED THEMES
# =============================================================================

DEFAULT_THEME = Theme(
    name="default",
    display_name="Default",
    primary="#7cb7ff",
    secondary="#9ed0ff",
    background="#0d1117",
    surface="#121826",
    surface_darken_1="#0f1521",
    panel="#161f2e",
    panel_alt="#101723",
    foreground="#e6e9f2",
    text="#e6e9f2",
    text_muted="#9aa3b5",
    success="#5fd1a2",
    warning="#f5c169",
    error="#ff8787",
    error_bg="#3d1515",
    border_muted="#243146",
    border_strong="#32405a",
    primary_darken_2="#4f8fd6",
)

DARK_THEME = Theme(
    name="dark",
    display_name="Dark",
    primary="#58a6ff",
    secondary="#79c0ff",
    background="#010409",
    surface="#0d1117",
    surface_darken_1="#0a0e14",
    panel="#161b22",
    panel_alt="#0d1117",
    foreground="#c9d1d9",
    text="#c9d1d9",
    text_muted="#8b949e",
    success="#3fb950",
    warning="#d29922",
    error="#f85149",
    error_bg="#4a1818",
    border_muted="#30363d",
    border_strong="#6e7681",
    primary_darken_2="#1f6feb",
)

LIGHT_THEME = Theme(
    name="light",
    display_name="Light",
    primary="#0969da",
    secondary="#218bff",
    background="#ffffff",
    surface="#f6f8fa",
    surface_darken_1="#eaeef2",
    panel="#f6f8fa",
    panel_alt="#ffffff",
    foreground="#24292f",
    text="#24292f",
    text_muted="#57606a",
    success="#1a7f37",
    warning="#9a6700",
    error="#cf222e",
    error_bg="#ffebe9",
    border_muted="#d0d7de",
    border_strong="#babbbd",
    primary_darken_2="#0550ae",
)

HIGH_CONTRAST_THEME = Theme(
    name="high_contrast",
    display_name="High Contrast",
    primary="#ffffff",
    secondary="#ffffff",
    background="#000000",
    surface="#000000",
    surface_darken_1="#000000",
    panel="#000000",
    panel_alt="#000000",
    foreground="#ffffff",
    text="#ffffff",
    text_muted="#ffffff",
    success="#00ff00",
    warning="#ffff00",
    error="#ff0000",
    error_bg="#330000",
    border_muted="#ffffff",
    border_strong="#ffffff",
    primary_darken_2="#ffffff",
)

DRACULA_THEME = Theme(
    name="dracula",
    display_name="Dracula",
    primary="#bd93f9",
    secondary="#ff79c6",
    background="#282a36",
    surface="#44475a",
    surface_darken_1="#3d3f4d",
    panel="#44475a",
    panel_alt="#282a36",
    foreground="#f8f8f2",
    text="#f8f8f2",
    text_muted="#6272a4",
    success="#50fa7b",
    warning="#f1fa8c",
    error="#ff5555",
    error_bg="#442a2a",
    border_muted="#44475a",
    border_strong="#6272a4",
    primary_darken_2="#8e40ff",
)

NORD_THEME = Theme(
    name="nord",
    display_name="Nord",
    primary="#88c0d0",
    secondary="#81a1c1",
    background="#2e3440",
    surface="#3b4252",
    surface_darken_1="#353b49",
    panel="#434c5e",
    panel_alt="#3b4252",
    foreground="#d8dee9",
    text="#d8dee9",
    text_muted="#4c566a",
    success="#a3be8c",
    warning="#ebcb8b",
    error="#bf616a",
    error_bg="#4c3737",
    border_muted="#4c566a",
    border_strong="#616e88",
    primary_darken_2="#5e81ac",
)

# Theme registry
THEMES: Dict[str, Theme] = {
    "default": DEFAULT_THEME,
    "dark": DARK_THEME,
    "light": LIGHT_THEME,
    "high_contrast": HIGH_CONTRAST_THEME,
    "dracula": DRACULA_THEME,
    "nord": NORD_THEME,
}


def get_theme(name: str) -> Theme:
    """Get a theme by name.

    Args:
        name: Theme name

    Returns:
        Theme instance

    Raises:
        KeyError: If theme not found
    """
    if name not in THEMES:
        raise KeyError(f"Theme '{name}' not found. Available themes: {list(THEMES.keys())}")
    return THEMES[name]


def list_themes() -> list[str]:
    """List available theme names.

    Returns:
        List of theme names
    """
    return list(THEMES.keys())


def register_theme(theme: Theme) -> None:
    """Register a custom theme.

    Args:
        theme: Theme instance to register
    """
    THEMES[theme.name] = theme


def create_custom_theme(
    name: str,
    display_name: str,
    base_theme: str = "default",
    **overrides: Any,
) -> Theme:
    """Create a custom theme based on an existing theme.

    Args:
        name: Custom theme name
        display_name: Human-readable name
        base_theme: Base theme to inherit from
        **overrides: Color overrides (e.g., primary="#ff0000")

    Returns:
        New Theme instance
    """
    base = get_theme(base_theme)
    theme_dict = base.to_dict()
    theme_dict.update(overrides)

    return Theme(
        name=name,
        display_name=display_name,
        **theme_dict,
    )
