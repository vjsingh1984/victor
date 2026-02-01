"""Keybinding Configuration System for Victor TUI.

Provides customizable keyboard shortcuts with preset configurations
and support for user-defined keybindings.
"""

from dataclasses import dataclass
from typing import Optional


# =============================================================================
# DEFAULT KEYBINDINGS
# =============================================================================

DEFAULT_KEYBINDINGS: dict[str, str] = {
    # Navigation
    "quit": "ctrl+c",
    "clear": "ctrl+l",
    "focus_input": "escape",
    "scroll_up": "ctrl+up",
    "scroll_down": "ctrl+down",
    "scroll_top": "ctrl+home",
    "scroll_bottom": "ctrl+end",
    # Panels
    "toggle_thinking": "ctrl+t",
    "toggle_tools": "ctrl+y",
    "toggle_details": "ctrl+d",
    # Actions
    "cancel_stream": "ctrl+x",
    "show_help": "ctrl+slash",
    # Sessions
    "resume_any_session": "ctrl+g",
    "resume_project_session": "ctrl+p",
    "resume_session": "ctrl+r",
    "save_session": "ctrl+s",
    "export_session": "ctrl+e",
    # Themes
    "next_theme": "ctrl+right",
    "prev_theme": "ctrl+left",
    # Input (not remappable)
    "submit": "enter",
    "newline": "shift+enter",
    "history_prev": "up",
    "history_next": "down",
}


# =============================================================================
# KEYBINDING PRESETS
# =============================================================================

VIM_PRESET: dict[str, str] = {
    **DEFAULT_KEYBINDINGS,
    "quit": "ctrl+q",
    "clear": "ctrl+d",
    "toggle_thinking": "ctrl+w",
    "toggle_tools": "ctrl+b",
    "show_help": "f1",
}


EMACS_PRESET: dict[str, str] = {
    **DEFAULT_KEYBINDINGS,
    "quit": "ctrl+x,ctrl+c",
    "clear": "ctrl+x,k",
    "toggle_thinking": "ctrl+x,ctrl+t",
    "show_help": "ctrl+h",
}


# =============================================================================
# KEYBINDING CONFIG CLASS
# =============================================================================


@dataclass
class KeybindingConfig:
    """Keybinding configuration.

    Attributes:
        bindings: Dictionary mapping action names to key combinations
        preset_name: Name of the preset (if any)
    """

    bindings: dict[str, str]
    preset_name: Optional[str] = None

    def get_binding(self, action: str) -> Optional[str]:
        """Get keybinding for an action.

        Args:
            action: Action name

        Returns:
            Key combination string or None if not found
        """
        return self.bindings.get(action)

    def set_binding(self, action: str, keys: str) -> None:
        """Set keybinding for an action.

        Args:
            action: Action name
            keys: Key combination (e.g., "ctrl+x")
        """
        self.bindings[action] = keys

    def remove_binding(self, action: str) -> None:
        """Remove a keybinding.

        Args:
            action: Action name to remove
        """
        if action in self.bindings:
            del self.bindings[action]

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary.

        Returns:
            Dictionary of bindings
        """
        return self.bindings.copy()

    @classmethod
    def from_dict(
        cls, bindings: dict[str, str], preset_name: Optional[str] = None
    ) -> "KeybindingConfig":
        """Create from dictionary.

        Args:
            bindings: Dictionary of bindings
            preset_name: Optional preset name

        Returns:
            KeybindingConfig instance
        """
        return cls(bindings=bindings.copy(), preset_name=preset_name)


# =============================================================================
# PRESET REGISTRY
# =============================================================================

KEYBINDING_PRESETS: dict[str, dict[str, str]] = {
    "default": DEFAULT_KEYBINDINGS,
    "vim": VIM_PRESET,
    "emacs": EMACS_PRESET,
}


def get_preset(name: str) -> dict[str, str]:
    """Get a keybinding preset by name.

    Args:
        name: Preset name

    Returns:
        Dictionary of keybindings

    Raises:
        KeyError: If preset not found
    """
    if name not in KEYBINDING_PRESETS:
        raise KeyError(
            f"Keybinding preset '{name}' not found. Available: {list(KEYBINDING_PRESETS.keys())}"
        )
    return KEYBINDING_PRESETS[name].copy()


def create_config_from_preset(preset_name: str) -> KeybindingConfig:
    """Create a KeybindingConfig from a preset.

    Args:
        preset_name: Name of the preset

    Returns:
        KeybindingConfig instance
    """
    bindings = get_preset(preset_name)
    return KeybindingConfig(bindings=bindings, preset_name=preset_name)


def list_presets() -> list[str]:
    """List available preset names.

    Returns:
        List of preset names
    """
    return list(KEYBINDING_PRESETS.keys())


def register_preset(name: str, bindings: dict[str, str]) -> None:
    """Register a custom keybinding preset.

    Args:
        name: Preset name
        bindings: Dictionary of keybindings
    """
    KEYBINDING_PRESETS[name] = bindings.copy()


# =============================================================================
# VALIDATION
# =============================================================================


def validate_binding(binding: str) -> bool:
    """Validate a keybinding string.

    Args:
        binding: Key combination string (e.g., "ctrl+x", "shift+enter")

    Returns:
        True if valid, False otherwise
    """
    if not binding:
        return False

    # Basic validation: must contain modifier and key, or just key
    # Supported modifiers: ctrl, shift, alt, meta
    # Supported keys: a-z, 0-9, f1-f12, home, end, up, down, left, right, etc.

    parts = binding.lower().split("+")
    if len(parts) == 0:
        return False

    # Validate each part
    modifiers = {"ctrl", "shift", "alt", "meta"}
    valid_keys = {
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "f1",
        "f2",
        "f3",
        "f4",
        "f5",
        "f6",
        "f7",
        "f8",
        "f9",
        "f10",
        "f11",
        "f12",
        "home",
        "end",
        "up",
        "down",
        "left",
        "right",
        "pageup",
        "pagedown",
        "space",
        "tab",
        "enter",
        "escape",
        "esc",
        "slash",
        "backslash",
        "comma",
        "period",
        "semicolon",
        "quote",
    }

    # All parts except the last must be modifiers
    for part in parts[:-1]:
        if part not in modifiers:
            return False

    # Last part must be a valid key
    if parts[-1] not in valid_keys and parts[-1] not in modifiers:
        return False

    return True


def validate_bindings(bindings: dict[str, str]) -> tuple[bool, list[str]]:
    """Validate a dictionary of bindings.

    Args:
        bindings: Dictionary of bindings

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    for action, binding in bindings.items():
        if not validate_binding(binding):
            errors.append(f"Invalid binding for '{action}': {binding}")

    return len(errors) == 0, errors
