from __future__ import annotations

import sys

import pytest

from victor.core.symbols import ICONS, IconSet, get_symbol, get_text_symbol


def test_symbol_registry_is_ui_free_import_surface():
    """Core symbol lookup must not import UI modules as a side effect."""
    loaded_before = set(sys.modules)

    assert get_symbol("success", use_emoji=False, with_color=False) == "+"
    assert "victor.ui.emoji" not in set(sys.modules) - loaded_before


def test_get_symbol_matches_iconset_rendering():
    icon = IconSet("✓", "+", "green")

    assert icon.get(use_emoji=True, with_color=True) == "[green]✓[/]"
    assert get_symbol("success", use_emoji=True, with_color=True) == "[green]✓[/]"
    assert get_text_symbol("success") == "+"


def test_all_symbols_have_text_and_emoji_alternatives():
    for name, icon in ICONS.items():
        assert icon.emoji, f"{name} missing emoji alternative"
        assert icon.text, f"{name} missing text alternative"


def test_unknown_symbol_raises_key_error():
    with pytest.raises(KeyError, match="Unknown icon"):
        get_symbol("missing")
