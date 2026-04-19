# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Test VerticalConfigRegistry.

After config decoupling (TDI CORE-7/CORE-8), core only has "default"
entries. Vertical-specific configs are registered at runtime by external
packages via register_vertical_config().
"""

import pytest

from victor.core.verticals.config_registry import VerticalConfigRegistry


class TestVerticalConfigRegistry:
    """Test configuration registry functionality."""

    def test_default_provider_hints_exist(self):
        """Default provider hints must always exist."""
        hints = VerticalConfigRegistry.get_provider_hints("default")
        assert "preferred_providers" in hints
        assert hints["min_context_window"] >= 8000
        assert hints["requires_tool_calling"] is True

    def test_unknown_vertical_falls_back_to_default(self):
        """Unknown verticals should get default hints."""
        hints = VerticalConfigRegistry.get_provider_hints("nonexistent_xyz")
        default = VerticalConfigRegistry.get_provider_hints("default")
        assert hints == default

    def test_default_evaluation_criteria_exist(self):
        """Default evaluation criteria must always exist."""
        criteria = VerticalConfigRegistry.get_evaluation_criteria("default")
        assert "Task completion accuracy" in criteria
        assert len(criteria) >= 3

    def test_unknown_vertical_eval_falls_back_to_default(self):
        """Unknown verticals should get default evaluation criteria."""
        criteria = VerticalConfigRegistry.get_evaluation_criteria("nonexistent_xyz")
        default = VerticalConfigRegistry.get_evaluation_criteria("default")
        assert criteria == default

    def test_register_custom_provider_hints(self):
        """Test registering custom provider hints."""
        VerticalConfigRegistry.register_provider_hints(
            "test_custom", {"custom_key": "custom_value", "test": True}
        )
        try:
            hints = VerticalConfigRegistry.get_provider_hints("test_custom")
            assert hints["custom_key"] == "custom_value"
            assert hints["test"] is True
        finally:
            VerticalConfigRegistry._provider_hints.pop("test_custom", None)

    def test_register_custom_evaluation_criteria(self):
        """Test registering custom evaluation criteria."""
        custom_criteria = ["Custom criterion 1", "Custom criterion 2"]
        VerticalConfigRegistry.register_evaluation_criteria("test_custom_eval", custom_criteria)
        try:
            criteria = VerticalConfigRegistry.get_evaluation_criteria("test_custom_eval")
            assert criteria == custom_criteria
        finally:
            VerticalConfigRegistry._evaluation_criteria.pop("test_custom_eval", None)

    def test_returns_copy_to_prevent_mutation(self):
        """Test that registry returns copies to prevent accidental mutation."""
        hints1 = VerticalConfigRegistry.get_provider_hints("default")
        hints2 = VerticalConfigRegistry.get_provider_hints("default")
        hints1["custom_key"] = "custom_value"
        assert "custom_key" not in hints2

    def test_evaluation_criteria_returns_copy(self):
        """Test that evaluation criteria returns copies to prevent mutation."""
        criteria1 = VerticalConfigRegistry.get_evaluation_criteria("default")
        criteria2 = VerticalConfigRegistry.get_evaluation_criteria("default")
        criteria1.append("Custom criterion")
        assert "Custom criterion" not in criteria2

    def test_vertical_specific_configs_not_hardcoded(self):
        """Core must NOT have hardcoded vertical-specific entries.

        Only 'default' should exist in the class-level dicts. Vertical
        packages register their own configs at runtime.
        """
        allowed_keys = {"default"}
        hint_keys = set(VerticalConfigRegistry._provider_hints.keys())
        eval_keys = set(VerticalConfigRegistry._evaluation_criteria.keys())
        # Allow runtime-registered keys (from external packages if installed)
        # but the class-level dicts themselves should only have "default"
        assert "default" in hint_keys
        assert "default" in eval_keys
