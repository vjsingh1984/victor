# Copyright 2025 Vijaykumar Singh
#
# Licensed under the Apache License, Version 2.0

"""Unit tests for CapabilityAdapter (DIP compliance).

Tests for adapter that bridges legacy direct orchestrator writes
with new VerticalContext-based config storage (TDD approach).
"""

import os
import pytest
from typing import Any, Dict, Optional
from unittest.mock import Mock, patch, call

from victor.core.verticals.context import VerticalContext
from victor.core.verticals.capability_adapter import (
    CapabilityAdapter,
    get_capability_adapter,
    LegacyWriteMode,
)


class TestCapabilityAdapterInit:
    """Test suite for CapabilityAdapter initialization."""

    def test_init_with_context(self):
        """Should initialize with VerticalContext."""
        context = VerticalContext(name="coding")
        adapter = CapabilityAdapter(context)

        assert adapter.context is context
        assert adapter.legacy_mode == LegacyWriteMode.BACKWARD_COMPATIBLE

    def test_init_with_legacy_mode(self):
        """Should initialize with specified legacy mode."""
        context = VerticalContext(name="coding")
        adapter = CapabilityAdapter(context, legacy_mode=LegacyWriteMode.CONTEXT_ONLY)

        assert adapter.legacy_mode == LegacyWriteMode.CONTEXT_ONLY

    def test_init_with_feature_flag(self):
        """Should respect VICTOR_USE_CONTEXT_CONFIG feature flag."""
        context = VerticalContext(name="coding")

        # Test with flag=false (default: backward compatible)
        with patch.dict(os.environ, {"VICTOR_USE_CONTEXT_CONFIG": "false"}):
            adapter = CapabilityAdapter(context)
            assert adapter.legacy_mode == LegacyWriteMode.BACKWARD_COMPATIBLE

        # Test with flag=true (context only)
        with patch.dict(os.environ, {"VICTOR_USE_CONTEXT_CONFIG": "true"}):
            adapter = CapabilityAdapter(context)
            assert adapter.legacy_mode == LegacyWriteMode.CONTEXT_ONLY


class TestSetCapabilityConfig:
    """Test suite for set_capability_config method."""

    def test_stores_in_context(self):
        """Should store config in VerticalContext."""
        context = VerticalContext(name="coding")
        adapter = CapabilityAdapter(context)

        adapter.set_capability_config("code_style", {"formatter": "black"})

        assert context.get_capability_config("code_style") == {"formatter": "black"}

    def test_stores_in_orchestrator_legacy_mode(self):
        """Should also store in orchestrator in BACKWARD_COMPATIBLE mode."""
        context = VerticalContext(name="coding")
        orchestrator = Mock()
        adapter = CapabilityAdapter(context, legacy_mode=LegacyWriteMode.BACKWARD_COMPATIBLE)

        adapter.set_capability_config(orchestrator, "code_style", {"formatter": "black"})

        # Check both storage locations
        assert context.get_capability_config("code_style") == {"formatter": "black"}
        assert orchestrator.code_style == {"formatter": "black"}

    def test_context_only_mode_skips_orchestrator(self):
        """Should skip orchestrator writes in CONTEXT_ONLY mode."""
        context = VerticalContext(name="coding")
        orchestrator = Mock()
        adapter = CapabilityAdapter(context, legacy_mode=LegacyWriteMode.CONTEXT_ONLY)

        adapter.set_capability_config(orchestrator, "code_style", {"formatter": "black"})

        # Check only context storage
        assert context.get_capability_config("code_style") == {"formatter": "black"}
        assert not hasattr(orchestrator, "code_style")

    def test_emits_deprecation_warning_in_compat_mode(self):
        """Should emit deprecation warning when writing to orchestrator."""
        context = VerticalContext(name="coding")
        orchestrator = Mock()
        adapter = CapabilityAdapter(context, legacy_mode=LegacyWriteMode.BACKWARD_COMPATIBLE)

        with pytest.warns(DeprecationWarning, match="Direct orchestrator attribute.*deprecated"):
            adapter.set_capability_config(orchestrator, "code_style", {"formatter": "black"})

    def test_nested_dict_updates(self):
        """Should handle nested dict updates correctly."""
        context = VerticalContext(name="coding")
        orchestrator = Mock()
        orchestrator.safety_config = {}
        adapter = CapabilityAdapter(context, legacy_mode=LegacyWriteMode.BACKWARD_COMPATIBLE)

        # Simulate git safety config pattern
        adapter.set_capability_config(
            orchestrator,
            "safety_config.git",
            {"require_tests_before_commit": True, "allowed_branches": ["main", "develop"]},
        )

        # Check nested update happened
        assert orchestrator.safety_config["git"] == {
            "require_tests_before_commit": True,
            "allowed_branches": ["main", "develop"],
        }


class TestGetCapabilityConfig:
    """Test suite for get_capability_config method."""

    def test_retrieves_from_context(self):
        """Should retrieve from VerticalContext."""
        context = VerticalContext(name="coding")
        context.set_capability_config("code_style", {"formatter": "black"})
        adapter = CapabilityAdapter(context)

        result = adapter.get_capability_config("code_style")

        assert result == {"formatter": "black"}

    def test_falls_back_to_orchestrator_compat_mode(self):
        """Should fall back to orchestrator in BACKWARD_COMPATIBLE mode."""
        context = VerticalContext(name="coding")
        orchestrator = Mock()
        orchestrator.code_style = {"formatter": "ruff"}

        adapter = CapabilityAdapter(context, legacy_mode=LegacyWriteMode.BACKWARD_COMPATIBLE)

        result = adapter.get_capability_config(orchestrator, "code_style")

        assert result == {"formatter": "ruff"}

    def test_returns_default_if_not_found(self):
        """Should return default value if config not found."""
        context = VerticalContext(name="coding")
        adapter = CapabilityAdapter(context)

        result = adapter.get_capability_config("nonexistent", default={})

        assert result == {}

    def test_context_only_mode_no_fallback(self):
        """Should not fall back to orchestrator in CONTEXT_ONLY mode."""
        context = VerticalContext(name="coding")
        orchestrator = Mock()
        orchestrator.code_style = {"formatter": "ruff"}

        adapter = CapabilityAdapter(context, legacy_mode=LegacyWriteMode.CONTEXT_ONLY)

        result = adapter.get_capability_config(orchestrator, "code_style")

        # Should not find it in context-only mode
        assert result is None


class TestApplyCapabilityConfigs:
    """Test suite for apply_capability_configs method."""

    def test_applies_multiple_configs(self):
        """Should apply multiple configs at once."""
        context = VerticalContext(name="coding")
        orchestrator = Mock()
        adapter = CapabilityAdapter(context, legacy_mode=LegacyWriteMode.BACKWARD_COMPATIBLE)

        configs = {
            "code_style": {"formatter": "black"},
            "test_config": {"framework": "pytest"},
            "lsp_config": {"languages": ["python"]},
        }

        adapter.apply_capability_configs(orchestrator, configs)

        # Check all configs stored
        assert context.get_capability_config("code_style") == {"formatter": "black"}
        assert context.get_capability_config("test_config") == {"framework": "pytest"}
        assert context.get_capability_config("lsp_config") == {"languages": ["python"]}

        # Check orchestrator has them too (backward compat)
        assert orchestrator.code_style == {"formatter": "black"}
        assert orchestrator.test_config == {"framework": "pytest"}
        assert orchestrator.lsp_config == {"languages": ["python"]}

    def test_context_only_mode_skips_orchestrator(self):
        """Should only store in context in CONTEXT_ONLY mode."""
        context = VerticalContext(name="coding")
        orchestrator = Mock()
        adapter = CapabilityAdapter(context, legacy_mode=LegacyWriteMode.CONTEXT_ONLY)

        configs = {
            "code_style": {"formatter": "black"},
            "test_config": {"framework": "pytest"},
        }

        adapter.apply_capability_configs(orchestrator, configs)

        # Check only context storage
        assert context.get_capability_config("code_style") == {"formatter": "black"}
        assert not hasattr(orchestrator, "code_style")


class TestDeprecationWarnings:
    """Test suite for deprecation warning behavior."""

    def test_warns_on_legacy_attribute_access(self):
        """Should warn when accessing orchestrator attributes directly."""
        context = VerticalContext(name="coding")
        orchestrator = Mock()
        orchestrator.code_style = {"formatter": "black"}

        adapter = CapabilityAdapter(context, legacy_mode=LegacyWriteMode.BACKWARD_COMPATIBLE)

        with pytest.warns(DeprecationWarning):
            _ = adapter.get_capability_config(orchestrator, "code_style")

    def test_no_warning_in_context_only_mode(self):
        """Should not warn in CONTEXT_ONLY mode."""
        context = VerticalContext(name="coding")
        orchestrator = Mock()

        adapter = CapabilityAdapter(context, legacy_mode=LegacyWriteMode.CONTEXT_ONLY)

        # Should not emit warning
        with pytest.warns(None) as warning_list:
            adapter.set_capability_config(orchestrator, "code_style", {"formatter": "black"})

        assert len(warning_list) == 0


class TestMigrationHelpers:
    """Test suite for migration helper methods."""

    def test_migrate_from_orchestrator(self):
        """Should migrate all configs from orchestrator to context."""
        context = VerticalContext(name="coding")
        orchestrator = Mock()
        orchestrator.code_style = {"formatter": "black"}
        orchestrator.test_config = {"framework": "pytest"}
        orchestrator.lsp_config = {"languages": ["python"]}

        adapter = CapabilityAdapter(context)

        migrated = adapter.migrate_from_orchestrator(orchestrator)

        # Check migration happened
        assert context.get_capability_config("code_style") == {"formatter": "black"}
        assert context.get_capability_config("test_config") == {"framework": "pytest"}
        assert context.get_capability_config("lsp_config") == {"languages": ["python"]}
        assert migrated == 3  # 3 configs migrated

    def test_clear_orchestrator_configs(self):
        """Should clear configs from orchestrator after migration."""
        context = VerticalContext(name="coding")
        orchestrator = Mock()
        orchestrator.code_style = {"formatter": "black"}

        adapter = CapabilityAdapter(context)

        adapter.migrate_from_orchestrator(orchestrator)
        adapter.clear_orchestrator_configs(orchestrator, ["code_style"])

        # Check orchestrator attributes deleted
        del orchestrator.code_style
        assert not hasattr(orchestrator, "code_style")


class TestGetCapabilityAdapterFactory:
    """Test suite for get_capability_adapter factory."""

    def test_returns_adapter_for_context(self):
        """Should return CapabilityAdapter for VerticalContext."""
        context = VerticalContext(name="coding")

        adapter = get_capability_adapter(context)

        assert isinstance(adapter, CapabilityAdapter)
        assert adapter.context is context

    def test_caches_adapter_instance(self):
        """Should cache and return same adapter instance."""
        context = VerticalContext(name="coding")

        adapter1 = get_capability_adapter(context)
        adapter2 = get_capability_adapter(context)

        assert adapter1 is adapter2


class TestLegacyWriteMode:
    """Test suite for LegacyWriteMode enum."""

    def test_backward_compatible_mode(self):
        """BACKWARD_COMPATIBLE should write to both locations."""
        assert LegacyWriteMode.BACKWARD_COMPATIBLE.value == "backward_compatible"

    def test_context_only_mode(self):
        """CONTEXT_ONLY should write only to context."""
        assert LegacyWriteMode.CONTEXT_ONLY.value == "context_only"


class TestRealWorldIntegrationPatterns:
    """Test suite for real-world integration patterns from coding capabilities."""

    def test_configure_git_safety_pattern(self):
        """Should handle git safety configuration pattern."""
        context = VerticalContext(name="coding")
        orchestrator = Mock()
        orchestrator.safety_config = {}
        adapter = CapabilityAdapter(context, legacy_mode=LegacyWriteMode.BACKWARD_COMPATIBLE)

        # Pattern from victor/coding/capabilities.py:94-98
        adapter.set_capability_config(
            orchestrator,
            "safety_config",
            {
                "git": {
                    "require_tests_before_commit": True,
                    "allowed_branches": ["main", "develop"],
                }
            },
        )

        # Verify stored in context
        assert context.get_capability_config("safety_config") == {
            "git": {
                "require_tests_before_commit": True,
                "allowed_branches": ["main", "develop"],
            }
        }

        # Verify stored in orchestrator (backward compat)
        assert orchestrator.safety_config["git"]["require_tests_before_commit"] is True

    def test_configure_code_style_pattern(self):
        """Should handle code style configuration pattern."""
        context = VerticalContext(name="coding")
        orchestrator = Mock()
        adapter = CapabilityAdapter(context, legacy_mode=LegacyWriteMode.BACKWARD_COMPATIBLE)

        # Pattern from victor/coding/capabilities.py:120-127
        adapter.set_capability_config(
            orchestrator,
            "code_style",
            {
                "formatter": "black",
                "linter": "ruff",
                "max_line_length": 100,
                "enforce_type_hints": True,
            },
        )

        # Verify both locations
        assert context.get_capability_config("code_style")["formatter"] == "black"
        assert orchestrator.code_style["formatter"] == "black"

    def test_configure_test_requirements_pattern(self):
        """Should handle test requirements configuration pattern."""
        context = VerticalContext(name="coding")
        orchestrator = Mock()
        adapter = CapabilityAdapter(context, legacy_mode=LegacyWriteMode.BACKWARD_COMPATIBLE)

        # Pattern from victor/coding/capabilities.py:169-175
        adapter.set_capability_config(
            orchestrator,
            "test_config",
            {
                "min_coverage": 0.8,
                "required_patterns": ["test_*.py"],
                "framework": "pytest",
                "run_on_edit": False,
            },
        )

        # Verify both locations
        assert context.get_capability_config("test_config")["min_coverage"] == 0.8
        assert orchestrator.test_config["framework"] == "pytest"
