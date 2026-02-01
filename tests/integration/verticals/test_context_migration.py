# Copyright 2025 Vijaykumar Singh
#
# Licensed under the Apache License, Version 2.0

"""Integration tests for DIP compliance migration.

Tests vertical integration with context-based config storage,
verifying backward compatibility and new context-only mode.
"""

import os
from unittest.mock import Mock, patch

from victor.core.verticals.context import VerticalContext
from victor.core.verticals.capability_adapter import (
    CapabilityAdapter,
    get_capability_adapter,
    LegacyWriteMode,
)


class TestContextMigrationIntegration:
    """Integration tests for context-based config migration."""

    def test_coding_vertical_with_context_backward_compat(self):
        """Test coding vertical integration in backward compatible mode."""
        context = VerticalContext(name="coding")
        orchestrator = Mock()

        # Create adapter in backward compatible mode
        adapter = CapabilityAdapter(context, LegacyWriteMode.BACKWARD_COMPATIBLE)

        # Configure code style (mimicking actual vertical behavior)
        adapter.set_capability_config(
            orchestrator,
            "code_style",
            {
                "formatter": "black",
                "linter": "ruff",
                "max_line_length": 100,
            },
        )

        # Verify both locations (backward compat)
        assert context.get_capability_config("code_style") == {
            "formatter": "black",
            "linter": "ruff",
            "max_line_length": 100,
        }
        assert orchestrator.code_style == {
            "formatter": "black",
            "linter": "ruff",
            "max_line_length": 100,
        }

    def test_coding_vertical_with_context_only_mode(self):
        """Test coding vertical integration in context-only mode."""
        context = VerticalContext(name="coding")

        # Use a real object instead of Mock to avoid auto-attribute creation
        class FakeOrchestrator:
            pass

        orchestrator = FakeOrchestrator()

        # Create adapter in context-only mode
        adapter = CapabilityAdapter(context, LegacyWriteMode.CONTEXT_ONLY)

        # Configure code style
        adapter.set_capability_config(
            orchestrator,
            "code_style",
            {
                "formatter": "black",
                "linter": "ruff",
                "max_line_length": 100,
            },
        )

        # Verify only context storage
        assert context.get_capability_config("code_style") == {
            "formatter": "black",
            "linter": "ruff",
            "max_line_length": 100,
        }
        # Orchestrator should NOT have the attribute in context-only mode
        assert not hasattr(orchestrator, "code_style")

    def test_multiple_configs_in_single_context(self):
        """Test storing multiple capability configs in one context."""
        context = VerticalContext(name="coding")
        orchestrator = Mock()
        adapter = CapabilityAdapter(context, LegacyWriteMode.BACKWARD_COMPATIBLE)

        # Configure multiple capabilities (realistic vertical setup)
        configs = {
            "code_style": {"formatter": "black", "linter": "ruff"},
            "test_config": {"framework": "pytest", "min_coverage": 0.8},
            "lsp_config": {"languages": ["python", "typescript"]},
        }

        adapter.apply_capability_configs(orchestrator, configs)

        # Verify all configs stored
        for name, config in configs.items():
            assert context.get_capability_config(name) == config
            # Also in orchestrator (backward compat)
            assert getattr(orchestrator, name) == config

    def test_config_retrieval_with_fallback(self):
        """Test config retrieval with context-to-orchestrator fallback."""
        context = VerticalContext(name="coding")
        orchestrator = Mock()

        # Set config only in orchestrator (legacy scenario)
        orchestrator.code_style = {"formatter": "ruff", "linter": "flake8"}

        # Create adapter and retrieve
        adapter = CapabilityAdapter(context, LegacyWriteMode.BACKWARD_COMPATIBLE)
        retrieved = adapter.get_capability_config(orchestrator, "code_style")

        # Should fall back to orchestrator
        assert retrieved == {"formatter": "ruff", "linter": "flake8"}

    def test_nested_config_pattern(self):
        """Test nested config pattern (e.g., safety_config.git)."""
        context = VerticalContext(name="coding")
        orchestrator = Mock()
        orchestrator.safety_config = {}
        adapter = CapabilityAdapter(context, LegacyWriteMode.BACKWARD_COMPATIBLE)

        # Configure nested git safety (pattern from victor/coding/capabilities.py:94-98)
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

        # Verify stored in orchestrator (nested dict update)
        assert orchestrator.safety_config["git"] == {
            "require_tests_before_commit": True,
            "allowed_branches": ["main", "develop"],
        }


class TestFeatureFlagIntegration:
    """Integration tests for VICTOR_USE_CONTEXT_CONFIG feature flag."""

    def test_backward_compat_mode_with_false_flag(self):
        """Test VICTOR_USE_CONTEXT_CONFIG=false enables backward compat mode."""
        context = VerticalContext(name="coding")
        orchestrator = Mock()

        # Set flag to false
        with patch.dict(os.environ, {"VICTOR_USE_CONTEXT_CONFIG": "false"}):
            adapter = CapabilityAdapter(context)
            assert adapter.legacy_mode == LegacyWriteMode.BACKWARD_COMPATIBLE

            # Should write to both locations
            adapter.set_capability_config(orchestrator, "test_config", {"key": "value"})

            assert context.get_capability_config("test_config") == {"key": "value"}
            assert hasattr(orchestrator, "test_config")

    def test_context_only_mode_with_true_flag(self):
        """Test VICTOR_USE_CONTEXT_CONFIG=true enables context-only mode."""
        context = VerticalContext(name="coding")

        # Use a real object instead of Mock to avoid auto-attribute creation
        class FakeOrchestrator:
            pass

        orchestrator = FakeOrchestrator()

        # Set flag to true
        with patch.dict(os.environ, {"VICTOR_USE_CONTEXT_CONFIG": "true"}):
            adapter = CapabilityAdapter(context)
            assert adapter.legacy_mode == LegacyWriteMode.CONTEXT_ONLY

            # Should write only to context
            adapter.set_capability_config(orchestrator, "test_config", {"key": "value"})

            assert context.get_capability_config("test_config") == {"key": "value"}
            # Orchestrator should NOT have attribute (real object will raise AttributeError)
            assert not hasattr(orchestrator, "test_config")


class TestMigrationPath:
    """Tests for gradual migration from legacy to context-only."""

    def test_migrate_from_orchestrator_to_context(self):
        """Test migrating existing orchestrator configs to context."""
        context = VerticalContext(name="coding")

        # Use a real object instead of Mock to avoid auto-attribute creation
        class FakeOrchestrator:
            def __init__(self):
                self.code_style = {"formatter": "black"}
                self.test_config = {"framework": "pytest"}

        orchestrator = FakeOrchestrator()

        # Migrate to context
        adapter = CapabilityAdapter(context, LegacyWriteMode.CONTEXT_ONLY)
        migrated_count = adapter.migrate_from_orchestrator(
            orchestrator, config_names=["code_style", "test_config", "nonexistent"]
        )

        # Should migrate 2 configs
        assert migrated_count == 2

        # Verify in context
        assert context.get_capability_config("code_style") == {"formatter": "black"}
        assert context.get_capability_config("test_config") == {"framework": "pytest"}

    def test_clear_orchestrator_after_migration(self):
        """Test clearing orchestrator configs after migration."""
        context = VerticalContext(name="coding")
        orchestrator = Mock()

        # Set up configs
        orchestrator.code_style = {"formatter": "black"}

        # Migrate
        adapter = CapabilityAdapter(context, LegacyWriteMode.CONTEXT_ONLY)
        adapter.migrate_from_orchestrator(orchestrator, ["code_style"])

        # Clear from orchestrator
        adapter.clear_orchestrator_configs(orchestrator, ["code_style"])

        # Verify orchestrator cleaned
        assert not hasattr(orchestrator, "code_style")

        # Verify context still has it
        assert context.get_capability_config("code_style") == {"formatter": "black"}


class TestRealWorldVerticalIntegration:
    """Real-world vertical integration scenarios."""

    def test_coding_vertical_full_config(self):
        """Test full coding vertical configuration with all capabilities."""
        context = VerticalContext(name="coding")
        orchestrator = Mock()
        adapter = CapabilityAdapter(context, LegacyWriteMode.BACKWARD_COMPATIBLE)

        # Simulate full coding vertical setup
        configs = {
            "code_style": {
                "formatter": "black",
                "linter": "ruff",
                "max_line_length": 100,
                "enforce_type_hints": True,
            },
            "test_config": {
                "min_coverage": 0.8,
                "framework": "pytest",
                "run_on_edit": False,
            },
            "lsp_config": {
                "languages": ["python", "typescript", "javascript"],
                "features": {
                    "hover": True,
                    "references": True,
                    "symbols": True,
                },
            },
            "refactor_config": {
                "enable_rename": True,
                "enable_extract": True,
                "require_tests": True,
            },
        }

        adapter.apply_capability_configs(orchestrator, configs)

        # Verify all configs accessible from context
        for name, config in configs.items():
            retrieved = context.get_capability_config(name)
            assert retrieved == config, f"Config {name} mismatch"

        # Verify backward compat (orchestrator also has them)
        assert orchestrator.code_style == configs["code_style"]
        assert orchestrator.test_config == configs["test_config"]

    def test_research_vertical_config(self):
        """Test research vertical configuration patterns."""
        context = VerticalContext(name="research")
        orchestrator = Mock()
        adapter = CapabilityAdapter(context, LegacyWriteMode.BACKWARD_COMPATIBLE)

        # Research vertical configs
        configs = {
            "citation_style": {"format": "apa", "url_style": "short"},
            "source_verification": {"strict": True, "timeout": 30},
        }

        adapter.apply_capability_configs(orchestrator, configs)

        # Verify
        assert context.get_capability_config("citation_style")["format"] == "apa"
        assert orchestrator.citation_style["format"] == "apa"

    def test_devops_vertical_config(self):
        """Test devops vertical configuration patterns."""
        context = VerticalContext(name="devops")
        orchestrator = Mock()
        adapter = CapabilityAdapter(context, LegacyWriteMode.BACKWARD_COMPATIBLE)

        # Devops vertical configs
        configs = {
            "docker_config": {"runtime": "python:3.12", "port": 8000},
            "ci_config": {"pipeline": "github_actions", "run_tests": True},
        }

        adapter.apply_capability_configs(orchestrator, configs)

        # Verify
        assert context.get_capability_config("docker_config")["runtime"] == "python:3.12"
        assert orchestrator.docker_config["runtime"] == "python:3.12"


class TestConcurrencyAndIsolation:
    """Tests for concurrent access and isolation."""

    def test_multiple_verticals_dont_interfere(self):
        """Test that multiple verticals maintain separate configs."""
        coding_context = VerticalContext(name="coding")
        research_context = VerticalContext(name="research")
        orchestrator = Mock()

        # Configure coding
        coding_adapter = get_capability_adapter(coding_context)
        coding_adapter.set_capability_config(orchestrator, "code_style", {"formatter": "black"})

        # Configure research
        research_adapter = get_capability_adapter(research_context)
        research_adapter.set_capability_config(orchestrator, "citation_style", {"format": "apa"})

        # Verify isolation
        assert coding_context.get_capability_config("code_style") == {"formatter": "black"}
        assert research_context.get_capability_config("citation_style") == {"format": "apa"}

        # Contexts shouldn't have each other's configs
        assert coding_context.get_capability_config("citation_style") is None
        assert research_context.get_capability_config("code_style") is None

    def test_adapter_caching_per_context(self):
        """Test that adapter caching works correctly per context."""
        context1 = VerticalContext(name="coding")
        context2 = VerticalContext(name="coding")

        # Get adapters
        adapter1 = get_capability_adapter(context1)
        adapter2 = get_capability_adapter(context2)

        # Should be different instances (different context objects)
        assert adapter1 is not adapter2

        # Same context should return same adapter
        adapter1_again = get_capability_adapter(context1)
        assert adapter1 is adapter1_again


class TestBackwardCompatibilityGuarantees:
    """Tests ensuring backward compatibility guarantees."""

    def test_legacy_code_still_works_without_context(self):
        """Test that legacy code without context parameter still works."""
        context = VerticalContext(name="coding")
        orchestrator = Mock()
        adapter = CapabilityAdapter(context, LegacyWriteMode.BACKWARD_COMPATIBLE)

        # Legacy code might not have context at all
        # Adapter should still work with just orchestrator
        adapter.set_capability_config(orchestrator, "legacy_config", {"old_way": True})

        # Verify backward compat
        assert context.get_capability_config("legacy_config") == {"old_way": True}
        assert orchestrator.legacy_config == {"old_way": True}

    def test_getter_with_legacy_orchestrator_only(self):
        """Test getting config when only orchestrator has it (no context)."""
        context = VerticalContext(name="coding")
        orchestrator = Mock()
        orchestrator.legacy_value = {"exists": True}

        adapter = CapabilityAdapter(context, LegacyWriteMode.BACKWARD_COMPATIBLE)

        # Should fall back to orchestrator
        result = adapter.get_capability_config(orchestrator, "legacy_value")

        assert result == {"exists": True}
