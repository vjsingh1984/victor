"""Tests for feature flag management."""

import os
from unittest.mock import patch

import pytest

from victor.core.language_capabilities import (
    FeatureFlagManager,
    LanguageFeatureFlags,
)


@pytest.fixture(autouse=True)
def reset_manager():
    """Reset feature flag manager singleton."""
    FeatureFlagManager.reset_instance()
    yield
    FeatureFlagManager.reset_instance()


class TestFeatureFlagManager:
    """Tests for FeatureFlagManager."""

    def test_singleton_instance(self):
        """Manager should be a singleton."""
        manager1 = FeatureFlagManager.instance()
        manager2 = FeatureFlagManager.instance()
        assert manager1 is manager2

    def test_default_flags(self):
        """Should have sensible defaults."""
        manager = FeatureFlagManager.instance()

        assert manager.is_indexing_enabled("python")
        assert manager.is_validation_enabled("python")
        assert manager.is_native_ast_enabled("python")
        assert manager.is_tree_sitter_enabled("python")
        assert manager.is_lsp_enabled("python")
        assert not manager.is_strict_mode()
        assert manager.is_cache_enabled()
        assert manager.is_parallel_enabled()

    def test_disable_indexing_globally(self):
        """Should disable indexing globally."""
        manager = FeatureFlagManager.instance()
        manager.set_global_flag("indexing_enabled", False)

        assert not manager.is_indexing_enabled("python")
        assert not manager.is_indexing_enabled("typescript")

    def test_disable_validation_globally(self):
        """Should disable validation globally."""
        manager = FeatureFlagManager.instance()
        manager.set_global_flag("validation_enabled", False)

        assert not manager.is_validation_enabled("python")
        assert not manager.is_validation_enabled("typescript")

    def test_language_specific_override(self):
        """Should allow language-specific overrides."""
        manager = FeatureFlagManager.instance()

        # Disable validation for rust only
        manager.set_language_flags(
            "rust",
            LanguageFeatureFlags(validation_enabled=False)
        )

        assert not manager.is_validation_enabled("rust")
        assert manager.is_validation_enabled("python")  # Others unaffected

    def test_disable_language(self):
        """Should disable all features for a language."""
        manager = FeatureFlagManager.instance()
        manager.disable_language("php")

        assert not manager.is_indexing_enabled("php")
        assert not manager.is_validation_enabled("php")
        assert not manager.is_native_ast_enabled("php")
        assert not manager.is_tree_sitter_enabled("php")
        assert not manager.is_lsp_enabled("php")

    def test_enable_language(self):
        """Should re-enable language by clearing override."""
        manager = FeatureFlagManager.instance()

        # Disable then re-enable
        manager.disable_language("ruby")
        manager.enable_language("ruby")

        assert manager.is_indexing_enabled("ruby")
        assert manager.is_validation_enabled("ruby")

    def test_get_cache_ttl(self):
        """Should get cache TTL."""
        manager = FeatureFlagManager.instance()
        ttl = manager.get_cache_ttl()

        assert isinstance(ttl, int)
        assert ttl > 0

    def test_to_dict(self):
        """Should export flags to dict."""
        manager = FeatureFlagManager.instance()
        data = manager.to_dict()

        assert "global" in data
        assert "language_overrides" in data
        assert data["global"]["indexing_enabled"] is True
        assert data["global"]["validation_enabled"] is True

    def test_get_language_flags(self):
        """Should get language-specific flags."""
        manager = FeatureFlagManager.instance()

        # No override initially
        flags = manager.get_language_flags("python")
        assert flags is None

        # After setting override
        manager.set_language_flags(
            "python",
            LanguageFeatureFlags(validation_enabled=False)
        )

        flags = manager.get_language_flags("python")
        assert flags is not None
        assert not flags.validation_enabled


class TestEnvironmentVariables:
    """Tests for environment variable configuration."""

    def test_env_indexing_disabled(self):
        """Should read indexing flag from env."""
        with patch.dict(os.environ, {"VICTOR_INDEXING_ENABLED": "false"}):
            FeatureFlagManager.reset_instance()
            manager = FeatureFlagManager.instance()

            assert not manager.is_indexing_enabled("python")

    def test_env_validation_disabled(self):
        """Should read validation flag from env."""
        with patch.dict(os.environ, {"VICTOR_VALIDATION_ENABLED": "false"}):
            FeatureFlagManager.reset_instance()
            manager = FeatureFlagManager.instance()

            assert not manager.is_validation_enabled("python")

    def test_env_strict_mode(self):
        """Should read strict mode from env."""
        with patch.dict(os.environ, {"VICTOR_STRICT_VALIDATION": "true"}):
            FeatureFlagManager.reset_instance()
            manager = FeatureFlagManager.instance()

            assert manager.is_strict_mode()

    def test_env_cache_ttl(self):
        """Should read cache TTL from env."""
        with patch.dict(os.environ, {"VICTOR_LANG_CACHE_TTL": "600"}):
            FeatureFlagManager.reset_instance()
            manager = FeatureFlagManager.instance()

            assert manager.get_cache_ttl() == 600

    def test_env_cache_disabled(self):
        """Should read cache enabled from env."""
        with patch.dict(os.environ, {"VICTOR_LANG_CACHE_ENABLED": "false"}):
            FeatureFlagManager.reset_instance()
            manager = FeatureFlagManager.instance()

            assert not manager.is_cache_enabled()
