"""Tests for unified settings (VictorSettings).

Tests cover:
- Settings precedence (CLI > env > .env > settings.yaml > profiles.yaml > defaults)
- Field validation
- Type safety
- from_sources() classmethod
- Integration with DI container
"""

import pytest
import yaml

from victor.config.unified_settings import VictorSettings


class TestVictorSettingsBasic:
    """Test basic VictorSettings functionality."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        settings = VictorSettings()

        assert settings.default_provider == "ollama"
        assert settings.default_model == "qwen3-coder:30b"
        assert settings.default_temperature == 0.7
        assert settings.default_max_tokens == 4096
        assert settings.airgapped_mode is False
        assert settings.use_semantic_tool_selection is True
        assert settings.tool_cache_enabled is True
        assert settings.tool_cache_ttl == 600
        assert settings.analytics_enabled is True

    def test_provider_validation(self):
        """Test provider name validation."""
        # Valid providers
        valid_providers = [
            "ollama",
            "anthropic",
            "openai",
            "google",
            "groq",
            "lmstudio",
            "vllm",
            "deepseek",
        ]
        for provider in valid_providers:
            settings = VictorSettings(default_provider=provider)
            assert settings.default_provider == provider

        # Invalid provider
        with pytest.raises(ValueError, match="Invalid provider"):
            VictorSettings(default_provider="invalid_provider")

    def test_write_approval_mode_validation(self):
        """Test write approval mode validation."""
        # Valid modes
        for mode in ["off", "risky_only", "all_writes"]:
            settings = VictorSettings(write_approval_mode=mode)
            assert settings.write_approval_mode == mode

        # Invalid mode
        with pytest.raises(ValueError, match="Invalid write_approval_mode"):
            VictorSettings(write_approval_mode="invalid_mode")

    def test_tool_validation_mode_validation(self):
        """Test tool validation mode validation."""
        # Valid modes
        for mode in ["strict", "lenient", "off"]:
            settings = VictorSettings(tool_validation_mode=mode)
            assert settings.tool_validation_mode == mode

        # Invalid mode
        with pytest.raises(ValueError, match="Invalid tool_validation_mode"):
            VictorSettings(tool_validation_mode="invalid_mode")

    def test_context_compaction_strategy_validation(self):
        """Test context compaction strategy validation."""
        # Valid strategies
        for strategy in ["simple", "tiered", "semantic", "hybrid"]:
            settings = VictorSettings(context_compaction_strategy=strategy)
            assert settings.context_compaction_strategy == strategy

        # Invalid strategy
        with pytest.raises(ValueError, match="Invalid context_compaction_strategy"):
            VictorSettings(context_compaction_strategy="invalid_strategy")

    def test_hybrid_search_weights_validation(self):
        """Test that hybrid search weights must sum to 1.0."""
        # Valid: weights sum to 1.0
        settings = VictorSettings(
            enable_hybrid_search=True,
            hybrid_search_semantic_weight=0.7,
            hybrid_search_keyword_weight=0.3,
        )
        assert settings.hybrid_search_semantic_weight == 0.7
        assert settings.hybrid_search_keyword_weight == 0.3

        # Invalid: weights don't sum to 1.0
        with pytest.raises(ValueError, match="must sum to 1.0"):
            VictorSettings(
                enable_hybrid_search=True,
                hybrid_search_semantic_weight=0.7,
                hybrid_search_keyword_weight=0.5,
            )

    def test_numeric_constraints(self):
        """Test numeric field constraints."""
        # Temperature range
        settings = VictorSettings(default_temperature=0.0)
        assert settings.default_temperature == 0.0
        settings = VictorSettings(default_temperature=2.0)
        assert settings.default_temperature == 2.0

        with pytest.raises(ValueError):
            VictorSettings(default_temperature=-0.1)
        with pytest.raises(ValueError):
            VictorSettings(default_temperature=2.1)

        # Max tokens must be positive
        settings = VictorSettings(default_max_tokens=100)
        assert settings.default_max_tokens == 100

        with pytest.raises(ValueError):
            VictorSettings(default_max_tokens=0)


class TestVictorSettingsPrecedence:
    """Test settings precedence (CLI > env > .env > settings.yaml > profiles.yaml > defaults)."""

    def test_from_sources_defaults_only(self):
        """Test from_sources with no overrides (uses defaults)."""
        settings = VictorSettings.from_sources()

        assert settings.default_provider == "ollama"
        assert settings.default_model == "qwen3-coder:30b"
        assert settings.tool_cache_ttl == 600

    def test_from_sources_with_profiles_yaml(self, tmp_path):
        """Test from_sources with profiles.yaml."""
        # Create temporary profiles.yaml
        profiles_yaml = tmp_path / "profiles.yaml"
        profiles_data = {
            "profiles": {
                "test": {
                    "default_provider": "anthropic",
                    "default_model": "claude-opus-4",
                    "default_temperature": 0.5,
                }
            }
        }
        with open(profiles_yaml, "w") as f:
            yaml.dump(profiles_data, f)

        settings = VictorSettings.from_sources(
            profile_name="test",
            config_dir=tmp_path,
        )

        assert settings.default_provider == "anthropic"
        assert settings.default_model == "claude-opus-4"
        assert settings.default_temperature == 0.5

    def test_from_sources_with_settings_yaml(self, tmp_path):
        """Test from_sources with settings.yaml."""
        # Create temporary settings.yaml
        settings_yaml = tmp_path / "settings.yaml"
        settings_data = {
            "default_provider": "openai",
            "default_model": "gpt-4",
            "tool_cache_ttl": 300,
        }
        with open(settings_yaml, "w") as f:
            yaml.dump(settings_data, f)

        settings = VictorSettings.from_sources(config_dir=tmp_path)

        assert settings.default_provider == "openai"
        assert settings.default_model == "gpt-4"
        assert settings.tool_cache_ttl == 300

    def test_from_sources_settings_yaml_overrides_profiles_yaml(self, tmp_path):
        """Test that settings.yaml takes precedence over profiles.yaml."""
        # Create profiles.yaml
        profiles_yaml = tmp_path / "profiles.yaml"
        profiles_data = {
            "profiles": {
                "test": {
                    "default_provider": "anthropic",
                    "tool_cache_ttl": 600,
                }
            }
        }
        with open(profiles_yaml, "w") as f:
            yaml.dump(profiles_data, f)

        # Create settings.yaml (higher precedence)
        settings_yaml = tmp_path / "settings.yaml"
        settings_data = {
            "default_provider": "openai",  # Should override profile
            # tool_cache_ttl not specified, should use profile value
        }
        with open(settings_yaml, "w") as f:
            yaml.dump(settings_data, f)

        settings = VictorSettings.from_sources(
            profile_name="test",
            config_dir=tmp_path,
        )

        assert settings.default_provider == "openai"  # From settings.yaml
        assert settings.tool_cache_ttl == 600  # From profiles.yaml

    def test_from_sources_cli_args_highest_precedence(self, tmp_path):
        """Test that CLI args have highest precedence."""
        # Create settings.yaml
        settings_yaml = tmp_path / "settings.yaml"
        settings_data = {
            "default_provider": "openai",
            "tool_cache_ttl": 300,
        }
        with open(settings_yaml, "w") as f:
            yaml.dump(settings_data, f)

        # CLI args should override everything
        cli_args = {
            "default_provider": "groq",
            "tool_cache_ttl": 120,
        }

        settings = VictorSettings.from_sources(
            cli_args=cli_args,
            config_dir=tmp_path,
        )

        assert settings.default_provider == "groq"  # From CLI
        assert settings.tool_cache_ttl == 120  # From CLI

    def test_from_sources_env_vars(self, tmp_path, monkeypatch):
        """Test that environment variables work."""
        # Set environment variables
        monkeypatch.setenv("VICTOR_DEFAULT_PROVIDER", "lmstudio")
        monkeypatch.setenv("VICTOR_TOOL_CACHE_TTL", "450")

        settings = VictorSettings.from_sources(config_dir=tmp_path)

        assert settings.default_provider == "lmstudio"
        assert settings.tool_cache_ttl == 450

    def test_from_sources_cli_overrides_env(self, tmp_path, monkeypatch):
        """Test that CLI args override environment variables."""
        # Set environment variable
        monkeypatch.setenv("VICTOR_DEFAULT_PROVIDER", "lmstudio")

        # CLI args should override env
        cli_args = {"default_provider": "vllm"}

        settings = VictorSettings.from_sources(
            cli_args=cli_args,
            config_dir=tmp_path,
        )

        assert settings.default_provider == "vllm"  # From CLI, not env

    def test_from_sources_filters_none_values(self, tmp_path):
        """Test that None values in CLI args are filtered out."""
        cli_args = {
            "default_provider": "anthropic",
            "default_model": None,  # Should be ignored
            "tool_cache_ttl": None,  # Should be ignored
        }

        settings = VictorSettings.from_sources(
            cli_args=cli_args,
            config_dir=tmp_path,
        )

        assert settings.default_provider == "anthropic"  # From CLI
        assert settings.default_model == "qwen3-coder:30b"  # From defaults
        assert settings.tool_cache_ttl == 600  # From defaults


class TestVictorSettingsTypeSafety:
    """Test type safety of VictorSettings."""

    def test_direct_attribute_access(self):
        """Test that attributes can be accessed directly without getattr."""
        settings = VictorSettings()

        # Should work without getattr
        assert isinstance(settings.default_provider, str)
        assert isinstance(settings.tool_cache_ttl, int)
        assert isinstance(settings.use_semantic_tool_selection, bool)
        assert isinstance(settings.tool_cache_allowlist, list)

    def test_optional_fields(self):
        """Test that optional fields can be None."""
        settings = VictorSettings()

        assert settings.anthropic_api_key is None
        assert settings.openai_api_key is None
        assert settings.log_file is None
        assert settings.codebase_persist_directory is None

    def test_field_types_validated(self):
        """Test that field types are validated."""
        # Valid types
        settings = VictorSettings(
            default_provider="ollama",
            tool_cache_ttl=300,
            use_semantic_tool_selection=True,
        )
        assert settings.default_provider == "ollama"
        assert settings.tool_cache_ttl == 300
        assert settings.use_semantic_tool_selection is True

        # Invalid type for integer field
        with pytest.raises(ValueError):
            VictorSettings(tool_cache_ttl="not_an_int")

        # Invalid type for boolean field
        with pytest.raises(ValueError):
            VictorSettings(use_semantic_tool_selection="not_a_bool")


class TestVictorSettingsEdgeCases:
    """Test edge cases and error handling."""

    def test_nonexistent_profile(self, tmp_path):
        """Test loading a nonexistent profile."""
        profiles_yaml = tmp_path / "profiles.yaml"
        profiles_data = {
            "profiles": {
                "test": {
                    "default_provider": "anthropic",
                }
            }
        }
        with open(profiles_yaml, "w") as f:
            yaml.dump(profiles_data, f)

        # Request nonexistent profile - should use defaults
        settings = VictorSettings.from_sources(
            profile_name="nonexistent",
            config_dir=tmp_path,
        )

        assert settings.default_provider == "ollama"  # Default, not from profile

    def test_invalid_yaml_graceful_handling(self, tmp_path):
        """Test that invalid YAML is handled gracefully."""
        # Create invalid YAML
        settings_yaml = tmp_path / "settings.yaml"
        with open(settings_yaml, "w") as f:
            f.write("{ invalid yaml content")

        # Should not crash, should use defaults
        settings = VictorSettings.from_sources(config_dir=tmp_path)
        assert settings.default_provider == "ollama"

    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed (for forward compatibility)."""
        # Pydantic with extra='allow' should accept unknown fields
        settings = VictorSettings(unknown_field="value")
        # Unknown field should be accessible
        assert hasattr(settings, "unknown_field")

    def test_model_copy(self):
        """Test that model_copy works for overriding values."""
        original = VictorSettings(default_provider="ollama")
        updated = original.model_copy(update={"default_provider": "anthropic"})

        assert original.default_provider == "ollama"
        assert updated.default_provider == "anthropic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
