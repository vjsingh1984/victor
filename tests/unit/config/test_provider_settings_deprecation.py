"""Tests for ProviderSettings string method deprecation warnings."""

import pytest

from victor.config.settings import ProviderSettings


@pytest.fixture
def provider():
    return ProviderSettings(default_provider="ollama")


class TestProviderSettingsDeprecation:
    """Verify all 8 string-like methods emit DeprecationWarning."""

    def test_lower_deprecated(self, provider):
        with pytest.warns(DeprecationWarning, match="ProviderSettings.lower"):
            assert provider.lower() == "ollama"

    def test_upper_deprecated(self, provider):
        with pytest.warns(DeprecationWarning, match="ProviderSettings.upper"):
            assert provider.upper() == "OLLAMA"

    def test_title_deprecated(self, provider):
        with pytest.warns(DeprecationWarning, match="ProviderSettings.title"):
            assert provider.title() == "Ollama"

    def test_startswith_deprecated(self, provider):
        with pytest.warns(DeprecationWarning, match="ProviderSettings.startswith"):
            assert provider.startswith("oll") is True

    def test_endswith_deprecated(self, provider):
        with pytest.warns(DeprecationWarning, match="ProviderSettings.endswith"):
            assert provider.endswith("ama") is True

    def test_replace_deprecated(self, provider):
        with pytest.warns(DeprecationWarning, match="ProviderSettings.replace"):
            assert provider.replace("ollama", "openai") == "openai"

    def test_split_deprecated(self, provider):
        with pytest.warns(DeprecationWarning, match="ProviderSettings.split"):
            assert provider.split("l") == ["o", "", "ama"]

    def test_strip_deprecated(self, provider):
        p = ProviderSettings(default_provider="  ollama  ")
        with pytest.warns(DeprecationWarning, match="ProviderSettings.strip"):
            assert p.strip() == "ollama"

    def test_str_not_deprecated(self, provider):
        """__str__ should NOT emit a deprecation warning."""
        # No warning context manager — just verify it works
        assert str(provider) == "ollama"

    def test_eq_not_deprecated(self, provider):
        """__eq__ should NOT emit a deprecation warning."""
        assert provider == "ollama"

    def test_hash_not_deprecated(self, provider):
        """__hash__ should NOT emit a deprecation warning."""
        assert hash(provider) == hash("ollama")
