# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for extension cache coordination (Phase 3.3: Extension Cache Coordination).

Tests that refresh_plugins() properly clears the extension cache.
"""

from unittest.mock import MagicMock, patch

from victor.core.verticals.extension_loader import VerticalExtensionLoader
from victor.core.verticals.vertical_loader import VerticalLoader


class TestClearExtensionCache:
    """Test clear_extension_cache method."""

    def test_clear_extension_cache_clears_all(self):
        """clear_extension_cache(clear_all=True) should clear all entries."""
        # Add some test data to the cache
        VerticalExtensionLoader._extensions_cache["CodingAssistant:middleware"] = MagicMock()
        VerticalExtensionLoader._extensions_cache["ResearchAssistant:middleware"] = MagicMock()
        VerticalExtensionLoader._extensions_cache["DataAnalysisAssistant:middleware"] = MagicMock()

        initial_count = len(VerticalExtensionLoader._extensions_cache)

        # Clear all
        VerticalExtensionLoader.clear_extension_cache(clear_all=True)

        # All should be cleared
        assert len(VerticalExtensionLoader._extensions_cache) == 0

    def test_clear_extension_cache_clears_specific_vertical(self):
        """clear_extension_cache(clear_all=False) should clear only for that vertical."""
        # Add test data for multiple verticals
        VerticalExtensionLoader._extensions_cache["CodingAssistant:middleware"] = MagicMock()
        VerticalExtensionLoader._extensions_cache["CodingAssistant:safety"] = MagicMock()
        VerticalExtensionLoader._extensions_cache["ResearchAssistant:middleware"] = MagicMock()

        # Clear only CodingAssistant (by calling from CodingAssistant class)
        from victor.coding.assistant import CodingAssistant

        CodingAssistant.clear_extension_cache(clear_all=False)

        # Should clear 2 CodingAssistant entries but not Research
        assert "CodingAssistant:middleware" not in VerticalExtensionLoader._extensions_cache
        assert "CodingAssistant:safety" not in VerticalExtensionLoader._extensions_cache
        assert "ResearchAssistant:middleware" in VerticalExtensionLoader._extensions_cache

    def test_clear_extension_cache_clears_when_empty(self):
        """clear_extension_cache should handle empty cache gracefully."""
        VerticalExtensionLoader._extensions_cache.clear()

        # Should not raise
        VerticalExtensionLoader.clear_extension_cache(clear_all=True)
        assert len(VerticalExtensionLoader._extensions_cache) == 0


class TestRefreshPluginsClearsExtensionCache:
    """Test that refresh_plugins() calls clear_extension_cache.

    Phase 3.3: Added extension cache clearing to refresh_plugins().
    """

    def test_refresh_plugins_clears_extension_cache(self):
        """refresh_plugins should call clear_extension_cache."""
        loader = VerticalLoader()

        # Add some test data to extension cache
        VerticalExtensionLoader._extensions_cache["TestVertical:middleware"] = MagicMock()
        VerticalExtensionLoader._extensions_cache["TestVertical:safety"] = MagicMock()

        # Mock the other parts of refresh_plugins
        with patch.object(loader, "_discovered_verticals", None):
            with patch.object(loader, "_discovered_tools", None):
                with patch(
                    "victor.core.verticals.vertical_loader.get_entry_point_cache"
                ) as mock_cache_get:
                    mock_cache = MagicMock()
                    mock_cache_get.return_value = mock_cache

                    # Call refresh_plugins
                    loader.refresh_plugins()

                    # Verify extension cache was cleared
                    assert len(VerticalExtensionLoader._extensions_cache) == 0

    def test_refresh_plugins_clears_all_extension_cache(self):
        """refresh_plugins should clear extension cache for all verticals."""
        loader = VerticalLoader()

        # Add test data for multiple verticals
        VerticalExtensionLoader._extensions_cache["CodingAssistant:middleware"] = MagicMock()
        VerticalExtensionLoader._extensions_cache["ResearchAssistant:middleware"] = MagicMock()
        VerticalExtensionLoader._extensions_cache["DataAnalysisAssistant:middleware"] = MagicMock()

        with patch.object(loader, "_discovered_verticals", None):
            with patch.object(loader, "_discovered_tools", None):
                with patch(
                    "victor.core.verticals.vertical_loader.get_entry_point_cache"
                ) as mock_cache_get:
                    mock_cache = MagicMock()
                    mock_cache_get.return_value = mock_cache

                    loader.refresh_plugins()

                    # All should be cleared
                    assert (
                        "CodingAssistant:middleware"
                        not in VerticalExtensionLoader._extensions_cache
                    )
                    assert (
                        "ResearchAssistant:middleware"
                        not in VerticalExtensionLoader._extensions_cache
                    )
                    assert (
                        "DataAnalysisAssistant:middleware"
                        not in VerticalExtensionLoader._extensions_cache
                    )

    def test_refresh_plugins_invalidates_entry_point_cache(self):
        """refresh_plugins should still invalidate entry point cache."""
        loader = VerticalLoader()

        with patch.object(loader, "_discovered_verticals", None):
            with patch.object(loader, "_discovered_tools", None):
                with patch(
                    "victor.core.verticals.vertical_loader.get_entry_point_cache"
                ) as mock_cache_get:
                    mock_cache = MagicMock()
                    mock_cache_get.return_value = mock_cache

                    loader.refresh_plugins()

                    # Verify entry point cache invalidation
                    mock_cache.invalidate.assert_any_call("victor.verticals")
                    mock_cache.invalidate.assert_any_call("victor.tools")


class TestExtensionCacheConsistency:
    """Test that extension cache stays consistent across operations."""

    def test_extension_cache_key_format(self):
        """Extension cache keys should use format 'ClassName:key'."""
        from victor.coding.assistant import CodingAssistant

        # Getting an extension should use the correct cache key format
        VerticalExtensionLoader._extensions_cache.clear()

        # Get middleware (should cache it)
        middleware = CodingAssistant.get_middleware()

        # Check cache key format
        cache_keys = list(VerticalExtensionLoader._extensions_cache.keys())
        assert any(key.startswith("CodingAssistant:") for key in cache_keys)

    def test_different_verticals_separate_cache_entries(self):
        """Different verticals should have separate cache entries."""
        from victor.coding.assistant import CodingAssistant
        from victor.research.assistant import ResearchAssistant

        VerticalExtensionLoader._extensions_cache.clear()

        # Get extensions that are actually cached
        CodingAssistant.get_safety_extension()
        ResearchAssistant.get_safety_extension()

        # Should have separate cache entries
        coding_keys = [
            k
            for k in VerticalExtensionLoader._extensions_cache.keys()
            if k.startswith("CodingAssistant:")
        ]
        research_keys = [
            k
            for k in VerticalExtensionLoader._extensions_cache.keys()
            if k.startswith("ResearchAssistant:")
        ]

        # Safety extensions should be cached
        assert len(coding_keys) > 0
        assert len(research_keys) > 0

    def test_cache_cleared_on_refresh(self):
        """Cached extensions should be cleared after refresh_plugins."""
        from victor.coding.assistant import CodingAssistant

        # Get an extension (caches it)
        middleware1 = CodingAssistant.get_middleware()

        # Verify it's cached
        cache_keys_before = list(VerticalExtensionLoader._extensions_cache.keys())
        assert len(cache_keys_before) > 0

        # Refresh plugins
        loader = VerticalLoader()
        with patch.object(loader, "_discovered_verticals", None):
            with patch.object(loader, "_discovered_tools", None):
                with patch(
                    "victor.core.verticals.vertical_loader.get_entry_point_cache"
                ) as mock_cache_get:
                    mock_cache = MagicMock()
                    mock_cache_get.return_value = mock_cache
                    loader.refresh_plugins()

        # Verify cache is cleared
        cache_keys_after = list(VerticalExtensionLoader._extensions_cache.keys())
        assert len(cache_keys_after) == 0

        # Getting extension again should create new cache entry
        middleware2 = CodingAssistant.get_middleware()
        cache_keys_new = list(VerticalExtensionLoader._extensions_cache.keys())
        assert len(cache_keys_new) > 0
