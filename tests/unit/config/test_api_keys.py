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

"""Tests for victor.config.api_keys module.

Tests the APIKeysProxy class and ensures backward compatibility
for code accessing victor.config.api_keys.api_keys.
"""

import pytest

from victor.config.api_keys import api_keys, APIKeysProxy, APIKeyManager


class TestAPIKeysAttribute:
    """Test the module-level api_keys attribute."""

    def test_api_keys_attribute_exists(self):
        """Test that api_keys attribute is available at module level."""
        # Import the module
        import victor.config.api_keys as api_keys_module

        # Verify api_keys attribute exists
        assert hasattr(
            api_keys_module, "api_keys"
        ), "victor.config.api_keys module should have 'api_keys' attribute"

    def test_api_keys_is_api_keys_proxy(self):
        """Test that api_keys is an APIKeysProxy instance."""
        assert isinstance(api_keys, APIKeysProxy), "api_keys should be an instance of APIKeysProxy"

    def test_api_keys_has_get_method(self):
        """Test that api_keys has a get() method."""
        assert hasattr(api_keys, "get"), "api_keys should have a get() method"
        assert callable(api_keys.get), "api_keys.get should be callable"

    def test_api_keys_get_returns_none_for_missing_provider(self):
        """Test that api_keys.get() returns None for missing provider."""
        result = api_keys.get("nonexistent_provider_xyz")
        assert result is None, "api_keys.get() should return None for non-existent provider"

    def test_api_keys_get_with_default(self):
        """Test that api_keys.get() returns default value when specified."""
        result = api_keys.get("nonexistent_provider_xyz", default="test_default")
        assert (
            result == "test_default"
        ), "api_keys.get() should return default value when provider not found"

    def test_api_keys_supports_getitem(self):
        """Test that api_keys supports dictionary-style access."""
        # This should not raise an exception
        result = api_keys["anthropic"]
        # Result may be None if not configured, but access should work
        assert result is None or isinstance(result, str)

    def test_api_keys_supports_contains(self):
        """Test that api_keys supports 'in' operator."""
        # Should not raise exception
        result = "anthropic" in api_keys
        assert isinstance(result, bool)

    def test_api_keys_has_keys_method(self):
        """Test that api_keys has a keys() method."""
        assert hasattr(api_keys, "keys"), "api_keys should have keys() method"
        assert callable(api_keys.keys), "api_keys.keys() should be callable"

        # Call it to ensure it works
        result = api_keys.keys()
        assert isinstance(result, list), "api_keys.keys() should return a list"

    def test_api_keys_has_items_method(self):
        """Test that api_keys has an items() method."""
        assert hasattr(api_keys, "items"), "api_keys should have items() method"
        assert callable(api_keys.items), "api_keys.items() should be callable"

        # Call it to ensure it works
        result = api_keys.items()
        assert isinstance(result, list), "api_keys.items() should return a list"

    def test_api_keys_has_values_method(self):
        """Test that api_keys has a values() method."""
        assert hasattr(api_keys, "values"), "api_keys should have values() method"
        assert callable(api_keys.values), "api_keys.values() should be callable"

        # Call it to ensure it works
        result = api_keys.values()
        assert isinstance(result, list), "api_keys.values() should return a list"

    def test_api_keys_repr(self):
        """Test that api_keys has a useful repr."""
        result = repr(api_keys)
        assert isinstance(result, str), "repr(api_keys) should return a string"
        assert "APIKeysProxy" in result, "repr(api_keys) should contain 'APIKeysProxy'"


class TestAPIKeysProxy:
    """Test the APIKeysProxy class directly."""

    def test_api_keys_proxy_initialization(self):
        """Test that APIKeysProxy can be instantiated."""
        proxy = APIKeysProxy()
        assert proxy is not None

    def test_api_keys_proxy_get_method(self):
        """Test APIKeysProxy.get() method."""
        proxy = APIKeysProxy()
        result = proxy.get("test_provider")
        assert result is None  # No keys configured in test

    def test_api_keys_proxy_getitem(self):
        """Test APIKeysProxy.__getitem__() method."""
        proxy = APIKeysProxy()
        result = proxy["test_provider"]
        assert result is None  # No keys configured in test

    def test_api_keys_proxy_contains(self):
        """Test APIKeysProxy.__contains__() method."""
        proxy = APIKeysProxy()
        result = "test_provider" in proxy
        assert isinstance(result, bool)
        assert result is False  # No keys configured in test

    def test_api_keys_proxy_keys_returns_list(self):
        """Test that APIKeysProxy.keys() returns a list."""
        proxy = APIKeysProxy()
        result = proxy.keys()
        assert isinstance(result, list)

    def test_api_keys_proxy_items_returns_list(self):
        """Test that APIKeysProxy.items() returns a list."""
        proxy = APIKeysProxy()
        result = proxy.items()
        assert isinstance(result, list)

    def test_api_keys_proxy_values_returns_list(self):
        """Test that APIKeysProxy.values() returns a list."""
        proxy = APIKeysProxy()
        result = proxy.values()
        assert isinstance(result, list)


class TestAPIKeysManagerIntegration:
    """Test integration with APIKeyManager."""

    def test_api_keys_proxy_uses_manager(self):
        """Test that APIKeysProxy uses APIKeyManager internally."""
        proxy = APIKeysProxy()
        # Access internal manager
        manager = proxy._get_manager()
        assert isinstance(manager, APIKeyManager)

    def test_api_keys_proxy_manager_reuse(self):
        """Test that APIKeysProxy reuses the same manager instance."""
        proxy = APIKeysProxy()
        manager1 = proxy._get_manager()
        manager2 = proxy._get_manager()
        assert manager1 is manager2, "APIKeysProxy should reuse the same APIKeyManager instance"


class TestBackwardCompatibility:
    """Test backward compatibility scenarios."""

    def test_legacy_dict_style_access(self):
        """Test legacy dictionary-style access patterns."""
        # These patterns should not raise exceptions
        key = api_keys.get("anthropic")
        key = api_keys["anthropic"]
        has_key = "anthropic" in api_keys
        all_keys = api_keys.keys()

    def test_module_level_access(self):
        """Test accessing api_keys directly from module."""
        import victor.config.api_keys as api_keys_module

        # This is the critical test - ensures backward compatibility
        api_keys_obj = api_keys_module.api_keys
        assert api_keys_obj is not None

    def test_from_import(self):
        """Test importing api_keys using from statement."""
        from victor.config.api_keys import api_keys as imported_keys

        assert imported_keys is not None
        assert isinstance(imported_keys, APIKeysProxy)
