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

"""Tests for lazy vertical loading."""

import pytest
import threading
import time
from unittest.mock import MagicMock, patch

from victor.core.verticals.lazy_loader import LazyVerticalProxy, LazyVerticalLoader, LoadTrigger


class TestLazyVerticalProxy:
    """Tests for LazyVerticalProxy."""

    def test_lazy_loading_on_first_access(self):
        """Test that vertical is loaded on first access."""

        class MockVertical:
            name = "test"
            loaded = False

            @classmethod
            def _load(cls):
                cls.loaded = True
                return cls

        proxy = LazyVerticalProxy(vertical_name="test", loader=MockVertical._load)

        # Should not be loaded initially
        assert not proxy.is_loaded()
        assert not MockVertical.loaded

        # Access should trigger loading
        vertical = proxy.load()
        assert proxy.is_loaded()
        assert MockVertical.loaded
        assert vertical == MockVertical

    def test_cached_after_first_load(self):
        """Test that loaded vertical is cached after first access."""
        load_count = 0

        def loader():
            nonlocal load_count
            load_count += 1
            return MagicMock(name="test")

        proxy = LazyVerticalProxy(vertical_name="test", loader=loader)

        # First load
        v1 = proxy.load()
        assert load_count == 1

        # Second load should use cache
        v2 = proxy.load()
        assert load_count == 1
        assert v1 is v2

    def test_unload(self):
        """Test unloading a vertical."""
        load_count = 0

        def loader():
            nonlocal load_count
            load_count += 1
            return MagicMock(name="test")

        proxy = LazyVerticalProxy(vertical_name="test", loader=loader)

        # Load
        proxy.load()
        assert load_count == 1
        assert proxy.is_loaded()

        # Unload
        proxy.unload()
        assert not proxy.is_loaded()

        # Load again should call loader
        proxy.load()
        assert load_count == 2

    def test_proxy_attribute_access(self):
        """Test that proxy forwards attribute access to loaded vertical."""

        class MockVertical:
            name = "test"

            def get_tools(self):
                return ["read", "write"]

        proxy = LazyVerticalProxy(vertical_name="test", loader=lambda: MockVertical)

        # Attribute access should trigger loading
        assert not proxy.is_loaded()
        tools = proxy.get_tools()
        assert proxy.is_loaded()
        assert tools == ["read", "write"]

    def test_proxy_repr(self):
        """Test proxy string representation."""
        proxy = LazyVerticalProxy(vertical_name="test", loader=lambda: MagicMock)

        # Unloaded
        assert "unloaded" in repr(proxy)

        # Loaded
        proxy.load()
        assert "loaded" in repr(proxy)

    def test_thread_safety(self):
        """Test that proxy is thread-safe."""
        load_count = 0
        lock = threading.Lock()

        def loader():
            nonlocal load_count
            with lock:
                load_count += 1
            # Simulate slow load
            time.sleep(0.01)
            return MagicMock(name="test")

        proxy = LazyVerticalProxy(vertical_name="test", loader=loader)

        def access_proxy():
            proxy.load()

        # Spawn multiple threads
        threads = [threading.Thread(target=access_proxy) for _ in range(10)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Should only load once despite multiple threads
        assert load_count == 1

    def test_recursive_loading_detection(self):
        """Test that recursive loading is detected and prevented."""

        class BadLoader:
            loading = False

            @classmethod
            def recursive_loader(cls):
                if cls.loading:
                    raise RuntimeError("Recursive!")
                cls.loading = True
                # Trigger another load (would cause infinite recursion without protection)
                proxy.load()
                return MagicMock()

        proxy = LazyVerticalProxy(
            vertical_name="bad", loader=BadLoader.recursive_loader
        )

        with pytest.raises(RuntimeError, match="Recursive loading"):
            proxy.load()


class TestLazyVerticalLoader:
    """Tests for LazyVerticalLoader."""

    def test_register_and_get_vertical(self):
        """Test registering and getting a vertical."""
        loader = LazyVerticalLoader()

        mock_vertical = MagicMock(name="test")

        loader.register_vertical("test", lambda: mock_vertical)

        # Should not be loaded yet (ON_DEMAND trigger)
        assert not loader.is_loaded("test")

        # Get should load it
        vertical = loader.get_vertical("test")
        assert loader.is_loaded("test")
        assert vertical == mock_vertical

    def test_eager_loading(self):
        """Test eager loading mode."""
        loader = LazyVerticalLoader(load_trigger=LoadTrigger.EAGER)

        mock_vertical = MagicMock(name="test")
        loaded = []

        def load_fn():
            loaded.append(True)
            return mock_vertical

        loader.register_vertical("test", load_fn)

        # Should be loaded immediately
        assert loader.is_loaded("test")
        assert len(loaded) == 1

    def test_unload_vertical(self):
        """Test unloading a vertical."""
        loader = LazyVerticalLoader()

        mock_vertical = MagicMock(name="test")
        loader.register_vertical("test", lambda: mock_vertical)

        # Load it
        loader.get_vertical("test")
        assert loader.is_loaded("test")

        # Unload it
        result = loader.unload_vertical("test")
        assert result is True
        assert not loader.is_loaded("test")

        # Unload again should return False
        result = loader.unload_vertical("test")
        assert result is False

    def test_unload_all(self):
        """Test unloading all verticals."""
        loader = LazyVerticalLoader()

        loader.register_vertical("test1", lambda: MagicMock(name="test1"))
        loader.register_vertical("test2", lambda: MagicMock(name="test2"))

        # Load both
        loader.get_vertical("test1")
        loader.get_vertical("test2")

        assert loader.get_loaded_count() == 2

        # Unload all
        count = loader.unload_all()
        assert count == 2
        assert loader.get_loaded_count() == 0

    def test_list_loaded(self):
        """Test listing loaded verticals."""
        loader = LazyVerticalLoader()

        loader.register_vertical("test1", lambda: MagicMock(name="test1"))
        loader.register_vertical("test2", lambda: MagicMock(name="test2"))

        # Load only one
        loader.get_vertical("test1")

        loaded = loader.list_loaded()
        assert loaded == {"test1"}

    def test_list_registered(self):
        """Test listing registered verticals."""
        loader = LazyVerticalLoader()

        loader.register_vertical("test1", lambda: MagicMock(name="test1"))
        loader.register_vertical("test2", lambda: MagicMock(name="test2"))

        registered = loader.list_registered()
        assert set(registered) == {"test1", "test2"}

    def test_auto_trigger_in_development(self):
        """Test AUTO trigger in development mode."""
        with patch.dict("os.environ", {"VICTOR_PROFILE": "development"}):
            loader = LazyVerticalLoader(load_trigger=LoadTrigger.AUTO)
            # AUTO in development should resolve to EAGER
            mock_vertical = MagicMock(name="test")
            loaded = []

            def load_fn():
                loaded.append(True)
                return mock_vertical

            loader.register_vertical("test", load_fn)
            assert loader.is_loaded("test")
            assert len(loaded) == 1

    def test_auto_trigger_in_production(self):
        """Test AUTO trigger in production mode."""
        with patch.dict("os.environ", {"VICTOR_PROFILE": "production"}):
            loader = LazyVerticalLoader(load_trigger=LoadTrigger.AUTO)
            # AUTO in production should resolve to ON_DEMAND
            mock_vertical = MagicMock(name="test")
            loaded = []

            def load_fn():
                loaded.append(True)
                return mock_vertical

            loader.register_vertical("test", load_fn)
            assert not loader.is_loaded("test")
            assert len(loaded) == 0

            # Load on demand
            loader.get_vertical("test")
            assert loader.is_loaded("test")
            assert len(loaded) == 1

    def test_get_lazy_loader_singleton(self):
        """Test global lazy loader singleton."""
        from victor.core.verticals.lazy_loader import get_lazy_loader

        loader1 = get_lazy_loader()
        loader2 = get_lazy_loader()

        assert loader1 is loader2


class TestVerticalLoaderIntegration:
    """Integration tests with VerticalLoader."""

    def test_load_with_lazy_parameter(self):
        """Test loading vertical with lazy parameter."""
        from victor.core.verticals.vertical_loader import VerticalLoader

        loader = VerticalLoader()

        # Eager load
        vertical_eager = loader.load("coding", lazy=False)
        # Should be actual vertical class, not proxy
        assert not hasattr(vertical_eager, "_load_lock")

        # Lazy load
        vertical_lazy = loader.load("coding", lazy=True)
        # Should be proxy
        assert hasattr(vertical_lazy, "_load_lock")
        assert hasattr(vertical_lazy, "load")

    def test_configure_lazy_mode_from_settings(self):
        """Test configuring lazy mode from settings."""
        from victor.core.verticals.vertical_loader import VerticalLoader
        from victor.config.settings import Settings

        loader = VerticalLoader()

        # Test with lazy mode
        settings = Settings(vertical_loading_mode="lazy")
        loader.configure_lazy_mode(settings)
        assert loader._lazy_mode is True

        # Test with eager mode
        settings = Settings(vertical_loading_mode="eager")
        loader.configure_lazy_mode(settings)
        assert loader._lazy_mode is False

        # Test with auto mode
        settings = Settings(vertical_loading_mode="auto")
        loader.configure_lazy_mode(settings)
        # Auto resolves based on VICTOR_PROFILE
        # Default is development, so should be eager
        assert loader._lazy_mode is False
