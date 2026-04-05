"""Tests for ExtensionCacheManager."""

from victor.core.verticals.extension_cache_manager import ExtensionCacheManager


class TestExtensionCacheManager:
    def setup_method(self):
        self.cache = ExtensionCacheManager()

    def test_get_or_create_caches_result(self):
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return "value"

        result1 = self.cache.get_or_create("ns", "key", factory)
        result2 = self.cache.get_or_create("ns", "key", factory)
        assert result1 == "value"
        assert result2 == "value"
        assert call_count == 1  # Factory called only once

    def test_different_namespaces_independent(self):
        self.cache.get_or_create("ns1", "key", lambda: "a")
        self.cache.get_or_create("ns2", "key", lambda: "b")
        found1, val1 = self.cache.get_if_cached("ns1", "key")
        found2, val2 = self.cache.get_if_cached("ns2", "key")
        assert val1 == "a"
        assert val2 == "b"

    def test_get_if_cached_miss(self):
        found, val = self.cache.get_if_cached("ns", "nonexistent")
        assert found is False
        assert val is None

    def test_get_if_cached_hit(self):
        self.cache.get_or_create("ns", "key", lambda: 42)
        found, val = self.cache.get_if_cached("ns", "key")
        assert found is True
        assert val == 42

    def test_load_optional_caches_non_none(self):
        call_count = 0

        def loader():
            nonlocal call_count
            call_count += 1
            return "resolved"

        result1 = self.cache.load_optional("ns", "ext", loader)
        result2 = self.cache.load_optional("ns", "ext", loader)
        assert result1 == "resolved"
        assert result2 == "resolved"
        assert call_count == 1

    def test_load_optional_does_not_cache_none(self):
        call_count = 0

        def loader():
            nonlocal call_count
            call_count += 1
            return None

        result1 = self.cache.load_optional("ns", "ext", loader)
        result2 = self.cache.load_optional("ns", "ext", loader)
        assert result1 is None
        assert result2 is None
        assert call_count == 2  # Called twice since None is not cached

    def test_invalidate_specific_key(self):
        self.cache.get_or_create("ns", "a", lambda: 1)
        self.cache.get_or_create("ns", "b", lambda: 2)
        removed = self.cache.invalidate("ns", "a")
        assert removed == 1
        assert self.cache.get_if_cached("ns", "a") == (False, None)
        assert self.cache.get_if_cached("ns", "b") == (True, 2)

    def test_invalidate_namespace(self):
        self.cache.get_or_create("ns1", "a", lambda: 1)
        self.cache.get_or_create("ns1", "b", lambda: 2)
        self.cache.get_or_create("ns2", "c", lambda: 3)
        removed = self.cache.invalidate("ns1")
        assert removed == 2
        assert self.cache.get_if_cached("ns2", "c") == (True, 3)

    def test_invalidate_all(self):
        self.cache.get_or_create("ns1", "a", lambda: 1)
        self.cache.get_or_create("ns2", "b", lambda: 2)
        removed = self.cache.invalidate()
        assert removed == 2
        assert self.cache.get_if_cached("ns1", "a") == (False, None)
        assert self.cache.get_if_cached("ns2", "b") == (False, None)
