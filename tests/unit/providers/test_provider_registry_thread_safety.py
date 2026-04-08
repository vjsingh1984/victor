"""Tests for ProviderRegistry thread safety."""

import threading
from unittest.mock import patch, MagicMock

from victor.providers.registry import (
    ProviderRegistry,
    _lazy_provider_specs,
    _LazyProviderSpec,
    _registry_instance,
)


class TestProviderRegistryThreadSafety:
    """Verify thread-safe lazy provider materialization."""

    def setup_method(self):
        # Save original state
        self._original_specs = dict(_lazy_provider_specs)
        self._original_registry = dict(_registry_instance._items)

    def teardown_method(self):
        # Restore original state
        _lazy_provider_specs.clear()
        _lazy_provider_specs.update(self._original_specs)
        _registry_instance._items.clear()
        _registry_instance._items.update(self._original_registry)

    def test_concurrent_get_same_provider(self):
        """10 concurrent threads calling get() for the same provider should import once."""
        import_count = {"count": 0}
        lock = threading.Lock()

        fake_provider = type("FakeProvider", (), {})

        def fake_import(module_path):
            with lock:
                import_count["count"] += 1
            mod = MagicMock()
            mod.FakeProvider = fake_provider
            return mod

        # Register a fake lazy provider
        _lazy_provider_specs["test_thread"] = _LazyProviderSpec(
            module_path="fake.module",
            class_name="FakeProvider",
        )

        errors = []
        results = []

        def get_provider():
            try:
                result = ProviderRegistry.get("test_thread")
                results.append(result)
            except Exception as e:
                errors.append(e)

        with patch(
            "victor.providers.registry.importlib.import_module", side_effect=fake_import
        ):
            threads = [threading.Thread(target=get_provider) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

        assert not errors, f"Errors occurred: {errors}"
        assert (
            import_count["count"] == 1
        ), f"Import should happen exactly once, got {import_count['count']}"
        assert len(results) == 10
        # All threads should get the same class
        assert all(r is fake_provider for r in results)
