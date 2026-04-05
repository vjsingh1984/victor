"""Tests for host-owned vertical cache invalidation helpers."""

from __future__ import annotations

from victor.core.verticals.cache_invalidation import (
    VerticalRuntimeInvalidationReason,
    invalidate_vertical_runtime_state,
)


def test_invalidate_vertical_runtime_state_calls_all_runtime_reset_hooks(monkeypatch) -> None:
    """Invalidation should clear config caches, registry discovery, and loader caches."""

    calls: list[str] = []

    monkeypatch.setattr(
        "victor.core.verticals.cache_invalidation.VerticalBase.clear_config_cache",
        lambda *, clear_all=False: calls.append(f"config:{clear_all}"),
    )
    monkeypatch.setattr(
        "victor.core.verticals.cache_invalidation.VerticalRegistry.reset_discovery",
        lambda: calls.append("registry"),
    )

    class _Loader:
        def refresh_plugins(self) -> None:
            calls.append("loader")

    monkeypatch.setattr(
        "victor.core.verticals.cache_invalidation.get_vertical_loader",
        lambda: _Loader(),
    )

    result = invalidate_vertical_runtime_state(
        VerticalRuntimeInvalidationReason.INSTALL,
        package_name="victor-security",
    )

    assert calls == ["config:True", "registry", "loader"]
    assert result.reason == VerticalRuntimeInvalidationReason.INSTALL
    assert result.package_name == "victor-security"
    assert result.config_cache_cleared is True
    assert result.registry_reset is True
    assert result.loader_refreshed is True
    assert result.successful is True


def test_invalidate_vertical_runtime_state_collects_partial_failures(monkeypatch) -> None:
    """Invalidation should remain best-effort and surface partial failures."""

    monkeypatch.setattr(
        "victor.core.verticals.cache_invalidation.VerticalBase.clear_config_cache",
        lambda *, clear_all=False: (_ for _ in ()).throw(RuntimeError("config boom")),
    )
    monkeypatch.setattr(
        "victor.core.verticals.cache_invalidation.VerticalRegistry.reset_discovery",
        lambda: None,
    )

    class _Loader:
        def refresh_plugins(self) -> None:
            raise RuntimeError("loader boom")

    monkeypatch.setattr(
        "victor.core.verticals.cache_invalidation.get_vertical_loader",
        lambda: _Loader(),
    )

    result = invalidate_vertical_runtime_state(VerticalRuntimeInvalidationReason.RELOAD)

    assert result.reason == VerticalRuntimeInvalidationReason.RELOAD
    assert result.config_cache_cleared is False
    assert result.registry_reset is True
    assert result.loader_refreshed is False
    assert result.successful is False
    assert any(error.startswith("config_cache:") for error in result.errors)
    assert any(error.startswith("loader_refresh:") for error in result.errors)
