"""Tests for EmbeddingRegistry lifecycle management (close_all / reset)."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.storage.vector_stores.registry import EmbeddingRegistry


@pytest.fixture(autouse=True)
def _clean_cache():
    """Ensure provider cache is empty before and after each test."""
    EmbeddingRegistry._provider_cache.clear()
    yield
    EmbeddingRegistry._provider_cache.clear()


def _make_provider(name: str = "mock") -> AsyncMock:
    """Return an AsyncMock that behaves like a BaseEmbeddingProvider."""
    provider = AsyncMock()
    provider.__class__ = MagicMock
    provider.__class__.__name__ = name
    return provider


class TestCloseAll:
    """Tests for EmbeddingRegistry.close_all()."""

    async def test_close_all_calls_close_on_all_providers(self):
        """close_all() should invoke close() on every cached provider."""
        p1 = _make_provider("provider-a")
        p2 = _make_provider("provider-b")
        EmbeddingRegistry._provider_cache["key-a"] = p1
        EmbeddingRegistry._provider_cache["key-b"] = p2

        await EmbeddingRegistry.close_all()

        p1.close.assert_awaited_once()
        p2.close.assert_awaited_once()
        assert len(EmbeddingRegistry._provider_cache) == 0

    async def test_close_all_handles_provider_exception(self):
        """If one provider.close() raises, others should still be closed."""
        p1 = _make_provider("failing")
        p1.close.side_effect = RuntimeError("connection lost")
        p2 = _make_provider("healthy")

        EmbeddingRegistry._provider_cache["fail"] = p1
        EmbeddingRegistry._provider_cache["ok"] = p2

        await EmbeddingRegistry.close_all()

        p1.close.assert_awaited_once()
        p2.close.assert_awaited_once()
        assert len(EmbeddingRegistry._provider_cache) == 0

    async def test_close_all_on_empty_cache(self):
        """close_all() on an empty cache should succeed without error."""
        await EmbeddingRegistry.close_all()
        assert len(EmbeddingRegistry._provider_cache) == 0


class TestResetWarning:
    """Tests for EmbeddingRegistry.reset() warning on active providers."""

    def test_reset_warns_when_cache_non_empty(self, caplog):
        """reset() with active providers should log a warning."""
        EmbeddingRegistry._provider_cache["key"] = _make_provider()

        with caplog.at_level(logging.WARNING):
            EmbeddingRegistry.reset()

        assert any(
            "reset() called with 1 active providers" in rec.message for rec in caplog.records
        )
        assert len(EmbeddingRegistry._provider_cache) == 0

    def test_reset_no_warning_when_cache_empty(self, caplog):
        """reset() with empty cache should not emit a warning."""
        with caplog.at_level(logging.WARNING):
            EmbeddingRegistry.reset()

        warning_records = [
            r for r in caplog.records if r.levelno == logging.WARNING and "reset()" in r.message
        ]
        assert len(warning_records) == 0
