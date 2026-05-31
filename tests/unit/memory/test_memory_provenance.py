"""Unit tests for MemoryProvenance, MemoryQuery.query_id, and adapter stamping (Item 8)."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.storage.memory.unified import (
    MemoryProvenance,
    MemoryQuery,
    MemoryResult,
    MemoryType,
)

# ---------------------------------------------------------------------------
# MemoryQuery.query_id
# ---------------------------------------------------------------------------


class TestMemoryQueryId:
    def test_generates_unique_query_id(self):
        q1 = MemoryQuery(query="hello")
        q2 = MemoryQuery(query="hello")
        assert q1.query_id != q2.query_id

    def test_query_id_is_hex_string(self):
        q = MemoryQuery(query="test")
        assert isinstance(q.query_id, str)
        int(q.query_id, 16)  # raises ValueError if not hex

    def test_query_id_has_sufficient_length(self):
        q = MemoryQuery(query="test")
        assert len(q.query_id) >= 32  # uuid4().hex is 32 chars


# ---------------------------------------------------------------------------
# MemoryProvenance value object
# ---------------------------------------------------------------------------


class TestMemoryProvenance:
    def test_provenance_is_frozen(self):
        prov = MemoryProvenance(
            store_class="EntityMemory", query_id="abc123", lane=MemoryType.ENTITY
        )
        with pytest.raises((AttributeError, TypeError)):
            prov.store_class = "other"  # type: ignore[misc]

    def test_provenance_defaults(self):
        prov = MemoryProvenance(store_class="MyStore", query_id="qid", lane=MemoryType.CONVERSATION)
        assert prov.store_id is None
        assert prov.adapter_version == "1"


# ---------------------------------------------------------------------------
# MemoryResult backward compatibility
# ---------------------------------------------------------------------------


class TestMemoryResultBackwardCompat:
    def test_memory_result_without_provenance_is_backward_compatible(self):
        r = MemoryResult(
            source=MemoryType.ENTITY,
            content="hello",
            relevance=0.9,
        )
        assert r.provenance is None

    def test_memory_result_accepts_provenance(self):
        prov = MemoryProvenance(store_class="EntityMemory", query_id="q1", lane=MemoryType.ENTITY)
        r = MemoryResult(source=MemoryType.ENTITY, content="hello", relevance=0.5, provenance=prov)
        assert r.provenance is prov

    def test_replace_stamps_provenance_on_existing_result(self):
        r = MemoryResult(source=MemoryType.ENTITY, content="x", relevance=0.7)
        prov = MemoryProvenance(store_class="X", query_id="y", lane=MemoryType.ENTITY)
        stamped = replace(r, provenance=prov)
        assert stamped.provenance == prov
        assert r.provenance is None  # original unchanged


# ---------------------------------------------------------------------------
# EntityMemoryAdapter provenance stamping
# ---------------------------------------------------------------------------


class TestEntityAdapterProvenance:
    async def test_entity_adapter_stamps_provenance(self):
        from victor.storage.memory.adapters import EntityMemoryAdapter

        mock_entity = MagicMock()
        mock_entity.id = "e1"
        mock_entity.name = "auth"
        mock_entity.entity_type = MagicMock(value="function")
        mock_entity.confidence = 0.9
        mock_entity.mentions = 1
        mock_entity.source = "test"
        mock_entity.last_seen = None

        mock_memory = AsyncMock()
        mock_memory.search = AsyncMock(return_value=[mock_entity])
        mock_memory._initialized = True
        mock_memory.__class__.__name__ = "EntityMemory"

        adapter = EntityMemoryAdapter(mock_memory)
        query = MemoryQuery(query="auth")
        results = await adapter.search(query)

        assert results, "expected at least one result"
        assert results[0].provenance is not None
        assert results[0].provenance.query_id == query.query_id
        assert results[0].provenance.lane == MemoryType.ENTITY
        assert "EntityMemory" in results[0].provenance.store_class

    async def test_provenance_query_id_matches_query(self):
        from victor.storage.memory.adapters import EntityMemoryAdapter

        mock_entity = MagicMock()
        mock_entity.id = "e2"
        mock_entity.name = "login"
        mock_entity.entity_type = MagicMock(value="method")
        mock_entity.confidence = 0.8
        mock_entity.mentions = 0
        mock_entity.source = "test"
        mock_entity.last_seen = None

        mock_memory = AsyncMock()
        mock_memory.search = AsyncMock(return_value=[mock_entity])
        mock_memory._initialized = True

        adapter = EntityMemoryAdapter(mock_memory)
        query = MemoryQuery(query="login")
        results = await adapter.search(query)

        assert results[0].provenance.query_id == query.query_id
