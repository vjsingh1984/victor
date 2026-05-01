from pathlib import Path

import pytest

from victor.core.database import reset_project_database
from victor.storage.graph.memory_store import MemoryGraphStore
from victor.storage.graph.sqlite_store import SqliteGraphStore


@pytest.mark.asyncio
async def test_memory_graph_store_lifecycle_methods() -> None:
    store = MemoryGraphStore()

    await store.initialize()
    await store.close()


@pytest.mark.asyncio
async def test_sqlite_graph_store_accepts_project_db_path(tmp_path: Path) -> None:
    db_path = tmp_path / "custom_state" / "project.db"
    store = SqliteGraphStore(db_path)

    try:
        await store.initialize()
        assert store.db_path == db_path
    finally:
        await store.close()
        reset_project_database(db_path)
