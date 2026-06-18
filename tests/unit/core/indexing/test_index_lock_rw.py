"""Tests for reader/writer locking added to fix index-lock starvation (A1).

Covers the in-process ``_AsyncRWLock`` and the cross-process shared/exclusive
``FileLock`` so a "shared" acquisition is genuinely concurrent at both levels,
while writers stay exclusive.
"""

import asyncio

import pytest

from victor.core.indexing.index_lock import FileLock, _AsyncRWLock

# ---------------------------------------------------------------------------
# In-process reader/writer lock.
# ---------------------------------------------------------------------------


class TestAsyncRWLock:
    @pytest.mark.asyncio
    async def test_multiple_readers_concurrent(self):
        lock = _AsyncRWLock()
        await lock.acquire_read()
        await lock.acquire_read()  # second reader must NOT block
        assert lock._readers == 2
        assert lock.is_held() is True
        await lock.release_read()
        await lock.release_read()
        assert lock.is_held() is False

    @pytest.mark.asyncio
    async def test_writer_excludes_readers(self):
        lock = _AsyncRWLock()
        await lock.acquire_write()
        reader = asyncio.create_task(lock.acquire_read())
        await asyncio.sleep(0.05)
        assert reader.done() is False  # blocked while writer holds
        await lock.release_write()
        await asyncio.wait_for(reader, timeout=1.0)  # now proceeds
        await lock.release_read()

    @pytest.mark.asyncio
    async def test_writer_waits_for_readers(self):
        lock = _AsyncRWLock()
        await lock.acquire_read()
        writer = asyncio.create_task(lock.acquire_write())
        await asyncio.sleep(0.05)
        assert writer.done() is False  # writer waits for the reader to drain
        await lock.release_read()
        await asyncio.wait_for(writer, timeout=1.0)
        await lock.release_write()

    @pytest.mark.asyncio
    async def test_writer_preference_blocks_new_readers(self):
        """A queued writer blocks new readers so it cannot be starved."""
        lock = _AsyncRWLock()
        await lock.acquire_read()
        writer = asyncio.create_task(lock.acquire_write())
        await asyncio.sleep(0.05)  # let the writer queue
        late_reader = asyncio.create_task(lock.acquire_read())
        await asyncio.sleep(0.05)
        assert late_reader.done() is False  # held off by the waiting writer
        await lock.release_read()
        await asyncio.wait_for(writer, timeout=1.0)
        await lock.release_write()
        await asyncio.wait_for(late_reader, timeout=1.0)
        await lock.release_read()


# ---------------------------------------------------------------------------
# Cross-process file lock: shared vs exclusive (separate descriptions contend
# even within one process).
# ---------------------------------------------------------------------------


class TestFileLockSharedExclusive:
    def test_exclusive_excludes_exclusive(self, tmp_path):
        lf = tmp_path / "index.lock"
        a, b = FileLock(lf), FileLock(lf)
        assert a.acquire(shared=False) is True
        assert b.acquire(timeout=0.2, shared=False) is False
        a.release()
        assert b.acquire(timeout=0.2, shared=False) is True
        b.release()

    def test_shared_allows_concurrent_readers(self, tmp_path):
        lf = tmp_path / "index.lock"
        a, b = FileLock(lf), FileLock(lf)
        assert a.acquire(shared=True) is True
        assert b.acquire(timeout=0.2, shared=True) is True
        a.release()
        b.release()

    def test_shared_blocks_exclusive(self, tmp_path):
        lf = tmp_path / "index.lock"
        reader, writer = FileLock(lf), FileLock(lf)
        assert reader.acquire(shared=True) is True
        assert writer.acquire(timeout=0.2, shared=False) is False
        reader.release()
        assert writer.acquire(timeout=0.2, shared=False) is True
        writer.release()

    def test_lock_file_persists_after_release(self, tmp_path):
        """The lock file is intentionally not unlinked (inode-stability)."""
        lf = tmp_path / "index.lock"
        lock = FileLock(lf)
        lock.acquire(shared=False)
        lock.release()
        assert lf.exists()
