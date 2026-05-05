import types

import pytest

from victor.core.indexing.index_lock import IndexLockRegistry


@pytest.mark.asyncio
async def test_acquire_lock_releases_file_lock_after_context(tmp_path, monkeypatch):
    registry = IndexLockRegistry()
    root = tmp_path / "repo"
    root.mkdir()
    acquire_calls = []
    release_calls = []

    monkeypatch.setattr(
        "victor.config.settings.get_project_paths",
        lambda _path: types.SimpleNamespace(project_victor_dir=root / ".victor"),
    )
    def _fake_acquire(self, timeout=300.0):
        acquire_calls.append((self.lock_file, timeout))
        self._lock_fd = 1
        return True

    monkeypatch.setattr("victor.core.indexing.index_lock.FileLock.acquire", _fake_acquire)

    def _fake_release(self):
        if self._lock_fd is None:
            return
        release_calls.append(self.lock_file)
        self._lock_fd = None

    monkeypatch.setattr("victor.core.indexing.index_lock.FileLock.release", _fake_release)

    path_lock = await registry.acquire_lock(root)
    async with path_lock:
        pass

    assert len(acquire_calls) == 1
    assert len(release_calls) == 1
    assert acquire_calls[0][0] == release_calls[0]


@pytest.mark.asyncio
async def test_acquire_lock_marks_usage_on_context_exit(tmp_path, monkeypatch):
    registry = IndexLockRegistry()
    root = tmp_path / "repo"
    root.mkdir()

    monkeypatch.setattr(
        "victor.config.settings.get_project_paths",
        lambda _path: types.SimpleNamespace(project_victor_dir=root / ".victor"),
    )
    def _fake_acquire_noop(self, timeout=300.0):
        self._lock_fd = 1
        return True

    monkeypatch.setattr("victor.core.indexing.index_lock.FileLock.acquire", _fake_acquire_noop)

    def _noop_release(self):
        self._lock_fd = None

    monkeypatch.setattr("victor.core.indexing.index_lock.FileLock.release", _noop_release)

    path_lock = await registry.acquire_lock(root)
    async with path_lock:
        pass

    stats = registry.get_stats()["lock_details"][str(root.resolve())]
    assert stats["last_used"] >= stats["created_at"]
