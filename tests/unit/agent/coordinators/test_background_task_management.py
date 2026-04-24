"""Tests for SessionCoordinator.create_background_task (extracted from orchestrator)."""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

from victor.agent.coordinators.session_coordinator import SessionCoordinator as LegacySessionCoordinator
from victor.agent.services.session_compat import SessionCoordinator


def test_session_coordinator_module_reexports_service_shim():
    assert LegacySessionCoordinator is SessionCoordinator


class TestCreateBackgroundTask:
    @pytest.fixture
    def task_state(self):
        return {
            "tasks": set(),
            "lock": threading.Lock(),
        }

    @pytest.mark.asyncio
    async def test_creates_and_tracks_task(self, task_state):
        async def noop():
            pass

        task = SessionCoordinator.create_background_task(
            noop(), "test-task", task_state["tasks"], task_state["lock"]
        )

        assert task is not None
        assert task.get_name() == "test-task"
        assert task in task_state["tasks"]

        await task
        # After completion, done callback should discard the task
        # Give the callback a moment to fire
        await asyncio.sleep(0)
        assert task not in task_state["tasks"]

    @pytest.mark.asyncio
    async def test_removes_task_on_completion(self, task_state):
        async def quick():
            return 42

        task = SessionCoordinator.create_background_task(
            quick(), "quick-task", task_state["tasks"], task_state["lock"]
        )

        assert task in task_state["tasks"]
        await task
        await asyncio.sleep(0)
        assert task not in task_state["tasks"]

    @pytest.mark.asyncio
    async def test_removes_task_on_exception(self, task_state):
        async def failing():
            raise ValueError("boom")

        task = SessionCoordinator.create_background_task(
            failing(), "fail-task", task_state["tasks"], task_state["lock"]
        )

        assert task in task_state["tasks"]
        with pytest.raises(ValueError, match="boom"):
            await task
        await asyncio.sleep(0)
        assert task not in task_state["tasks"]

    def test_returns_none_without_event_loop(self, task_state):
        async def noop():
            pass

        coro = noop()
        task = SessionCoordinator.create_background_task(
            coro, "no-loop", task_state["tasks"], task_state["lock"]
        )

        assert task is None
        assert len(task_state["tasks"]) == 0

    def test_closes_coroutine_when_no_loop(self, task_state):
        async def noop():
            pass

        coro = noop()
        SessionCoordinator.create_background_task(
            coro, "close-test", task_state["tasks"], task_state["lock"]
        )

        # Coroutine should have been closed (no ResourceWarning)
        # If it wasn't closed, Python would warn about never-awaited coroutine

    @pytest.mark.asyncio
    async def test_multiple_concurrent_tasks(self, task_state):
        results = []

        async def worker(n):
            results.append(n)

        tasks = []
        for i in range(5):
            t = SessionCoordinator.create_background_task(
                worker(i), f"worker-{i}", task_state["tasks"], task_state["lock"]
            )
            tasks.append(t)

        assert len(task_state["tasks"]) == 5
        await asyncio.gather(*tasks)
        await asyncio.sleep(0)
        assert len(task_state["tasks"]) == 0
        assert sorted(results) == [0, 1, 2, 3, 4]


class TestSessionCoordinatorServiceShim:
    def test_recover_session_delegates_to_bound_service(self):
        coordinator = SessionCoordinator(session_state_manager=MagicMock())
        service = MagicMock()
        service.recover_session.return_value = True
        coordinator.bind_session_service(service)

        assert coordinator.recover_session("session-123") is True
        service.recover_session.assert_called_once_with("session-123")

    @pytest.mark.asyncio
    async def test_save_checkpoint_delegates_to_bound_service(self):
        coordinator = SessionCoordinator(session_state_manager=MagicMock())
        service = MagicMock()
        service.save_checkpoint = AsyncMock(return_value="ckpt-123")
        coordinator.bind_session_service(service)

        result = await coordinator.save_checkpoint("before", ["manual"])

        assert result == "ckpt-123"
        service.save_checkpoint.assert_awaited_once_with("before", ["manual"])
