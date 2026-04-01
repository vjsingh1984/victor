from __future__ import annotations

from unittest.mock import Mock, patch

import victor.core.events.sync_wrapper as sync_wrapper_module
from victor.core.events.protocols import MessagingEvent


class TestSyncEventWrapperBridge:
    def test_publish_uses_shared_sync_bridge_without_running_loop(self) -> None:
        backend = Mock()
        coro = object()
        backend.publish.return_value = coro
        wrapper = sync_wrapper_module.SyncEventWrapper(backend)
        event = MessagingEvent(topic="tool.start", data={"tool": "read_file"})

        with (
            patch.object(sync_wrapper_module.asyncio, "get_running_loop", side_effect=RuntimeError),
            patch.object(sync_wrapper_module, "run_sync", return_value=True) as mock_run_sync,
            patch.object(sync_wrapper_module, "run_sync_in_thread") as mock_run_sync_in_thread,
        ):
            result = wrapper.publish(event)

        backend.publish.assert_called_once_with(event)
        mock_run_sync.assert_called_once_with(coro)
        mock_run_sync_in_thread.assert_not_called()
        assert result is True

    def test_publish_uses_thread_bridge_when_loop_running(self) -> None:
        backend = Mock()
        coro = object()
        backend.publish.return_value = coro
        wrapper = sync_wrapper_module.SyncEventWrapper(backend)
        event = MessagingEvent(topic="tool.end", data={"tool": "read_file"})

        with (
            patch.object(sync_wrapper_module.asyncio, "get_running_loop", return_value=object()),
            patch.object(sync_wrapper_module, "run_sync") as mock_run_sync,
            patch.object(
                sync_wrapper_module, "run_sync_in_thread", return_value=True
            ) as mock_run_sync_in_thread,
        ):
            result = wrapper.publish(event)

        backend.publish.assert_called_once_with(event)
        mock_run_sync.assert_not_called()
        mock_run_sync_in_thread.assert_called_once_with(coro, timeout=5.0)
        assert result is True

    def test_subscribe_uses_shared_sync_bridge(self) -> None:
        backend = Mock()
        handle = object()
        coro = object()
        backend.subscribe.return_value = coro
        wrapper = sync_wrapper_module.SyncEventWrapper(backend)
        handler = Mock()

        with patch.object(sync_wrapper_module, "run_sync", return_value=handle) as mock_run_sync:
            result = wrapper.subscribe("tool.*", handler)

        backend.subscribe.assert_called_once()
        subscribe_args = backend.subscribe.call_args.args
        assert subscribe_args[0] == "tool.*"
        assert callable(subscribe_args[1])
        mock_run_sync.assert_called_once_with(coro)
        assert result is handle
