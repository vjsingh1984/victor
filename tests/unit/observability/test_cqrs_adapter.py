# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for CQRS adapter async/sync startup boundaries."""

from unittest.mock import AsyncMock, MagicMock, patch

from victor.observability.cqrs_adapter import AdapterConfig, CQRSEventAdapter


class TestCQRSEventAdapterStartup:
    """Tests for adapter startup loop-bridging behavior."""

    def test_start_runs_async_subscriptions_without_running_loop(self):
        """Sync startup should bridge async subscriptions directly."""
        adapter = CQRSEventAdapter(
            event_bus=object(),
            event_dispatcher=object(),
            config=AdapterConfig(
                enable_observability_to_cqrs=True,
                enable_cqrs_to_observability=True,
            ),
        )
        adapter._async_subscribe_observability = AsyncMock(return_value=None)
        adapter._async_subscribe_cqrs = AsyncMock(return_value=None)

        adapter.start()

        adapter._async_subscribe_observability.assert_awaited_once_with()
        adapter._async_subscribe_cqrs.assert_awaited_once_with()
        assert adapter.is_active is True

    def test_start_schedules_async_subscriptions_on_running_loop(self):
        """Startup should schedule background subscriptions when a loop is active."""
        adapter = CQRSEventAdapter(
            event_bus=object(),
            event_dispatcher=object(),
            config=AdapterConfig(
                enable_observability_to_cqrs=True,
                enable_cqrs_to_observability=True,
            ),
        )
        adapter._async_subscribe_observability = AsyncMock(return_value=None)
        adapter._async_subscribe_cqrs = AsyncMock(return_value=None)

        scheduled = []
        loop = MagicMock()

        def capture_task(coro):
            scheduled.append(coro)
            coro.close()
            return MagicMock()

        loop.create_task.side_effect = capture_task

        with patch("victor.observability.cqrs_adapter.asyncio.get_running_loop", return_value=loop):
            adapter.start()

        assert len(scheduled) == 2
        assert adapter._async_subscribe_observability.call_count == 1
        assert adapter._async_subscribe_cqrs.call_count == 1
        assert adapter.is_active is True
