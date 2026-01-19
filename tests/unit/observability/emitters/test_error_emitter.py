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

"""Tests for ErrorEventEmitter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from victor.observability.emitters.error_emitter import ErrorEventEmitter
from victor.core.events import ObservabilityBus, InMemoryEventBackend


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================


class TestErrorEventEmitterInit:
    """Tests for ErrorEventEmitter initialization."""

    def test_init_with_default_bus(self):
        """Test initialization with default bus (None)."""
        emitter = ErrorEventEmitter()
        assert emitter._bus is None
        assert emitter._sync_wrapper is None
        assert emitter.is_enabled()

    def test_init_with_custom_bus(self):
        """Test initialization with custom ObservabilityBus."""
        mock_bus = MagicMock(spec=ObservabilityBus)
        emitter = ErrorEventEmitter(bus=mock_bus)
        assert emitter._bus is mock_bus
        assert emitter._sync_wrapper is None
        assert emitter.is_enabled()

    def test_init_state(self):
        """Test initial state after initialization."""
        emitter = ErrorEventEmitter()
        assert emitter.is_enabled()
        emitter.disable()
        assert not emitter.is_enabled()
        emitter.enable()
        assert emitter.is_enabled()


# =============================================================================
# BUS RETRIEVAL TESTS
# =============================================================================


class TestGetBus:
    """Tests for _get_bus method."""

    def test_get_bus_returns_instance_when_set(self):
        """Test _get_bus returns the bus instance when set."""
        mock_bus = MagicMock(spec=ObservabilityBus)
        emitter = ErrorEventEmitter(bus=mock_bus)
        assert emitter._get_bus() is mock_bus

    def test_get_bus_fallback_to_di_container(self):
        """Test _get_bus falls back to DI container when not set."""
        emitter = ErrorEventEmitter()

        # Mock the get_observability_bus function
        with patch("victor.core.events.get_observability_bus") as mock_get:
            mock_bus = MagicMock(spec=ObservabilityBus)
            mock_get.return_value = mock_bus

            result = emitter._get_bus()
            assert result is mock_bus
            mock_get.assert_called_once()

    def test_get_bus_returns_none_on_exception(self):
        """Test _get_bus returns None when exception occurs."""
        emitter = ErrorEventEmitter(bus=None)

        with patch("victor.core.events.get_observability_bus") as mock_get:
            mock_get.side_effect = Exception("DI container error")

            result = emitter._get_bus()
            assert result is None


# =============================================================================
# SYNC WRAPPER TESTS
# =============================================================================


class TestGetSyncWrapper:
    """Tests for _get_sync_wrapper method."""

    def test_get_sync_wrapper_returns_cached(self):
        """Test _get_sync_wrapper returns cached wrapper."""
        mock_backend = MagicMock()
        mock_bus = MagicMock(spec=ObservabilityBus)
        mock_bus.backend = mock_backend

        emitter = ErrorEventEmitter(bus=mock_bus)

        # First call creates wrapper and caches it
        result1 = emitter._get_sync_wrapper()
        result2 = emitter._get_sync_wrapper()

        # Both calls should return the same cached wrapper
        assert result1 is result2
        assert result1 is not None

    def test_get_sync_wrapper_without_bus(self):
        """Test _get_sync_wrapper returns None when no bus available."""
        emitter = ErrorEventEmitter(bus=None)
        with patch("victor.core.events.get_observability_bus") as mock_get:
            mock_get.side_effect = Exception("No bus available")
            result = emitter._get_sync_wrapper()
            assert result is None


# =============================================================================
# ASYNC EMIT TESTS
# =============================================================================


class TestEmitAsync:
    """Tests for emit_async method."""

    @pytest.mark.asyncio
    async def test_emit_async_when_disabled(self):
        """Test emit_async returns False when emitter is disabled."""
        mock_bus = MagicMock(spec=ObservabilityBus)
        emitter = ErrorEventEmitter(bus=mock_bus)
        emitter.disable()

        result = await emitter.emit_async("error.raised", {"error": "test"})
        assert result is False
        mock_bus.emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_emit_async_without_bus(self):
        """Test emit_async returns False when no bus available."""
        emitter = ErrorEventEmitter(bus=None)
        with patch("victor.core.events.get_observability_bus") as mock_get:
            mock_get.side_effect = Exception("No bus")

            result = await emitter.emit_async("error.raised", {"error": "test"})
            assert result is False

    @pytest.mark.asyncio
    async def test_emit_async_adds_category(self):
        """Test emit_async adds category to data."""
        mock_bus = MagicMock(spec=ObservabilityBus)
        mock_bus.emit = AsyncMock(return_value=True)

        emitter = ErrorEventEmitter(bus=mock_bus)

        result = await emitter.emit_async("error.raised", {"error": "test error"})

        assert result is True
        mock_bus.emit.assert_called_once()
        call_args = mock_bus.emit.call_args
        assert call_args[0][0] == "error.raised"
        assert "category" in call_args[0][1]
        assert call_args[0][1]["category"] == "error"
        assert call_args[0][1]["error"] == "test error"

    @pytest.mark.asyncio
    async def test_emit_async_handles_exception(self):
        """Test emit_async handles exceptions gracefully."""
        mock_bus = MagicMock(spec=ObservabilityBus)
        mock_bus.emit = AsyncMock(side_effect=Exception("Emission failed"))

        emitter = ErrorEventEmitter(bus=mock_bus)

        result = await emitter.emit_async("error.raised", {"error": "test"})
        assert result is False

    @pytest.mark.asyncio
    async def test_emit_async_with_bus_none(self):
        """Test emit_async when bus is None."""
        emitter = ErrorEventEmitter(bus=None)
        with patch("victor.core.events.get_observability_bus", return_value=None):
            result = await emitter.emit_async("error.raised", {"error": "test"})
            assert result is False


# =============================================================================
# SYNC EMIT TESTS
# =============================================================================


class TestEmit:
    """Tests for emit method (sync wrapper)."""

    def test_emit_without_bus(self):
        """Test emit handles no bus gracefully."""
        emitter = ErrorEventEmitter(bus=None)
        with patch("victor.core.events.get_observability_bus", return_value=None):
            # Should not raise exception
            emitter.emit("error.raised", {"error": "test"})

    def test_emit_calls_emit_helper(self):
        """Test emit uses emit_event_sync helper."""
        mock_bus = MagicMock(spec=ObservabilityBus)
        mock_bus.backend = MagicMock()
        mock_bus.backend._is_connected = True

        emitter = ErrorEventEmitter(bus=mock_bus)

        with patch("victor.core.events.emit_helper.emit_event_sync") as mock_emit_sync:
            emitter.emit("error.raised", {"error": "test error", "recoverable": True})

            mock_emit_sync.assert_called_once()
            call_args = mock_emit_sync.call_args

            # Verify the call arguments (correlation_id has a default, so we check what was actually passed)
            assert call_args[0][0] is mock_bus
            assert call_args[1]["topic"] == "error.raised"
            assert call_args[1]["data"]["error"] == "test error"
            assert call_args[1]["source"] == "ErrorEventEmitter"

    def test_emit_handles_exception(self):
        """Test emit handles exceptions gracefully."""
        mock_bus = MagicMock(spec=ObservabilityBus)
        mock_bus.backend = MagicMock()
        mock_bus.backend._is_connected = True

        emitter = ErrorEventEmitter(bus=mock_bus)

        with patch(
            "victor.core.events.emit_helper.emit_event_sync", side_effect=Exception("Helper failed")
        ):
            # Should not raise exception
            emitter.emit("error.raised", {"error": "test"})


# =============================================================================
# ERROR ASYNC TESTS
# =============================================================================


class TestErrorAsync:
    """Tests for error_async method."""

    @pytest.mark.asyncio
    async def test_error_async_basic(self):
        """Test error_async with basic error."""
        mock_bus = MagicMock(spec=ObservabilityBus)
        mock_bus.emit = AsyncMock(return_value=True)

        emitter = ErrorEventEmitter(bus=mock_bus)

        test_error = ValueError("Test error")
        result = await emitter.error_async(test_error, recoverable=True)

        assert result is True
        mock_bus.emit.assert_called_once()
        call_args = mock_bus.emit.call_args

        assert call_args[0][0] == "error.raised"
        data = call_args[0][1]
        assert data["error"] == "Test error"
        assert data["error_type"] == "ValueError"
        assert data["recoverable"] is True
        assert "traceback" in data
        assert "category" in data
        assert data["category"] == "error"

    @pytest.mark.asyncio
    async def test_error_async_with_context(self):
        """Test error_async with context."""
        mock_bus = MagicMock(spec=ObservabilityBus)
        mock_bus.emit = AsyncMock(return_value=True)

        emitter = ErrorEventEmitter(bus=mock_bus)

        test_error = RuntimeError("Runtime error")
        context = {"component": "tool_executor", "tool": "read_file"}

        result = await emitter.error_async(test_error, recoverable=False, context=context)

        assert result is True
        call_args = mock_bus.emit.call_args
        data = call_args[0][1]
        assert data["context"] == context
        assert data["recoverable"] is False

    @pytest.mark.asyncio
    async def test_error_async_with_metadata(self):
        """Test error_async with additional metadata."""
        mock_bus = MagicMock(spec=ObservabilityBus)
        mock_bus.emit = AsyncMock(return_value=True)

        emitter = ErrorEventEmitter(bus=mock_bus)

        test_error = Exception("Test error")
        result = await emitter.error_async(
            test_error,
            recoverable=True,
            context={"key": "value"},
            agent_id="agent-123",
            session_id="session-456",
        )

        assert result is True
        call_args = mock_bus.emit.call_args
        data = call_args[0][1]
        assert data["agent_id"] == "agent-123"
        assert data["session_id"] == "session-456"
        assert data["context"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_error_async_traceback_truncation(self):
        """Test that traceback is truncated to last 2000 characters."""
        mock_bus = MagicMock(spec=ObservabilityBus)
        mock_bus.emit = AsyncMock(return_value=True)

        emitter = ErrorEventEmitter(bus=mock_bus)

        # Create a long traceback
        long_traceback = "x" * 3000

        with patch(
            "victor.observability.emitters.error_emitter.traceback.format_exc",
            return_value=long_traceback,
        ):
            test_error = Exception("Test")
            await emitter.error_async(test_error, recoverable=True)

            call_args = mock_bus.emit.call_args
            data = call_args[0][1]
            assert len(data["traceback"]) <= 2000

    @pytest.mark.asyncio
    async def test_error_async_empty_traceback(self):
        """Test error_async handles empty traceback."""
        mock_bus = MagicMock(spec=ObservabilityBus)
        mock_bus.emit = AsyncMock(return_value=True)

        emitter = ErrorEventEmitter(bus=mock_bus)

        with patch(
            "victor.observability.emitters.error_emitter.traceback.format_exc", return_value=""
        ):
            test_error = Exception("Test")
            await emitter.error_async(test_error, recoverable=True)

            call_args = mock_bus.emit.call_args
            data = call_args[0][1]
            assert data["traceback"] is None

    @pytest.mark.asyncio
    async def test_error_async_when_disabled(self):
        """Test error_async when emitter is disabled."""
        mock_bus = MagicMock(spec=ObservabilityBus)
        mock_bus.emit = AsyncMock(return_value=True)

        emitter = ErrorEventEmitter(bus=mock_bus)
        emitter.disable()

        test_error = Exception("Test")
        result = await emitter.error_async(test_error, recoverable=True)

        assert result is False
        mock_bus.emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_async_without_bus(self):
        """Test error_async returns False when no bus."""
        emitter = ErrorEventEmitter(bus=None)
        with patch("victor.core.events.get_observability_bus", return_value=None):
            test_error = Exception("Test")
            result = await emitter.error_async(test_error, recoverable=True)
            assert result is False


# =============================================================================
# ERROR SYNC TESTS
# =============================================================================


class TestError:
    """Tests for error method (sync wrapper)."""

    def test_error_basic(self):
        """Test error method with basic error."""
        mock_bus = MagicMock(spec=ObservabilityBus)
        mock_bus.backend = MagicMock()
        mock_bus.backend._is_connected = True

        emitter = ErrorEventEmitter(bus=mock_bus)

        with patch("victor.core.events.emit_helper.emit_event_sync") as mock_emit_sync:
            test_error = ValueError("Test error")
            emitter.error(test_error, recoverable=True)

            mock_emit_sync.assert_called_once()
            call_args = mock_emit_sync.call_args

            assert call_args[1]["topic"] == "error.raised"
            data = call_args[1]["data"]
            assert data["error"] == "Test error"
            assert data["error_type"] == "ValueError"
            assert data["recoverable"] is True
            assert "traceback" in data

    def test_error_with_context(self):
        """Test error method with context."""
        mock_bus = MagicMock(spec=ObservabilityBus)
        mock_bus.backend = MagicMock()
        mock_bus.backend._is_connected = True

        emitter = ErrorEventEmitter(bus=mock_bus)

        with patch("victor.core.events.emit_helper.emit_event_sync") as mock_emit_sync:
            test_error = RuntimeError("Runtime error")
            context = {"component": "tool_executor", "tool": "write_file"}

            emitter.error(test_error, recoverable=False, context=context)

            call_args = mock_emit_sync.call_args
            data = call_args[1]["data"]
            assert data["context"] == context
            assert data["recoverable"] is False

    def test_error_with_metadata(self):
        """Test error method with metadata."""
        mock_bus = MagicMock(spec=ObservabilityBus)
        mock_bus.backend = MagicMock()
        mock_bus.backend._is_connected = True

        emitter = ErrorEventEmitter(bus=mock_bus)

        with patch("victor.core.events.emit_helper.emit_event_sync") as mock_emit_sync:
            test_error = Exception("Test")
            emitter.error(
                test_error,
                recoverable=True,
                agent_id="agent-123",
                session_id="session-456",
                extra_field="extra_value",
            )

            call_args = mock_emit_sync.call_args
            data = call_args[1]["data"]
            assert data["agent_id"] == "agent-123"
            assert data["session_id"] == "session-456"
            assert data["extra_field"] == "extra_value"

    def test_error_handles_traceback(self):
        """Test error method includes traceback."""
        mock_bus = MagicMock(spec=ObservabilityBus)
        mock_bus.backend = MagicMock()
        mock_bus.backend._is_connected = True

        emitter = ErrorEventEmitter(bus=mock_bus)

        with patch("victor.core.events.emit_helper.emit_event_sync") as mock_emit_sync:
            with patch(
                "victor.observability.emitters.error_emitter.traceback.format_exc",
                return_value="Traceback line 1\nTraceback line 2",
            ):
                test_error = Exception("Test")
                emitter.error(test_error, recoverable=True)

                call_args = mock_emit_sync.call_args
                data = call_args[1]["data"]
                assert "traceback" in data
                assert data["traceback"] is not None

    def test_error_without_bus(self):
        """Test error handles no bus gracefully."""
        emitter = ErrorEventEmitter(bus=None)
        with patch("victor.core.events.get_observability_bus", return_value=None):
            test_error = Exception("Test")
            # Should not raise exception
            emitter.error(test_error, recoverable=True)

    def test_error_handles_exception_in_emit(self):
        """Test error handles exceptions in emit gracefully."""
        mock_bus = MagicMock(spec=ObservabilityBus)
        mock_bus.backend = MagicMock()
        mock_bus.backend._is_connected = True

        emitter = ErrorEventEmitter(bus=mock_bus)

        with patch(
            "victor.core.events.emit_helper.emit_event_sync", side_effect=Exception("Emit failed")
        ):
            test_error = Exception("Test")
            # Should not raise exception
            emitter.error(test_error, recoverable=True)


# =============================================================================
# ENABLE/DISABLE TESTS
# =============================================================================


class TestEnableDisable:
    """Tests for enable/disable functionality."""

    def test_enable_disable(self):
        """Test enable and disable methods."""
        emitter = ErrorEventEmitter()

        assert emitter.is_enabled()

        emitter.disable()
        assert not emitter.is_enabled()

        emitter.enable()
        assert emitter.is_enabled()

    def test_disable_affects_emit_async(self):
        """Test that disable affects emit_async."""
        mock_bus = MagicMock(spec=ObservabilityBus)
        mock_bus.emit = AsyncMock(return_value=True)

        emitter = ErrorEventEmitter(bus=mock_bus)
        emitter.disable()

        import asyncio

        async def test():
            result = await emitter.emit_async("error.raised", {"error": "test"})
            return result

        result = asyncio.run(test())
        assert result is False
        mock_bus.emit.assert_not_called()

    def test_enable_reallows_emit_async(self):
        """Test that enable re-allows emit_async."""
        mock_bus = MagicMock(spec=ObservabilityBus)
        mock_bus.emit = AsyncMock(return_value=True)

        emitter = ErrorEventEmitter(bus=mock_bus)
        emitter.disable()
        emitter.enable()

        import asyncio

        async def test():
            result = await emitter.emit_async("error.raised", {"error": "test"})
            return result

        result = asyncio.run(test())
        assert result is True
        mock_bus.emit.assert_called_once()


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestErrorEventEmitterIntegration:
    """Integration tests for ErrorEventEmitter."""

    @pytest.mark.asyncio
    async def test_full_error_event_flow(self):
        """Test complete error event flow with real backend."""
        # Create real backend and bus
        backend = InMemoryEventBackend()
        await backend.connect()
        bus = ObservabilityBus(backend=backend)

        # Create emitter
        emitter = ErrorEventEmitter(bus=bus)

        # Track received events
        received_events = []

        async def handler(event):
            received_events.append(event)

        # Subscribe to error events
        await backend.subscribe("error.*", handler)

        # Emit error
        test_error = ValueError("Integration test error")
        result = await emitter.error_async(
            test_error,
            recoverable=True,
            context={"component": "test_component"},
            test_id="integration-123",
        )

        # Give event time to propagate
        import asyncio

        await asyncio.sleep(0.1)

        # Verify
        assert result is True
        assert len(received_events) == 1

        event = received_events[0]
        assert event.topic == "error.raised"
        assert event.data["error"] == "Integration test error"
        assert event.data["error_type"] == "ValueError"
        assert event.data["recoverable"] is True
        assert event.data["context"]["component"] == "test_component"
        assert event.data["test_id"] == "integration-123"
        assert event.data["category"] == "error"
        assert "traceback" in event.data

        await backend.disconnect()

    @pytest.mark.asyncio
    async def test_multiple_errors_sequence(self):
        """Test emitting multiple errors in sequence."""
        backend = InMemoryEventBackend()
        await backend.connect()
        bus = ObservabilityBus(backend=backend)

        emitter = ErrorEventEmitter(bus=bus)

        received_events = []

        async def handler(event):
            received_events.append(event)

        await backend.subscribe("error.*", handler)

        # Emit multiple errors
        errors = [
            ValueError("Error 1"),
            RuntimeError("Error 2"),
            TypeError("Error 3"),
        ]

        for error in errors:
            await emitter.error_async(error, recoverable=True)

        import asyncio

        await asyncio.sleep(0.1)

        assert len(received_events) == 3
        assert received_events[0].data["error"] == "Error 1"
        assert received_events[1].data["error"] == "Error 2"
        assert received_events[2].data["error"] == "Error 3"

        await backend.disconnect()

    def test_sync_error_method_integration(self):
        """Test sync error method with real backend."""
        # This test uses emit_event_sync which creates tasks
        # We'll verify it doesn't raise exceptions
        backend = InMemoryEventBackend()
        import asyncio

        # Connect in async context
        async def setup():
            await backend.connect()
            bus = ObservabilityBus(backend=backend)
            return bus

        bus = asyncio.run(setup())

        emitter = ErrorEventEmitter(bus=bus)

        # Emit error (should not raise)
        test_error = Exception("Sync test error")
        emitter.error(test_error, recoverable=True, context={"sync": True})

        # Give time for async task to complete
        async def wait():
            await asyncio.sleep(0.1)

        asyncio.run(wait())

        # The sync error method should not raise exceptions
        # We just verify the emitter is still enabled after the call
        assert emitter.is_enabled()

        # Clean up
        async def teardown():
            await backend.disconnect()

        asyncio.run(teardown())


# =============================================================================
# EDGE CASES TESTS
# =============================================================================


class TestErrorEventEmitterEdgeCases:
    """Edge case tests for ErrorEventEmitter."""

    @pytest.mark.asyncio
    async def test_error_with_none_context(self):
        """Test error with None context."""
        mock_bus = MagicMock(spec=ObservabilityBus)
        mock_bus.emit = AsyncMock(return_value=True)

        emitter = ErrorEventEmitter(bus=mock_bus)

        test_error = Exception("Test")
        await emitter.error_async(test_error, recoverable=True, context=None)

        call_args = mock_bus.emit.call_args
        data = call_args[0][1]
        assert data["context"] == {}

    @pytest.mark.asyncio
    async def test_error_with_empty_context(self):
        """Test error with empty context dict."""
        mock_bus = MagicMock(spec=ObservabilityBus)
        mock_bus.emit = AsyncMock(return_value=True)

        emitter = ErrorEventEmitter(bus=mock_bus)

        test_error = Exception("Test")
        await emitter.error_async(test_error, recoverable=True, context={})

        call_args = mock_bus.emit.call_args
        data = call_args[0][1]
        assert data["context"] == {}

    @pytest.mark.asyncio
    async def test_error_with_special_characters_in_error_message(self):
        """Test error with special characters in error message."""
        mock_bus = MagicMock(spec=ObservabilityBus)
        mock_bus.emit = AsyncMock(return_value=True)

        emitter = ErrorEventEmitter(bus=mock_bus)

        test_error = Exception("Error with special chars: \n\t\r\"'<>{}")
        await emitter.error_async(test_error, recoverable=True)

        call_args = mock_bus.emit.call_args
        data = call_args[0][1]
        assert "special chars" in data["error"]

    @pytest.mark.asyncio
    async def test_emit_with_empty_data(self):
        """Test emit with empty data dict."""
        mock_bus = MagicMock(spec=ObservabilityBus)
        mock_bus.emit = AsyncMock(return_value=True)

        emitter = ErrorEventEmitter(bus=mock_bus)

        result = await emitter.emit_async("error.raised", {})

        assert result is True
        call_args = mock_bus.emit.call_args
        assert call_args[0][1]["category"] == "error"

    @pytest.mark.asyncio
    async def test_error_with_very_long_context(self):
        """Test error with very large context dict."""
        mock_bus = MagicMock(spec=ObservabilityBus)
        mock_bus.emit = AsyncMock(return_value=True)

        emitter = ErrorEventEmitter(bus=mock_bus)

        large_context = {f"key_{i}": f"value_{i}" * 100 for i in range(100)}

        test_error = Exception("Test")
        await emitter.error_async(test_error, recoverable=True, context=large_context)

        call_args = mock_bus.emit.call_args
        data = call_args[0][1]
        assert data["context"] == large_context

    def test_multiple_enable_disable_calls(self):
        """Test multiple enable/disable calls."""
        emitter = ErrorEventEmitter()

        emitter.disable()
        emitter.disable()
        assert not emitter.is_enabled()

        emitter.enable()
        emitter.enable()
        assert emitter.is_enabled()
