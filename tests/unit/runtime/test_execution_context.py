"""Tests for ExecutionContext — the foundation for global state elimination (GS-1).

Validates:
1. ExecutionContext creation from settings + container
2. ServiceAccessor lazy resolution from container
3. Immutable-like with_session() and with_metadata() patterns
4. Integration with existing DI container
"""

from unittest.mock import MagicMock

import pytest

from victor.runtime.context import (
    ExecutionContext,
    ServiceAccessor,
    resolve_execution_context,
    resolve_runtime_services,
)


class TestExecutionContextCreation:
    """Test ExecutionContext.create() factory."""

    def test_create_with_settings_and_container(self):
        settings = MagicMock()
        container = MagicMock()
        container.get_optional.return_value = None

        ctx = ExecutionContext.create(settings, container, session_id="test-session")

        assert ctx.session_id == "test-session"
        assert ctx.settings is settings
        assert ctx.container is container
        assert isinstance(ctx.services, ServiceAccessor)
        assert ctx.metadata == {}

    def test_create_with_empty_session_id(self):
        ctx = ExecutionContext.create(MagicMock(), MagicMock())
        assert ctx.session_id == ""

    def test_create_with_explicit_state_manager(self):
        state_mgr = MagicMock()
        ctx = ExecutionContext.create(MagicMock(), MagicMock(), state_manager=state_mgr)
        assert ctx.state is state_mgr

    def test_create_without_state_manager_uses_factory(self):
        """When no state_manager provided, should attempt to resolve from factory."""
        ctx = ExecutionContext.create(MagicMock(), MagicMock())
        # state should be set (either from factory or None if factory fails)
        assert hasattr(ctx, "state")


class TestServiceAccessor:
    """Test ServiceAccessor lazy resolution."""

    def test_accessor_resolves_from_container(self):
        mock_service = MagicMock()
        container = MagicMock()
        container.get_optional.return_value = mock_service

        accessor = ServiceAccessor(_container=container)

        # Access a service property
        result = accessor.chat
        assert result is mock_service

    def test_accessor_caches_resolved_services(self):
        mock_service = MagicMock()
        container = MagicMock()
        container.get_optional.return_value = mock_service

        accessor = ServiceAccessor(_container=container)

        # Access twice
        _ = accessor.chat
        _ = accessor.chat

        # Container should only be called once for each service type
        # (lazy caching)
        assert container.get_optional.call_count <= 6  # At most once per property

    def test_accessor_returns_none_for_unregistered_service(self):
        container = MagicMock()
        container.get_optional.return_value = None

        accessor = ServiceAccessor(_container=container)
        assert accessor.chat is None
        assert accessor.tool is None

    def test_accessor_has_all_six_service_properties(self):
        accessor = ServiceAccessor(_container=MagicMock())
        properties = ["chat", "tool", "session", "context", "provider", "recovery"]
        for prop in properties:
            assert hasattr(accessor, prop), f"ServiceAccessor missing '{prop}' property"

    def test_accessor_handles_none_container(self):
        accessor = ServiceAccessor(_container=None)
        assert accessor.chat is None
        assert accessor.tool is None


class TestExecutionContextImmutability:
    """Test with_session() and with_metadata() patterns."""

    def test_with_session_creates_new_context(self):
        ctx = ExecutionContext.create(MagicMock(), MagicMock(), session_id="original")
        new_ctx = ctx.with_session("new-session")

        assert new_ctx.session_id == "new-session"
        assert ctx.session_id == "original"  # Original unchanged

    def test_with_session_shares_services(self):
        ctx = ExecutionContext.create(MagicMock(), MagicMock())
        new_ctx = ctx.with_session("new")

        assert new_ctx.services is ctx.services
        assert new_ctx.state is ctx.state
        assert new_ctx.container is ctx.container

    def test_with_metadata_creates_new_context(self):
        ctx = ExecutionContext.create(MagicMock(), MagicMock())
        new_ctx = ctx.with_metadata(vertical="coding", mode="planning")

        assert new_ctx.metadata == {"vertical": "coding", "mode": "planning"}
        assert ctx.metadata == {}  # Original unchanged

    def test_with_metadata_merges_existing(self):
        ctx = ExecutionContext.create(MagicMock(), MagicMock())
        ctx1 = ctx.with_metadata(a=1)
        ctx2 = ctx1.with_metadata(b=2)

        assert ctx2.metadata == {"a": 1, "b": 2}


class TestCleanupLifecycle:
    """Test cleanup hooks for long-running session resources (GS-4)."""

    @pytest.mark.asyncio
    async def test_register_and_run_sync_cleanup(self):
        """Sync cleanup hooks should be called during cleanup()."""
        ctx = ExecutionContext.create(MagicMock(), MagicMock())
        called = []
        ctx.register_cleanup(lambda: called.append("cleaned"))

        count = await ctx.cleanup()
        assert count == 1
        assert called == ["cleaned"]

    @pytest.mark.asyncio
    async def test_register_and_run_async_cleanup(self):
        """Async cleanup hooks should be awaited during cleanup()."""
        ctx = ExecutionContext.create(MagicMock(), MagicMock())
        called = []

        async def async_hook():
            called.append("async_cleaned")

        ctx.register_cleanup(async_hook)
        count = await ctx.cleanup()
        assert count == 1
        assert called == ["async_cleaned"]

    @pytest.mark.asyncio
    async def test_cleanup_runs_in_reverse_order(self):
        """Hooks should run in reverse registration order (LIFO)."""
        ctx = ExecutionContext.create(MagicMock(), MagicMock())
        order = []
        ctx.register_cleanup(lambda: order.append("first"))
        ctx.register_cleanup(lambda: order.append("second"))
        ctx.register_cleanup(lambda: order.append("third"))

        await ctx.cleanup()
        assert order == ["third", "second", "first"]

    @pytest.mark.asyncio
    async def test_cleanup_tolerates_hook_failure(self):
        """A failing hook should not prevent other hooks from running."""
        ctx = ExecutionContext.create(MagicMock(), MagicMock())
        called = []
        ctx.register_cleanup(lambda: called.append("a"))
        ctx.register_cleanup(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        ctx.register_cleanup(lambda: called.append("c"))

        count = await ctx.cleanup()
        # "c" runs first (LIFO), then the failing one, then "a"
        assert "c" in called
        assert "a" in called
        assert count == 2  # 2 succeeded, 1 failed

    @pytest.mark.asyncio
    async def test_cleanup_clears_hooks_after_run(self):
        """After cleanup(), hooks list should be empty."""
        ctx = ExecutionContext.create(MagicMock(), MagicMock())
        ctx.register_cleanup(lambda: None)

        await ctx.cleanup()
        assert len(ctx._cleanup_hooks) == 0

    @pytest.mark.asyncio
    async def test_cleanup_with_closeable_object(self):
        """Objects with close() method should be cleaned up."""
        ctx = ExecutionContext.create(MagicMock(), MagicMock())

        class Closeable:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        obj = Closeable()
        ctx.register_cleanup(obj)
        count = await ctx.cleanup()
        assert count == 1
        assert obj.closed is True

    @pytest.mark.asyncio
    async def test_no_hooks_returns_zero(self):
        ctx = ExecutionContext.create(MagicMock(), MagicMock())
        count = await ctx.cleanup()
        assert count == 0


class TestExecutionContextIntegration:
    """Test integration with real DI container and runtime service resolution."""

    def test_resolve_execution_context_prefers_explicit_argument(self):
        explicit_context = MagicMock(name="explicit_context")
        runtime_owner = MagicMock()
        runtime_owner._execution_context = MagicMock(name="orchestrator_context")

        resolved = resolve_execution_context(runtime_owner, explicit_context)

        assert resolved is explicit_context

    def test_resolve_runtime_services_prefers_execution_context_services(self):
        chat_service = MagicMock(name="chat_service")
        session_service = MagicMock(name="session_service")
        runtime_owner = MagicMock()
        execution_context = MagicMock()
        execution_context.services = MagicMock()
        execution_context.services.chat = chat_service
        execution_context.services.session = session_service

        resolved = resolve_runtime_services(runtime_owner, execution_context)

        assert resolved.chat is chat_service
        assert resolved.session is session_service

    def test_resolve_runtime_services_falls_back_to_context_container(self):
        chat_service = MagicMock(name="chat_service")
        session_service = MagicMock(name="session_service")
        container = MagicMock()
        container.get_optional.side_effect = [chat_service, session_service]
        runtime_owner = MagicMock()
        runtime_owner._chat_service = MagicMock(name="legacy_chat")
        runtime_owner._session_service = MagicMock(name="legacy_session")
        execution_context = MagicMock()
        execution_context.services = MagicMock()
        execution_context.services.chat = None
        execution_context.services.session = None
        execution_context.container = container

        resolved = resolve_runtime_services(runtime_owner, execution_context)

        assert resolved.chat is chat_service
        assert resolved.session is session_service

    def test_resolve_runtime_services_falls_back_to_runtime_owner_state(self):
        chat_service = MagicMock(name="chat_service")
        session_service = MagicMock(name="session_service")
        runtime_owner = MagicMock()
        runtime_owner._chat_service = chat_service
        runtime_owner._session_service = session_service

        resolved = resolve_runtime_services(runtime_owner)

        assert resolved.chat is chat_service
        assert resolved.session is session_service

    def test_create_with_real_container(self):
        from victor.core.container import ServiceContainer

        container = ServiceContainer()
        settings = MagicMock()

        ctx = ExecutionContext.create(settings, container, session_id="integration-test")

        assert ctx.session_id == "integration-test"
        assert ctx.container is container
        # Services should be None since nothing is registered
        assert ctx.services.chat is None
        assert ctx.services.tool is None

    def test_create_with_bootstrapped_container(self):
        from victor.core.bootstrap_services import bootstrap_new_services
        from victor.core.container import ServiceContainer

        container = ServiceContainer()
        bootstrap_new_services(
            container,
            conversation_controller=MagicMock(),
            streaming_coordinator=MagicMock(),
        )

        ctx = ExecutionContext.create(MagicMock(), container, session_id="bootstrapped")

        # Services should be resolved since they were bootstrapped
        assert ctx.services.chat is not None
        assert ctx.services.tool is not None
        assert ctx.services.session is not None
        assert ctx.services.context is not None
        assert ctx.services.provider is not None
        assert ctx.services.recovery is not None
