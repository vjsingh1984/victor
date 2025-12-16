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

"""Tests for CQRS pattern module."""

import asyncio
from dataclasses import dataclass
from typing import Optional

import pytest

from victor.core.cqrs import (
    # Base types
    Command,
    Query,
    CommandResult,
    QueryResult,
    # Handlers
    CommandHandler,
    QueryHandler,
    command_handler,
    query_handler,
    get_registered_command_handlers,
    get_registered_query_handlers,
    clear_handlers,
    # Middleware
    CommandMiddleware,
    QueryMiddleware,
    LoggingCommandMiddleware,
    LoggingQueryMiddleware,
    ValidationCommandMiddleware,
    RetryCommandMiddleware,
    CachingQueryMiddleware,
    # Buses
    CommandBus,
    QueryBus,
    # Mediator
    Mediator,
    # Read Models
    ReadModel,
    InMemoryReadModel,
    # Exceptions
    CQRSError,
    CommandError,
    QueryError,
    ValidationError,
    # Factories
    create_command_bus,
    create_query_bus,
    create_mediator,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class CreateUserCommand(Command):
    """Test command for creating a user."""

    name: str = ""
    email: str = ""


@dataclass
class UpdateUserCommand(Command):
    """Test command for updating a user."""

    user_id: str = ""
    name: str = ""


@dataclass
class DeleteUserCommand(Command):
    """Test command for deleting a user."""

    user_id: str = ""


@dataclass
class User:
    """Test user entity."""

    id: str = ""
    name: str = ""
    email: str = ""


@dataclass
class GetUserQuery(Query[User]):
    """Test query for getting a user."""

    user_id: str = ""


@dataclass
class ListUsersQuery(Query[list]):
    """Test query for listing users."""

    limit: int = 10


@pytest.fixture(autouse=True)
def cleanup_handlers():
    """Clear handler registry before and after each test."""
    clear_handlers()
    yield
    clear_handlers()


# =============================================================================
# Command Tests
# =============================================================================


class TestCommand:
    """Tests for Command base class."""

    def test_command_has_id(self):
        """Test command has unique ID."""
        cmd = CreateUserCommand(name="Alice", email="alice@test.com")

        assert cmd.command_id is not None
        assert len(cmd.command_id) > 0

    def test_command_has_timestamp(self):
        """Test command has timestamp."""
        cmd = CreateUserCommand(name="Alice", email="alice@test.com")

        assert cmd.timestamp is not None

    def test_correlation_id_defaults_to_command_id(self):
        """Test correlation_id defaults to command_id."""
        cmd = CreateUserCommand(name="Alice", email="alice@test.com")

        assert cmd.correlation_id == cmd.command_id

    def test_correlation_id_can_be_set(self):
        """Test correlation_id can be explicitly set."""
        cmd = CreateUserCommand(
            name="Alice", email="alice@test.com", correlation_id="custom-correlation"
        )

        assert cmd.correlation_id == "custom-correlation"

    def test_different_commands_have_different_ids(self):
        """Test different commands have unique IDs."""
        cmd1 = CreateUserCommand(name="Alice", email="alice@test.com")
        cmd2 = CreateUserCommand(name="Bob", email="bob@test.com")

        assert cmd1.command_id != cmd2.command_id


# =============================================================================
# Query Tests
# =============================================================================


class TestQuery:
    """Tests for Query base class."""

    def test_query_has_id(self):
        """Test query has unique ID."""
        query = GetUserQuery(user_id="user-1")

        assert query.query_id is not None
        assert len(query.query_id) > 0

    def test_query_has_timestamp(self):
        """Test query has timestamp."""
        query = GetUserQuery(user_id="user-1")

        assert query.timestamp is not None

    def test_different_queries_have_different_ids(self):
        """Test different queries have unique IDs."""
        q1 = GetUserQuery(user_id="user-1")
        q2 = GetUserQuery(user_id="user-2")

        assert q1.query_id != q2.query_id


# =============================================================================
# Handler Registration Tests
# =============================================================================


class TestHandlerRegistration:
    """Tests for handler registration decorators."""

    def test_command_handler_decorator(self):
        """Test command_handler decorator registers handler."""

        @command_handler(CreateUserCommand)
        async def handle_create(cmd: CreateUserCommand) -> str:
            return "user-123"

        handlers = get_registered_command_handlers()
        assert CreateUserCommand in handlers

    def test_query_handler_decorator(self):
        """Test query_handler decorator registers handler."""

        @query_handler(GetUserQuery)
        async def handle_get(query: GetUserQuery) -> User:
            return User(id=query.user_id, name="Test", email="test@test.com")

        handlers = get_registered_query_handlers()
        assert GetUserQuery in handlers

    def test_clear_handlers(self):
        """Test clearing handler registry."""

        @command_handler(CreateUserCommand)
        async def handle_create(cmd: CreateUserCommand) -> str:
            return "user-123"

        @query_handler(GetUserQuery)
        async def handle_get(query: GetUserQuery) -> User:
            return User(id=query.user_id, name="Test", email="test@test.com")

        clear_handlers()

        assert len(get_registered_command_handlers()) == 0
        assert len(get_registered_query_handlers()) == 0


# =============================================================================
# CommandBus Tests
# =============================================================================


class TestCommandBus:
    """Tests for CommandBus."""

    @pytest.mark.asyncio
    async def test_execute_registered_command(self):
        """Test executing a registered command."""
        bus = CommandBus()

        async def handle_create(cmd: CreateUserCommand) -> str:
            return f"user-{cmd.name}"

        bus.register(CreateUserCommand, handle_create)

        result = await bus.execute(CreateUserCommand(name="Alice", email="alice@test.com"))

        assert result.success is True
        assert result.result == "user-Alice"

    @pytest.mark.asyncio
    async def test_execute_unregistered_command(self):
        """Test executing unregistered command returns error."""
        bus = CommandBus()

        result = await bus.execute(CreateUserCommand(name="Alice", email="alice@test.com"))

        assert result.success is False
        assert "No handler registered" in result.error

    @pytest.mark.asyncio
    async def test_execute_returns_command_id(self):
        """Test result contains command ID."""
        bus = CommandBus()

        async def handle_create(cmd: CreateUserCommand) -> str:
            return "user-123"

        bus.register(CreateUserCommand, handle_create)
        cmd = CreateUserCommand(name="Alice", email="alice@test.com")

        result = await bus.execute(cmd)

        assert result.command_id == cmd.command_id

    @pytest.mark.asyncio
    async def test_execute_measures_time(self):
        """Test result contains execution time."""
        bus = CommandBus()

        async def handle_create(cmd: CreateUserCommand) -> str:
            await asyncio.sleep(0.01)  # Small delay
            return "user-123"

        bus.register(CreateUserCommand, handle_create)

        result = await bus.execute(CreateUserCommand(name="Alice", email="alice@test.com"))

        assert result.execution_time_ms >= 10  # At least 10ms

    @pytest.mark.asyncio
    async def test_execute_handler_exception(self):
        """Test handler exception returns error result."""
        bus = CommandBus()

        async def handle_create(cmd: CreateUserCommand) -> str:
            raise ValueError("Invalid user data")

        bus.register(CreateUserCommand, handle_create)

        result = await bus.execute(CreateUserCommand(name="Alice", email="alice@test.com"))

        assert result.success is False
        assert "Invalid user data" in result.error

    @pytest.mark.asyncio
    async def test_register_from_decorators(self):
        """Test registering handlers from decorators."""

        @command_handler(CreateUserCommand)
        async def handle_create(cmd: CreateUserCommand) -> str:
            return "decorated-user"

        bus = CommandBus()
        bus.register_from_decorators()

        result = await bus.execute(CreateUserCommand(name="Alice", email="alice@test.com"))

        assert result.success is True
        assert result.result == "decorated-user"

    @pytest.mark.asyncio
    async def test_register_class_handler(self):
        """Test registering class-based handler."""

        class CreateUserHandler(CommandHandler[str]):
            async def handle(self, command: Command) -> str:
                return "class-handler-result"

        bus = CommandBus()
        bus.register(CreateUserCommand, CreateUserHandler())

        result = await bus.execute(CreateUserCommand(name="Alice", email="alice@test.com"))

        assert result.success is True
        assert result.result == "class-handler-result"


# =============================================================================
# QueryBus Tests
# =============================================================================


class TestQueryBus:
    """Tests for QueryBus."""

    @pytest.mark.asyncio
    async def test_execute_registered_query(self):
        """Test executing a registered query."""
        bus = QueryBus()

        async def handle_get(query: GetUserQuery) -> User:
            return User(id=query.user_id, name="Alice", email="alice@test.com")

        bus.register(GetUserQuery, handle_get)

        result = await bus.execute(GetUserQuery(user_id="user-123"))

        assert result.success is True
        assert result.data.name == "Alice"

    @pytest.mark.asyncio
    async def test_execute_unregistered_query(self):
        """Test executing unregistered query returns error."""
        bus = QueryBus()

        result = await bus.execute(GetUserQuery(user_id="user-123"))

        assert result.success is False
        assert "No handler registered" in result.error

    @pytest.mark.asyncio
    async def test_execute_returns_query_id(self):
        """Test result contains query ID."""
        bus = QueryBus()

        async def handle_get(query: GetUserQuery) -> User:
            return User(id=query.user_id, name="Alice", email="alice@test.com")

        bus.register(GetUserQuery, handle_get)
        query = GetUserQuery(user_id="user-123")

        result = await bus.execute(query)

        assert result.query_id == query.query_id

    @pytest.mark.asyncio
    async def test_register_from_decorators(self):
        """Test registering handlers from decorators."""

        @query_handler(GetUserQuery)
        async def handle_get(query: GetUserQuery) -> User:
            return User(id=query.user_id, name="Decorated", email="dec@test.com")

        bus = QueryBus()
        bus.register_from_decorators()

        result = await bus.execute(GetUserQuery(user_id="user-123"))

        assert result.success is True
        assert result.data.name == "Decorated"


# =============================================================================
# Middleware Tests
# =============================================================================


class TestCommandMiddleware:
    """Tests for command middleware."""

    @pytest.mark.asyncio
    async def test_logging_middleware(self):
        """Test logging middleware executes handler."""
        bus = CommandBus()
        bus.use(LoggingCommandMiddleware())

        call_count = 0

        async def handle_create(cmd: CreateUserCommand) -> str:
            nonlocal call_count
            call_count += 1
            return "user-123"

        bus.register(CreateUserCommand, handle_create)

        result = await bus.execute(CreateUserCommand(name="Alice", email="alice@test.com"))

        assert result.success is True
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_validation_middleware(self):
        """Test validation middleware validates command."""
        bus = CommandBus()
        validation_middleware = ValidationCommandMiddleware()

        def validate_create(cmd: CreateUserCommand) -> None:
            if not cmd.email or "@" not in cmd.email:
                raise ValidationError("Invalid email")

        validation_middleware.register_validator(CreateUserCommand, validate_create)
        bus.use(validation_middleware)

        async def handle_create(cmd: CreateUserCommand) -> str:
            return "user-123"

        bus.register(CreateUserCommand, handle_create)

        # Valid command
        result = await bus.execute(CreateUserCommand(name="Alice", email="alice@test.com"))
        assert result.success is True

        # Invalid command
        result = await bus.execute(CreateUserCommand(name="Bob", email="invalid"))
        assert result.success is False
        assert "Invalid email" in result.error

    @pytest.mark.asyncio
    async def test_retry_middleware(self):
        """Test retry middleware retries failed commands."""
        bus = CommandBus()
        bus.use(RetryCommandMiddleware(max_retries=2, retry_delay=0.01))

        attempt_count = 0

        async def handle_create(cmd: CreateUserCommand) -> str:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise ValueError("Temporary error")
            return "user-123"

        bus.register(CreateUserCommand, handle_create)

        result = await bus.execute(CreateUserCommand(name="Alice", email="alice@test.com"))

        assert result.success is True
        assert attempt_count == 2

    @pytest.mark.asyncio
    async def test_retry_middleware_exhausted(self):
        """Test retry middleware fails after max retries."""
        bus = CommandBus()
        bus.use(RetryCommandMiddleware(max_retries=2, retry_delay=0.01))

        async def handle_create(cmd: CreateUserCommand) -> str:
            raise ValueError("Permanent error")

        bus.register(CreateUserCommand, handle_create)

        result = await bus.execute(CreateUserCommand(name="Alice", email="alice@test.com"))

        assert result.success is False

    @pytest.mark.asyncio
    async def test_multiple_middleware(self):
        """Test multiple middleware in pipeline."""
        bus = CommandBus()

        calls = []

        class FirstMiddleware(CommandMiddleware):
            async def execute(self, command, next_handler):
                calls.append("first-before")
                result = await next_handler(command)
                calls.append("first-after")
                return result

        class SecondMiddleware(CommandMiddleware):
            async def execute(self, command, next_handler):
                calls.append("second-before")
                result = await next_handler(command)
                calls.append("second-after")
                return result

        bus.use(FirstMiddleware())
        bus.use(SecondMiddleware())

        async def handle_create(cmd: CreateUserCommand) -> str:
            calls.append("handler")
            return "user-123"

        bus.register(CreateUserCommand, handle_create)

        await bus.execute(CreateUserCommand(name="Alice", email="alice@test.com"))

        assert calls == [
            "first-before",
            "second-before",
            "handler",
            "second-after",
            "first-after",
        ]


class TestQueryMiddleware:
    """Tests for query middleware."""

    @pytest.mark.asyncio
    async def test_caching_middleware(self):
        """Test caching middleware caches results."""
        bus = QueryBus()
        bus.use(CachingQueryMiddleware(ttl=60.0))

        call_count = 0

        async def handle_get(query: GetUserQuery) -> User:
            nonlocal call_count
            call_count += 1
            return User(id=query.user_id, name="Alice", email="alice@test.com")

        bus.register(GetUserQuery, handle_get)

        # First call
        result1 = await bus.execute(GetUserQuery(user_id="user-123"))
        # Second call (should be cached)
        result2 = await bus.execute(GetUserQuery(user_id="user-123"))

        assert result1.success is True
        assert result2.success is True
        assert call_count == 1  # Handler called only once

    @pytest.mark.asyncio
    async def test_caching_middleware_different_queries(self):
        """Test different queries are cached separately."""
        bus = QueryBus()
        bus.use(CachingQueryMiddleware(ttl=60.0))

        call_count = 0

        async def handle_get(query: GetUserQuery) -> User:
            nonlocal call_count
            call_count += 1
            return User(id=query.user_id, name=f"User-{query.user_id}", email="test@test.com")

        bus.register(GetUserQuery, handle_get)

        await bus.execute(GetUserQuery(user_id="user-1"))
        await bus.execute(GetUserQuery(user_id="user-2"))

        assert call_count == 2  # Different queries, both executed

    @pytest.mark.asyncio
    async def test_caching_middleware_invalidation(self):
        """Test cache invalidation."""
        cache_middleware = CachingQueryMiddleware(ttl=60.0)
        bus = QueryBus()
        bus.use(cache_middleware)

        call_count = 0

        async def handle_get(query: GetUserQuery) -> User:
            nonlocal call_count
            call_count += 1
            return User(id=query.user_id, name="Alice", email="alice@test.com")

        bus.register(GetUserQuery, handle_get)

        await bus.execute(GetUserQuery(user_id="user-123"))
        cache_middleware.invalidate(GetUserQuery)
        await bus.execute(GetUserQuery(user_id="user-123"))

        assert call_count == 2  # Called twice due to invalidation


# =============================================================================
# Mediator Tests
# =============================================================================


class TestMediator:
    """Tests for Mediator."""

    @pytest.mark.asyncio
    async def test_send_command(self):
        """Test sending command through mediator."""
        mediator = Mediator()

        async def handle_create(cmd: CreateUserCommand) -> str:
            return f"user-{cmd.name}"

        mediator.register_command(CreateUserCommand, handle_create)

        result = await mediator.send(CreateUserCommand(name="Alice", email="alice@test.com"))

        assert isinstance(result, CommandResult)
        assert result.success is True
        assert result.result == "user-Alice"

    @pytest.mark.asyncio
    async def test_send_query(self):
        """Test sending query through mediator."""
        mediator = Mediator()

        async def handle_get(query: GetUserQuery) -> User:
            return User(id=query.user_id, name="Alice", email="alice@test.com")

        mediator.register_query(GetUserQuery, handle_get)

        result = await mediator.send(GetUserQuery(user_id="user-123"))

        assert isinstance(result, QueryResult)
        assert result.success is True
        assert result.data.name == "Alice"

    @pytest.mark.asyncio
    async def test_register_from_decorators(self):
        """Test registering from decorators."""

        @command_handler(CreateUserCommand)
        async def handle_create(cmd: CreateUserCommand) -> str:
            return "decorated"

        @query_handler(GetUserQuery)
        async def handle_get(query: GetUserQuery) -> User:
            return User(id=query.user_id, name="Decorated", email="dec@test.com")

        mediator = Mediator()
        mediator.register_from_decorators()

        cmd_result = await mediator.send(CreateUserCommand(name="Alice", email="alice@test.com"))
        query_result = await mediator.send(GetUserQuery(user_id="user-123"))

        assert cmd_result.success is True
        assert query_result.success is True

    @pytest.mark.asyncio
    async def test_access_buses(self):
        """Test accessing command and query buses."""
        mediator = Mediator()

        assert mediator.command_bus is not None
        assert mediator.query_bus is not None


# =============================================================================
# Read Model Tests
# =============================================================================


class TestInMemoryReadModel:
    """Tests for InMemoryReadModel."""

    @pytest.mark.asyncio
    async def test_get_returns_none_for_missing(self):
        """Test get returns None for missing item."""
        model = InMemoryReadModel[User]()

        result = await model.get("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_returns_item(self):
        """Test get returns stored item."""
        model = InMemoryReadModel[User]()
        user = User(id="user-1", name="Alice", email="alice@test.com")
        model.data["user-1"] = user

        result = await model.get("user-1")

        assert result == user

    @pytest.mark.asyncio
    async def test_get_all(self):
        """Test get_all returns all items."""
        model = InMemoryReadModel[User]()
        model.data["user-1"] = User(id="user-1", name="Alice", email="alice@test.com")
        model.data["user-2"] = User(id="user-2", name="Bob", email="bob@test.com")

        result = await model.get_all()

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_rebuild_clears_data(self):
        """Test rebuild clears data."""
        model = InMemoryReadModel[User]()
        model.data["user-1"] = User(id="user-1", name="Alice", email="alice@test.com")

        await model.rebuild()

        assert len(model.data) == 0


# =============================================================================
# Factory Tests
# =============================================================================


class TestFactories:
    """Tests for factory functions."""

    def test_create_command_bus_default(self):
        """Test create_command_bus with defaults."""
        bus = create_command_bus()

        assert isinstance(bus, CommandBus)

    def test_create_command_bus_with_retry(self):
        """Test create_command_bus with retry middleware."""
        bus = create_command_bus(with_retry=True, max_retries=5)

        assert isinstance(bus, CommandBus)
        # Has middleware registered (internal check via execute)

    def test_create_query_bus_default(self):
        """Test create_query_bus with defaults."""
        bus = create_query_bus()

        assert isinstance(bus, QueryBus)

    def test_create_query_bus_with_caching(self):
        """Test create_query_bus with caching middleware."""
        bus = create_query_bus(with_caching=True, cache_ttl=120.0)

        assert isinstance(bus, QueryBus)

    def test_create_mediator(self):
        """Test create_mediator."""
        mediator = create_mediator()

        assert isinstance(mediator, Mediator)
        assert mediator.command_bus is not None
        assert mediator.query_bus is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestCQRSIntegration:
    """Integration tests for CQRS pattern."""

    @pytest.mark.asyncio
    async def test_full_cqrs_workflow(self):
        """Test complete CQRS workflow."""
        # Storage
        users: dict = {}

        # Command handler
        async def handle_create(cmd: CreateUserCommand) -> str:
            user_id = f"user-{len(users) + 1}"
            users[user_id] = User(id=user_id, name=cmd.name, email=cmd.email)
            return user_id

        async def handle_update(cmd: UpdateUserCommand) -> None:
            if cmd.user_id in users:
                users[cmd.user_id].name = cmd.name

        async def handle_delete(cmd: DeleteUserCommand) -> None:
            users.pop(cmd.user_id, None)

        # Query handler
        async def handle_get(query: GetUserQuery) -> Optional[User]:
            return users.get(query.user_id)

        async def handle_list(query: ListUsersQuery) -> list:
            return list(users.values())[: query.limit]

        # Create mediator
        mediator = Mediator()
        mediator.register_command(CreateUserCommand, handle_create)
        mediator.register_command(UpdateUserCommand, handle_update)
        mediator.register_command(DeleteUserCommand, handle_delete)
        mediator.register_query(GetUserQuery, handle_get)
        mediator.register_query(ListUsersQuery, handle_list)

        # Execute workflow
        # Create user
        create_result = await mediator.send(CreateUserCommand(name="Alice", email="alice@test.com"))
        assert create_result.success is True
        user_id = create_result.result

        # Query user
        get_result = await mediator.send(GetUserQuery(user_id=user_id))
        assert get_result.success is True
        assert get_result.data.name == "Alice"

        # Update user
        update_result = await mediator.send(
            UpdateUserCommand(user_id=user_id, name="Alice Updated")
        )
        assert update_result.success is True

        # Verify update
        get_result2 = await mediator.send(GetUserQuery(user_id=user_id))
        assert get_result2.data.name == "Alice Updated"

        # Delete user
        delete_result = await mediator.send(DeleteUserCommand(user_id=user_id))
        assert delete_result.success is True

        # Verify deletion
        get_result3 = await mediator.send(GetUserQuery(user_id=user_id))
        assert get_result3.data is None

    @pytest.mark.asyncio
    async def test_cqrs_with_middleware_pipeline(self):
        """Test CQRS with full middleware pipeline."""
        # Create mediator with middleware
        mediator = create_mediator(
            with_logging=True,
            with_retry=True,
            with_caching=True,
        )

        users: dict = {}

        async def handle_create(cmd: CreateUserCommand) -> str:
            user_id = f"user-{len(users) + 1}"
            users[user_id] = User(id=user_id, name=cmd.name, email=cmd.email)
            return user_id

        async def handle_get(query: GetUserQuery) -> Optional[User]:
            return users.get(query.user_id)

        mediator.register_command(CreateUserCommand, handle_create)
        mediator.register_query(GetUserQuery, handle_get)

        # Execute
        create_result = await mediator.send(CreateUserCommand(name="Bob", email="bob@test.com"))
        assert create_result.success is True

        get_result = await mediator.send(GetUserQuery(user_id=create_result.result))
        assert get_result.success is True
        assert get_result.data.name == "Bob"
