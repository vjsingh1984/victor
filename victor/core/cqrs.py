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

"""CQRS (Command Query Responsibility Segregation) pattern implementation.

This module provides a comprehensive CQRS implementation that:
- Separates read and write operations for scalability
- Supports command handlers for write operations
- Supports query handlers for read operations
- Integrates with Event Sourcing for eventual consistency
- Provides middleware pipeline for cross-cutting concerns

Design Patterns:
- CQRS: Separate read/write paths
- Mediator: Route commands/queries to handlers
- Pipeline: Middleware for cross-cutting concerns
- Factory: Handler registration and resolution

Example:
    from victor.core.cqrs import (
        Command, Query, CommandBus, QueryBus,
        command_handler, query_handler
    )

    # Define a command
    @dataclass
    class CreateUserCommand(Command):
        name: str
        email: str

    # Define command handler
    @command_handler(CreateUserCommand)
    async def handle_create_user(command: CreateUserCommand) -> str:
        user_id = await user_service.create(command.name, command.email)
        return user_id

    # Execute command
    bus = CommandBus()
    user_id = await bus.execute(CreateUserCommand(name="Alice", email="alice@example.com"))

    # Define a query
    @dataclass
    class GetUserQuery(Query[User]):
        user_id: str

    # Define query handler
    @query_handler(GetUserQuery)
    async def handle_get_user(query: GetUserQuery) -> User:
        return await user_repo.get(query.user_id)

    # Execute query
    query_bus = QueryBus()
    user = await query_bus.execute(GetUserQuery(user_id="123"))
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    Awaitable,
)
from uuid import uuid4

logger = logging.getLogger(__name__)


# =============================================================================
# Base Types
# =============================================================================


TResult = TypeVar("TResult")


@dataclass
class Command:
    """Base class for commands (write operations).

    Commands represent intent to change system state.
    They should be named imperatively (CreateUser, UpdateOrder, etc.)
    """

    command_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Set correlation_id to command_id if not provided."""
        if self.correlation_id is None:
            self.correlation_id = self.command_id


@dataclass
class Query(Generic[TResult]):
    """Base class for queries (read operations).

    Queries represent a request for data without side effects.
    They should be named as questions (GetUser, FindOrders, etc.)
    """

    query_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CommandResult(Generic[TResult]):
    """Result of command execution."""

    success: bool
    result: Optional[TResult] = None
    error: Optional[str] = None
    command_id: str = ""
    execution_time_ms: float = 0.0
    events: List[Any] = field(default_factory=list)


@dataclass
class QueryResult(Generic[TResult]):
    """Result of query execution."""

    success: bool
    data: Optional[TResult] = None
    error: Optional[str] = None
    query_id: str = ""
    execution_time_ms: float = 0.0
    cached: bool = False


# =============================================================================
# Handler Types
# =============================================================================


class CommandHandler(ABC, Generic[TResult]):
    """Abstract base for command handlers."""

    @abstractmethod
    async def handle(self, command: Command) -> TResult:
        """Handle a command and return result."""
        ...


class QueryHandler(ABC, Generic[TResult]):
    """Abstract base for query handlers."""

    @abstractmethod
    async def handle(self, query: Query[TResult]) -> TResult:
        """Handle a query and return result."""
        ...


# Handler function types
CommandHandlerFunc = Callable[[Command], Awaitable[Any]]
QueryHandlerFunc = Callable[[Query[TResult]], Awaitable[TResult]]


# =============================================================================
# Handler Registry
# =============================================================================

# Global registries for decorator-based registration
_COMMAND_HANDLERS: Dict[Type[Command], CommandHandlerFunc] = {}
_QUERY_HANDLERS: Dict[Type[Query], QueryHandlerFunc] = {}


def command_handler(
    command_type: Type[Command],
) -> Callable[[CommandHandlerFunc], CommandHandlerFunc]:
    """Decorator to register a command handler function.

    Example:
        @command_handler(CreateUserCommand)
        async def handle_create_user(command: CreateUserCommand) -> str:
            return await create_user(command.name, command.email)
    """

    def decorator(func: CommandHandlerFunc) -> CommandHandlerFunc:
        _COMMAND_HANDLERS[command_type] = func
        logger.debug(f"Registered command handler for {command_type.__name__}")
        return func

    return decorator


def query_handler(
    query_type: Type[Query],
) -> Callable[[QueryHandlerFunc], QueryHandlerFunc]:
    """Decorator to register a query handler function.

    Example:
        @query_handler(GetUserQuery)
        async def handle_get_user(query: GetUserQuery) -> User:
            return await user_repo.get(query.user_id)
    """

    def decorator(func: QueryHandlerFunc) -> QueryHandlerFunc:
        _QUERY_HANDLERS[query_type] = func
        logger.debug(f"Registered query handler for {query_type.__name__}")
        return func

    return decorator


def get_registered_command_handlers() -> Dict[Type[Command], CommandHandlerFunc]:
    """Get all registered command handlers."""
    return _COMMAND_HANDLERS.copy()


def get_registered_query_handlers() -> Dict[Type[Query], QueryHandlerFunc]:
    """Get all registered query handlers."""
    return _QUERY_HANDLERS.copy()


def clear_handlers() -> None:
    """Clear all registered handlers (useful for testing)."""
    _COMMAND_HANDLERS.clear()
    _QUERY_HANDLERS.clear()


# =============================================================================
# Middleware
# =============================================================================


class CommandMiddleware(ABC):
    """Middleware for command processing pipeline."""

    @abstractmethod
    async def execute(
        self,
        command: Command,
        next_handler: Callable[[Command], Awaitable[Any]],
    ) -> Any:
        """Process command and call next handler."""
        ...


class QueryMiddleware(ABC):
    """Middleware for query processing pipeline."""

    @abstractmethod
    async def execute(
        self,
        query: Query[TResult],
        next_handler: Callable[[Query[TResult]], Awaitable[TResult]],
    ) -> TResult:
        """Process query and call next handler."""
        ...


class LoggingCommandMiddleware(CommandMiddleware):
    """Logs command execution."""

    async def execute(
        self,
        command: Command,
        next_handler: Callable[[Command], Awaitable[Any]],
    ) -> Any:
        command_type = type(command).__name__
        logger.info(f"Executing command: {command_type} (id={command.command_id})")
        start = time.perf_counter()

        try:
            result = await next_handler(command)
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(f"Command {command_type} completed in {elapsed:.2f}ms")
            return result
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(f"Command {command_type} failed after {elapsed:.2f}ms: {e}")
            raise


class LoggingQueryMiddleware(QueryMiddleware):
    """Logs query execution."""

    async def execute(
        self,
        query: Query[TResult],
        next_handler: Callable[[Query[TResult]], Awaitable[TResult]],
    ) -> TResult:
        query_type = type(query).__name__
        logger.info(f"Executing query: {query_type} (id={query.query_id})")
        start = time.perf_counter()

        try:
            result = await next_handler(query)
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(f"Query {query_type} completed in {elapsed:.2f}ms")
            return result
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(f"Query {query_type} failed after {elapsed:.2f}ms: {e}")
            raise


class ValidationCommandMiddleware(CommandMiddleware):
    """Validates commands before execution."""

    def __init__(
        self,
        validators: Optional[Dict[Type[Command], Callable[[Command], None]]] = None,
    ) -> None:
        self._validators = validators or {}

    def register_validator(
        self, command_type: Type[Command], validator: Callable[[Command], None]
    ) -> None:
        """Register a validator for a command type."""
        self._validators[command_type] = validator

    async def execute(
        self,
        command: Command,
        next_handler: Callable[[Command], Awaitable[Any]],
    ) -> Any:
        command_type = type(command)
        if command_type in self._validators:
            self._validators[command_type](command)
        return await next_handler(command)


class RetryCommandMiddleware(CommandMiddleware):
    """Retries failed commands."""

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 0.1,
        retryable_exceptions: Optional[tuple] = None,
    ) -> None:
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._retryable_exceptions = retryable_exceptions or (Exception,)

    async def execute(
        self,
        command: Command,
        next_handler: Callable[[Command], Awaitable[Any]],
    ) -> Any:
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            try:
                return await next_handler(command)
            except self._retryable_exceptions as e:
                last_error = e
                if attempt < self._max_retries:
                    logger.warning(
                        f"Command {type(command).__name__} failed (attempt {attempt + 1}), "
                        f"retrying in {self._retry_delay}s: {e}"
                    )
                    await asyncio.sleep(self._retry_delay)

        raise last_error  # type: ignore


class CachingQueryMiddleware(QueryMiddleware):
    """Caches query results."""

    def __init__(self, cache: Optional[Dict[str, Any]] = None, ttl: float = 60.0) -> None:
        self._cache = cache if cache is not None else {}
        self._ttl = ttl
        self._timestamps: Dict[str, float] = {}

    def _get_cache_key(self, query: Query) -> str:
        """Generate cache key from query."""
        query_dict = {k: v for k, v in query.__dict__.items() if k not in ("query_id", "timestamp")}
        return f"{type(query).__name__}:{hash(frozenset(query_dict.items()))}"

    async def execute(
        self,
        query: Query[TResult],
        next_handler: Callable[[Query[TResult]], Awaitable[TResult]],
    ) -> TResult:
        key = self._get_cache_key(query)

        # Check cache
        if key in self._cache:
            cached_at = self._timestamps.get(key, 0)
            if time.time() - cached_at < self._ttl:
                logger.debug(f"Cache hit for {type(query).__name__}")
                return self._cache[key]
            else:
                # Expired
                del self._cache[key]
                del self._timestamps[key]

        # Execute and cache
        result = await next_handler(query)
        self._cache[key] = result
        self._timestamps[key] = time.time()
        return result

    def invalidate(self, query_type: Optional[Type[Query]] = None) -> None:
        """Invalidate cache entries."""
        if query_type is None:
            self._cache.clear()
            self._timestamps.clear()
        else:
            prefix = f"{query_type.__name__}:"
            keys_to_remove = [k for k in self._cache if k.startswith(prefix)]
            for key in keys_to_remove:
                del self._cache[key]
                self._timestamps.pop(key, None)


# =============================================================================
# Command Bus
# =============================================================================


class CommandBus:
    """Routes commands to their handlers with middleware support.

    The command bus is responsible for:
    - Routing commands to registered handlers
    - Executing middleware pipeline
    - Returning command results

    Example:
        bus = CommandBus()
        bus.register(CreateUserCommand, create_user_handler)
        result = await bus.execute(CreateUserCommand(name="Alice"))
    """

    def __init__(self) -> None:
        """Initialize command bus."""
        self._handlers: Dict[Type[Command], CommandHandlerFunc] = {}
        self._middleware: List[CommandMiddleware] = []
        self._lock = asyncio.Lock()

    def register(
        self,
        command_type: Type[Command],
        handler: Union[CommandHandlerFunc, CommandHandler],
    ) -> None:
        """Register a handler for a command type.

        Args:
            command_type: Type of command to handle
            handler: Handler function or class instance
        """
        if isinstance(handler, CommandHandler):
            self._handlers[command_type] = handler.handle
        else:
            self._handlers[command_type] = handler
        logger.debug(f"Registered handler for {command_type.__name__}")

    def register_from_decorators(self) -> None:
        """Register all handlers from decorator registry."""
        for command_type, handler in _COMMAND_HANDLERS.items():
            self._handlers[command_type] = handler

    def use(self, middleware: CommandMiddleware) -> "CommandBus":
        """Add middleware to the pipeline.

        Args:
            middleware: Middleware instance

        Returns:
            Self for chaining
        """
        self._middleware.append(middleware)
        return self

    async def execute(self, command: Command) -> CommandResult:
        """Execute a command through the handler pipeline.

        Args:
            command: Command to execute

        Returns:
            CommandResult with success status and result/error
        """
        command_type = type(command)
        handler = self._handlers.get(command_type)

        if handler is None:
            return CommandResult(
                success=False,
                error=f"No handler registered for {command_type.__name__}",
                command_id=command.command_id,
            )

        start = time.perf_counter()

        # Build middleware chain
        async def final_handler(cmd: Command) -> Any:
            return await handler(cmd)

        chain = final_handler
        for middleware in reversed(self._middleware):

            def create_next(mw: CommandMiddleware, next_fn: Callable) -> Callable:
                async def wrapped(cmd: Command) -> Any:
                    return await mw.execute(cmd, next_fn)

                return wrapped

            chain = create_next(middleware, chain)

        try:
            result = await chain(command)
            elapsed = (time.perf_counter() - start) * 1000

            return CommandResult(
                success=True,
                result=result,
                command_id=command.command_id,
                execution_time_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(f"Command {command_type.__name__} failed: {e}")

            return CommandResult(
                success=False,
                error=str(e),
                command_id=command.command_id,
                execution_time_ms=elapsed,
            )


# =============================================================================
# Query Bus
# =============================================================================


class QueryBus:
    """Routes queries to their handlers with middleware support.

    The query bus is responsible for:
    - Routing queries to registered handlers
    - Executing middleware pipeline (caching, logging)
    - Returning query results

    Example:
        bus = QueryBus()
        bus.register(GetUserQuery, get_user_handler)
        result = await bus.execute(GetUserQuery(user_id="123"))
    """

    def __init__(self) -> None:
        """Initialize query bus."""
        self._handlers: Dict[Type[Query], QueryHandlerFunc] = {}
        self._middleware: List[QueryMiddleware] = []

    def register(
        self,
        query_type: Type[Query],
        handler: Union[QueryHandlerFunc, QueryHandler],
    ) -> None:
        """Register a handler for a query type.

        Args:
            query_type: Type of query to handle
            handler: Handler function or class instance
        """
        if isinstance(handler, QueryHandler):
            self._handlers[query_type] = handler.handle
        else:
            self._handlers[query_type] = handler
        logger.debug(f"Registered handler for {query_type.__name__}")

    def register_from_decorators(self) -> None:
        """Register all handlers from decorator registry."""
        for query_type, handler in _QUERY_HANDLERS.items():
            self._handlers[query_type] = handler

    def use(self, middleware: QueryMiddleware) -> "QueryBus":
        """Add middleware to the pipeline.

        Args:
            middleware: Middleware instance

        Returns:
            Self for chaining
        """
        self._middleware.append(middleware)
        return self

    async def execute(self, query: Query[TResult]) -> QueryResult[TResult]:
        """Execute a query through the handler pipeline.

        Args:
            query: Query to execute

        Returns:
            QueryResult with success status and data/error
        """
        query_type = type(query)
        handler = self._handlers.get(query_type)

        if handler is None:
            return QueryResult(
                success=False,
                error=f"No handler registered for {query_type.__name__}",
                query_id=query.query_id,
            )

        start = time.perf_counter()

        # Build middleware chain
        async def final_handler(q: Query[TResult]) -> TResult:
            return await handler(q)

        chain = final_handler
        for middleware in reversed(self._middleware):

            def create_next(
                mw: QueryMiddleware, next_fn: Callable
            ) -> Callable[[Query[TResult]], Awaitable[TResult]]:
                async def wrapped(q: Query[TResult]) -> TResult:
                    return await mw.execute(q, next_fn)

                return wrapped

            chain = create_next(middleware, chain)

        try:
            data = await chain(query)
            elapsed = (time.perf_counter() - start) * 1000

            return QueryResult(
                success=True,
                data=data,
                query_id=query.query_id,
                execution_time_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(f"Query {query_type.__name__} failed: {e}")

            return QueryResult(
                success=False,
                error=str(e),
                query_id=query.query_id,
                execution_time_ms=elapsed,
            )


# =============================================================================
# Mediator (Unified Dispatcher)
# =============================================================================


class Mediator:
    """Unified dispatcher for both commands and queries.

    Provides a single entry point for all CQRS operations.

    Example:
        mediator = Mediator()
        mediator.register_command(CreateUserCommand, create_user_handler)
        mediator.register_query(GetUserQuery, get_user_handler)

        # Send command
        user_id = await mediator.send(CreateUserCommand(name="Alice"))

        # Send query
        user = await mediator.send(GetUserQuery(user_id=user_id))
    """

    def __init__(
        self,
        command_bus: Optional[CommandBus] = None,
        query_bus: Optional[QueryBus] = None,
    ) -> None:
        """Initialize mediator.

        Args:
            command_bus: Optional command bus instance
            query_bus: Optional query bus instance
        """
        self._command_bus = command_bus or CommandBus()
        self._query_bus = query_bus or QueryBus()

    @property
    def command_bus(self) -> CommandBus:
        """Get command bus."""
        return self._command_bus

    @property
    def query_bus(self) -> QueryBus:
        """Get query bus."""
        return self._query_bus

    def register_command(
        self,
        command_type: Type[Command],
        handler: Union[CommandHandlerFunc, CommandHandler],
    ) -> None:
        """Register a command handler."""
        self._command_bus.register(command_type, handler)

    def register_query(
        self,
        query_type: Type[Query],
        handler: Union[QueryHandlerFunc, QueryHandler],
    ) -> None:
        """Register a query handler."""
        self._query_bus.register(query_type, handler)

    def register_from_decorators(self) -> None:
        """Register all handlers from decorators."""
        self._command_bus.register_from_decorators()
        self._query_bus.register_from_decorators()

    async def send(
        self, message: Union[Command, Query[TResult]]
    ) -> Union[CommandResult, QueryResult[TResult]]:
        """Send a command or query for execution.

        Args:
            message: Command or Query to execute

        Returns:
            CommandResult or QueryResult
        """
        if isinstance(message, Command):
            return await self._command_bus.execute(message)
        elif isinstance(message, Query):
            return await self._query_bus.execute(message)
        else:
            raise TypeError(f"Unknown message type: {type(message)}")


# =============================================================================
# Read Model Support
# =============================================================================


class ReadModel(ABC):
    """Base class for read models (projections).

    Read models are optimized views of data for queries.
    They are typically denormalized and updated from events.
    """

    @abstractmethod
    async def rebuild(self) -> None:
        """Rebuild the read model from events."""
        ...


class InMemoryReadModel(ReadModel, Generic[TResult]):
    """Simple in-memory read model.

    Example:
        class UserReadModel(InMemoryReadModel[User]):
            async def apply_event(self, event: Event) -> None:
                if isinstance(event, UserCreatedEvent):
                    self.data[event.user_id] = User(...)
    """

    def __init__(self) -> None:
        """Initialize read model."""
        self.data: Dict[str, TResult] = {}

    async def get(self, id: str) -> Optional[TResult]:
        """Get item by ID."""
        return self.data.get(id)

    async def get_all(self) -> List[TResult]:
        """Get all items."""
        return list(self.data.values())

    async def rebuild(self) -> None:
        """Rebuild read model - override in subclass."""
        self.data.clear()


# =============================================================================
# Exceptions
# =============================================================================


class CQRSError(Exception):
    """Base exception for CQRS errors."""

    pass


class CommandError(CQRSError):
    """Error during command execution."""

    pass


class QueryError(CQRSError):
    """Error during query execution."""

    pass


class HandlerNotFoundError(CQRSError):
    """Handler not found for command/query."""

    pass


class ValidationError(CQRSError):
    """Command validation error."""

    pass


# =============================================================================
# Factory Functions
# =============================================================================


def create_command_bus(
    with_logging: bool = True,
    with_retry: bool = False,
    max_retries: int = 3,
) -> CommandBus:
    """Create a configured command bus.

    Args:
        with_logging: Add logging middleware
        with_retry: Add retry middleware
        max_retries: Max retry attempts

    Returns:
        Configured CommandBus
    """
    bus = CommandBus()

    if with_logging:
        bus.use(LoggingCommandMiddleware())

    if with_retry:
        bus.use(RetryCommandMiddleware(max_retries=max_retries))

    return bus


def create_query_bus(
    with_logging: bool = True,
    with_caching: bool = False,
    cache_ttl: float = 60.0,
) -> QueryBus:
    """Create a configured query bus.

    Args:
        with_logging: Add logging middleware
        with_caching: Add caching middleware
        cache_ttl: Cache TTL in seconds

    Returns:
        Configured QueryBus
    """
    bus = QueryBus()

    if with_logging:
        bus.use(LoggingQueryMiddleware())

    if with_caching:
        bus.use(CachingQueryMiddleware(ttl=cache_ttl))

    return bus


def create_mediator(
    with_logging: bool = True,
    with_retry: bool = False,
    with_caching: bool = False,
) -> Mediator:
    """Create a configured mediator.

    Args:
        with_logging: Add logging middleware
        with_retry: Add retry middleware for commands
        with_caching: Add caching middleware for queries

    Returns:
        Configured Mediator
    """
    command_bus = create_command_bus(with_logging=with_logging, with_retry=with_retry)
    query_bus = create_query_bus(with_logging=with_logging, with_caching=with_caching)
    return Mediator(command_bus=command_bus, query_bus=query_bus)
