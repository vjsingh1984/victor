# Core Design Patterns

Victor's core infrastructure (`victor/core/`) implements enterprise design patterns for robust, scalable, and maintainable code. This document covers the four foundational patterns and how to use them.

## Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         Victor Core Patterns                                │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │  Event Sourcing │     │    Repository   │     │  Unit of Work   │       │
│  │                 │     │                 │     │                 │       │
│  │ • Event Store   │     │ • CRUD Ops      │     │ • Transactions  │       │
│  │ • Aggregates    │     │ • Specifications│     │ • Identity Map  │       │
│  │ • Projections   │     │ • Caching       │     │ • Change Track  │       │
│  └────────┬────────┘     └────────┬────────┘     └────────┬────────┘       │
│           │                       │                       │                 │
│           └───────────────────────┼───────────────────────┘                 │
│                                   │                                         │
│                         ┌─────────▼─────────┐                               │
│                         │       CQRS        │                               │
│                         │                   │                               │
│                         │ • Command Bus     │                               │
│                         │ • Query Bus       │                               │
│                         │ • Mediator        │                               │
│                         │ • Read Models     │                               │
│                         └───────────────────┘                               │
└────────────────────────────────────────────────────────────────────────────┘
```

## Event Sourcing

Event Sourcing stores all changes to application state as a sequence of events. Instead of storing current state, we store the history of state changes.

### When to Use
- Need complete audit trail
- Require temporal queries ("state at time X")
- Complex domain with business rules
- Need to replay/debug state transitions

### Core Components

```python
from victor.core import (
    Event,
    EventStore,
    InMemoryEventStore,
    SQLiteEventStore,
    Aggregate,
    EventDispatcher,
    Projection,
)
```

### Basic Usage

```python
from dataclasses import dataclass
from victor.core import Event, InMemoryEventStore, Aggregate, EventDispatcher

# Define domain events
@dataclass
class TaskCreatedEvent(Event):
    task_id: str = ""
    title: str = ""

@dataclass
class TaskCompletedEvent(Event):
    task_id: str = ""

# Create event store
store = InMemoryEventStore()

# Store events
event = TaskCreatedEvent(task_id="task-1", title="Implement feature")
await store.append("task-1", event)

# Load events
events = await store.get_events("task-1")
```

### Aggregates

Aggregates encapsulate domain logic and apply events:

```python
from victor.core import Aggregate

class TaskAggregate(Aggregate):
    def __init__(self):
        super().__init__()
        self.title = ""
        self.completed = False

    def create(self, task_id: str, title: str):
        self.apply(TaskCreatedEvent(task_id=task_id, title=title))

    def complete(self):
        if not self.completed:
            self.apply(TaskCompletedEvent(task_id=self.id))

    def _apply_TaskCreatedEvent(self, event: TaskCreatedEvent):
        self.id = event.task_id
        self.title = event.title

    def _apply_TaskCompletedEvent(self, event: TaskCompletedEvent):
        self.completed = True

# Usage
task = TaskAggregate()
task.create("task-1", "Build feature")
task.complete()

# Save to store
for event in task.pending_events:
    await store.append(task.id, event)
task.mark_events_committed()
```

### Projections

Build read-optimized views from events:

```python
from victor.core import Projection, EventDispatcher

class TaskListProjection(Projection):
    def __init__(self):
        self.tasks = {}

    async def handle_TaskCreatedEvent(self, event: TaskCreatedEvent):
        self.tasks[event.task_id] = {
            "id": event.task_id,
            "title": event.title,
            "completed": False,
        }

    async def handle_TaskCompletedEvent(self, event: TaskCompletedEvent):
        if event.task_id in self.tasks:
            self.tasks[event.task_id]["completed"] = True

# Wire up
dispatcher = EventDispatcher()
projection = TaskListProjection()
dispatcher.register(projection)

# Dispatch events
await dispatcher.dispatch(TaskCreatedEvent(task_id="1", title="Test"))
print(projection.tasks)  # {"1": {"id": "1", "title": "Test", "completed": False}}
```

### Persistent Event Store

```python
from victor.core import SQLiteEventStore

# SQLite-backed persistence
store = SQLiteEventStore("events.db")
await store.initialize()

await store.append("aggregate-1", event)
events = await store.get_events("aggregate-1")
```

---

## Repository Pattern

The Repository pattern abstracts data access, providing collection-like interfaces for domain entities.

### When to Use
- Decouple domain logic from persistence
- Enable easy testing with in-memory stores
- Support multiple storage backends
- Apply consistent data access patterns

### Core Components

```python
from victor.core import (
    Entity,
    Repository,
    InMemoryRepository,
    SQLiteRepository,
    CachedRepository,
    # Specification pattern
    BaseSpecification,
    AttributeSpecification,
    AndSpecification,
    OrSpecification,
    NotSpecification,
)
```

### Basic Usage

```python
from dataclasses import dataclass
from victor.core import Entity, InMemoryRepository

@dataclass
class User(Entity):
    name: str = ""
    email: str = ""
    active: bool = True

# Create repository
repo = InMemoryRepository[User]()

# CRUD operations
user = User(name="Alice", email="alice@example.com")
await repo.add(user)

found = await repo.get(user.id)
print(found.name)  # "Alice"

user.email = "alice@newdomain.com"
await repo.update(user)

await repo.delete(user.id)
```

### Specification Pattern

Query using composable specifications:

```python
from victor.core import AttributeSpecification, AndSpecification

# Simple attribute match
active_spec = AttributeSpecification("active", True)
active_users = await repo.find(active_spec)

# Composite specifications
name_spec = AttributeSpecification("name", "Alice")
combined = AndSpecification(active_spec, name_spec)
alice_if_active = await repo.find(combined)

# Custom specification
class EmailDomainSpec(BaseSpecification[User]):
    def __init__(self, domain: str):
        self.domain = domain

    def is_satisfied_by(self, entity: User) -> bool:
        return entity.email.endswith(f"@{self.domain}")

corp_users = await repo.find(EmailDomainSpec("corp.com"))
```

### Cached Repository

Add caching layer for performance:

```python
from victor.core import InMemoryRepository, CachedRepository, create_cached_repository

# Manual setup
base_repo = InMemoryRepository[User]()
cached_repo = CachedRepository(base_repo, ttl_seconds=300)

# Or use factory
cached_repo = create_cached_repository(
    "memory",
    entity_type=User,
    cache_ttl=300,
)
```

### SQLite Repository

```python
from victor.core import SQLiteRepository

repo = SQLiteRepository[User](
    db_path="users.db",
    table_name="users",
)
await repo.initialize()
```

---

## Unit of Work

Unit of Work tracks changes across multiple repositories and commits them atomically.

### When to Use
- Coordinate changes across multiple entities
- Need transaction semantics
- Implement domain operations touching multiple aggregates
- Ensure data consistency

### Core Components

```python
from victor.core import (
    UnitOfWork,
    SQLiteUnitOfWork,
    CompositeUnitOfWork,
    EntityState,
    TrackedEntity,
    transactional,
)
```

### Basic Usage

```python
from victor.core import UnitOfWork, InMemoryRepository

@dataclass
class Order(Entity):
    user_id: str = ""
    total: float = 0.0

# Setup
user_repo = InMemoryRepository[User]()
order_repo = InMemoryRepository[Order]()

uow = UnitOfWork()
uow.register_repository(user_repo, User)
uow.register_repository(order_repo, Order)

# Track changes
user = User(name="Bob", email="bob@example.com")
order = Order(user_id=user.id, total=99.99)

uow.register_new(user)
uow.register_new(order)

# Commit atomically
await uow.commit()

# Both persisted or neither
```

### Context Manager

```python
async with UnitOfWork() as uow:
    uow.register_repository(user_repo, User)

    user = User(name="Charlie")
    uow.register_new(user)

    await uow.commit()
    # Auto-rollback on exception
```

### Transactional Decorator

```python
from victor.core import transactional

async with transactional((user_repo, User), (order_repo, Order)) as uow:
    user = User(name="Dana")
    order = Order(user_id=user.id, total=50.0)

    uow.register_new(user)
    uow.register_new(order)
    # Auto-commit on success, auto-rollback on error
```

### Change Tracking

```python
uow = UnitOfWork()
uow.register_repository(user_repo, User)

# Attach existing entity
user = await user_repo.get("user-1")
uow.attach(user)

# Modify
user.email = "new@email.com"
uow.register_modified(user)

# Check changes
print(uow.has_changes())  # True
print(uow.pending_count)  # 1

await uow.commit()
```

### Rollback

```python
uow = UnitOfWork()
uow.attach(user)
original_email = user.email

user.email = "changed@email.com"
uow.register_modified(user)

await uow.rollback()
print(user.email)  # original_email (reverted)
```

---

## CQRS (Command Query Responsibility Segregation)

CQRS separates read and write operations into distinct models, enabling independent optimization and scaling.

### When to Use
- Read and write patterns differ significantly
- Need to scale reads independently from writes
- Complex domain logic for writes, simple reads
- Event-driven architecture

### Core Components

```python
from victor.core import (
    # Commands
    Command,
    CommandResult,
    CommandHandler,
    CommandBus,
    command_handler,
    # Queries
    Query,
    QueryResult,
    QueryHandler,
    QueryBus,
    query_handler,
    # Mediator
    Mediator,
    # Middleware
    LoggingCommandMiddleware,
    ValidationCommandMiddleware,
    RetryCommandMiddleware,
    CachingQueryMiddleware,
    # Read Models
    ReadModel,
    InMemoryReadModel,
)
```

### Commands

```python
from dataclasses import dataclass
from victor.core import Command, CommandResult, CommandBus

@dataclass
class CreateUserCommand(Command):
    name: str = ""
    email: str = ""

# Handler function
async def handle_create_user(cmd: CreateUserCommand) -> CommandResult:
    # Create user logic
    user_id = "user-" + cmd.email.split("@")[0]
    return CommandResult(success=True, data={"user_id": user_id})

# Setup bus
bus = CommandBus()
bus.register(CreateUserCommand, handle_create_user)

# Execute
result = await bus.execute(CreateUserCommand(name="Eve", email="eve@example.com"))
print(result.data["user_id"])  # "user-eve"
```

### Handler Classes

```python
from victor.core import CommandHandler

class CreateUserHandler(CommandHandler[CreateUserCommand]):
    def __init__(self, user_repo):
        self.user_repo = user_repo

    async def handle(self, command: CreateUserCommand) -> CommandResult:
        user = User(name=command.name, email=command.email)
        await self.user_repo.add(user)
        return CommandResult(success=True, data={"user_id": user.id})

# Register
bus.register(CreateUserCommand, CreateUserHandler(user_repo))
```

### Decorator Registration

```python
from victor.core import command_handler, query_handler

@command_handler(CreateUserCommand)
async def create_user(cmd: CreateUserCommand) -> CommandResult:
    return CommandResult(success=True)

@query_handler(GetUserQuery)
async def get_user(query: GetUserQuery) -> QueryResult:
    return QueryResult(success=True, data={"name": "Alice"})

# Auto-registered handlers
from victor.core import get_registered_command_handlers
handlers = get_registered_command_handlers()
```

### Queries

```python
from victor.core import Query, QueryResult, QueryBus

@dataclass
class GetUserByIdQuery(Query[dict]):
    user_id: str = ""

async def handle_get_user(query: GetUserByIdQuery) -> QueryResult[dict]:
    # Fetch user logic
    user_data = {"id": query.user_id, "name": "Frank"}
    return QueryResult(success=True, data=user_data)

query_bus = QueryBus()
query_bus.register(GetUserByIdQuery, handle_get_user)

result = await query_bus.execute(GetUserByIdQuery(user_id="user-1"))
print(result.data["name"])  # "Frank"
```

### Mediator

Unified dispatch for both commands and queries:

```python
from victor.core import Mediator, create_mediator

mediator = create_mediator()

# Register handlers
mediator.register_command(CreateUserCommand, handle_create_user)
mediator.register_query(GetUserByIdQuery, handle_get_user)

# Send either type
cmd_result = await mediator.send(CreateUserCommand(name="Grace", email="grace@ex.com"))
query_result = await mediator.send(GetUserByIdQuery(user_id="user-1"))
```

### Middleware

Add cross-cutting concerns:

```python
from victor.core import (
    CommandBus,
    LoggingCommandMiddleware,
    ValidationCommandMiddleware,
    RetryCommandMiddleware,
)

bus = CommandBus()

# Chain middleware (executed in order)
bus.use(LoggingCommandMiddleware())
bus.use(ValidationCommandMiddleware())
bus.use(RetryCommandMiddleware(max_retries=3))

bus.register(CreateUserCommand, handle_create_user)
```

### Query Caching

```python
from victor.core import QueryBus, CachingQueryMiddleware

query_bus = QueryBus()
query_bus.use(CachingQueryMiddleware(ttl_seconds=60))
query_bus.register(GetUserByIdQuery, handle_get_user)

# First call hits handler
result1 = await query_bus.execute(GetUserByIdQuery(user_id="1"))

# Second call returns cached result
result2 = await query_bus.execute(GetUserByIdQuery(user_id="1"))
```

### Read Models

Optimized views for queries:

```python
from victor.core import InMemoryReadModel

class UserListReadModel(InMemoryReadModel):
    """Read model for user listings."""

    async def get_all_active(self):
        return [u for u in self._data.values() if u.get("active")]

    async def get_by_email_domain(self, domain: str):
        return [
            u for u in self._data.values()
            if u.get("email", "").endswith(f"@{domain}")
        ]

read_model = UserListReadModel()
await read_model.add("user-1", {"name": "Alice", "email": "a@corp.com", "active": True})

active = await read_model.get_all_active()
corp_users = await read_model.get_by_email_domain("corp.com")
```

---

## Integration Patterns

### Event Sourcing + CQRS

```python
# Commands modify aggregates
@command_handler(CreateTaskCommand)
async def create_task(cmd: CreateTaskCommand) -> CommandResult:
    task = TaskAggregate()
    task.create(cmd.task_id, cmd.title)

    for event in task.pending_events:
        await event_store.append(task.id, event)
        await dispatcher.dispatch(event)  # Update projections

    task.mark_events_committed()
    return CommandResult(success=True, data={"task_id": task.id})

# Queries use projections
@query_handler(GetTasksQuery)
async def get_tasks(query: GetTasksQuery) -> QueryResult:
    return QueryResult(success=True, data=projection.tasks)
```

### Repository + Unit of Work

```python
async with transactional((user_repo, User), (order_repo, Order)) as uow:
    # Business logic
    user = User(name="Helen")
    order = Order(user_id=user.id, total=100.0)

    uow.register_new(user)
    uow.register_new(order)
    # Auto-commit on success
```

### Full Stack Example

```python
from victor.core import (
    Event, InMemoryEventStore, Aggregate, EventDispatcher, Projection,
    Entity, InMemoryRepository, UnitOfWork,
    Command, CommandResult, Query, QueryResult, Mediator,
)

# Domain Events
@dataclass
class AccountCreatedEvent(Event):
    account_id: str = ""
    owner: str = ""

@dataclass
class MoneyDepositedEvent(Event):
    account_id: str = ""
    amount: float = 0.0

# Aggregate
class BankAccount(Aggregate):
    def __init__(self):
        super().__init__()
        self.owner = ""
        self.balance = 0.0

    def create(self, account_id: str, owner: str):
        self.apply(AccountCreatedEvent(account_id=account_id, owner=owner))

    def deposit(self, amount: float):
        self.apply(MoneyDepositedEvent(account_id=self.id, amount=amount))

    def _apply_AccountCreatedEvent(self, e):
        self.id = e.account_id
        self.owner = e.owner

    def _apply_MoneyDepositedEvent(self, e):
        self.balance += e.amount

# Projection
class AccountBalanceProjection(Projection):
    def __init__(self):
        self.balances = {}

    async def handle_AccountCreatedEvent(self, e):
        self.balances[e.account_id] = 0.0

    async def handle_MoneyDepositedEvent(self, e):
        self.balances[e.account_id] = self.balances.get(e.account_id, 0) + e.amount

# Commands & Queries
@dataclass
class DepositCommand(Command):
    account_id: str = ""
    amount: float = 0.0

@dataclass
class GetBalanceQuery(Query[float]):
    account_id: str = ""

# Setup
store = InMemoryEventStore()
dispatcher = EventDispatcher()
projection = AccountBalanceProjection()
dispatcher.register(projection)
mediator = Mediator()

async def handle_deposit(cmd: DepositCommand) -> CommandResult:
    events = await store.get_events(cmd.account_id)
    account = BankAccount()
    account.load_from_history(events)

    account.deposit(cmd.amount)

    for event in account.pending_events:
        await store.append(account.id, event)
        await dispatcher.dispatch(event)

    account.mark_events_committed()
    return CommandResult(success=True)

async def handle_get_balance(query: GetBalanceQuery) -> QueryResult[float]:
    balance = projection.balances.get(query.account_id, 0.0)
    return QueryResult(success=True, data=balance)

mediator.register_command(DepositCommand, handle_deposit)
mediator.register_query(GetBalanceQuery, handle_get_balance)

# Use
await mediator.send(DepositCommand(account_id="acc-1", amount=100.0))
result = await mediator.send(GetBalanceQuery(account_id="acc-1"))
print(result.data)  # 100.0
```

---

## API Reference

### Event Sourcing

| Class | Description |
|-------|-------------|
| `Event` | Base event class with id, timestamp, aggregate_id |
| `EventStore` | Abstract event store interface |
| `InMemoryEventStore` | In-memory event storage |
| `SQLiteEventStore` | SQLite-backed event storage |
| `Aggregate` | Base aggregate with event application |
| `EventDispatcher` | Dispatch events to projections |
| `Projection` | Base projection for read models |

### Repository

| Class | Description |
|-------|-------------|
| `Entity` | Base entity with id, created_at, updated_at |
| `Repository` | Abstract repository interface |
| `InMemoryRepository` | In-memory repository |
| `SQLiteRepository` | SQLite-backed repository |
| `CachedRepository` | Caching decorator for repositories |
| `BaseSpecification` | Base for query specifications |
| `AttributeSpecification` | Match entity attribute |
| `AndSpecification` | Combine specs with AND |
| `OrSpecification` | Combine specs with OR |
| `NotSpecification` | Negate a specification |

### Unit of Work

| Class | Description |
|-------|-------------|
| `UnitOfWork` | Track and commit changes atomically |
| `SQLiteUnitOfWork` | SQLite transaction support |
| `CompositeUnitOfWork` | Coordinate multiple units |
| `EntityState` | Enum: CLEAN, NEW, MODIFIED, DELETED |
| `TrackedEntity` | Entity wrapper with state tracking |
| `transactional` | Context manager for transactions |

### CQRS

| Class | Description |
|-------|-------------|
| `Command` | Base command with id, timestamp |
| `CommandResult` | Command execution result |
| `CommandHandler` | Abstract command handler |
| `CommandBus` | Route commands to handlers |
| `Query` | Base query with generic result type |
| `QueryResult` | Query execution result |
| `QueryHandler` | Abstract query handler |
| `QueryBus` | Route queries to handlers |
| `Mediator` | Unified command/query dispatch |
| `ReadModel` | Abstract read model interface |
| `InMemoryReadModel` | In-memory read model |

### Middleware

| Class | Description |
|-------|-------------|
| `CommandMiddleware` | Base command middleware |
| `LoggingCommandMiddleware` | Log command execution |
| `ValidationCommandMiddleware` | Validate commands |
| `RetryCommandMiddleware` | Retry failed commands |
| `QueryMiddleware` | Base query middleware |
| `LoggingQueryMiddleware` | Log query execution |
| `CachingQueryMiddleware` | Cache query results |

---

## Related Documentation

- [STATE_MACHINE.md](./STATE_MACHINE.md) - Conversation state management
- [VERTICALS.md](./VERTICALS.md) - Domain-specific assistants
- [ARCHITECTURE_DEEP_DIVE.md](./ARCHITECTURE_DEEP_DIVE.md) - Overall architecture
