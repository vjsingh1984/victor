# Enhanced Dependency Injection Container

An advanced DI container for Victor with automatic dependency resolution through constructor inspection.

## Features

### Auto-Resolution
- **Constructor Injection**: Automatically inject dependencies based on type hints
- **Interface Mapping**: Register implementations as interfaces using `as_interfaces`
- **Factory Functions**: Support for custom factory functions with manual resolution

### Lifecycle Management
- **Singleton**: One instance for the entire application
- **Scoped**: One instance per scope (e.g., per request)
- **Transient**: New instance every time

### Developer Experience
- **Circular Dependency Detection**: Detects and reports circular dependencies at runtime
- **Clear Error Messages**: Helpful error messages with resolution chain
- **Thread-Safe**: Concurrent access to singleton services
- **Lazy Initialization**: Services created only when needed

## Installation

The DI container is part of Victor framework:

```python
from victor.framework.di import DIContainer, ServiceLifetime
```

## Quick Start

```python
from victor.framework.di import DIContainer, ServiceLifetime

# Define services
class Logger:
    def log(self, message: str):
        print(message)

class Database:
    def __init__(self, logger: Logger):
        self.logger = logger

class UserService:
    def __init__(self, db: Database):
        self.db = db

# Create container and register services
container = (
    DIContainer()
    .register(Logger, lifetime=ServiceLifetime.SINGLETON)
    .register(Database, lifetime=ServiceLifetime.SINGLETON)
    .register(UserService, lifetime=ServiceLifetime.TRANSIENT)
)

# Resolve with auto-injected dependencies
user_service = container.get(UserService)
# Logger and Database are automatically injected!
```

## Registration Patterns

### 1. Simple Registration

```python
container.register(Logger)
```

### 2. With Custom Factory

```python
container.register(
    Logger,
    factory=lambda: Logger(prefix="[CUSTOM] ")
)
```

### 3. As Interface

```python
container.register(
    FileLogger,
    as_interfaces=[ILogger]
)
```

### 4. Container-Aware Factory

```python
def create_cache(cont: DIContainer) -> Cache:
    logger = cont.get(ILogger)
    return Cache(logger=logger)

container.register_factory(ICache, create_cache)
```

### 5. Instance Registration

```python
logger = Logger()
container.register_instance(ILogger, logger)
```

## Lifecycle Examples

### Singleton

```python
container.register(Config, lifetime=ServiceLifetime.SINGLETON)

config1 = container.get(Config)
config2 = container.get(Config)
assert config1 is config2  # Same instance
```

### Scoped

```python
container.register(RequestContext, lifetime=ServiceLifetime.SCOPED)

with container.create_scope() as scope1:
    ctx1 = scope1.get(RequestContext)

with container.create_scope() as scope2:
    ctx2 = scope2.get(RequestContext)
    assert ctx2 is not ctx1  # Different instances
```

### Transient

```python
container.register(Repository, lifetime=ServiceLifetime.TRANSIENT)

repo1 = container.get(Repository)
repo2 = container.get(Repository)
assert repo1 is not repo2  # Different instances
```

## Interface Mapping

Register a class as multiple interfaces:

```python
class AdvancedLogger:
    def log(self, msg): pass
    def flush(self): pass

container.register(
    AdvancedLogger,
    as_interfaces=[ILogger, IFlushable]
)

# Can resolve as any registered interface
logger = container.get(ILogger)
flushable = container.get(IFlushable)
assert isinstance(logger, AdvancedLogger)
assert isinstance(flushable, AdvancedLogger)
assert logger is flushable  # Same instance
```

## Migration from victor.core.container

The enhanced DI container is designed to coexist with the existing `ServiceContainer`. Key differences:

### Before (Manual Factory)

```python
# victor/core/container pattern
from victor.core.container import ServiceContainer, ServiceLifetime

container = ServiceContainer()
container.register(
    Cache,
    lambda c: Cache(c.get(Logger)),
    ServiceLifetime.SINGLETON
)
```

### After (Auto-Resolution)

```python
# victor/framework/di pattern
from victor.framework.di import DIContainer, ServiceLifetime

container = DIContainer()
container.register(Logger, lifetime=ServiceLifetime.SINGLETON)
container.register(Cache, lifetime=ServiceLifetime.SINGLETON)
# Dependencies auto-injected!
```

## Migration Benefits

1. **Less Boilerplate**: No manual factory functions needed
2. **Type Safety**: Constructor dependencies are type-hinted
3. **Testability**: Easy to inject mock dependencies
4. **Clearer Graph**: All dependencies visible in constructor
5. **Backward Compatible**: Can coexist with existing code

## Error Handling

### Missing Dependency

```python
container.register(Database)  # Requires Cache
try:
    container.get(Database)
except DIError as e:
    print(e)
    # "Failed to create instance of Database: Service not registered: ICache (required by: Database)"
```

### Circular Dependency

```python
# ServiceA depends on ServiceB, ServiceB depends on ServiceA
container.register(ServiceA)
container.register(ServiceB)

try:
    container.get(ServiceA)
except CircularDependencyError as e:
    print(e)
    # "Circular dependency detected: ServiceA -> ServiceB -> ServiceA"
```

## Best Practices

1. **Use Interfaces**: Depend on abstractions, not concrete implementations
2. **Singleton for Stateless**: Use singleton for stateless services
3. **Scoped for Request**: Use scoped for request-level data
4. **Transient for Stateful**: Use transient for stateful services
5. **Type Hints**: Always type hint constructor dependencies
6. **Avoid Circular**: Design to avoid circular dependencies

## Testing

```python
def test_user_service():
    container = DIContainer()
    container.register(Logger)
    container.register(Database)
    container.register(UserService)

    service = container.get(UserService)
    assert service.db.logger is not None
```

## Thread Safety

The container is thread-safe for singleton resolution:

```python
from concurrent.futures import ThreadPoolExecutor

container = DIContainer()
container.register(Config, lifetime=ServiceLifetime.SINGLETON)

def get_config():
    return container.get(Config)

with ThreadPoolExecutor(max_workers=10) as executor:
    configs = list(executor.map(lambda _: get_config(), range(100)))
    # All configs are the same instance
    assert all(c is configs[0] for c in configs)
```

## Performance Considerations

- **Lazy Initialization**: Services created only when first requested
- **Thread-Safe Locks**: Minimal locking for singleton resolution
- **Factory Caching**: Descriptors cached after first resolution
- **Scope Isolation**: Scoped services don't affect singletons

## Comparison with victor.core.container

| Feature | victor.core.container | victor.framework.di |
|---------|----------------------|-------------------|
| Manual Factory | ✅ | ✅ |
| Auto-Resolution | ❌ | ✅ |
| Constructor Injection | ❌ | ✅ |
| Lifecycle Management | ✅ | ✅ |
| Scoped Services | ✅ | ✅ |
| Circular Detection | ❌ | ✅ |
| Thread Safety | ✅ | ✅ |

## Future Enhancements

Potential improvements for Phase 4:

- [ ] Async constructor injection
- [ ] Generic type resolution
- [ ] Conditional registration
- [ ] Decorator-based registration
- [ ] Lifecycle events (on_resolve, on_dispose)
- [ ] Named registrations
- [ ] Child containers
- [ ] Performance monitoring

## Contributing

When adding new features:

1. Add tests in `tests/unit/framework/di/test_container.py`
2. Update this README with examples
3. Ensure backward compatibility
4. Run `make test` before committing

## License

Apache License 2.0 - See LICENSE file for details
