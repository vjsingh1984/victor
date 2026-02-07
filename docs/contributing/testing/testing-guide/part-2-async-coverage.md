# Testing Guide - Part 2

**Part 2 of 3:** Async Testing, Coverage, and Error Handling

---

## Navigation

- [Part 1: Philosophy & Mocks](part-1-philosophy-mocks.md)
- **[Part 2: Async & Coverage](#)** (Current)
- [Part 3: Scenarios & Best Practices](part-3-scenarios-best-practices.md)
- [**Complete Guide**](../TESTING_GUIDE.md)

---

### Basic Async Tests

All async coordinator methods must be tested with `@pytest.mark.asyncio`:

```python
@pytest.mark.asyncio
async def test_async_method_success(coordinator: ChatCoordinator):
    """Test async method handles successful execution."""
    # Arrange
    user_message = "Hello"

    # Act
    response = await coordinator.chat(user_message)

    # Assert
    assert response.content == "Expected response"
```

### Testing Async Generators

For streaming methods that return async generators:

```python
@pytest.mark.asyncio
async def test_streaming_chat(coordinator: ChatCoordinator):
    """Test streaming chat yields chunks correctly."""
    # Arrange
    user_message = "Stream this"

    # Act
    chunks = []
    async for chunk in coordinator.stream_chat(user_message):
        chunks.append(chunk)

    # Assert
    assert len(chunks) > 0
    assert chunks[0].content == "First chunk"
```

### Mocking Async Methods

Use `AsyncMock` for async methods:

```python
@pytest.fixture
def mock_provider(self) -> Mock:
    """Create mock provider with async methods."""
    provider = Mock()
    provider.chat = AsyncMock(
        return_value=CompletionResponse(
            content="Response",
            role="assistant"
        )
    )
    provider.stream = self._create_async_stream()
    return provider

def _create_async_stream(self):
    """Create async stream generator."""
    async def stream():
        yield StreamChunk(content="Chunk 1")
        yield StreamChunk(content="Chunk 2")
    return stream
```

### Testing Async Error Handling

```python
@pytest.mark.asyncio
async def test_async_error_handling(coordinator: ChatCoordinator):
    """Test async method handles errors correctly."""
    # Arrange
    coordinator._orchestrator.provider.chat = AsyncMock(
        side_effect=ProviderConnectionError("Connection failed")
    )

    # Act & Assert
    with pytest.raises(ProviderConnectionError):
        await coordinator.chat("Test message")
```

### Testing Async Context Managers

```python
@pytest.mark.asyncio
async def test_async_context_manager(coordinator: SessionCoordinator):
    """Test async context manager behavior."""
    # Act
    async with coordinator.create_session() as session:
        # Assert within context
        assert session is not None
        assert session.active is True

    # Assert after context exit
    assert session.active is False
```

### Testing Concurrent Execution

```python
@pytest.mark.asyncio
async def test_concurrent_execution(coordinator: ToolCoordinator):
    """Test coordinator handles concurrent execution correctly."""
    # Arrange
    tasks = [
        coordinator.execute_tool("read_file", {"path": f"file{i}.txt"})
        for i in range(10)
    ]

    # Act
    results = await asyncio.gather(*tasks)

    # Assert
    assert len(results) == 10
    assert all(r.success for r in results)
```

## Coverage Targets and Guidelines

### Coverage Targets by Coordinator Type

| Coordinator Type | Target Coverage | Priority |
|-----------------|----------------|----------|
| ChatCoordinator | >75% | CRITICAL |
| ToolCoordinator | >80% | CRITICAL |
| ContextCoordinator | >75% | HIGH |
| AnalyticsCoordinator | >70% | MEDIUM |
| PromptCoordinator | >75% | HIGH |
| SessionCoordinator | >75% | HIGH |
| ProviderCoordinator | >75% | HIGH |
| ModeCoordinator | >70% | MEDIUM |
| ToolSelectionCoordinator | >75% | HIGH |
| CheckpointCoordinator | >70% | MEDIUM |
| EvaluationCoordinator | >65% | MEDIUM |
| MetricsCoordinator | >70% | MEDIUM |
| YAMLWorkflowCoordinator | >75% | HIGH |
| GraphExecutionCoordinator | >75% | HIGH |
| HITLCoordinator | >70% | MEDIUM |
| CacheCoordinator | >80% | HIGH |

### Measuring Coverage

Run coverage for specific coordinators:

```bash
# Coverage for single coordinator
pytest --cov=victor.agent.coordinators.chat_coordinator \
       --cov-report=term-missing \
       tests/unit/agent/coordinators/test_chat_coordinator.py

# Coverage for all coordinators
pytest --cov=victor.agent.coordinators \
       --cov-report=html \
       --cov-report=term-missing \
       tests/unit/agent/coordinators/

# HTML report
open htmlcov/index.html
```

### Coverage Goals

- **Statement Coverage**: Every line of code executed
- **Branch Coverage**: Every conditional branch tested
- **Path Coverage**: Every execution path tested (where feasible)

### What to Test

**DO Test:**
- Public API methods
- Error conditions and edge cases
- Integration points between coordinators
- State management and transitions
- Async behavior and concurrency
- Resource cleanup and lifecycle

**DON'T Test:**
- Private methods (test via public API)
- Implementation details
- External dependencies (use mocks)
- Trivial getters/setters
- Python/standard library functionality

## Common Test Scenarios

### Initialization Tests

```python
class TestCoordinatorInitialization:
    """Test coordinator initialization."""

    def test_initialization_with_defaults(self):
        """Test coordinator initializes with default config."""
        coordinator = CoordinatorName()
        assert coordinator._config.default_value == "expected"

    def test_initialization_with_custom_config(self):
        """Test coordinator accepts custom configuration."""
        config = CoordinatorConfig(custom_value="test")
        coordinator = CoordinatorName(config=config)
        assert coordinator._config.custom_value == "test"

    def test_initialization_validates_dependencies(self):
        """Test coordinator validates required dependencies."""
        with pytest.raises(ValueError, match="required dependency"):
            CoordinatorName(dependency=None)
```

### Success Path Tests

```python
@pytest.mark.asyncio
async def test_successful_execution(coordinator: CoordinatorName):
    """Test coordinator executes successfully with valid input."""
    # Arrange
    input_data = {"key": "value"}

    # Act
    result = await coordinator.execute(input_data)

    # Assert
    assert result.success is True
    assert result.output is not None
```

### Error Handling Tests

```python
@pytest.mark.asyncio
async def test_handles_invalid_input(coordinator: CoordinatorName):
    """Test coordinator validates input correctly."""
    # Arrange
    invalid_input = {"invalid": "data"}

    # Act & Assert
    with pytest.raises(ValidationError, match="Invalid input"):
        await coordinator.execute(invalid_input)

@pytest.mark.asyncio
async def test_handles_dependency_failure(coordinator: CoordinatorName):
    """Test coordinator handles dependency failures gracefully."""
    # Arrange
    coordinator._dependency.process = AsyncMock(
        side_effect=DependencyError("Service unavailable")
    )

    # Act & Assert
    with pytest.raises(DependencyError):
        await coordinator.execute({"key": "value"})
```

### Edge Case Tests

```python
@pytest.mark.asyncio
async def test_handles_empty_input(coordinator: CoordinatorName):
    """Test coordinator handles empty input gracefully."""
    # Act
    result = await coordinator.execute({})

    # Assert
    assert result is not None

@pytest.mark.asyncio
async def test_handles_large_input(coordinator: CoordinatorName):
    """Test coordinator handles large input efficiently."""
    # Arrange
    large_input = {"data": "x" * 1000000}

    # Act
    result = await coordinator.execute(large_input)

    # Assert
    assert result is not None

@pytest.mark.asyncio
async def test_handles_concurrent_requests(coordinator: CoordinatorName):
    """Test coordinator handles concurrent requests correctly."""
    # Arrange
    tasks = [coordinator.execute({"id": i}) for i in range(100)]

    # Act
    results = await asyncio.gather(*tasks)

    # Assert
    assert len(results) == 100
```

### State Management Tests

```python
@pytest.mark.asyncio
async def test_state_transitions(coordinator: CoordinatorName):
    """Test coordinator state management."""
    # Assert initial state
    assert coordinator.state == State.INITIAL

    # Act
    await coordinator.start()

    # Assert state changed
    assert coordinator.state == State.RUNNING

    # Act
    await coordinator.stop()

    # Assert final state
    assert coordinator.state == State.STOPPED
```

## Error Handling Testing
