# Testing Guide - Part 3

**Part 3 of 3:** Test Scenarios, Performance, and Best Practices

---

## Navigation

- [Part 1: Philosophy & Mocks](part-1-philosophy-mocks.md)
- [Part 2: Async & Coverage](part-2-async-coverage.md)
- **[Part 3: Scenarios & Best Practices](#)** (Current)
- [**Complete Guide**](../TESTING_GUIDE.md)

---

### Testing Exception Propagation

```python
@pytest.mark.asyncio
async def test_propagates_provider_errors(coordinator: ChatCoordinator):
    """Test coordinator propagates provider errors correctly."""
    # Arrange
    coordinator._orchestrator.provider.chat = AsyncMock(
        side_effect=ProviderRateLimitError("Rate limit exceeded")
    )

    # Act & Assert
    with pytest.raises(ProviderRateLimitError):
        await coordinator.chat("Test message")
```

### Testing Error Recovery

```python
@pytest.mark.asyncio
async def test_recovers_from_transient_errors(coordinator: ChatCoordinator):
    """Test coordinator recovers from transient errors."""
    # Arrange
    call_count = 0

    async def flaky_provider(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ProviderConnectionError("Connection failed")
        return CompletionResponse(content="Success", role="assistant")

    coordinator._orchestrator.provider.chat = AsyncMock(side_effect=flaky_provider)

    # Act
    response = await coordinator.chat("Test message")

    # Assert
    assert response.content == "Success"
    assert call_count == 3
```

### Testing Fallback Behavior

```python
@pytest.mark.asyncio
async def test_falls_back_to_secondary_provider(coordinator: ChatCoordinator):
    """Test coordinator falls back to secondary provider."""
    # Arrange
    primary_provider = AsyncMock(
        side_effect=ProviderError("Primary failed")
    )
    secondary_provider = AsyncMock(
        return_value=CompletionResponse(content="Fallback response")
    )

    coordinator._primary_provider = primary_provider
    coordinator._secondary_provider = secondary_provider

    # Act
    response = await coordinator.chat_with_fallback("Test")

    # Assert
    assert response.content == "Fallback response"
    primary_provider.assert_called_once()
    secondary_provider.assert_called_once()
```

### Testing Timeout Handling

```python
@pytest.mark.asyncio
async def test_handles_timeout(coordinator: ToolCoordinator):
    """Test coordinator handles operation timeouts."""
    # Arrange
    async def slow_tool(*args, **kwargs):
        await asyncio.sleep(10)
        return ToolResult(success=True)

    coordinator._pipeline._execute_single_tool = AsyncMock(side_effect=slow_tool)

    # Act & Assert
    with pytest.raises(asyncio.TimeoutError):
        async with asyncio.timeout(1.0):
            await coordinator.execute_tool("slow_tool", {})
```

## Performance Testing

### Testing Execution Time

```python
@pytest.mark.asyncio
async def test_execution_time(coordinator: ToolCoordinator):
    """Test coordinator executes within acceptable time."""
    # Arrange
    start_time = time.time()

    # Act
    await coordinator.execute_tool("fast_tool", {})

    # Assert
    elapsed = time.time() - start_time
    assert elapsed < 1.0, f"Execution took {elapsed}s, expected < 1.0s"
```

### Testing Memory Usage

```python
@pytest.mark.asyncio
async def test_memory_usage(coordinator: ContextCoordinator):
    """Test coordinator manages memory efficiently."""
    # Arrange
    import tracemalloc
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()

    # Act
    for i in range(1000):
        await coordinator.compact_context({"messages": []})

    # Assert
    snapshot2 = tracemalloc.take_snapshot()
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    assert len(top_stats) < 100, "Memory usage grew too much"
    tracemalloc.stop()
```

### Testing Concurrency Performance

```python
@pytest.mark.asyncio
async def test_concurrent_performance(coordinator: ToolCoordinator):
    """Test coordinator handles concurrent operations efficiently."""
    # Arrange
    start_time = time.time()

    # Act
    tasks = [
        coordinator.execute_tool("tool", {"id": i})
        for i in range(100)
    ]
    results = await asyncio.gather(*tasks)

    # Assert
    elapsed = time.time() - start_time
    assert len(results) == 100
    assert elapsed < 5.0, f"100 operations took {elapsed}s"
```

## Integration Testing

### Testing Coordinator Interactions

```python
@pytest.mark.asyncio
async def test_tool_and_chat_coordination(chat_coord: ChatCoordinator,
                                          tool_coord: ToolCoordinator):
    """Test ChatCoordinator and ToolCoordinator work together."""
    # Arrange
    chat_coord._orchestrator.provider.chat = AsyncMock(
        return_value=CompletionResponse(
            content="I'll use a tool",
            tool_calls=[ToolCall(
                id="call_1",
                name="read_file",
                arguments={"path": "test.txt"}
            )]
        )
    )
    tool_coord._pipeline.execute_tool_calls = AsyncMock(
        return_value=[
            ToolResult(tool_name="read_file", success=True, output="Content")
        ]
    )

    # Act
    response = await chat_coord.chat("Read test.txt")

    # Assert
    assert response.content is not None
    tool_coord._pipeline.execute_tool_calls.assert_called_once()
```

### Testing End-to-End Workflows

```python
@pytest.mark.asyncio
async def test_end_to_end_workflow(orchestrator: Mock):
    """Test complete workflow through multiple coordinators."""
    # Arrange
    chat_coord = ChatCoordinator(orchestrator)
    tool_coord = ToolCoordinator(
        orchestrator._tool_pipeline,
        orchestrator._tool_registry
    )
    context_coord = ContextCoordinator(orchestrator)

    # Act
    response = await chat_coord.chat("Process this file")

    # Assert
    assert response is not None
    assert context_coord.context_size < 10000  # Context compacted
```

### Testing with Real Dependencies

```python
@pytest.mark.integration
@pytest.mark.asyncio
@requires_ollama()  # Skip if Ollama not available
async def test_with_real_provider():
    """Test coordinator with real LLM provider."""
    # Arrange
    from victor.providers.ollama import OllamaProvider
    provider = OllamaProvider(model="qwen2.5-coder:7b")
    orch = Mock()
    orch.provider = provider
    orch.conversation = Mock()

    coordinator = ChatCoordinator(orch)

    # Act
    response = await coordinator.chat("Say 'Hello, World!'")

    # Assert
    assert "Hello" in response.content
```

## Best Practices Summary

1. **Use Fixtures**: Create reusable fixtures for common setup
2. **Mock External Dependencies**: Isolate coordinators from external services
3. **Test Async Properly**: Always use `@pytest.mark.asyncio` for async tests
4. **Descriptive Names**: Make test names self-documenting
5. **AAA Pattern**: Structure tests with Arrange-Act-Assert
6. **Test Public APIs**: Focus on interface, not implementation
7. **Error Handling**: Test both success and failure paths
8. **Edge Cases**: Test boundaries and unusual inputs
9. **State Management**: Verify state transitions
10. **Performance**: Ensure acceptable performance characteristics

## Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/)
- [Python unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
- [Victor Architecture Documentation](../architecture/overview.md)
- [Testing Quick Start Guide](TESTING_QUICK_START.md)

---

## See Also

- [Documentation Home](../../README.md)


**Last Updated:** February 01, 2026
**Reading Time:** 6 minutes
