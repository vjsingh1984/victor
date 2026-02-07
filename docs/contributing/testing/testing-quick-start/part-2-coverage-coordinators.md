# Testing Quick Start - Part 2

**Part 2 of 3:** Coverage and Coordinator-Specific Patterns

---

## Navigation

- [Part 1: Templates & Patterns](part-1-templates-patterns-mocks.md)
- **[Part 2: Coverage & Coordinators](#)** (Current)
- [Part 3: Troubleshooting & Reference](part-3-troubleshooting-reference.md)
- [**Complete Guide**](../TESTING_QUICK_START.md)

---

### Run All Coordinator Tests

```bash
# Run all coordinator tests
pytest tests/unit/agent/coordinators/ -v

# Run with coverage
pytest tests/unit/agent/coordinators/ \
    --cov=victor.agent.coordinators \
    --cov-report=term-missing \
    --cov-report=html

# Run specific coordinator test
pytest tests/unit/agent/coordinators/test_chat_coordinator.py -v

# Run specific test
pytest tests/unit/agent/coordinators/test_chat_coordinator.py::TestChatCoordinatorChat::test_chat_with_tools -v
```

### Coverage for Specific Coordinator

```bash
# ChatCoordinator coverage
pytest --cov=victor.agent.coordinators.chat_coordinator \
       --cov-report=term-missing \
       tests/unit/agent/coordinators/test_chat_coordinator.py

# ToolCoordinator coverage
pytest --cov=victor.agent.coordinators.tool_coordinator \
       --cov-report=term-missing \
       tests/unit/agent/coordinators/test_tool_coordinator.py
```

### Generate HTML Coverage Report

```bash
# Generate detailed HTML report
pytest tests/unit/agent/coordinators/ \
    --cov=victor.agent.coordinators \
    --cov-report=html \
    --cov-report=term

# Open report
open htmlcov/index.html
```

### Run Tests in Parallel

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel (4 workers)
pytest tests/unit/agent/coordinators/ -n 4

# Run with coverage in parallel
pytest tests/unit/agent/coordinators/ \
    -n 4 \
    --cov=victor.agent.coordinators \
    --cov-report=term-missing
```

### Run Tests by Marker

```bash
# Run only unit tests
pytest -m unit tests/unit/agent/coordinators/

# Run only integration tests
pytest -m integration tests/unit/agent/coordinators/

# Run only slow tests
pytest -m slow tests/unit/agent/coordinators/

# Skip slow tests
pytest -m "not slow" tests/unit/agent/coordinators/
```

### Debug Failed Tests

```bash
# Run with verbose output
pytest tests/unit/agent/coordinators/test_chat_coordinator.py -vv

# Run with pdb on failure
pytest tests/unit/agent/coordinators/test_chat_coordinator.py --pdb

# Show local variables on failure
pytest tests/unit/agent/coordinators/test_chat_coordinator.py -l

# Stop on first failure
pytest tests/unit/agent/coordinators/ -x

# Show print statements
pytest tests/unit/agent/coordinators/ -s
```

## Coordinator-Specific Patterns

### ChatCoordinator Patterns

```python
"""ChatCoordinator test patterns."""

@pytest.mark.asyncio
async def test_chat_with_no_tools(coordinator: ChatCoordinator):
    """Test chat without tool calls."""
    # Arrange
    coordinator._orchestrator.provider.chat = AsyncMock(
        return_value=CompletionResponse(
            content="Response",
            role="assistant",
            tool_calls=None
        )
    )

    # Act
    response = await coordinator.chat("Hello")

    # Assert
    assert response.content == "Response"
    assert response.tool_calls is None


@pytest.mark.asyncio
async def test_chat_with_tools(coordinator: ChatCoordinator):
    """Test chat with tool calls."""
    # Arrange
    tool_call = ToolCall(
        id="call_1",
        name="read_file",
        arguments={"path": "test.txt"}
    )

    coordinator._orchestrator.provider.chat = AsyncMock(
        return_value=CompletionResponse(
            content="I'll read the file",
            role="assistant",
            tool_calls=[tool_call]
        )
    )

    coordinator._orchestrator._handle_tool_calls = AsyncMock(
        return_value=[ToolResult(
            tool_name="read_file",
            success=True,
            output="File content"
        )]
    )

    # Act
    response = await coordinator.chat("Read test.txt")

    # Assert
    assert response.content is not None
    coordinator._orchestrator._handle_tool_calls.assert_called_once()
```

### ToolCoordinator Patterns

```python
"""ToolCoordinator test patterns."""

@pytest.mark.asyncio
async def test_execute_single_tool(coordinator: ToolCoordinator):
    """Test single tool execution."""
    # Arrange
    coordinator._pipeline._execute_single_tool = AsyncMock(
        return_value=ToolResult(
            tool_name="read_file",
            success=True,
            output="Content"
        )
    )

    # Act
    result = await coordinator.execute_tool(
        "read_file",
        {"path": "test.txt"}
    )

    # Assert
    assert result.success is True
    assert result.output == "Content"


@pytest.mark.asyncio
async def test_batch_tool_execution(coordinator: ToolCoordinator):
    """Test batch tool execution."""
    # Arrange
    tool_calls = [
        ToolCall(id="1", name="read_file", arguments={"path": "file1.txt"}),
        ToolCall(id="2", name="read_file", arguments={"path": "file2.txt"}),
    ]

    coordinator._pipeline.execute_tool_calls = AsyncMock(
        return_value=[
            ToolResult(tool_name="read_file", success=True, output="Content 1"),
            ToolResult(tool_name="read_file", success=True, output="Content 2"),
        ]
    )

    # Act
    results = await coordinator.execute_batch(tool_calls)

    # Assert
    assert len(results) == 2
    assert all(r.success for r in results)


@pytest.mark.asyncio
async def test_budget_enforcement(coordinator: ToolCoordinator):
    """Test tool budget is enforced."""
    # Arrange
    coordinator._config.default_budget = 5
    coordinator._tools_used = 5

    # Act
    can_execute = await coordinator.can_execute_tool("any_tool")

    # Assert
    assert can_execute is False
```

### ContextCoordinator Patterns

```python
"""ContextCoordinator test patterns."""

@pytest.mark.asyncio
async def test_context_budget_check(coordinator: ContextCoordinator):
    """Test context budget checking."""
    # Arrange
    context = {"token_count": 8000}
    budget = ContextBudget(max_tokens=10000)

    # Act
    exceeds_budget = await coordinator.exceeds_budget(context, budget)

    # Assert
    assert exceeds_budget is False


@pytest.mark.asyncio
async def test_context_compaction(coordinator: ContextCoordinator):
    """Test context compaction."""
    # Arrange
    large_context = {"messages": ["msg"] * 1000, "token_count": 15000}

    coordinator._compactor = AsyncMock(
        return_value=CompactionResult(
            compacted_context={"messages": ["msg"] * 100},
            tokens_saved=14000,
            messages_removed=900,
            strategy_used="truncation"
        )
    )

    # Act
    result = await coordinator.compact(large_context, budget=1000)

    # Assert
    assert result.tokens_saved == 14000
    assert result.messages_removed == 900


@pytest.mark.asyncio
async def test_semantic_compaction(coordinator: ContextCoordinator):
    """Test semantic-aware compaction."""
    # Arrange
    context = {
        "messages": [
            {"role": "system", "content": "Important system message"},
            {"role": "user", "content": "Old message"},
            {"role": "user", "content": "Recent important message"},
        ]
    }

    coordinator._semantic_compactor = AsyncMock(
        return_value=CompactionResult(
            compacted_context={
                "messages": [
                    {"role": "system", "content": "Important system message"},
                    {"role": "user", "content": "Recent important message"},
                ]
            },
            tokens_saved=1000,
            messages_removed=1,
            strategy_used="semantic"
        )
    )

    # Act
    result = await coordinator.compact_semantic(context, budget=2000)

    # Assert
    assert len(result.compacted_context["messages"]) == 2
```

### ProviderCoordinator Patterns

```python
"""ProviderCoordinator test patterns."""

@pytest.mark.asyncio
async def test_switch_provider(coordinator: ProviderCoordinator):
    """Test switching providers."""
    # Arrange
    old_provider = coordinator._current_provider
    new_provider = Mock(model="new-model")

    # Act
    await coordinator.switch_provider(new_provider)

    # Assert
    assert coordinator._current_provider == new_provider
    assert coordinator._current_provider.model == "new-model"


@pytest.mark.asyncio
async def test_fallback_provider(coordinator: ProviderCoordinator):
    """Test fallback to secondary provider."""
    # Arrange
    primary = AsyncMock(side_effect=ProviderError("Failed"))
    secondary = AsyncMock(
        return_value=CompletionResponse(content="Fallback")
    )

    coordinator._primary_provider = primary
    coordinator._secondary_provider = secondary

    # Act
    result = await coordinator.chat_with_fallback("Test")

    # Assert
    assert result.content == "Fallback"
    primary.assert_called_once()
    secondary.assert_called_once()
```

### AnalyticsCoordinator Patterns

```python
"""AnalyticsCoordinator test patterns."""

@pytest.mark.asyncio
async def test_metrics_collection(coordinator: AnalyticsCoordinator):
    """Test metrics collection."""
    # Arrange
    coordinator._metrics_collector = Mock()

    # Act
    await coordinator.track_event("tool_execution", {
        "tool_name": "read_file",
        "duration": 0.5,
        "success": True
    })

    # Assert
    coordinator._metrics_collector.record.assert_called_once()


@pytest.mark.asyncio
async def test_session_analytics(coordinator: AnalyticsCoordinator):
    """Test session-level analytics."""
    # Act
    analytics = await coordinator.get_session_analytics()

    # Assert
    assert analytics.total_messages == 0
    assert analytics.total_tool_calls == 0
    assert analytics.total_duration > 0
```

### PromptCoordinator Patterns

```python
"""PromptCoordinator test patterns."""

@pytest.mark.asyncio
async def test_prompt_building(coordinator: PromptCoordinator):
    """Test system prompt building."""
    # Arrange
    contributors = [
        Mock(contribute=lambda: "System instruction"),
        Mock(contribute=lambda: "Safety guideline"),
    ]

    # Act
    prompt = await coordinator.build_prompt(contributors)

    # Assert
    assert "System instruction" in prompt
    assert "Safety guideline" in prompt


@pytest.mark.asyncio
async def test_prompt_caching(coordinator: PromptCoordinator):
    """Test prompt caching."""
    # Arrange
    contributors = [Mock(contribute=lambda: "Instruction")]

    # Act - first call
    prompt1 = await coordinator.build_prompt(contributors)

    # Act - second call (should be cached)
    prompt2 = await coordinator.build_prompt(contributors)

    # Assert
    assert prompt1 == prompt2
    # Verify cache was used
```

## Troubleshooting
