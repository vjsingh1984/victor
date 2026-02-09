# Step Handler Examples - Part 3

**Part 3 of 3:** Advanced Patterns and Testing Examples

---

## Navigation

- [Part 1: Basic through Workflow](part-1-basic-middleware-workflow.md)
- [Part 2: Safety & Config](part-2-safety-config.md)
- **[Part 3: Advanced & Testing](#)** (Current)
- [**Complete Guide**](../step_handler_examples.md)

---

## Advanced Patterns

### Example 15: Handler Composition

Compose multiple validation steps:

```python
class CompositeValidationHandler(BaseStepHandler):
    """Compose multiple validation steps."""

    @property
    def name(self) -> str:
        return "composite_validation"

    @property
    def order(self) -> int:
        return 8  # Early, after capability_config (5)

    def __init__(self):
        super().__init__()
        self._validators = [
            self._validate_tools,
            self._validate_prompts,
            self._validate_config,
        ]

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Run all validation steps."""
        passed = 0
        failed = 0

        for validator in self._validators:
            try:
                validator(orchestrator, vertical, context, result)
                passed += 1
            except ValidationError as e:
                result.add_error(f"Validation failed: {e}")
                failed += 1

        result.add_info(f"Validation: {passed} passed, {failed} failed")

    def _validate_tools(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Validate tools."""
        tools = vertical.get_tools()
        if not tools:
            raise ValidationError("No tools configured")

    def _validate_prompts(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Validate system prompt."""
        prompt = vertical.get_system_prompt()
        if len(prompt) < 10:
            raise ValidationError("System prompt too short")

    def _validate_config(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Validate configuration."""
        config = vertical.get_config()
        if not config:
            raise ValidationError("No configuration")


class ValidationError(Exception):
    """Validation error."""
    pass
```text

### Example 16: Async Handler

Load resources asynchronously:

```python
class AsyncResourceLoaderHandler(BaseStepHandler):
    """Load resources asynchronously."""

    @property
    def name(self) -> str:
        return "async_resource_loader"

    @property
    def order(self) -> int:
        return 9  # Early, before tools (10)

    async def apply_async(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
        strict_mode: bool = False,
    ) -> None:
        """Apply resources asynchronously."""
        # Load resources asynchronously
        resources = await self._load_resources_async(vertical)

        # Apply to context
        context.apply_resources(resources)

        result.add_info(f"Loaded {len(resources)} resources asynchronously")

    async def _load_resources_async(
        self,
        vertical: Type[VerticalBase],
    ) -> List[Any]:
        """Load resources asynchronously."""
        # Simulate async loading
        import asyncio

        await asyncio.sleep(0.1)  # Simulate I/O

        # Return loaded resources
        return [{"name": "resource1"}, {"name": "resource2"}]
```

### Example 17: Conditional Handler with Retry

Retry operations on failure:

```python
class RetryableOperationHandler(BaseStepHandler):
    """Retry operations on failure."""

    @property
    def name(self) -> str:
        return "retryable_operation"

    @property
    def order(self) -> int:
        return 70  # After framework (60)

    def __init__(self, max_retries: int = 3):
        super().__init__()
        self._max_retries = max_retries

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Perform operation with retry."""
        for attempt in range(self._max_retries):
            try:
                # Attempt operation
                self._perform_operation(orchestrator, vertical, context)
                result.add_info(f"Operation succeeded on attempt {attempt + 1}")
                return
            except Exception as e:
                if attempt < self._max_retries - 1:
                    result.add_warning(f"Attempt {attempt + 1} failed: {e}")
                    continue
                else:
                    result.add_error(f"Operation failed after {self._max_retries} attempts")
                    raise

    def _perform_operation(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
    ) -> None:
        """Perform the actual operation."""
        # Simulate flaky operation
        import random

        if random.random() < 0.7:  # 70% failure rate
            raise RuntimeError("Operation failed")

        # Success
        context.apply_operation_result({"status": "success"})
```text

---

## Testing Examples

### Example 18: Testable Handler

Write a handler designed for easy testing:

```python
class TestableHandler(BaseStepHandler):
    """Handler designed for easy testing."""

    @property
    def name(self) -> str:
        return "testable"

    @property
    def order(self) -> int:
        return 25

    def __init__(self, validator=None, processor=None):
        """Initialize with injectable dependencies."""
        super().__init__()
        self._validator = validator or self._default_validator
        self._processor = processor or self._default_processor

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply with testable components."""
        # Validate using injected validator
        if not self._validator(vertical):
            result.add_error("Validation failed")
            return

        # Process using injected processor
        processed = self._processor(vertical.get_data())

        # Apply result
        context.apply_data(processed)
        result.add_info("Processing successful")

    def _default_validator(self, vertical: Type[VerticalBase]) -> bool:
        """Default validation logic."""
        return len(vertical.get_tools()) > 0

    def _default_processor(self, data: Any) -> Any:
        """Default processing logic."""
        return data


# Test with mocks
def test_with_mocks():
    """Test handler with mocked dependencies."""
    # Create mock validator
    mock_validator = MagicMock(return_value=True)

    # Create mock processor
    mock_processor = MagicMock(return_value="processed")

    # Create handler with mocks
    handler = TestableHandler(
        validator=mock_validator,
        processor=mock_processor,
    )

    # Test
    handler._do_apply(orchestrator, vertical, context, result)

    # Verify
    mock_validator.assert_called_once_with(vertical)
    mock_processor.assert_called_once()
```

### Example 19: Handler with Test Hooks

Include hooks for test-specific behavior:

```python
class HandlerWithTestHooks(BaseStepHandler):
    """Handler with test-specific hooks."""

    @property
    def name(self) -> str:
        return "test_hooks"

    @property
    def order(self) -> int:
        return 25

    def __init__(self, test_mode: bool = False):
        """Initialize with optional test mode."""
        super().__init__()
        self._test_mode = test_mode

    def _do_apply(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Apply with test-specific behavior."""
        if self._test_mode:
            # Test mode behavior
            self._apply_test_behavior(orchestrator, context, result)
        else:
            # Production behavior
            self._apply_production_behavior(orchestrator, vertical, context, result)

    def _apply_test_behavior(
        self,
        orchestrator: Any,
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Test-specific behavior."""
        # Use mock data in tests
        context.apply_data({"test": True})
        result.add_info("Test mode applied")

    def _apply_production_behavior(
        self,
        orchestrator: Any,
        vertical: Type[VerticalBase],
        context: VerticalContext,
        result: IntegrationResult,
    ) -> None:
        """Production behavior."""
        # Real logic in production
        data = vertical.get_data()
        context.apply_data(data)
        result.add_info("Production mode applied")


# Test with test mode enabled
def test_handler_with_test_mode():
    """Test handler in test mode."""
    handler = HandlerWithTestHooks(test_mode=True)

    handler._do_apply(orchestrator, vertical, context, result)

    assert context.data == {"test": True}
    assert "Test mode" in result.info[0]
```text

---

## Summary

These examples demonstrate common step handler patterns:

**Key Patterns:**
1. **Validation**: Validate before applying (Example 4, 11, 15)
2. **Filtering**: Filter based on conditions (Example 5, 6)
3. **Composition**: Compose multiple operations (Example 15)
4. **Async**: Load resources asynchronously (Example 16)
5. **Retry**: Retry on failure (Example 17)
6. **Testability**: Design for testing (Example 18, 19)

**Best Practices:**
- Use clear, descriptive names
- Choose appropriate order values
- Handle errors gracefully
- Provide step details for observability
- Design for testability

**Next Steps:**
- Review [Step Handler Guide](step_handler_guide.md) for concepts
- See [Migration Guide](step_handler_migration.md) for migration patterns
- Check [Quick Reference](step_handler_quick_reference.md) for API details

**Questions?** See main guide or troubleshooting section

---

**Last Updated:** February 01, 2026
**Reading Time:** 2 min
