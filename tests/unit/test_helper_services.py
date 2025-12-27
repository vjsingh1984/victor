"""Test Phase 2C helper/adapter service resolution from DI container."""

import pytest
from victor.agent.service_provider import OrchestratorServiceProvider
from victor.core.container import ServiceContainer
from victor.config.settings import Settings
from victor.agent.protocols import (
    SystemPromptBuilderProtocol,
    ToolSelectorProtocol,
    ToolExecutorProtocol,
    ToolOutputFormatterProtocol,
    ParallelExecutorProtocol,
    ResponseCompleterProtocol,
    StreamingHandlerProtocol,
)


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings(
        provider="anthropic",
        model="claude-opus-4",
        api_key="test-key",
    )


@pytest.fixture
def container(settings):
    """Create DI container with registered services."""
    container = ServiceContainer()
    provider = OrchestratorServiceProvider(settings)
    provider.register_singleton_services(container)
    provider.register_scoped_services(container)
    return container


def test_system_prompt_builder_resolution(container):
    """Test SystemPromptBuilder resolves from DI container."""
    service = container.get(SystemPromptBuilderProtocol)
    assert service is not None
    print(f"✓ SystemPromptBuilder resolved: {type(service).__name__}")


def test_tool_selector_resolution(container):
    """Test ToolSelector resolves from DI container."""
    service = container.get(ToolSelectorProtocol)
    assert service is not None
    print(f"✓ ToolSelector resolved: {type(service).__name__}")


def test_tool_executor_resolution(container):
    """Test ToolExecutor resolves from DI container."""
    service = container.get(ToolExecutorProtocol)
    assert service is not None
    print(f"✓ ToolExecutor resolved: {type(service).__name__}")


def test_tool_output_formatter_resolution(container):
    """Test ToolOutputFormatter resolves from DI container."""
    service = container.get(ToolOutputFormatterProtocol)
    assert service is not None
    print(f"✓ ToolOutputFormatter resolved: {type(service).__name__}")


def test_parallel_executor_resolution(container):
    """Test ParallelExecutor resolves from DI container."""
    service = container.get(ParallelExecutorProtocol)
    assert service is not None
    print(f"✓ ParallelExecutor resolved: {type(service).__name__}")


def test_response_completer_resolution(container):
    """Test ResponseCompleter resolves from DI container."""
    service = container.get(ResponseCompleterProtocol)
    assert service is not None
    print(f"✓ ResponseCompleter resolved: {type(service).__name__}")


def test_streaming_handler_resolution(container):
    """Test StreamingHandler resolves from DI container (SCOPED)."""
    # For scoped services, need to create a scope
    with container.create_scope() as scope:
        service = scope.get(StreamingHandlerProtocol)
        assert service is not None
        print(f"✓ StreamingHandler resolved: {type(service).__name__} (scoped)")


def test_all_helper_services_resolve(container):
    """Verify all 7 Phase 2C helper/adapter services resolve correctly."""
    protocols = [
        (SystemPromptBuilderProtocol, "SystemPromptBuilder", False),
        (ToolSelectorProtocol, "ToolSelector", False),
        (ToolExecutorProtocol, "ToolExecutor", False),
        (ToolOutputFormatterProtocol, "ToolOutputFormatter", False),
        (ParallelExecutorProtocol, "ParallelExecutor", False),
        (ResponseCompleterProtocol, "ResponseCompleter", False),
        (StreamingHandlerProtocol, "StreamingHandler", True),  # SCOPED
    ]

    print("\n=== Phase 2C Helper/Adapter Services Resolution ===")
    for protocol, name, is_scoped in protocols:
        if is_scoped:
            with container.create_scope() as scope:
                service = scope.get(protocol)
                assert service is not None, f"{name} failed to resolve"
                print(f"✓ {name:30s} resolved: {type(service).__name__} (scoped)")
        else:
            service = container.get(protocol)
            assert service is not None, f"{name} failed to resolve"
            print(f"✓ {name:30s} resolved: {type(service).__name__}")
    print("=" * 55)
    print("✓ All 7 Phase 2C services resolved successfully\n")
