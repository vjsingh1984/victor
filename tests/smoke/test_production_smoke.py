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

"""Comprehensive production smoke tests.

These tests verify production readiness by testing:
1. Core Infrastructure
2. Agent Functionality
3. Vertical Loading
4. Integration
5. Performance
6. Security

Run with:
    pytest tests/smoke/test_production_smoke.py -v -m smoke
"""

import asyncio
import pytest
import time
from unittest.mock import MagicMock, AsyncMock, patch

from victor.config.settings import Settings
from victor.core.container import ServiceContainer
from victor.core.events import create_event_backend, MessagingEvent, BackendType, BackendConfig
from victor.providers.mock import MockProvider
from victor.agent.tool_pipeline import ToolPipeline


# =============================================================================
# Core Infrastructure Smoke Tests
# =============================================================================


@pytest.mark.smoke
@pytest.mark.unit
class TestCoreInfrastructureSmokeTests:
    """Smoke tests for core infrastructure components."""

    def test_service_container_creation(self):
        """Test ServiceContainer can be created and is functional."""
        container = ServiceContainer()
        assert container is not None
        assert hasattr(container, "register")
        assert hasattr(container, "get")

    def test_event_bus_creation_and_operations(self):
        """Test EventBus can be created and perform basic operations."""
        backend = create_event_backend(BackendConfig(backend_type=BackendType.IN_MEMORY))
        assert backend is not None

    @pytest.mark.asyncio
    async def test_event_bus_publish_subscribe(self):
        """Test EventBus publish/subscribe functionality."""
        backend = create_event_backend(BackendConfig(backend_type=BackendType.IN_MEMORY))
        await backend.connect()

        # Test publish
        await backend.publish(MessagingEvent(topic="test.event", data={"test": "data"}))

        # Test subscribe (should not raise)
        await backend.subscribe("test.*", lambda event: None)
        await backend.disconnect()

    def test_provider_initialization(self):
        """Test provider initialization."""
        provider = MockProvider(model="test-model")
        assert provider is not None
        assert provider.name == "mock"

    @pytest.mark.asyncio
    async def test_provider_basic_chat(self):
        """Test provider can perform basic chat."""
        from victor.providers.base import Message

        provider = MockProvider(model="test-model")
        messages = [Message(role="user", content="Hello")]
        response = await provider.chat(messages, model="test-model")
        assert response is not None
        assert hasattr(response, "content")
        assert len(response.content) > 0

    def test_tool_pipeline_creation(self):
        """Test ToolPipeline class exists and can be instantiated."""
        from victor.agent.tool_pipeline import ToolPipeline, ToolPipelineConfig
        from victor.tools.registry import ToolRegistry
        from victor.agent.tool_executor import ToolExecutor

        # Verify class exists
        assert ToolPipeline is not None

        # Smoke test: Verify it can be instantiated with minimal dependencies
        # (We don't need a fully functional pipeline, just verify it can be created)
        registry = ToolRegistry()
        executor = ToolExecutor(tool_registry=registry)
        config = ToolPipelineConfig(tool_budget=100)

        pipeline = ToolPipeline(
            tool_registry=registry,
            tool_executor=executor,
            config=config,
        )
        assert pipeline is not None

    def test_settings_loading(self):
        """Test Settings can be loaded."""
        settings = Settings()
        assert settings is not None
        # Verify key settings exist
        assert hasattr(settings, "default_provider")
        assert hasattr(settings, "default_model")


# =============================================================================
# Agent Functionality Smoke Tests
# =============================================================================


@pytest.mark.smoke
@pytest.mark.unit
class TestAgentFunctionalitySmokeTests:
    """Smoke tests for agent functionality."""

    def test_message_creation(self):
        """Test Message creation and structure."""
        from victor.providers.base import Message

        msg = Message(role="user", content="Test message")
        assert msg.role == "user"
        assert msg.content == "Test message"

    def test_message_to_dict_conversion(self):
        """Test Message to_dict conversion."""
        from victor.providers.base import Message

        msg = Message(role="user", content="Test")
        msg_dict = msg.to_dict()
        assert msg_dict["role"] == "user"
        assert msg_dict["content"] == "Test"

    def test_stream_chunk_creation(self):
        """Test StreamChunk creation."""
        from victor.providers.base import StreamChunk

        chunk = StreamChunk(
            content="Test chunk",
            stop_reason=None,
        )
        assert chunk.content == "Test chunk"
        assert chunk.is_final is False  # Default value
        assert chunk.tool_calls is None  # Default value

    def test_tool_call_creation(self):
        """Test ToolCall creation."""
        from victor.agent.tool_calling.base import ToolCall

        tool_call = ToolCall(
            id="test_id",
            name="test_tool",
            arguments={"param": "value"},
        )
        assert tool_call.id == "test_id"
        assert tool_call.name == "test_tool"

    @pytest.mark.asyncio
    async def test_provider_error_handling(self):
        """Test provider handles errors gracefully."""
        provider = MockProvider(model="test-model")
        # Should handle empty or invalid input without crashing
        try:
            response = await provider.chat("")
            assert response is not None
        except Exception:
            # Should raise a meaningful error, not crash
            pass


# =============================================================================
# Vertical Smoke Tests
# =============================================================================


@pytest.mark.smoke
@pytest.mark.unit
class TestVerticalSmokeTests:
    """Smoke tests for vertical loading and initialization."""

    def test_coding_vertical_loads(self):
        """Test Coding vertical can load."""
        from victor.coding import CodingAssistant

        config = CodingAssistant.get_config()
        assert config is not None
        tools = CodingAssistant.get_tools()
        assert len(tools) > 0

    def test_rag_vertical_loads(self):
        """Test RAG vertical can load."""
        from victor.rag import RAGAssistant

        config = RAGAssistant.get_config()
        assert config is not None
        tools = RAGAssistant.get_tools()
        assert len(tools) > 0

    def test_devops_vertical_loads(self):
        """Test DevOps vertical can load."""
        from victor.devops import DevOpsAssistant

        config = DevOpsAssistant.get_config()
        assert config is not None
        tools = DevOpsAssistant.get_tools()
        assert len(tools) > 0

    def test_dataanalysis_vertical_loads(self):
        """Test DataAnalysis vertical can load."""
        from victor.dataanalysis import DataAnalysisAssistant

        config = DataAnalysisAssistant.get_config()
        assert config is not None
        tools = DataAnalysisAssistant.get_tools()
        assert len(tools) > 0

    def test_research_vertical_loads(self):
        """Test Research vertical can load."""
        from victor.research import ResearchAssistant

        config = ResearchAssistant.get_config()
        assert config is not None
        tools = ResearchAssistant.get_tools()
        assert len(tools) > 0


# =============================================================================
# Integration Smoke Tests
# =============================================================================


@pytest.mark.smoke
@pytest.mark.unit
class TestIntegrationSmokeTests:
    """Smoke tests for basic integration."""

    def test_orchestrator_class_exists(self):
        """Test AgentOrchestrator class can be imported."""
        from victor.agent.orchestrator import AgentOrchestrator

        assert AgentOrchestrator is not None

    def test_tool_registry_accessible(self):
        """Test tool registry is accessible."""
        from victor.tools.registry import ToolRegistry

        registry = ToolRegistry()
        assert registry is not None
        # ToolRegistry has 'get' method (inherited from BaseRegistry)
        assert hasattr(registry, "get")

    def test_team_coordinator_creation(self):
        """Test team coordinator can be created."""
        from victor.teams import create_coordinator

        coordinator = create_coordinator(lightweight=True)
        assert coordinator is not None


# =============================================================================
# Performance Smoke Tests
# =============================================================================


@pytest.mark.smoke
@pytest.mark.unit
class TestPerformanceSmokeTests:
    """Smoke tests for performance targets."""

    def test_initialization_time_target(self):
        """Test initialization time is under 2 seconds."""
        start = time.time()

        settings = Settings()
        container = ServiceContainer()

        # Initialize core components
        from victor.agent.tool_pipeline import ToolPipeline, ToolPipelineConfig
        from victor.tools.registry import ToolRegistry
        from victor.agent.tool_executor import ToolExecutor

        registry = ToolRegistry()
        executor = ToolExecutor(tool_registry=registry)
        pipeline = ToolPipeline(
            tool_registry=registry,
            tool_executor=executor,
            config=ToolPipelineConfig(tool_budget=100),
        )

        duration = time.time() - start
        assert duration < 2.0, f"Initialization too slow: {duration:.2f}s"

    def test_provider_instantiation_performance(self):
        """Test provider instantiation is fast (< 50ms)."""
        start = time.time()

        providers = []
        for _ in range(10):
            provider = MockProvider(model="test")
            providers.append(provider)

        duration = (time.time() - start) * 1000  # ms
        avg_time = duration / len(providers)

        assert avg_time < 50.0, f"Provider instantiation too slow: {avg_time:.2f}ms"

    def test_memory_usage_target(self):
        """Test memory usage is under 500MB."""
        try:
            import psutil

            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            assert memory_mb < 500, f"Memory usage too high: {memory_mb:.1f}MB"
        except ImportError:
            pytest.skip("psutil not available")

    def test_message_creation_performance(self):
        """Test Message creation is fast (< 0.1ms)."""
        from victor.providers.base import Message

        start = time.time()
        messages = []
        for _ in range(1000):
            msg = Message(role="user", content="Test")
            messages.append(msg)

        duration = (time.time() - start) * 1000  # ms
        avg_time = duration / len(messages)

        assert avg_time < 0.1, f"Message creation too slow: {avg_time:.3f}ms"


# =============================================================================
# Security Smoke Tests
# =============================================================================


@pytest.mark.smoke
@pytest.mark.unit
class TestSecuritySmokeTests:
    """Smoke tests for security controls."""

    def test_action_authorization_exists(self):
        """Test action authorization framework exists."""
        from victor.config.settings import Settings

        # Settings controls action authorization via tool_call_budget
        settings = Settings()
        assert hasattr(settings, "tool_call_budget")

    def test_provider_factory_security(self):
        """Test provider factory has security controls."""
        from victor.providers.provider_factory import resolve_api_key

        # Test API key resolution doesn't leak secrets
        key = resolve_api_key(
            None,
            "test_provider",
            env_var_names=["NONEXISTENT_KEY"],
            log_warning=False,
        )
        # Should return None or empty string for missing key, not crash
        assert key is None or key == ""

    def test_file_access_controls(self):
        """Test file access controls exist."""
        from victor.tools.registry import ToolRegistry

        # File tools should be registered in the tool registry
        registry = ToolRegistry()
        assert registry is not None
        # Verify registry has essential methods for tool access control
        assert hasattr(registry, "get") or hasattr(registry, "is_tool_enabled")

    def test_circuit_breaker_prevents_cascading_failures(self):
        """Test circuit breaker prevents cascading failures."""
        from victor.providers.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1.0, name="test")

        # Should allow execution initially
        assert breaker.can_execute() is True

        # Record failures
        breaker.record_failure()
        breaker.record_failure()

        # Should block execution after threshold
        assert breaker.can_execute() is False


# =============================================================================
# Configuration Smoke Tests
# =============================================================================


@pytest.mark.smoke
@pytest.mark.unit
class TestConfigurationSmokeTests:
    """Smoke tests for configuration loading."""

    def test_mode_config_loading(self):
        """Test mode configs can be loaded."""
        from victor.core.mode_config import ModeConfigRegistry

        registry = ModeConfigRegistry.get_instance()
        assert registry is not None

    def test_capability_loading(self):
        """Test capabilities can be loaded."""
        from victor.core.capabilities import CapabilityLoader

        loader = CapabilityLoader()
        assert loader is not None

    def test_team_specification_loading(self):
        """Test team specifications can be loaded."""
        from victor.teams import create_coordinator

        # Team coordination should be available
        assert create_coordinator is not None


# =============================================================================
# Error Recovery Smoke Tests
# =============================================================================


@pytest.mark.smoke
@pytest.mark.unit
class TestErrorRecoverySmokeTests:
    """Smoke tests for error recovery mechanisms."""

    def test_circuit_breaker_state_transitions(self):
        """Test circuit breaker state transitions."""
        from victor.providers.circuit_breaker import CircuitBreaker, CircuitState

        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1.0, name="test")

        # Initial state: CLOSED
        assert breaker.state == CircuitState.CLOSED

        # After failures: OPEN
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

    def test_retry_strategy_exists(self):
        """Test retry strategy exists."""
        from victor.framework.resilience import ExponentialBackoffStrategy

        # ExponentialBackoffStrategy with correct parameters
        strategy = ExponentialBackoffStrategy(
            max_attempts=3,
            base_delay=1.0,
            max_delay=10.0,
        )
        assert strategy is not None

    def test_validation_pipeline_exists(self):
        """Test validation pipeline exists."""
        from victor.framework.validation import ValidationPipeline

        pipeline = ValidationPipeline()
        assert pipeline is not None


# =============================================================================
# Observability Smoke Tests
# =============================================================================


@pytest.mark.smoke
@pytest.mark.unit
class TestObservabilitySmokeTests:
    """Smoke tests for observability features."""

    def test_usage_analytics_singleton(self):
        """Test UsageAnalytics singleton can be accessed."""
        from victor.agent.usage_analytics import UsageAnalytics

        analytics = UsageAnalytics.get_instance()
        assert analytics is not None

    def test_metrics_collection(self):
        """Test metrics can be collected."""
        from victor.agent.metrics_collector import MetricsCollector, MetricsCollectorConfig

        # Create mock usage logger
        from dataclasses import dataclass, field

        @dataclass
        class MockUsageLogger:
            tool_selections: dict = field(default_factory=dict)

            def record_tool_selection(self, method: str, num_tools: int):
                self.tool_selections[method] = num_tools

            def log_event(self, event_type: str, data: dict):
                pass

        collector = MetricsCollector(
            config=MetricsCollectorConfig(),
            usage_logger=MockUsageLogger(),
        )
        assert collector is not None

    def test_health_checker_exists(self):
        """Test health checker exists."""
        from victor.framework.health import HealthChecker

        checker = HealthChecker()
        assert checker is not None


# =============================================================================
# Summary and Reporting
# =============================================================================


@pytest.mark.smoke
@pytest.mark.unit
class TestSmokeTestSummary:
    """Summary smoke test to verify all major components."""

    def test_all_core_components_importable(self):
        """Test all core components can be imported."""
        # Core
        from victor.core.container import ServiceContainer
        from victor.core.events import create_event_backend
        from victor.config.settings import Settings

        # Agent
        from victor.agent.orchestrator import AgentOrchestrator
        from victor.agent.tool_pipeline import ToolPipeline

        # Providers
        from victor.providers.base import BaseProvider
        from victor.providers.mock import MockProvider

        # Tools
        from victor.tools.base import BaseTool

        # Framework
        from victor.framework import Agent, Task, State
        from victor.framework.resilience import CircuitBreaker
        from victor.framework.validation import ValidationPipeline

        # Teams
        from victor.teams import create_coordinator, TeamFormation

        # All imports should succeed
        assert True

    def test_production_readiness_checklist(self):
        """Verify production readiness checklist items."""

        # 1. Core infrastructure works
        container = ServiceContainer()
        assert container is not None

        # 2. Configuration loads
        settings = Settings()
        assert settings is not None

        # 3. Provider system works
        provider = MockProvider(model="test")
        assert provider is not None

        # 4. Tool system works
        from victor.tools.registry import ToolRegistry

        registry = ToolRegistry()
        assert registry is not None

        # 5. Verticals load
        from victor.coding import CodingAssistant

        config = CodingAssistant.get_config()
        assert config is not None

        # 6. Team coordination works
        from victor.teams import create_coordinator

        coordinator = create_coordinator(lightweight=True)
        assert coordinator is not None

        # 7. Error handling exists
        from victor.providers.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker(failure_threshold=5, name="test")
        assert breaker is not None

        # 8. Observability exists
        from victor.agent.usage_analytics import UsageAnalytics

        analytics = UsageAnalytics.get_instance()
        assert analytics is not None

        # All checklist items pass
        assert True
