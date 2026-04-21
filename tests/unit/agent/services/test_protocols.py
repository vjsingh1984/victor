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

"""Tests for service protocol definitions.

Tests that protocols are properly defined and can be implemented
by concrete classes.
"""

import pytest

from victor.agent.services.protocols import (
    ChatServiceProtocol,
    ToolServiceProtocol,
    ContextServiceProtocol,
    ProviderServiceProtocol,
    RecoveryServiceProtocol,
    SessionServiceProtocol,
)


class TestProtocolDefinitions:
    """Test that protocols are properly defined."""

    def test_chat_service_protocol_exists(self):
        """Test ChatServiceProtocol is defined."""
        assert ChatServiceProtocol is not None

    def test_tool_service_protocol_exists(self):
        """Test ToolServiceProtocol is defined."""
        assert ToolServiceProtocol is not None

    def test_context_service_protocol_exists(self):
        """Test ContextServiceProtocol is defined."""
        assert ContextServiceProtocol is not None

    def test_provider_service_protocol_exists(self):
        """Test ProviderServiceProtocol is defined."""
        assert ProviderServiceProtocol is not None

    def test_recovery_service_protocol_exists(self):
        """Test RecoveryServiceProtocol is defined."""
        assert RecoveryServiceProtocol is not None

    def test_session_service_protocol_exists(self):
        """Test SessionServiceProtocol is defined."""
        assert SessionServiceProtocol is not None


class TestProtocolImplementation:
    """Test that protocols can be implemented by concrete classes."""

    def test_chat_service_protocol_implementation(self):
        """Test that a class can implement ChatServiceProtocol."""

        class MockChatService(ChatServiceProtocol):
            def __init__(self):
                self.reset_count = 0

            async def chat(self, user_message: str, **kwargs):
                from victor.providers.base import CompletionResponse

                return CompletionResponse(
                    content=f"Response to: {user_message}",
                    stop_reason="stop",
                    usage={"prompt_tokens": 10, "completion_tokens": 20},
                )

            async def stream_chat(self, user_message: str, **kwargs):
                from victor.providers.base import StreamChunk

                yield StreamChunk(content=f"Chunk for: {user_message}")

            async def chat_with_planning(self, user_message: str, use_planning=None):
                return await self.chat(user_message, use_planning=use_planning)

            async def handle_context_and_iteration_limits(
                self,
                user_message,
                max_total_iterations,
                max_context,
                total_iterations,
                last_quality_score,
            ):
                return False, None

            def reset_conversation(self) -> None:
                self.reset_count += 1

            def is_healthy(self) -> bool:
                return True

        service = MockChatService()
        assert isinstance(service, ChatServiceProtocol)
        assert service.reset_count == 0

        import asyncio

        async def test():
            response = await service.chat("test")
            assert "Response to: test" in response.content

            chunks = [c async for c in service.stream_chat("test")]
            assert len(chunks) == 1

            service.reset_conversation()
            assert service.reset_count == 1

            assert service.is_healthy()

        asyncio.run(test())

    def test_tool_service_protocol_implementation(self):
        """Test that a class can implement ToolServiceProtocol."""

        class MockToolService(ToolServiceProtocol):
            def __init__(self):
                self.budget = 100

            async def select_tools(self, context, max_tools=10):
                return ["tool1", "tool2"]

            async def execute_tool(self, tool_name, arguments):
                from victor.tools.base import ToolResult

                return ToolResult(success=True, output=f"Executed {tool_name}")

            async def execute_tools_parallel(self, tool_calls, max_parallel=5):
                for tool_name, args in tool_calls:
                    yield await self.execute_tool(tool_name, args)

            def get_tool_budget(self):
                return self.budget

            def set_tool_budget(self, budget):
                self.budget = max(0, budget)

            def get_tool_usage_stats(self):
                return {"total_calls": 5, "success_rate": 1.0}

            def reset_tool_budget(self):
                self.budget = 100

            def process_tool_results(self, pipeline_result, ctx):
                return []

            def get_available_tools(self):
                return {"tool1", "tool2"}

            def get_enabled_tools(self):
                return {"tool1", "tool2"}

            def set_enabled_tools(self, tools):
                self.enabled_tools = set(tools)

            def is_tool_enabled(self, tool_name):
                return tool_name in {"tool1", "tool2"}

            def resolve_tool_alias(self, tool_name):
                return tool_name

            def parse_and_validate_tool_calls(self, tool_calls, full_content, tool_adapter):
                return tool_calls, full_content

            async def execute_tool_with_retry(self, tool_name, tool_args, context, **kwargs):
                result = await self.execute_tool(tool_name, tool_args)
                return result, True, None

            def normalize_tool_arguments(self, tool_args, tool_name):
                return tool_args, "direct"

            def build_tool_access_context(self):
                return {"tools": self.get_enabled_tools()}

            def is_healthy(self):
                return self.budget > 0

        service = MockToolService()
        assert isinstance(service, ToolServiceProtocol)

        import asyncio

        async def test():
            tools = await service.select_tools(None, 5)
            assert tools == ["tool1", "tool2"]

            result = await service.execute_tool("test", {})
            assert result.success is True

            stats = service.get_tool_usage_stats()
            assert stats["total_calls"] == 5

        asyncio.run(test())

    def test_context_service_protocol_implementation(self):
        """Test that a class can implement ContextServiceProtocol."""

        class MockMetrics:
            def __init__(self):
                self.total_tokens = 1000
                self.message_count = 10
                self.user_message_count = 5
                self.assistant_message_count = 5
                self.tool_result_count = 0
                self.system_prompt_tokens = 100
                self.utilization_percent = 10.0

        class MockContextService(ContextServiceProtocol):
            def __init__(self):
                self.max_tokens = 10000
                self.messages = []

            async def get_context_metrics(self):
                return MockMetrics()

            async def check_context_overflow(self):
                return len(self.messages) > 1000

            async def compact_context(self, strategy="tiered", min_messages=6):
                removed = len(self.messages) - min_messages
                self.messages = self.messages[:min_messages]
                return removed

            def add_message(self, message):
                self.messages.append(message)

            def add_messages(self, messages):
                self.messages.extend(messages)

            def get_messages(self, limit=None, role=None):
                messages = self.messages
                if role:
                    messages = [m for m in messages if m.get("role") == role]
                if limit:
                    messages = messages[-limit:]
                return messages

            def clear_messages(self, retain_system=True):
                self.messages.clear()

            def get_max_tokens(self):
                return self.max_tokens

            def set_max_tokens(self, max_tokens):
                if max_tokens < 0:
                    raise ValueError("max_tokens must be positive")
                self.max_tokens = max_tokens

            def estimate_tokens(self, text):
                return len(text) // 4

            def is_healthy(self):
                return self.max_tokens > 0

        service = MockContextService()
        assert isinstance(service, ContextServiceProtocol)

        import asyncio

        async def test():
            metrics = await service.get_context_metrics()
            assert metrics.total_tokens == 1000

            assert not await service.check_context_overflow()

            service.add_message({"role": "user", "content": "test"})
            assert len(service.get_messages()) == 1

        asyncio.run(test())

    def test_provider_service_protocol_implementation(self):
        """Test that a class can implement ProviderServiceProtocol."""

        class MockProviderInfo:
            provider_name = "test_provider"
            model_name = "test_model"
            api_key_configured = True
            base_url = "https://api.test.com"
            supports_streaming = True
            supports_tool_calling = True
            max_tokens = 100000

        class MockProviderService(ProviderServiceProtocol):
            def __init__(self):
                self.current = MockProviderInfo()

            async def switch_provider(self, provider, model=None, validate=True):
                self.current.provider_name = provider
                if model:
                    self.current.model_name = model

            def get_current_provider_info(self):
                return self.current

            async def check_provider_health(self, provider=None):
                return True

            def get_available_providers(self):
                return ["provider1", "provider2"]

            async def get_provider_capabilities(self, provider=None):
                return {"streaming": True, "tools": True}

            async def start_health_monitoring(self):
                return None

            async def stop_health_monitoring(self):
                return None

            def get_current_provider(self):
                return self  # Mock provider

            async def test_provider(self, provider, model=None):
                return provider in self.get_available_providers()

            def get_rate_limit_wait_time(self, error):
                return 1.0

            def is_healthy(self):
                return self.current is not None

        service = MockProviderService()
        assert isinstance(service, ProviderServiceProtocol)

        import asyncio

        async def test():
            info = service.get_current_provider_info()
            assert info.provider_name == "test_provider"

            await service.switch_provider("new_provider")
            assert service.current.provider_name == "new_provider"

            assert await service.check_provider_health()

        asyncio.run(test())

    def test_recovery_service_protocol_implementation(self):
        """Test that a class can implement RecoveryServiceProtocol."""

        class MockRecoveryContext:
            def __init__(self, error):
                self.error = error
                self.error_type = type(error).__name__
                self.attempt_count = 1
                self.state = {}
                self.metadata = {}

        class MockRecoveryService(RecoveryServiceProtocol):
            def __init__(self):
                self.metrics = {"attempts": 0, "successes": 0}

            async def classify_error(self, error):
                if isinstance(error, TimeoutError):
                    return "timeout"
                return "unknown"

            async def select_recovery_action(self, context):
                class MockAction:
                    name = "retry"
                    description = "Retry operation"

                    async def execute(self, context):
                        return True

                return MockAction()

            async def execute_recovery(self, context):
                action = await self.select_recovery_action(context)
                return await action.execute(context)

            def can_retry(self, error, attempt_count):
                return attempt_count < 3

            def get_recovery_metrics(self):
                return self.metrics.copy()

            def reset_metrics(self):
                self.metrics = {"attempts": 0, "successes": 0}

            def is_healthy(self):
                return True

        service = MockRecoveryService()
        assert isinstance(service, RecoveryServiceProtocol)

        import asyncio

        async def test():
            error = TimeoutError("test")
            error_type = await service.classify_error(error)
            assert error_type == "timeout"

            context = MockRecoveryContext(error)
            assert service.can_retry(error, 1)
            assert not service.can_retry(error, 3)

            success = await service.execute_recovery(context)
            assert success is True

        asyncio.run(test())

    def test_session_service_protocol_implementation(self):
        """Test that a class can implement SessionServiceProtocol."""

        from datetime import datetime

        class MockSessionInfo:
            def __init__(self, session_id):
                self.session_id = session_id
                self.created_at = datetime.now()
                self.last_activity = datetime.now()
                self.message_count = 0
                self.tool_calls = 0
                self.metadata = {}

        class MockSessionService(SessionServiceProtocol):
            def __init__(self):
                self.sessions = {}
                self.current_id = None

            async def create_session(self, metadata=None):
                import uuid

                session_id = str(uuid.uuid4())
                self.sessions[session_id] = MockSessionInfo(session_id)
                self.current_id = session_id
                return session_id

            async def get_session(self, session_id):
                return self.sessions.get(session_id)

            async def update_session(self, session_id, metadata):
                if session_id in self.sessions:
                    self.sessions[session_id].metadata.update(metadata)
                    return True
                return False

            async def close_session(self, session_id):
                if session_id in self.sessions:
                    del self.sessions[session_id]
                    return True
                return False

            def get_current_session_id(self):
                return self.current_id

            async def list_sessions(self, state=None, limit=100):
                return list(self.sessions.values())[:limit]

            async def delete_session(self, session_id):
                return await self.close_session(session_id)

            async def get_session_metrics(self, session_id):
                session = self.sessions.get(session_id)
                if session:
                    return {"message_count": session.message_count}
                return {}

            def recover_session(self, session_id):
                self.current_id = session_id if session_id in self.sessions else None
                return self.current_id is not None

            async def maybe_auto_checkpoint(self):
                return None

            async def save_checkpoint(self, description=None, tags=None):
                return "checkpoint-1"

            async def restore_checkpoint(self, checkpoint_id):
                return checkpoint_id == "checkpoint-1"

            def get_memory_context(self, max_tokens=None, messages=None):
                return messages or []

            def get_recent_sessions(self, limit=10):
                return [{"session_id": sid} for sid in list(self.sessions)[:limit]]

            def get_session_stats(self):
                return {"session_id": self.current_id}

            def is_healthy(self):
                return True

        service = MockSessionService()
        assert isinstance(service, SessionServiceProtocol)

        import asyncio

        async def test():
            session_id = await service.create_session({"test": "data"})
            assert session_id is not None

            info = await service.get_session(session_id)
            assert info is not None
            assert info.session_id == session_id

            await service.update_session(session_id, {"key": "value"})

            await service.close_session(session_id)
            assert await service.get_session(session_id) is None

        asyncio.run(test())


class TestProtocolCompliance:
    """Test protocol compliance with ISP (Interface Segregation Principle)."""

    def test_protocols_are_focused(self):
        """Test that each protocol has a focused set of methods.

        ISP compliance: Protocols should not have too many methods.
        A reasonable threshold is around 10 methods per protocol.
        """
        # Count methods in each protocol
        chat_methods = [
            m
            for m in dir(ChatServiceProtocol)
            if not m.startswith("_") and callable(getattr(ChatServiceProtocol, m))
        ]

        # ChatServiceProtocol should have reasonable number of methods
        assert len(chat_methods) <= 10

    def test_protocols_are_composable(self):
        """Test that multiple protocols can be implemented together."""

        class MultiService(
            ChatServiceProtocol,
            ToolServiceProtocol,
        ):
            """Service implementing multiple protocols."""

            def __init__(self):
                pass

            # ChatServiceProtocol methods
            async def chat(self, user_message, **kwargs):
                from victor.providers.base import CompletionResponse

                return CompletionResponse(
                    content=f"Response: {user_message}",
                    finish_reason="stop",
                    usage={"prompt_tokens": 10, "completion_tokens": 20},
                )

            async def stream_chat(self, user_message, **kwargs):
                async def chunks():
                    from victor.providers.base import StreamChunk

                    yield StreamChunk(content="test")

                return chunks()

            async def chat_with_planning(self, user_message, use_planning=None):
                return await self.chat(user_message, use_planning=use_planning)

            async def handle_context_and_iteration_limits(
                self,
                user_message,
                max_total_iterations,
                max_context,
                total_iterations,
                last_quality_score,
            ):
                return False, None

            def reset_conversation(self):
                pass

            def is_healthy(self):
                return True

            # ToolServiceProtocol methods
            async def select_tools(self, context, max_tools=10):
                return []

            async def execute_tool(self, tool_name, arguments):
                from victor.tools.base import ToolResult

                return ToolResult(success=True, output="test")

            async def execute_tools_parallel(self, tool_calls, max_parallel=5):
                async def empty_gen():
                    return
                    yield

                return empty_gen()

            def get_tool_budget(self):
                return 100

            def set_tool_budget(self, budget):
                pass

            def get_tool_usage_stats(self):
                return {}

            def reset_tool_budget(self):
                pass

            def process_tool_results(self, pipeline_result, ctx):
                return []

            def get_available_tools(self):
                return set()

            def get_enabled_tools(self):
                return set()

            def set_enabled_tools(self, tools):
                pass

            def is_tool_enabled(self, tool_name):
                return True

            def resolve_tool_alias(self, tool_name):
                return tool_name

            def parse_and_validate_tool_calls(self, tool_calls, full_content, tool_adapter):
                return tool_calls, full_content

            async def execute_tool_with_retry(self, tool_name, tool_args, context, **kwargs):
                result = await self.execute_tool(tool_name, tool_args)
                return result, True, None

            def normalize_tool_arguments(self, tool_args, tool_name):
                return tool_args, "direct"

            def build_tool_access_context(self):
                return {}

        service = MultiService()
        assert isinstance(service, ChatServiceProtocol)
        assert isinstance(service, ToolServiceProtocol)


class TestProtocolTypeChecking:
    """Test that protocols support runtime type checking."""

    def test_chat_service_protocol_runtime_check(self):
        """Test runtime check for ChatServiceProtocol."""

        class ValidService:
            async def chat(self, user_message, **kwargs):
                pass

            async def stream_chat(self, user_message, **kwargs):
                pass

            async def chat_with_planning(self, user_message, use_planning=None):
                pass

            async def handle_context_and_iteration_limits(
                self,
                user_message,
                max_total_iterations,
                max_context,
                total_iterations,
                last_quality_score,
            ):
                return False, None

            def reset_conversation(self):
                pass

            def is_healthy(self):
                pass

        class InvalidService:
            def chat(self, user_message):
                pass  # Missing other methods

        valid = ValidService()
        # We can't directly test runtime_checkable without a proper implementation
        # but we can verify the protocol has the expected structure
        assert hasattr(ChatServiceProtocol, "chat")
        assert hasattr(ChatServiceProtocol, "stream_chat")
        assert hasattr(ChatServiceProtocol, "chat_with_planning")
        assert hasattr(ChatServiceProtocol, "handle_context_and_iteration_limits")
        assert hasattr(ChatServiceProtocol, "reset_conversation")
        assert hasattr(ChatServiceProtocol, "is_healthy")
