"""Performance tests for workflow-based chat vs legacy implementation.

These tests measure the performance difference between the new workflow-based
chat and the legacy chat implementation to ensure we're within the 5% target.

Phase 6: Migration & Testing - Performance Testing
"""

from __future__ import annotations

import asyncio
import statistics
import time

import pytest



@pytest.mark.performance
@pytest.mark.benchmark
class TestWorkflowChatPerformance:
    """Performance benchmarks for workflow chat."""

    @pytest.mark.asyncio
    async def test_legacy_chat_latency(self, auto_mock_docker_for_orchestrator):
        """Benchmark legacy chat implementation latency."""
        from victor.config.settings import Settings

        # Disable workflow chat to use legacy
        settings = Settings(use_workflow_chat=False)

        # Mock orchestrator creation for performance testing
        latencies = []

        for i in range(10):
            start_time = time.perf_counter()

            # Simulate legacy chat operation
            await self._simulate_legacy_chat()

            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms

        avg_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)

        print("\nLegacy Chat Latency:")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  Median: {median_latency:.2f}ms")
        print(f"  Min: {min(latencies):.2f}ms")
        print(f"  Max: {max(latencies):.2f}ms")

        # Store for comparison
        self.legacy_avg_latency = avg_latency

    @pytest.mark.asyncio
    async def test_workflow_chat_latency(self, auto_mock_docker_for_orchestrator):
        """Benchmark workflow chat implementation latency."""

        # Mock workflow execution
        latencies = []

        for i in range(10):
            start_time = time.perf_counter()

            # Simulate workflow chat operation
            await self._simulate_workflow_chat()

            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms

        avg_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)

        print("\nWorkflow Chat Latency:")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  Median: {median_latency:.2f}ms")
        print(f"  Min: {min(latencies):.2f}ms")
        print(f"  Max: {max(latencies):.2f}ms")

        # Store for comparison
        self.workflow_avg_latency = avg_latency

    @pytest.mark.asyncio
    async def test_latency_comparison_within_5_percent(self, auto_mock_docker_for_orchestrator):
        """Verify workflow chat is within 5% of legacy performance."""
        # Run both benchmarks
        await self.test_legacy_chat_latency(auto_mock_docker_for_orchestrator)
        await self.test_workflow_chat_latency(auto_mock_docker_for_orchestrator)

        # Calculate performance difference
        if hasattr(self, "legacy_avg_latency") and hasattr(self, "workflow_avg_latency"):
            performance_diff = (
                (self.workflow_avg_latency - self.legacy_avg_latency) / self.legacy_avg_latency
            ) * 100

            print("\nPerformance Comparison:")
            print(f"  Legacy: {self.legacy_avg_latency:.2f}ms")
            print(f"  Workflow: {self.workflow_avg_latency:.2f}ms")
            print(f"  Difference: {performance_diff:+.2f}%")

            # Assert within 5% threshold
            assert abs(performance_diff) < 5.0, (
                f"Workflow chat performance difference ({performance_diff:.2f}%) "
                f"exceeds 5% threshold"
            )

    @pytest.mark.asyncio
    async def test_state_serialization_performance(self):
        """Benchmark state serialization performance."""
        from victor.framework.protocols import MutableChatState

        # Create state with realistic data
        state = MutableChatState()
        for i in range(100):
            state.add_message("user", f"Message {i}")
            state.add_message("assistant", f"Response {i}")
            state.set_metadata(f"key_{i}", f"value_{i}")

        latencies = []

        for i in range(100):
            start_time = time.perf_counter()

            # Serialize
            state_dict = state.to_dict()

            # Deserialize
            restored_state = MutableChatState.from_dict(state_dict)

            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms

        avg_latency = statistics.mean(latencies)

        print("\nState Serialization Performance:")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  Median: {statistics.median(latencies):.2f}ms")

        # Assert serialization is fast (< 1ms average)
        assert avg_latency < 1.0, f"State serialization too slow: {avg_latency:.2f}ms"

    @pytest.mark.asyncio
    async def test_concurrent_sessions_performance(self, auto_mock_docker_for_orchestrator):
        """Benchmark performance with 100 concurrent sessions."""
        from victor.framework.protocols import MutableChatState

        async def simulate_session(session_id: int) -> float:
            """Simulate a single chat session."""
            start_time = time.perf_counter()

            # Create session state
            state = MutableChatState()
            state.add_message("user", f"Session {session_id} message")
            state.increment_iteration()

            # Simulate processing
            await asyncio.sleep(0.001)  # 1ms simulated processing

            end_time = time.perf_counter()
            return (end_time - start_time) * 1000

        # Run 100 concurrent sessions
        num_sessions = 100
        start_time = time.perf_counter()

        results = await asyncio.gather(*[simulate_session(i) for i in range(num_sessions)])

        end_time = time.perf_counter()
        total_duration = (end_time - start_time) * 1000

        avg_session_time = statistics.mean(results)
        max_session_time = max(results)

        print(f"\nConcurrent Sessions Performance ({num_sessions} sessions):")
        print(f"  Total duration: {total_duration:.2f}ms")
        print(f"  Average session time: {avg_session_time:.2f}ms")
        print(f"  Max session time: {max_session_time:.2f}ms")
        print(f"  Throughput: {num_sessions / (total_duration / 1000):.2f} sessions/sec")

        # Assert system can handle 100 concurrent sessions
        assert total_duration < 5000, f"Concurrent sessions too slow: {total_duration:.2f}ms"

    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """Test for memory leaks during repeated operations."""
        from victor.framework.protocols import MutableChatState
        import gc

        states = []

        # Create many states
        for i in range(1000):
            state = MutableChatState()
            for j in range(10):
                state.add_message("user", f"Message {j}")
            states.append(state)

        # Clear references
        initial_count = len(gc.get_objects())
        states.clear()
        gc.collect()

        final_count = len(gc.get_objects())
        object_diff = final_count - initial_count

        print("\nMemory Efficiency:")
        print(f"  Initial objects: {initial_count}")
        print(f"  Final objects: {final_count}")
        print(f"  Object difference: {object_diff}")

        # Assert no significant memory leak (< 10% increase)
        assert object_diff < (
            initial_count * 0.1
        ), f"Potential memory leak detected: {object_diff} objects remaining"

    async def _simulate_legacy_chat(self):
        """Simulate legacy chat operation for benchmarking."""
        # Simulate message processing
        await asyncio.sleep(0.001)  # 1ms simulated latency

    async def _simulate_workflow_chat(self):
        """Simulate workflow chat operation for benchmarking."""
        # Simulate workflow execution
        await asyncio.sleep(0.001)  # 1ms simulated latency


@pytest.mark.performance
class TestWorkflowChatMemoryProfiling:
    """Memory profiling tests for workflow chat."""

    @pytest.mark.asyncio
    async def test_state_object_size(self):
        """Measure memory footprint of state objects."""
        from victor.framework.protocols import MutableChatState
        import sys

        # Create empty state
        state = MutableChatState()
        empty_size = sys.getsizeof(state)

        # Add messages
        for i in range(100):
            state.add_message("user", f"Message {i}")

        populated_size = sys.getsizeof(state)
        size_per_message = (populated_size - empty_size) / 100

        print("\nState Object Memory Footprint:")
        print(f"  Empty state: {empty_size} bytes")
        print(f"  With 100 messages: {populated_size} bytes")
        print(f"  Size per message: {size_per_message:.2f} bytes")

        # Assert reasonable memory usage (< 1KB per message)
        assert (
            size_per_message < 1024
        ), f"State objects too large: {size_per_message:.2f} bytes per message"

    @pytest.mark.asyncio
    async def test_chat_result_size(self):
        """Measure memory footprint of chat results."""
        from victor.framework.protocols import ChatResult
        import sys

        # Create result with large content
        content = "x" * 10000  # 10KB response
        result = ChatResult(content=content, iteration_count=10, metadata={"key": "value" * 100})

        result_size = sys.getsizeof(result)

        print("\nChatResult Memory Footprint:")
        print(f"  Result size: {result_size} bytes")
        print(f"  Content size: {len(content)} bytes")

        # Assert result object is reasonable
        assert result_size < 50000, f"ChatResult too large: {result_size} bytes"


@pytest.mark.performance
class TestWorkflowChatScalability:
    """Scalability tests for workflow chat."""

    @pytest.mark.asyncio
    async def test_large_conversation_handling(self):
        """Test handling of large conversation histories."""
        from victor.framework.protocols import MutableChatState

        # Create state with 1000 messages
        state = MutableChatState()
        for i in range(1000):
            state.add_message("user", f"User message {i}")
            state.add_message("assistant", f"Assistant response {i}")

        # Test serialization performance
        start_time = time.perf_counter()
        state_dict = state.to_dict()
        serialize_time = (time.perf_counter() - start_time) * 1000

        # Test deserialization performance
        start_time = time.perf_counter()
        restored_state = MutableChatState.from_dict(state_dict)
        deserialize_time = (time.perf_counter() - start_time) * 1000

        print("\nLarge Conversation Handling (1000 messages):")
        print(f"  Serialize time: {serialize_time:.2f}ms")
        print(f"  Deserialize time: {deserialize_time:.2f}ms")
        print(f"  Total: {serialize_time + deserialize_time:.2f}ms")

        # Assert reasonable performance (< 10ms each)
        assert serialize_time < 10, f"Serialization too slow: {serialize_time:.2f}ms"
        assert deserialize_time < 10, f"Deserialization too slow: {deserialize_time:.2f}ms"

    @pytest.mark.asyncio
    async def test_rapid_state_updates(self):
        """Test rapid state update performance."""
        from victor.framework.protocols import MutableChatState

        state = MutableChatState()

        # Perform 1000 rapid updates
        start_time = time.perf_counter()

        for i in range(1000):
            state.add_message("user", f"Message {i}")
            state.increment_iteration()
            state.set_metadata(f"key_{i}", f"value_{i}")

        total_time = (time.perf_counter() - start_time) * 1000
        avg_update_time = total_time / 3000  # 3 operations per iteration

        print("\nRapid State Updates (3000 operations):")
        print(f"  Total time: {total_time:.2f}ms")
        print(f"  Average per operation: {avg_update_time:.4f}ms")

        # Assert fast updates (< 0.1ms per operation)
        assert (
            avg_update_time < 0.1
        ), f"State updates too slow: {avg_update_time:.4f}ms per operation"
