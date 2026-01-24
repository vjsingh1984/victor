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

"""Usage examples for provider mocks.

This file demonstrates common patterns for using the mock providers
in your tests.
"""

import pytest
from victor.core.errors import ProviderRateLimitError, ProviderTimeoutError
from victor.providers.base import Message

# Import mocks directly for standalone execution
try:
    from tests.mocks import (
        FailingProvider,
        MockBaseProvider,
        ProviderTestHelpers,
        StreamingTestProvider,
        ToolCallMockProvider,
    )
except ImportError:
    # When running as script
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from tests.mocks import (
        FailingProvider,
        MockBaseProvider,
        ProviderTestHelpers,
        StreamingTestProvider,
        ToolCallMockProvider,
    )


# =============================================================================
# Example 1: Basic Mock Provider Usage
# =============================================================================


async def example_basic_usage():
    """Example: Basic mock provider for simple testing."""
    # Create a mock provider with custom response
    provider = MockBaseProvider(response_text="Hello, world!")

    # Prepare test messages
    messages = ProviderTestHelpers.create_test_messages("Say hello")

    # Get response
    response = await provider.chat(messages, model="test-model")

    # Assertions
    assert response.content == "Hello, world!"
    assert response.role == "assistant"
    assert provider.call_count == 1


# =============================================================================
# Example 2: Testing Streaming Responses
# =============================================================================


async def example_streaming_test():
    """Example: Testing streaming response handling."""
    # Create streaming provider with controlled chunks
    provider = StreamingTestProvider(
        chunks=["The", "quick", "brown", "fox"],
        chunk_delay=0.01,  # Small delay for realism
    )

    messages = ProviderTestHelpers.create_test_messages("Tell me a story")

    # Collect all chunks
    chunks = await ProviderTestHelpers.collect_stream_chunks(provider, messages, model="test")

    # Verify chunks
    assert len(chunks) > 0
    assert chunks[-1].is_final

    # Reassemble content
    full_content = "".join(c.content for c in chunks if c.content)
    assert full_content == "Thequickbrownfox"


# =============================================================================
# Example 3: Testing Error Handling
# =============================================================================


async def example_error_handling():
    """Example: Testing error handling and retry logic."""
    # Provider that fails after 2 successful calls
    provider = FailingProvider(
        error_type="rate_limit",
        fail_after=2,  # First 2 calls succeed
    )

    messages = ProviderTestHelpers.create_test_messages("Test")

    # First call succeeds
    response1 = await provider.chat(messages, model="test")
    assert "Success" in response1.content

    # Second call succeeds
    response2 = await provider.chat(messages, model="test")
    assert "Success" in response2.content

    # Third call fails
    try:
        await provider.chat(messages, model="test")
        raise AssertionError("Unreachable code reached"), "Should have raised ProviderRateLimitError"
    except ProviderRateLimitError as e:
        assert e.retry_after == 60  # Default retry_after
        print(f"Rate limited! Retry after {e.retry_after}s")


# =============================================================================
# Example 4: Testing Tool Calling
# =============================================================================


async def example_tool_calling():
    """Example: Testing tool calling functionality."""
    # Create tool call
    tool_call = ProviderTestHelpers.create_test_tool_call(
        name="search_web",
        arguments={"query": "Python testing", "limit": 5},
    )

    # Provider returns tool calls
    provider = ToolCallMockProvider(
        response_text="I'll search the web for you.",
        tool_calls=[tool_call],
    )

    messages = ProviderTestHelpers.create_test_messages("Search Python testing")

    # Get response with tool calls
    response = await provider.chat(messages, model="test")

    # Verify tool calls
    assert response.tool_calls is not None
    assert len(response.tool_calls) == 1

    tc = response.tool_calls[0]
    assert tc["function"]["name"] == "search_web"
    assert "query" in tc["function"]["arguments"]


# =============================================================================
# Example 5: Testing Multi-Turn Conversations
# =============================================================================


async def example_multi_turn_conversation():
    """Example: Testing multi-turn conversation with tool calls."""
    # Define conversation sequence
    conversation = [
        {
            "content": "I'll search for that information.",
            "tool_calls": [ProviderTestHelpers.create_test_tool_call("search", {"query": "test"})],
        },
        {
            "content": "Based on the search results, here's what I found...",
            "tool_calls": None,  # No tool calls in second response
        },
    ]

    provider = ToolCallMockProvider(call_sequence=conversation)
    messages = ProviderTestHelpers.create_test_messages("Find information")

    # First turn - tool call
    response1 = await provider.chat(messages, model="test")
    assert response1.tool_calls is not None

    # Second turn - final response
    response2 = await provider.chat(messages, model="test")
    assert "Based on the search" in response2.content
    assert response2.tool_calls is None  # None when empty list


# =============================================================================
# Example 6: Testing with Custom Token Usage
# =============================================================================


async def example_custom_token_usage():
    """Example: Testing with custom token usage data."""
    custom_usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "cache_read_input_tokens": 25,
    }

    provider = MockBaseProvider(
        response_text="Response with custom usage",
        token_usage=custom_usage,
    )

    messages = ProviderTestHelpers.create_test_messages("Test")

    response = await provider.chat(messages, model="test")

    # Verify custom usage
    assert response.usage["total_tokens"] == 150
    assert response.usage["cache_read_input_tokens"] == 25


# =============================================================================
# Example 7: Testing Circuit Breaker Behavior
# =============================================================================


async def example_circuit_breaker():
    """Example: Testing circuit breaker with failing provider."""
    # Provider that fails immediately
    provider = FailingProvider(
        error_type="timeout",
        use_circuit_breaker=True,
        circuit_breaker_failure_threshold=3,  # Open after 3 failures
    )

    # Circuit breaker should be closed initially
    assert not provider.is_circuit_open()

    messages = ProviderTestHelpers.create_test_messages("Test")

    # First few failures open circuit after threshold
    for i in range(5):  # Exceed failure threshold of 3
        try:
            await provider.chat(messages, model="test")
        except ProviderTimeoutError:
            pass
        # After 3 failures, circuit should open
        if i >= 3:
            assert provider.is_circuit_open(), f"Circuit should be open after {i+1} failures"

    # Circuit should be open now
    assert provider.is_circuit_open()

    # Get circuit breaker stats
    stats = provider.get_circuit_breaker_stats()
    assert stats is not None
    print(f"  Circuit breaker state: {stats['state']}")


# =============================================================================
# Example 8: Testing Provider Switching
# =============================================================================


async def example_provider_switching():
    """Example: Testing switching between providers."""
    # Create two different mock providers
    provider1 = MockBaseProvider(response_text="Response from provider 1")
    provider2 = MockBaseProvider(response_text="Response from provider 2")

    messages = ProviderTestHelpers.create_test_messages("Test")

    # Use provider 1
    response1 = await provider1.chat(messages, model="model-1")
    assert "provider 1" in response1.content

    # Switch to provider 2
    response2 = await provider2.chat(messages, model="model-2")
    assert "provider 2" in response2.content

    # Verify call counts
    assert provider1.call_count == 1
    assert provider2.call_count == 1


# =============================================================================
# Example 9: Testing Request Parameters
# =============================================================================


async def example_request_parameters():
    """Example: Testing that request parameters are passed correctly."""
    provider = MockBaseProvider(response_text="Test response")

    messages = ProviderTestHelpers.create_test_messages("Test")

    # Make request with specific parameters
    response = await provider.chat(
        messages,
        model="gpt-4",
        temperature=0.5,
        max_tokens=1000,
    )

    # Verify parameters were recorded
    last_request = provider.last_request
    assert last_request["model"] == "gpt-4"
    assert last_request["temperature"] == 0.5
    assert last_request["max_tokens"] == 1000


# =============================================================================
# Example 10: Testing Response Validation
# =============================================================================


async def example_response_validation():
    """Example: Using helper functions to validate responses."""
    provider = ProviderTestHelpers.create_simple_mock("Valid response")
    messages = ProviderTestHelpers.create_test_messages("Test")

    # Get response
    response = await provider.chat(messages, model="test")

    # Validate response structure
    ProviderTestHelpers.assert_valid_response(response)

    # For streaming
    streaming_provider = ProviderTestHelpers.create_streaming_mock(chunks=["a", "b", "c"])

    chunks = await ProviderTestHelpers.collect_stream_chunks(
        streaming_provider, messages, model="test"
    )

    # Validate stream chunks
    ProviderTestHelpers.assert_valid_stream_chunks(chunks)


# =============================================================================
# Run Examples (for demonstration)
# =============================================================================


if __name__ == "__main__":
    import asyncio
    import sys
    from pathlib import Path

    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    async def run_examples():
        """Run all examples."""
        print("Running mock provider examples...\n")

        await example_basic_usage()
        print("✓ Basic usage example")

        await example_streaming_test()
        print("✓ Streaming test example")

        await example_error_handling()
        print("✓ Error handling example")

        await example_tool_calling()
        print("✓ Tool calling example")

        await example_multi_turn_conversation()
        print("✓ Multi-turn conversation example")

        await example_custom_token_usage()
        print("✓ Custom token usage example")

        await example_circuit_breaker()
        print("✓ Circuit breaker example")

        await example_provider_switching()
        print("✓ Provider switching example")

        await example_request_parameters()
        print("✓ Request parameters example")

        await example_response_validation()
        print("✓ Response validation example")

        print("\nAll examples completed successfully!")

    # Run examples
    asyncio.run(run_examples())
