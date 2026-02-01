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

"""LLM Provider abstraction layer."""

from victor.core.errors import (
    ProviderError,
    ProviderNotFoundError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from victor.providers.base import (
    BaseProvider,
    Message,
    CompletionResponse,
    StreamChunk,
    # ISP Protocol classes
    StreamingProvider,
    ToolCallingProvider,
    # Helper functions
    is_streaming_provider,
    is_tool_calling_provider,
)
from victor.providers.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitBreakerRegistry,
    CircuitState,
)
from victor.providers.registry import ProviderRegistry

__all__ = [
    # Base classes
    "BaseProvider",
    "Message",
    "CompletionResponse",
    "StreamChunk",
    # ISP Protocol classes (Interface Segregation Principle)
    "StreamingProvider",
    "ToolCallingProvider",
    # Helper functions for capability checking
    "is_streaming_provider",
    "is_tool_calling_provider",
    # Error classes
    "ProviderError",
    "ProviderNotFoundError",
    "ProviderAuthError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitBreakerError",
    "CircuitBreakerRegistry",
    "CircuitState",
    # Registry
    "ProviderRegistry",
]
