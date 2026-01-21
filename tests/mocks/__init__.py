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

"""Test mocks and fixtures for Victor testing.

This package provides high-quality mock implementations for testing
Victor components without external dependencies.
"""

from tests.mocks.provider_mocks import (
    FailingProvider,
    MockBaseProvider,
    ProviderTestHelpers,
    StreamingTestProvider,
    ToolCallMockProvider,
)

__all__ = [
    "MockBaseProvider",
    "FailingProvider",
    "StreamingTestProvider",
    "ToolCallMockProvider",
    "ProviderTestHelpers",
]
