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

"""Test utilities and helper functions for Victor AI test suite."""

# Use relative import to avoid ModuleNotFoundError during pytest collection
try:
    from .test_helpers import (
        create_test_completion_response,
        create_mock_provider,
        create_test_messages,
        create_test_tool_definition,
        create_mock_orchestrator,
        assert_completion_valid,
        assert_provider_called,
    )
except ImportError:
    # Fallback to absolute import for non-test contexts
    from tests.utils.test_helpers import (
        create_test_completion_response,
        create_mock_provider,
        create_test_messages,
        create_test_tool_definition,
        create_mock_orchestrator,
        assert_completion_valid,
        assert_provider_called,
    )

__all__ = [
    "create_test_completion_response",
    "create_mock_provider",
    "create_test_messages",
    "create_test_tool_definition",
    "create_mock_orchestrator",
    "assert_completion_valid",
    "assert_provider_called",
]
