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

"""Real execution integration tests package.

This package contains integration tests that execute with real LLM providers
and real file operations (no mocking).

Test Categories:
- test_real_tool_execution.py: Core tool execution tests
- test_real_conversation_flow.py: Multi-turn conversation tests
- test_real_error_scenarios.py: Error handling tests
- test_real_workflow_execution.py: Workflow execution tests (TODO)
- test_zai_provider.py: Cloud provider tests (TODO)

Requirements:
- Ollama running at localhost:11434
- Model: qwen3-coder-tools:30b or qwen2.5-coder:7b
- M1 Max hardware (or equivalent)
"""

__all__ = []
