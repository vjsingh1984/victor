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

"""Utility modules for agent orchestrator.

This package contains utility modules extracted from the orchestrator
to improve modularity, testability, and maintainability.

Modules:
- tool_detection: Tool name resolution, alias mapping, and shell variant detection
- conversions: Type conversion utilities for validation results, token usage, etc.
- helpers: General helper functions for string manipulation, formatting, etc.
"""

from victor.agent.utils.conversions import (
    token_usage_to_dict,
    validation_result_to_dict,
    message_to_dict,
)
from victor.agent.utils.helpers import (
    format_tool_output_for_log,
    extract_file_paths_from_text,
    extract_output_requirements_from_text,
)
from victor.agent.utils.tool_detection import (
    resolve_shell_variant,
    is_shell_alias,
    get_shell_aliases,
)

__all__ = [
    # Conversions
    "token_usage_to_dict",
    "validation_result_to_dict",
    "message_to_dict",
    # Helpers
    "format_tool_output_for_log",
    "extract_file_paths_from_text",
    "extract_output_requirements_from_text",
    # Tool detection
    "resolve_shell_variant",
    "is_shell_alias",
    "get_shell_aliases",
]
