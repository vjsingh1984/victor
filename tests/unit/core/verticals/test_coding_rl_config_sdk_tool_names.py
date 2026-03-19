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

"""Regression coverage for SDK-owned tool identifiers in coding RL config."""

from victor_sdk import ToolNames

from victor.verticals.contrib.coding.rl.config import CodingRLConfig


def test_coding_rl_config_uses_sdk_tool_names_for_task_mappings() -> None:
    """Task mappings should stay aligned with the SDK-owned tool registry."""
    config = CodingRLConfig()

    assert config.task_type_mappings["refactoring"] == [
        ToolNames.RENAME,
        ToolNames.EXTRACT,
        ToolNames.EDIT,
        ToolNames.READ,
    ]
    assert config.task_type_mappings["debugging"][:4] == [
        ToolNames.READ,
        ToolNames.GREP,
        ToolNames.SHELL,
        ToolNames.TEST,
    ]


def test_coding_rl_config_conflicting_tools_use_sdk_identifiers() -> None:
    """Conflicting tool declarations should use canonical SDK identifiers."""
    config = CodingRLConfig()

    assert config.conflicting_tools == {
        ToolNames.WRITE: {ToolNames.EDIT},
        ToolNames.EDIT: {ToolNames.WRITE},
    }
