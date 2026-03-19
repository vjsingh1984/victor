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

"""Regression coverage for coding definition-layer capability requirements."""

from victor_sdk import CapabilityIds

from victor.framework.vertical_runtime_adapter import VerticalRuntimeAdapter
from victor.verticals.contrib.coding.assistant import CodingAssistant


def test_coding_definition_declares_sdk_capability_requirements() -> None:
    """Coding should expose host/runtime needs through the SDK definition contract."""
    definition = CodingAssistant.get_definition()
    requirements = {
        requirement.capability_id: requirement for requirement in definition.capability_requirements
    }

    assert set(requirements) == {
        CapabilityIds.FILE_OPS,
        CapabilityIds.GIT,
        CapabilityIds.LSP,
    }
    assert requirements[CapabilityIds.FILE_OPS].optional is False
    assert requirements[CapabilityIds.GIT].optional is True
    assert requirements[CapabilityIds.LSP].optional is True


def test_coding_definition_capability_requirements_round_trip_into_runtime_config() -> None:
    """Runtime binding metadata should preserve the SDK capability requirements."""
    binding = VerticalRuntimeAdapter.build_runtime_binding(CodingAssistant)
    requirements = binding.runtime_config.metadata["capability_requirements"]

    assert [requirement["capability_id"] for requirement in requirements] == [
        CapabilityIds.FILE_OPS,
        CapabilityIds.GIT,
        CapabilityIds.LSP,
    ]


def test_coding_definition_metadata_round_trips_into_runtime_binding() -> None:
    """Coding metadata should be expressed through the SDK definition contract."""
    definition = CodingAssistant.get_definition()
    binding = VerticalRuntimeAdapter.build_runtime_binding(CodingAssistant)

    assert definition.metadata["supports_lsp"] is True
    assert definition.metadata["supports_git"] is True
    assert definition.metadata["max_file_size"] == 1_000_000
    assert "python" in definition.metadata["supported_languages"]
    assert binding.runtime_config.metadata["supports_lsp"] is True
    assert binding.runtime_config.metadata["supports_git"] is True
    assert binding.runtime_config.metadata["max_file_size"] == 1_000_000
    assert "python" in binding.runtime_config.metadata["supported_languages"]
