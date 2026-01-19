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

"""Built-in capability implementations.

This module provides the built-in capabilities that are included with Victor.
These are registered via entry points in pyproject.toml and can also be
loaded manually for testing.

External packages can follow the same pattern to register their own capabilities.
"""

from __future__ import annotations

from victor.agent.capabilities.base import CapabilityBase, CapabilitySpec


class EnabledToolsCapability(CapabilityBase):
    """Enabled tools capability."""

    @classmethod
    def get_spec(cls) -> CapabilitySpec:
        return CapabilitySpec(
            name="enabled_tools",
            method_name="set_enabled_tools",
            version="1.0",
            description="Set enabled tools for the agent",
        )


class ToolDependenciesCapability(CapabilityBase):
    """Tool dependencies capability."""

    @classmethod
    def get_spec(cls) -> CapabilitySpec:
        return CapabilitySpec(
            name="tool_dependencies",
            method_name="set_tool_dependencies",
            version="1.0",
            description="Set tool dependencies for workflow execution",
        )


class ToolSequencesCapability(CapabilityBase):
    """Tool sequences capability."""

    @classmethod
    def get_spec(cls) -> CapabilitySpec:
        return CapabilitySpec(
            name="tool_sequences",
            method_name="set_tool_sequences",
            version="1.0",
            description="Set tool sequences for workflow execution",
        )


class TieredToolConfigCapability(CapabilityBase):
    """Tiered tool config capability."""

    @classmethod
    def get_spec(cls) -> CapabilitySpec:
        return CapabilitySpec(
            name="tiered_tool_config",
            method_name="set_tiered_tool_config",
            version="1.0",
            description="Set tiered tool configuration from vertical",
        )


class VerticalMiddlewareCapability(CapabilityBase):
    """Vertical middleware capability."""

    @classmethod
    def get_spec(cls) -> CapabilitySpec:
        return CapabilitySpec(
            name="vertical_middleware",
            method_name="apply_vertical_middleware",
            version="1.0",
            description="Apply vertical middleware chain",
        )


class VerticalSafetyPatternsCapability(CapabilityBase):
    """Vertical safety patterns capability."""

    @classmethod
    def get_spec(cls) -> CapabilitySpec:
        return CapabilitySpec(
            name="vertical_safety_patterns",
            method_name="apply_vertical_safety_patterns",
            version="1.0",
            description="Apply vertical safety patterns",
        )


class VerticalContextCapability(CapabilityBase):
    """Vertical context capability."""

    @classmethod
    def get_spec(cls) -> CapabilitySpec:
        return CapabilitySpec(
            name="vertical_context",
            method_name="set_vertical_context",
            version="1.0",
            description="Set vertical context",
        )


class RlHooksCapability(CapabilityBase):
    """RL hooks capability."""

    @classmethod
    def get_spec(cls) -> CapabilitySpec:
        return CapabilitySpec(
            name="rl_hooks",
            method_name="set_rl_hooks",
            version="1.0",
            description="Set RL hooks for outcome recording",
        )


class TeamSpecsCapability(CapabilityBase):
    """Team specs capability."""

    @classmethod
    def get_spec(cls) -> CapabilitySpec:
        return CapabilitySpec(
            name="team_specs",
            method_name="set_team_specs",
            version="1.0",
            description="Set team specifications",
        )


class ModeConfigsCapability(CapabilityBase):
    """Mode configs capability."""

    @classmethod
    def get_spec(cls) -> CapabilitySpec:
        return CapabilitySpec(
            name="mode_configs",
            method_name="set_mode_configs",
            version="1.0",
            description="Set mode configurations",
        )


class DefaultBudgetCapability(CapabilityBase):
    """Default budget capability."""

    @classmethod
    def get_spec(cls) -> CapabilitySpec:
        return CapabilitySpec(
            name="default_budget",
            method_name="set_default_budget",
            version="1.0",
            description="Set default tool budget",
        )


class CustomPromptCapability(CapabilityBase):
    """Custom prompt capability."""

    @classmethod
    def get_spec(cls) -> CapabilitySpec:
        return CapabilitySpec(
            name="custom_prompt",
            method_name="set_custom_prompt",
            version="1.0",
            description="Set custom prompt",
        )


class PromptSectionCapability(CapabilityBase):
    """Prompt section capability."""

    @classmethod
    def get_spec(cls) -> CapabilitySpec:
        return CapabilitySpec(
            name="prompt_section",
            method_name="add_prompt_section",
            version="1.0",
            description="Add prompt section",
        )


class TaskTypeHintsCapability(CapabilityBase):
    """Task type hints capability."""

    @classmethod
    def get_spec(cls) -> CapabilitySpec:
        return CapabilitySpec(
            name="task_type_hints",
            method_name="set_task_type_hints",
            version="1.0",
            description="Set task type hints",
        )


class SafetyPatternsCapability(CapabilityBase):
    """Safety patterns capability."""

    @classmethod
    def get_spec(cls) -> CapabilitySpec:
        return CapabilitySpec(
            name="safety_patterns",
            method_name="add_safety_patterns",
            version="1.0",
            description="Add safety patterns",
        )


class EnrichmentStrategyCapability(CapabilityBase):
    """Enrichment strategy capability."""

    @classmethod
    def get_spec(cls) -> CapabilitySpec:
        return CapabilitySpec(
            name="enrichment_strategy",
            method_name="set_enrichment_strategy",
            version="1.0",
            description="Set enrichment strategy",
        )


__all__ = [
    "EnabledToolsCapability",
    "ToolDependenciesCapability",
    "ToolSequencesCapability",
    "TieredToolConfigCapability",
    "VerticalMiddlewareCapability",
    "VerticalSafetyPatternsCapability",
    "VerticalContextCapability",
    "RlHooksCapability",
    "TeamSpecsCapability",
    "ModeConfigsCapability",
    "DefaultBudgetCapability",
    "CustomPromptCapability",
    "PromptSectionCapability",
    "TaskTypeHintsCapability",
    "SafetyPatternsCapability",
    "EnrichmentStrategyCapability",
]
