#!/usr/bin/env python3
"""
Vertical Scaffolding Generator

This script generates vertical implementations from templates, reducing
code duplication by 65-70%. It creates all necessary files for a new
vertical including:
- assistant.py (main vertical class)
- prompts.py (prompt contributions)
- safety.py (safety patterns)
- escape_hatches.py (workflow escape hatches)
- handlers.py (workflow handlers)
- teams.py (team formations)
- __init__.py (package initialization)
- config/vertical.yaml (YAML configuration)

Usage:
    # Generate from template
    python scripts/generate_vertical.py --template security --output victor/security

    # Generate from existing vertical
    python scripts/generate_vertical.py --extract-from victor/coding --output templates/coding.yaml

    # List available templates
    python scripts/generate_vertical.py --list-templates

    # Validate a template
    python scripts/generate_vertical.py --validate templates/security.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from victor.framework.vertical_template import (
    VerticalTemplate,
    VerticalMetadata,
    ExtensionSpecs,
    PromptHintSpec,
    MiddlewareSpec,
)
from victor.framework.vertical_template_registry import VerticalTemplateRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# File Templates
# =============================================================================

ASSISTANT_PY_TEMPLATE = '''# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""{vertical_class_name} - {description}.

This vertical provides {capabilities_lower}.

Generated from template by scripts/generate_vertical.py.
"""

from __future__ import annotations

from typing import Any, Dict, List

from victor.core.verticals.base import StageDefinition, VerticalBase, VerticalConfig
from victor.core.verticals.protocols import (
    MiddlewareProtocol,
    ToolDependencyProviderProtocol,
)

# Phase 3: Import framework capabilities
from victor.framework.capabilities import (
    FileOperationsCapability,
    PromptContributionCapability,
)


class {vertical_class_name}(VerticalBase):
    """{description}.

    This vertical specializes in {capabilities_lower}.

    Example:
        from victor.{vertical_name} import {vertical_class_name}

        # Get vertical configuration
        config = {vertical_class_name}.get_config()

        # Create agent with this vertical
        agent = await Agent.create(
            tools=config.tools,
            vertical={vertical_class_name},
        )
    """

    name = "{vertical_name}"
    description = "{description}"
    version = "{version}"

    # =========================================================================
    # Phase 3: Framework Capabilities
    # =========================================================================
    # Reuse framework capabilities to reduce code duplication

    # Framework file operations capability (read, write, edit, grep)
    _file_ops = FileOperationsCapability()

    # Framework prompt contributions
    _prompt_contrib = PromptContributionCapability()

    # =========================================================================
    # Core Configuration
    # =========================================================================

    @classmethod
    def get_tools(cls) -> List[str]:
        """Get tools optimized for {vertical_name} tasks.

        Phase 3: Uses framework FileOperationsCapability for common file operations.

        Uses canonical tool names from victor.tools.tool_names.

        Returns:
            List of tool names including {vertical_name}-specific tools.
        """
        from victor.tools.tool_names import ToolNames

        # Start with framework file operations
        tools = cls._file_ops.get_tool_list()

        # Add {vertical_name}-specific tools
        tools.extend({tools_list})

        return tools

    @classmethod
    def get_system_prompt(cls) -> str:
        """Get {vertical_name}-focused system prompt.

        Returns:
            System prompt optimized for {vertical_name} tasks.
        """
        return """{system_prompt}"""

    @classmethod
    def get_stages(cls) -> Dict[str, StageDefinition]:
        """Get {vertical_name}-specific stage definitions.

        Uses canonical tool names from victor.tools.tool_names.

        Returns:
            Stage definitions optimized for {vertical_name} workflow.
        """
        from victor.tools.tool_names import ToolNames

        return {stages_dict}

    @classmethod
    def customize_config(cls, config: VerticalConfig) -> VerticalConfig:
        """Add {vertical_name}-specific configuration.

        Args:
            config: Base configuration.

        Returns:
            Customized configuration.
        """
        config.metadata["vertical_name"] = cls.name
        config.metadata["vertical_version"] = cls.version
        {custom_config}
        return config

    # =========================================================================
    # Extension Protocol Methods
    # =========================================================================
    # Most extension getters are auto-generated by VerticalExtensionLoaderMeta.
    # Only override for custom logic.

    @classmethod
    def get_middleware(cls) -> List[MiddlewareProtocol]:
        """Get {vertical_name}-specific middleware (cached).

        Returns:
            List of middleware implementations
        """
        {middleware_code}
        return cls._get_cached_extension("middleware", _create_middleware)

    @classmethod
    def get_tool_dependency_provider(cls):
        """Get {vertical_name} tool dependency provider (cached).

        Returns:
            Tool dependency provider
        """
        def _create():
            from victor.core.tool_dependency_loader import create_vertical_tool_dependency_provider
            return create_vertical_tool_dependency_provider("{vertical_name}")

        return cls._get_cached_extension("tool_dependency_provider", _create)

    @classmethod
    def get_handlers(cls) -> Dict[str, Any]:
        """Get compute handlers for {vertical_name} workflows.

        Returns:
            Dict mapping handler names to handler instances
        """
        from victor.{vertical_name}.handlers import HANDLERS
        return HANDLERS

    @classmethod
    def get_capability_configs(cls) -> Dict[str, Any]:
        """Get {vertical_name} capability configurations for centralized storage.

        Returns:
            Dict with {vertical_name} capability configurations
        """
        from victor.{vertical_name}.capabilities import get_capability_configs
        return get_capability_configs()

    # NOTE: The following getters are auto-generated by VerticalExtensionLoaderMeta:
    # - get_safety_extension()
    # - get_prompt_contributor()
    # - get_mode_config_provider()
    # - get_workflow_provider()
    # - get_tiered_tools()
    # - get_rl_config_provider()
    # - get_rl_hooks()
    # - get_team_spec_provider()
    # - get_capability_provider()
    #
    # get_extensions() is inherited from VerticalBase with full caching support.
    # To clear all caches, use cls.clear_config_cache().


__all__ = ["{vertical_class_name}"]
'''

PROMPTS_PY_TEMPLATE = '''# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""{vertical_class_name}-specific prompt contributions.

This module provides task type hints and system prompt sections
specific to {vertical_name} tasks. These are injected into
the framework via the PromptContributorProtocol.
"""

from __future__ import annotations

from typing import Dict

from victor.core.verticals.protocols import PromptContributorProtocol, TaskTypeHint


# Task-type-specific prompt hints for {vertical_name} tasks
{vertical_name_upper}_TASK_TYPE_HINTS: Dict[str, TaskTypeHint] = {{
{prompt_hints}
}}


# {vertical_class_name}-specific grounding rules
{vertical_name_upper}_GROUNDING_RULES = """
{grounding_rules}
""".strip()


# {vertical_class_name}-specific system prompt section
{vertical_name_upper}_SYSTEM_PROMPT_SECTION = """
{system_prompt_section}
""".strip()


class {vertical_class_name}PromptContributor(PromptContributorProtocol):
    """Prompt contributor for {vertical_name} vertical.

    Provides {vertical_name}-specific task type hints and system prompt sections
    for integration with the framework's prompt builder.
    """

    def __init__(self, use_extended_grounding: bool = False):
        """Initialize the prompt contributor.

        Args:
            use_extended_grounding: Whether to use extended grounding rules
        """
        self._use_extended_grounding = use_extended_grounding

    def get_task_type_hints(self) -> Dict[str, TaskTypeHint]:
        """Get {vertical_name}-specific task type hints.

        Returns:
            Dict mapping task types to their hints
        """
        return {vertical_name_upper}_TASK_TYPE_HINTS.copy()

    def get_system_prompt_section(self) -> str:
        """Get {vertical_name}-specific system prompt section.

        Returns:
            System prompt text for {vertical_name} tasks
        """
        return {vertical_name_upper}_SYSTEM_PROMPT_SECTION

    def get_grounding_rules(self) -> str:
        """Get {vertical_name}-specific grounding rules.

        Returns:
            Grounding rules text
        """
        if self._use_extended_grounding:
            return {vertical_name_upper}_GROUNDING_RULES
        return {vertical_name_upper}_GROUNDING_RULES

    def get_priority(self) -> int:
        """Get priority for prompt section ordering.

        Returns:
            Priority value
        """
        return 10


__all__ = [
    "{vertical_class_name}PromptContributor",
    "{vertical_name_upper}_TASK_TYPE_HINTS",
    "{vertical_name_upper}_GROUNDING_RULES",
    "{vertical_name_upper}_SYSTEM_PROMPT_SECTION",
]
'''

SAFETY_PY_TEMPLATE = '''# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""{vertical_class_name}-specific safety patterns.

This module defines dangerous operation patterns specific to
{vertical_name} tasks.
"""

from __future__ import annotations

from typing import Dict, List

from victor.core.verticals.protocols import SafetyExtensionProtocol, SafetyPattern


# {vertical_name}-specific dangerous patterns
{vertical_name_upper}_PATTERNS: List[SafetyPattern] = [
{safety_patterns}
]


class {vertical_class_name}SafetyExtension(SafetyExtensionProtocol):
    """Safety extension for {vertical_name} vertical.

    Provides {vertical_name}-specific dangerous operation patterns.
    """

    def __init__(self):
        """Initialize the safety extension."""
        pass

    def get_bash_patterns(self) -> List[SafetyPattern]:
        """Get {vertical_name}-specific bash command patterns.

        Returns:
            List of safety patterns for dangerous bash commands
        """
        return {vertical_name_upper}_PATTERNS

    def get_file_patterns(self) -> List[SafetyPattern]:
        """Get {vertical_name}-specific file operation patterns.

        Returns:
            List of safety patterns for file operations
        """
        return []

    def get_tool_restrictions(self) -> Dict[str, List[str]]:
        """Get tool-specific argument restrictions.

        Returns:
            Dict mapping tool names to restricted argument patterns
        """
        return {{}}

    def get_category(self) -> str:
        """Get the category name for these patterns.

        Returns:
            Category identifier
        """
        return "{vertical_name}"


__all__ = [
    "{vertical_class_name}SafetyExtension",
    "{vertical_name_upper}_PATTERNS",
]
'''

ESCAPE_HATCHES_PY_TEMPLATE = '''# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Escape hatches for {vertical_class_name} YAML workflows.

Complex conditions and transforms that cannot be expressed in YAML.
These are registered with the YAML workflow loader for use in condition nodes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


# =============================================================================
# Condition Functions
# =============================================================================


def example_condition(ctx: Dict[str, Any]) -> str:
    """Example condition function.

    Args:
        ctx: Workflow context

    Returns:
        Branch identifier
    """
    # Implement condition logic here
    return "default"


# =============================================================================
# Transform Functions
# =============================================================================


def example_transform(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Example transform function.

    Args:
        ctx: Workflow context

    Returns:
        Transformed context
    """
    # Implement transform logic here
    return ctx


# =============================================================================
# Registry Exports
# =============================================================================

# Conditions available in YAML workflows
CONDITIONS = {{
    "example_condition": example_condition,
}}

# Transforms available in YAML workflows
TRANSFORMS = {{
    "example_transform": example_transform,
}}

__all__ = [
    # Conditions
    "example_condition",
    # Transforms
    "example_transform",
    # Registries
    "CONDITIONS",
    "TRANSFORMS",
]
'''

HANDLERS_PY_TEMPLATE = '''# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Workflow handlers for {vertical_class_name}.

This module provides compute handlers for YAML workflow execution.
"""

from __future__ import annotations

from typing import Any, Dict


# =============================================================================
# Handler Functions
# =============================================================================


async def example_handler(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Example workflow handler.

    Args:
        ctx: Workflow context

    Returns:
        Handler result
    """
    # Implement handler logic here
    return {{"status": "success"}}


# =============================================================================
# Handler Registry
# =============================================================================

HANDLERS: Dict[str, Any] = {{
    "example_handler": example_handler,
}}

__all__ = [
    "example_handler",
    "HANDLERS",
]
'''

TEAMS_PY_TEMPLATE = '''# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Team formations for {vertical_class_name}.

This module provides persona definitions and team configurations
for multi-agent coordination.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class {vertical_class_name}Persona:
    """Persona definition for {vertical_name} team members."""

    name: str
    role: str
    expertise: str
    communication_style: str
    decision_approach: str


# =============================================================================
# Persona Definitions
# =============================================================================

{vertical_name_upper}_PERSONAS: Dict[str, {vertical_class_name}Persona] = {{
    "specialist": {vertical_class_name}Persona(
        name="{vertical_name} Specialist",
        role="specialist",
        expertise="{vertical_name} tasks",
        communication_style="direct",
        decision_approach="analytical",
    ),
}}

__all__ = [
    "{vertical_class_name}Persona",
    "{vertical_name_upper}_PERSONAS",
]
'''

CAPABILITIES_PY_TEMPLATE = '''# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Capability configurations for {vertical_class_name}.

This module provides capability configurations for the vertical.
"""

from __future__ import annotations

from typing import Any, Dict


def get_capability_configs() -> Dict[str, Any]:
    """Get {vertical_name} capability configurations.

    Returns:
        Dict with capability configurations
    """
    return {{
        # Add capability-specific configurations here
    }}


__all__ = [
    "get_capability_configs",
]
'''

INIT_PY_TEMPLATE = '''"""{vertical_class_name} - {description}.

This vertical provides specialized capabilities for {vertical_name} tasks.
"""

from victor.{vertical_name}.assistant import {vertical_class_name}

__all__ = ["{vertical_class_name}"
'''

VERTICAL_YAML_TEMPLATE = '''# Vertical Configuration for {vertical_name}
# Generated from template by scripts/generate_vertical.py

metadata:
  name: {vertical_name}
  description: {description}
  version: {version}
  category: {category}
  tags: {tags}

core:
  tools:
    list: {tools_list}

  system_prompt:
    source: inline
    text: |
      {system_prompt_indented}

  stages: {stages_yaml}
'''


# =============================================================================
# Generator Class
# =============================================================================


class VerticalGenerator:
    """Generate vertical implementation from template.

    Attributes:
        template: VerticalTemplate to generate from
        output_dir: Output directory for generated files
        overwrite: Whether to overwrite existing files
    """

    def __init__(
        self,
        template: VerticalTemplate,
        output_dir: str | Path,
        *,
        overwrite: bool = False,
    ):
        """Initialize the generator.

        Args:
            template: Template to generate from
            output_dir: Output directory path
            overwrite: If True, overwrite existing files
        """
        self.template = template
        self.output_dir = Path(output_dir)
        self.overwrite = overwrite

    def generate(self) -> bool:
        """Generate all vertical files.

        Returns:
            True if generation succeeded
        """
        logger.info(f"Generating vertical '{self.template.metadata.name}' to {self.output_dir}")

        # Validate template first
        errors = self.template.validate()
        if errors:
            logger.error(f"Template validation failed: {', '.join(errors)}")
            return False

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate files
        try:
            self._generate_assistant_py()
            self._generate_prompts_py()
            self._generate_safety_py()
            self._generate_escape_hatches_py()
            self._generate_handlers_py()
            self._generate_teams_py()
            self._generate_capabilities_py()
            self._generate_init_py()
            self._generate_vertical_yaml()

            logger.info(f"Successfully generated vertical '{self.template.metadata.name}'")
            return True

        except Exception as e:
            logger.error(f"Error generating vertical: {e}")
            return False

    def _generate_assistant_py(self) -> None:
        """Generate assistant.py file."""
        vertical_name = self.template.metadata.name
        class_name = to_class_name(vertical_name)
        description = self.template.metadata.description
        version = self.template.metadata.version

        # Format tools list
        tools_list = json.dumps(self.template.tools, indent=12)

        # Format stages dict
        stages_dict = self._format_stages_for_python()

        # Format system prompt
        system_prompt = self.template.system_prompt

        # Format middleware
        middleware_code = self._format_middleware()

        # Format custom config
        custom_config = self._format_custom_config()

        content = ASSISTANT_PY_TEMPLATE.format(
            vertical_name=vertical_name,
            vertical_class_name=class_name,
            description=description,
            version=version,
            capabilities_lower=description.lower(),
            tools_list=tools_list,
            system_prompt=system_prompt,
            stages_dict=stages_dict,
            middleware_code=middleware_code,
            custom_config=custom_config,
        )

        self._write_file("assistant.py", content)

    def _generate_prompts_py(self) -> None:
        """Generate prompts.py file."""
        vertical_name = self.template.metadata.name
        class_name = to_class_name(vertical_name)
        vertical_name_upper = vertical_name.upper()

        # Format prompt hints
        prompt_hints = self._format_prompt_hints()

        # Format grounding rules and system prompt section
        grounding_rules = self.template.custom_config.get("grounding_rules", "Base all responses on tool output.")
        system_prompt_section = self.template.custom_config.get("system_prompt_section", "Follow best practices.")

        content = PROMPTS_PY_TEMPLATE.format(
            vertical_name=vertical_name,
            vertical_class_name=class_name,
            vertical_name_upper=vertical_name_upper,
            prompt_hints=prompt_hints,
            grounding_rules=grounding_rules,
            system_prompt_section=system_prompt_section,
        )

        self._write_file("prompts.py", content)

    def _generate_safety_py(self) -> None:
        """Generate safety.py file."""
        vertical_name = self.template.metadata.name
        class_name = to_class_name(vertical_name)
        vertical_name_upper = vertical_name.upper()

        # Format safety patterns
        safety_patterns = self._format_safety_patterns()

        content = SAFETY_PY_TEMPLATE.format(
            vertical_name=vertical_name,
            vertical_class_name=class_name,
            vertical_name_upper=vertical_name_upper,
            safety_patterns=safety_patterns,
        )

        self._write_file("safety.py", content)

    def _generate_escape_hatches_py(self) -> None:
        """Generate escape_hatches.py file."""
        vertical_name = self.template.metadata.name
        class_name = to_class_name(vertical_name)

        content = ESCAPE_HATCHES_PY_TEMPLATE.format(
            vertical_name=vertical_name,
            vertical_class_name=class_name,
        )

        self._write_file("escape_hatches.py", content)

    def _generate_handlers_py(self) -> None:
        """Generate handlers.py file."""
        vertical_name = self.template.metadata.name
        class_name = to_class_name(vertical_name)

        content = HANDLERS_PY_TEMPLATE.format(
            vertical_name=vertical_name,
            vertical_class_name=class_name,
        )

        self._write_file("handlers.py", content)

    def _generate_teams_py(self) -> None:
        """Generate teams.py file."""
        vertical_name = self.template.metadata.name
        class_name = to_class_name(vertical_name)
        vertical_name_upper = vertical_name.upper()

        content = TEAMS_PY_TEMPLATE.format(
            vertical_name=vertical_name,
            vertical_class_name=class_name,
            vertical_name_upper=vertical_name_upper,
        )

        self._write_file("teams.py", content)

    def _generate_capabilities_py(self) -> None:
        """Generate capabilities.py file."""
        vertical_name = self.template.metadata.name
        class_name = to_class_name(vertical_name)

        content = CAPABILITIES_PY_TEMPLATE.format(
            vertical_name=vertical_name,
            vertical_class_name=class_name,
        )

        self._write_file("capabilities.py", content)

    def _generate_init_py(self) -> None:
        """Generate __init__.py file."""
        vertical_name = self.template.metadata.name
        class_name = to_class_name(vertical_name)
        description = self.template.metadata.description

        content = INIT_PY_TEMPLATE.format(
            vertical_name=vertical_name,
            vertical_class_name=class_name,
            description=description,
        )

        self._write_file("__init__.py", content)

    def _generate_vertical_yaml(self) -> None:
        """Generate config/vertical.yaml file."""
        config_dir = self.output_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)

        vertical_name = self.template.metadata.name
        class_name = to_class_name(vertical_name)
        description = self.template.metadata.description
        version = self.template.metadata.version

        # Format tools list for YAML
        tools_list = json.dumps(self.template.tools)

        # Indent system prompt for YAML
        system_prompt_indented = "\n      ".join(self.template.system_prompt.split("\n"))

        content = VERTICAL_YAML_TEMPLATE.format(
            vertical_name=vertical_name,
            vertical_class_name=class_name,
            description=description,
            version=version,
            category=self.template.metadata.category,
            tags=json.dumps(self.template.metadata.tags),
            tools_list=tools_list,
            system_prompt_indented=system_prompt_indented,
            stages_yaml="",  # TODO: Implement stages YAML formatting
        )

        yaml_path = config_dir / "vertical.yaml"
        self._write_file(yaml_path, content)

    def _write_file(self, path: str | Path, content: str) -> None:
        """Write content to file, checking for existence.

        Args:
            path: File path (relative to output_dir)
            content: File content
        """
        file_path = self.output_dir / path

        if file_path.exists() and not self.overwrite:
            logger.warning(f"File already exists, skipping: {file_path}")
            return

        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            f.write(content)

        logger.info(f"Generated: {file_path}")

    def _format_stages_for_python(self) -> str:
        """Format stages for Python code generation."""
        lines = ["{"]

        for stage_name, stage in self.template.stages.items():
            tools_str = json.dumps(list(stage.tools)) if stage.tools else "set()"
            keywords_str = json.dumps(stage.keywords)
            next_stages_str = json.dumps(list(stage.next_stages)) if stage.next_stages else "set()"

            lines.append(f'        "{stage_name}": StageDefinition(')
            lines.append(f'            name="{stage.name}",')
            lines.append(f'            description="{stage.description}",')
            lines.append(f'            tools={tools_str},')
            lines.append(f'            keywords={keywords_str},')
            lines.append(f'            next_stages={next_stages_str},')
            lines.append("        ),")

        lines.append("    }")
        return "\n".join(lines)

    def _format_middleware(self) -> str:
        """Format middleware for code generation."""
        if not self.template.extensions.middleware:
            return "        def _create_middleware() -> List[MiddlewareProtocol]:\n            return []"

        lines = ["        def _create_middleware() -> List[MiddlewareProtocol]:"]
        lines.append(f"            from victor.{self.template.metadata.name}.middleware import (")

        for i, mw in enumerate(self.template.extensions.middleware):
            if i > 0:
                lines.append("                ,")
            lines.append(f"                {mw.class_name},")

        lines.append("            )")
        lines.append("")
        lines.append("            return [")

        for mw in self.template.extensions.middleware:
            args = ", ".join(f"{k}={v}" for k, v in mw.config.items())
            lines.append(f"                {mw.class_name}({args})," if args else f"                {mw.class_name}(),")

        lines.append("            ]")

        return "\n".join(lines)

    def _format_custom_config(self) -> str:
        """Format custom config for code generation."""
        if not self.template.custom_config:
            return "        config.metadata['category'] = '" + self.template.metadata.category + "'"

        lines = []
        for key, value in self.template.custom_config.items():
            if isinstance(value, str):
                lines.append(f'        config.metadata["{key}"] = "{value}"')
            elif isinstance(value, (list, dict)):
                lines.append(f'        config.metadata["{key}"] = {json.dumps(value)}')
            else:
                lines.append(f'        config.metadata["{key}"] = {value}')

        return "\n".join(lines) if lines else "        pass"

    def _format_prompt_hints(self) -> str:
        """Format prompt hints for code generation."""
        if not self.template.extensions.prompt_hints:
            return "    # No task type hints defined"

        lines = []
        for hint in self.template.extensions.prompt_hints:
            tools_str = json.dumps(hint.priority_tools)
            lines.append(f'    "{hint.task_type}": TaskTypeHint(')
            lines.append(f'        task_type="{hint.task_type}",')
            lines.append(f'        hint={json.dumps(hint.hint)},')
            lines.append(f'        tool_budget={hint.tool_budget},')
            lines.append(f'        priority_tools={tools_str},')
            lines.append("    ),")

        return "\n".join(lines)

    def _format_safety_patterns(self) -> str:
        """Format safety patterns for code generation."""
        if not self.template.extensions.safety_patterns:
            return "    # No safety patterns defined\n"

        lines = []
        for pattern in self.template.extensions.safety_patterns:
            lines.append("    SafetyPattern(")
            lines.append(f'        name="{pattern.name}",')
            lines.append(f'        pattern=r"{pattern.pattern}",')
            lines.append(f'        description="{pattern.description}",')
            lines.append(f'        severity="{pattern.severity}",')
            lines.append(f'        category="{pattern.category}",')
            lines.append("    ),")

        return "\n".join(lines)


# =============================================================================
# Utility Functions
# =============================================================================


def to_class_name(vertical_name: str) -> str:
    """Convert vertical name to class name.

    Args:
        vertical_name: Vertical name (e.g., "coding", "security_audit")

    Returns:
        Class name (e.g., "CodingAssistant", "SecurityAuditAssistant")
    """
    parts = vertical_name.split("_")
    return "".join(p.capitalize() for p in parts) + "Assistant"


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate vertical implementations from templates"
    )

    parser.add_argument(
        "--template",
        type=str,
        help="Template name or path to template YAML file",
    )
    parser.add_argument(
        "--extract-from",
        type=str,
        help="Extract template from existing vertical",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for generated files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--list-templates",
        action="store_true",
        help="List available templates",
    )
    parser.add_argument(
        "--validate",
        type=str,
        help="Validate a template YAML file",
    )

    args = parser.parse_args()

    # List templates
    if args.list_templates:
        registry = VerticalTemplateRegistry.get_instance()
        templates = registry.list_names()
        print("Available templates:")
        for name in templates:
            print(f"  - {name}")
        return

    # Validate template
    if args.validate:
        registry = VerticalTemplateRegistry.get_instance()
        template = registry.load_from_yaml(args.validate)
        if template:
            errors = template.validate()
            if errors:
                print(f"Validation errors:")
                for error in errors:
                    print(f"  - {error}")
                return 1
            else:
                print("Template is valid!")
                return 0
        else:
            print(f"Failed to load template from {args.validate}")
            return 1

    # Generate from template
    if args.template:
        registry = VerticalTemplateRegistry.get_instance()

        # Try loading as file path first
        template_path = Path(args.template)
        if template_path.exists():
            template = registry.load_from_yaml(template_path)
        else:
            # Try loading from registry
            template = registry.get(args.template)

        if not template:
            print(f"Template not found: {args.template}")
            return 1

        generator = VerticalGenerator(
            template,
            args.output,
            overwrite=args.overwrite,
        )

        success = generator.generate()
        return 0 if success else 1

    # Extract from existing vertical
    if args.extract_from:
        print("Extract from existing vertical not yet implemented")
        return 1

    # No action specified
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
