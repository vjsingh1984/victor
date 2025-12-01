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

"""Configuration loading and validation for the agent."""

import logging
import os
from typing import Any, Dict, List, Optional, Set


from victor.config.settings import Settings
from victor.tools.base import ToolRegistry

logger = logging.getLogger(__name__)


# Core tools that should generally remain enabled for basic functionality
CORE_TOOLS: Set[str] = {
    "read_file",
    "write_file",
    "list_directory",
    "execute_bash",
}


class ConfigLoader:
    """Loads and validates agent configuration.

    Responsibilities:
    - Load tool configurations from profiles.yaml
    - Validate tool names against registry
    - Warn about disabled core tools
    - Resolve environment variables in configuration
    """

    def __init__(self, settings: Settings):
        """Initialize config loader.

        Args:
            settings: Application settings
        """
        self.settings = settings

    def load_tool_config(self, tool_registry: ToolRegistry) -> None:
        """Load and apply tool configurations.

        Loads tool enable/disable states from the 'tools' section in profiles.yaml.

        Expected formats:
        ```yaml
        tools:
          enabled:
            - read_file
            - write_file
          disabled:
            - code_review
        ```

        Or:
        ```yaml
        tools:
          code_review:
            enabled: false
        ```

        Args:
            tool_registry: Registry to configure
        """
        try:
            tool_config = self.settings.load_tool_config()
            if not tool_config:
                return

            registered_tools = {tool.name for tool in tool_registry.list_tools(only_enabled=False)}

            # Format 1: Lists of enabled/disabled tools
            self._apply_enabled_list(tool_config, tool_registry, registered_tools)
            self._apply_disabled_list(tool_config, tool_registry, registered_tools)

            # Format 2: Individual tool settings
            self._apply_individual_settings(tool_config, tool_registry, registered_tools)

            # Log final state
            self._log_tool_states(tool_registry)

        except Exception as e:
            logger.error(f"Failed to load tool configuration: {e}")

    def _apply_enabled_list(
        self, config: Dict[str, Any], registry: ToolRegistry, registered: Set[str]
    ) -> None:
        """Apply enabled list from configuration."""
        if "enabled" not in config:
            return

        enabled_tools = config.get("enabled", [])

        # Validate tool names
        invalid = [t for t in enabled_tools if t not in registered]
        if invalid:
            logger.warning(
                f"Invalid tool names in 'enabled' list: {', '.join(invalid)}. "
                f"Available: {', '.join(sorted(registered))}"
            )

        # Warn about missing core tools
        missing_core = CORE_TOOLS - set(enabled_tools)
        if missing_core:
            logger.warning(
                f"'enabled' list missing core tools: {', '.join(missing_core)}. "
                f"This may limit agent functionality."
            )

        # Disable all, then enable specified
        for tool in registry.list_tools(only_enabled=False):
            registry.disable_tool(tool.name)

        for tool_name in enabled_tools:
            if tool_name in registered:
                registry.enable_tool(tool_name)

    def _apply_disabled_list(
        self, config: Dict[str, Any], registry: ToolRegistry, registered: Set[str]
    ) -> None:
        """Apply disabled list from configuration."""
        if "disabled" not in config:
            return

        disabled_tools = config.get("disabled", [])

        # Validate tool names
        invalid = [t for t in disabled_tools if t not in registered]
        if invalid:
            logger.warning(
                f"Invalid tool names in 'disabled' list: {', '.join(invalid)}. "
                f"Available: {', '.join(sorted(registered))}"
            )

        # Warn about disabling core tools
        disabled_core = CORE_TOOLS & set(disabled_tools)
        if disabled_core:
            logger.warning(
                f"Disabling core tools: {', '.join(disabled_core)}. "
                f"This may limit agent functionality."
            )

        for tool_name in disabled_tools:
            if tool_name in registered:
                registry.disable_tool(tool_name)

    def _apply_individual_settings(
        self, config: Dict[str, Any], registry: ToolRegistry, registered: Set[str]
    ) -> None:
        """Apply individual tool settings from configuration."""
        for tool_name, tool_config in config.items():
            if not isinstance(tool_config, dict) or "enabled" not in tool_config:
                continue

            if tool_name not in registered:
                logger.warning(
                    f"Invalid tool name: '{tool_name}'. "
                    f"Available: {', '.join(sorted(registered))}"
                )
                continue

            if tool_config["enabled"]:
                registry.enable_tool(tool_name)
            else:
                registry.disable_tool(tool_name)
                if tool_name in CORE_TOOLS:
                    logger.warning(
                        f"Disabling core tool '{tool_name}'. "
                        f"This may limit agent functionality."
                    )

    def _log_tool_states(self, registry: ToolRegistry) -> None:
        """Log the final tool enabled/disabled states."""
        disabled = [name for name, enabled in registry.get_tool_states().items() if not enabled]
        if disabled:
            logger.info(f"Disabled tools: {', '.join(sorted(disabled))}")

    @staticmethod
    def resolve_env_vars(value: str) -> str:
        """Resolve environment variables in a string value.

        Supports ${VAR} and ${VAR:-default} syntax.

        Args:
            value: String that may contain env var references

        Returns:
            String with env vars resolved
        """
        import re

        def replacer(match: re.Match) -> str:
            var_expr = match.group(1)
            if ":-" in var_expr:
                var_name, default = var_expr.split(":-", 1)
                return os.environ.get(var_name, default)
            return os.environ.get(var_expr, "")

        return re.sub(r"\$\{([^}]+)\}", replacer, value)

    @staticmethod
    def resolve_endpoint_list(
        endpoints: List[str], env_var_prefix: Optional[str] = None
    ) -> List[str]:
        """Resolve and filter endpoint list.

        Args:
            endpoints: List of endpoint URLs (may contain env vars)
            env_var_prefix: Optional env var to prepend additional endpoints

        Returns:
            List of resolved, non-empty endpoint URLs
        """
        resolved: List[str] = []

        # Add env var endpoints first if specified
        if env_var_prefix:
            env_endpoints = os.environ.get(env_var_prefix, "")
            if env_endpoints:
                resolved.extend(ep.strip() for ep in env_endpoints.split(",") if ep.strip())

        # Resolve each configured endpoint
        for endpoint in endpoints:
            resolved_ep = ConfigLoader.resolve_env_vars(endpoint)
            if resolved_ep.strip():
                resolved.append(resolved_ep.strip())

        return resolved
