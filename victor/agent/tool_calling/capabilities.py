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

"""
Capability loader for model capabilities from YAML configuration.

Loads model capabilities from model_capabilities.yaml and provides
methods for querying capabilities by provider and model name.
"""

import fnmatch
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from victor.agent.tool_calling.base import ToolCallingCapabilities, ToolCallFormat

logger = logging.getLogger(__name__)


class ModelCapabilityLoader:
    """Loads and provides model capabilities from YAML configuration.

    Supports hierarchical capability resolution:
    1. Model-specific overrides (pattern matching)
    2. Provider-level defaults
    3. Global defaults

    Usage:
        loader = ModelCapabilityLoader()
        caps = loader.get_capabilities("ollama", "llama3.1:8b")
    """

    _instance: Optional["ModelCapabilityLoader"] = None
    _config: Optional[Dict[str, Any]] = None

    def __new__(cls) -> "ModelCapabilityLoader":
        """Singleton pattern for efficiency."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize loader and load configuration."""
        if self._config is None:
            self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        config_path = Path(__file__).parent.parent.parent / "config" / "model_capabilities.yaml"

        if not config_path.exists():
            logger.warning(f"Model capabilities config not found at {config_path}")
            self._config = {"defaults": {}, "providers": {}, "models": {}}
            return

        try:
            with open(config_path) as f:
                self._config = yaml.safe_load(f) or {}
            logger.debug(f"Loaded model capabilities from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load model capabilities: {e}")
            self._config = {"defaults": {}, "providers": {}, "models": {}}

    def reload(self) -> None:
        """Force reload configuration from file."""
        self._load_config()

    def get_capabilities(
        self,
        provider: str,
        model: str = "",
        format_hint: Optional[ToolCallFormat] = None,
    ) -> ToolCallingCapabilities:
        """Get capabilities for a provider/model combination.

        Resolution order (later overrides earlier):
        1. Global defaults
        2. Provider defaults
        3. Model-specific overrides (pattern matching)

        Args:
            provider: Provider name (ollama, lmstudio, vllm, etc.)
            model: Model name/identifier
            format_hint: Optional format hint to set

        Returns:
            ToolCallingCapabilities with resolved values
        """
        # Ensure config is loaded
        if self._config is None:
            self._load_config()

        # _config is guaranteed non-None after _load_config
        config_dict = self._config or {}

        # Start with global defaults
        config = dict(config_dict.get("defaults", {}))

        # Apply provider defaults
        provider_lower = provider.lower()
        provider_config = config_dict.get("providers", {}).get(provider_lower, {})
        config.update(provider_config)

        # Apply model-specific overrides
        if model:
            model_lower = model.lower()
            models_config = config_dict.get("models", {})

            # Find matching patterns (more specific patterns take precedence)
            matching_patterns = []
            for pattern, model_overrides in models_config.items():
                # Convert glob pattern to match
                if self._matches_pattern(model_lower, pattern.lower()):
                    # Score by specificity (longer patterns are more specific)
                    specificity = len(pattern.replace("*", ""))
                    matching_patterns.append((specificity, pattern, model_overrides))

            # Apply in order of specificity
            matching_patterns.sort(key=lambda x: x[0])
            for _, pattern, overrides in matching_patterns:
                logger.debug(f"Applying model pattern '{pattern}' for {model}")
                config.update(overrides)

        # Convert to ToolCallingCapabilities
        return self._config_to_capabilities(config, format_hint)

    def _matches_pattern(self, model_name: str, pattern: str) -> bool:
        """Check if model name matches a pattern.

        Supports:
        - Glob patterns (*, ?)
        - Prefix matching (llama3.1* matches llama3.1:8b)
        """
        # Try fnmatch first
        if fnmatch.fnmatch(model_name, pattern):
            return True

        # Check for prefix match (pattern without *)
        prefix = pattern.rstrip("*")
        if prefix and model_name.startswith(prefix):
            return True

        return False

    def _config_to_capabilities(
        self,
        config: Dict[str, Any],
        format_hint: Optional[ToolCallFormat] = None,
    ) -> ToolCallingCapabilities:
        """Convert config dict to ToolCallingCapabilities object."""
        # Map config keys to capability fields
        return ToolCallingCapabilities(
            native_tool_calls=config.get("native_tool_calls", False),
            streaming_tool_calls=config.get("streaming_tool_calls", False),
            parallel_tool_calls=config.get("parallel_tool_calls", False),
            tool_choice_param=config.get("tool_choice_param", False),
            json_fallback_parsing=config.get("json_fallback_parsing", True),
            xml_fallback_parsing=config.get("xml_fallback_parsing", True),
            thinking_mode=config.get("thinking_mode", False),
            requires_strict_prompting=config.get("requires_strict_prompting", True),
            tool_call_format=format_hint or ToolCallFormat.UNKNOWN,
            argument_format=config.get("argument_format", "json"),
            recommended_max_tools=config.get("recommended_max_tools", 20),
            recommended_tool_budget=config.get("recommended_tool_budget", 12),
        )

    def get_provider_names(self) -> list:
        """Get list of configured provider names."""
        config_dict = self._config or {}
        return list(config_dict.get("providers", {}).keys())

    def get_model_patterns(self) -> list:
        """Get list of configured model patterns."""
        config_dict = self._config or {}
        return list(config_dict.get("models", {}).keys())


# Module-level convenience function
def get_model_capabilities(
    provider: str,
    model: str = "",
    format_hint: Optional[ToolCallFormat] = None,
) -> ToolCallingCapabilities:
    """Get capabilities for a provider/model combination.

    Convenience function that uses the singleton loader.

    Args:
        provider: Provider name
        model: Model name
        format_hint: Optional format hint

    Returns:
        ToolCallingCapabilities
    """
    loader = ModelCapabilityLoader()
    return loader.get_capabilities(provider, model, format_hint)
