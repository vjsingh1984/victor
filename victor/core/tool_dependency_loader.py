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

"""YAML loader for tool dependency configurations.

This module provides functionality to load, validate, and convert YAML-based
tool dependency configurations into ToolDependencyConfig objects compatible
with BaseToolDependencyProvider.

Design Principles (SOLID):
    - SRP: Loader is separate from schema validation (tool_dependency_schema.py)
    - OCP: Loader can be extended without modification via hooks
    - DIP: Depends on abstractions (ToolDependencyConfig, ToolDependencySpec)

Features:
    - Load YAML files with schema validation
    - Convert validated specs to ToolDependencyConfig
    - Optional tool name canonicalization
    - Factory function for creating providers directly from YAML
    - Support for caching loaded configurations

Example:
    from pathlib import Path
    from victor.core.tool_dependency_loader import (
        load_tool_dependency_yaml,
        create_tool_dependency_provider,
    )

    # Load and create provider in one step
    provider = create_tool_dependency_provider(
        Path("victor/coding/tool_dependencies.yaml")
    )

    # Or load config separately for inspection
    config = load_tool_dependency_yaml(
        Path("victor/coding/tool_dependencies.yaml"),
        canonicalize=True,
    )
    print(config.required_tools)  # {"read", "write", "edit", "ls", "grep"}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml

from victor.core.tool_dependency_base import BaseToolDependencyProvider, ToolDependencyConfig
from victor.core.tool_dependency_schema import ToolDependencySpec
from victor.core.tool_types import ToolDependency

logger = logging.getLogger(__name__)


class ToolDependencyLoadError(Exception):
    """Exception raised when loading tool dependency YAML fails.

    Attributes:
        path: Path to the YAML file that failed to load.
        message: Detailed error message.
        cause: Original exception that caused the failure.
    """

    def __init__(
        self,
        path: Path,
        message: str,
        cause: Optional[Exception] = None,
    ):
        """Initialize the error.

        Args:
            path: Path to the YAML file that failed to load.
            message: Detailed error message.
            cause: Original exception that caused the failure.
        """
        self.path = path
        self.message = message
        self.cause = cause
        super().__init__(f"Failed to load {path}: {message}")


class ToolDependencyLoader:
    """Loader for YAML-based tool dependency configurations.

    This class provides methods to load, validate, and convert YAML
    configurations into ToolDependencyConfig objects.

    Attributes:
        canonicalize: Whether to canonicalize tool names during loading.

    Example:
        loader = ToolDependencyLoader(canonicalize=True)
        config = loader.load(Path("tool_dependencies.yaml"))
    """

    def __init__(self, canonicalize: bool = True):
        """Initialize the loader.

        Args:
            canonicalize: Whether to convert tool names to canonical form.
                Defaults to True for consistency with RL Q-values.
        """
        self._canonicalize = canonicalize
        self._cache: Dict[Path, ToolDependencyConfig] = {}

    def load(
        self,
        yaml_path: Path,
        use_cache: bool = True,
    ) -> ToolDependencyConfig:
        """Load a tool dependency YAML file and convert to config.

        Args:
            yaml_path: Path to the YAML configuration file.
            use_cache: Whether to use cached result if available.

        Returns:
            ToolDependencyConfig ready for use with BaseToolDependencyProvider.

        Raises:
            ToolDependencyLoadError: If loading or validation fails.
        """
        yaml_path = yaml_path.resolve()

        # Check cache
        if use_cache and yaml_path in self._cache:
            logger.debug(f"Using cached config for {yaml_path}")
            return self._cache[yaml_path]

        # Load and validate
        spec = self._load_and_validate(yaml_path)

        # Convert to config
        config = self._convert_to_config(spec)

        # Cache result
        self._cache[yaml_path] = config

        logger.info(f"Loaded tool dependency config for '{spec.vertical}' from {yaml_path}")
        return config

    def load_from_string(self, yaml_content: str) -> ToolDependencyConfig:
        """Load a tool dependency config from a YAML string.

        Useful for testing or embedding configs in code.

        Args:
            yaml_content: YAML configuration as a string.

        Returns:
            ToolDependencyConfig ready for use with BaseToolDependencyProvider.

        Raises:
            ToolDependencyLoadError: If parsing or validation fails.
        """
        try:
            data = yaml.safe_load(yaml_content)
            spec = ToolDependencySpec.model_validate(data)
            return self._convert_to_config(spec)
        except yaml.YAMLError as e:
            raise ToolDependencyLoadError(
                Path("<string>"),
                f"YAML parsing error: {e}",
                cause=e,
            )
        except Exception as e:
            raise ToolDependencyLoadError(
                Path("<string>"),
                f"Validation error: {e}",
                cause=e,
            )

    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        self._cache.clear()

    def _load_and_validate(self, yaml_path: Path) -> ToolDependencySpec:
        """Load and validate YAML file.

        Args:
            yaml_path: Path to the YAML file.

        Returns:
            Validated ToolDependencySpec.

        Raises:
            ToolDependencyLoadError: If loading or validation fails.
        """
        if not yaml_path.exists():
            raise ToolDependencyLoadError(
                yaml_path,
                "File not found",
            )

        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ToolDependencyLoadError(
                yaml_path,
                f"YAML parsing error: {e}",
                cause=e,
            )

        if data is None:
            raise ToolDependencyLoadError(
                yaml_path,
                "Empty YAML file",
            )

        try:
            spec = ToolDependencySpec.model_validate(data)
        except Exception as e:
            raise ToolDependencyLoadError(
                yaml_path,
                f"Schema validation error: {e}",
                cause=e,
            )

        return spec

    def _convert_to_config(self, spec: ToolDependencySpec) -> ToolDependencyConfig:
        """Convert validated spec to ToolDependencyConfig.

        Args:
            spec: Validated ToolDependencySpec.

        Returns:
            ToolDependencyConfig for use with BaseToolDependencyProvider.
        """
        # Convert dependencies
        dependencies = self._convert_dependencies(spec.dependencies)

        # Convert transitions
        transitions = self._convert_transitions(spec.transitions)

        # Convert clusters
        clusters = self._convert_clusters(spec.clusters)

        # Convert sequences
        sequences = self._convert_sequences(spec.sequences)

        # Convert tool sets
        required_tools = self._convert_tool_set(spec.required_tools)
        optional_tools = self._convert_tool_set(spec.optional_tools)
        default_sequence = self._convert_tool_list(spec.default_sequence)

        return ToolDependencyConfig(
            dependencies=dependencies,
            transitions=transitions,
            clusters=clusters,
            sequences=sequences,
            required_tools=required_tools,
            optional_tools=optional_tools,
            default_sequence=default_sequence,
        )

    def _convert_dependencies(
        self,
        deps: List[Any],
    ) -> List[ToolDependency]:
        """Convert dependency entries to ToolDependency objects.

        Args:
            deps: List of ToolDependencyEntry objects from spec.

        Returns:
            List of ToolDependency objects.
        """
        result = []
        for dep in deps:
            tool_name = self._canonicalize_name(dep.tool)
            depends_on = self._convert_tool_set(dep.depends_on)
            enables = self._convert_tool_set(dep.enables)

            result.append(
                ToolDependency(
                    tool_name=tool_name,
                    depends_on=depends_on,
                    enables=enables,
                    weight=dep.weight,
                )
            )
        return result

    def _convert_transitions(
        self,
        transitions: Dict[str, List[Any]],
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Convert transition mappings.

        Args:
            transitions: Dict of tool -> list of ToolTransition.

        Returns:
            Dict mapping tool -> list of (next_tool, probability) tuples.
        """
        result: Dict[str, List[Tuple[str, float]]] = {}
        for source, targets in transitions.items():
            source_name = self._canonicalize_name(source)
            result[source_name] = [(self._canonicalize_name(t.tool), t.weight) for t in targets]
        return result

    def _convert_clusters(
        self,
        clusters: Dict[str, List[str]],
    ) -> Dict[str, Set[str]]:
        """Convert cluster definitions.

        Args:
            clusters: Dict of cluster_name -> list of tools.

        Returns:
            Dict mapping cluster_name -> set of tools.
        """
        return {name: self._convert_tool_set(tools) for name, tools in clusters.items()}

    def _convert_sequences(
        self,
        sequences: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        """Convert sequence definitions.

        Args:
            sequences: Dict of sequence_name -> list of tools.

        Returns:
            Dict mapping sequence_name -> list of tools.
        """
        return {name: self._convert_tool_list(tools) for name, tools in sequences.items()}

    def _convert_tool_set(self, tools: List[str]) -> Set[str]:
        """Convert tool list to set, optionally canonicalizing names.

        Args:
            tools: List of tool names.

        Returns:
            Set of (optionally canonicalized) tool names.
        """
        if self._canonicalize:
            return {self._canonicalize_name(t) for t in tools}
        return set(tools)

    def _convert_tool_list(self, tools: List[str]) -> List[str]:
        """Convert tool list, optionally canonicalizing names.

        Args:
            tools: List of tool names.

        Returns:
            List of (optionally canonicalized) tool names.
        """
        if self._canonicalize:
            return [self._canonicalize_name(t) for t in tools]
        return list(tools)

    def _canonicalize_name(self, name: str) -> str:
        """Canonicalize a tool name if enabled.

        Args:
            name: Tool name to canonicalize.

        Returns:
            Canonical tool name if canonicalization enabled, else original.
        """
        if not self._canonicalize:
            return name

        try:
            from victor.framework.tool_naming import get_canonical_name

            return get_canonical_name(name)
        except ImportError:
            # If tool_naming not available, return as-is
            logger.warning("Tool naming module not available, skipping canonicalization")
            return name


# Module-level loader instance for convenience
_default_loader = ToolDependencyLoader(canonicalize=True)


def load_tool_dependency_yaml(
    yaml_path: Union[str, Path],
    canonicalize: bool = True,
    use_cache: bool = True,
) -> ToolDependencyConfig:
    """Load a tool dependency YAML file and convert to config.

    Convenience function that uses a module-level loader instance.

    Args:
        yaml_path: Path to the YAML configuration file.
        canonicalize: Whether to convert tool names to canonical form.
        use_cache: Whether to use cached result if available.

    Returns:
        ToolDependencyConfig ready for use with BaseToolDependencyProvider.

    Raises:
        ToolDependencyLoadError: If loading or validation fails.

    Example:
        config = load_tool_dependency_yaml("victor/coding/tool_dependencies.yaml")
        print(config.required_tools)  # {"read", "write", "edit", "ls", "grep"}
    """
    path = Path(yaml_path) if isinstance(yaml_path, str) else yaml_path

    if canonicalize != _default_loader._canonicalize:
        # Use a fresh loader if canonicalization setting differs
        loader = ToolDependencyLoader(canonicalize=canonicalize)
        return loader.load(path, use_cache=False)

    return _default_loader.load(path, use_cache=use_cache)


def create_tool_dependency_provider(
    yaml_path: Union[str, Path],
    canonicalize: bool = True,
) -> BaseToolDependencyProvider:
    """Create a tool dependency provider from a YAML configuration file.

    Factory function that loads a YAML file and returns a configured
    BaseToolDependencyProvider instance.

    Args:
        yaml_path: Path to the YAML configuration file.
        canonicalize: Whether to convert tool names to canonical form.

    Returns:
        BaseToolDependencyProvider configured with the YAML settings.

    Raises:
        ToolDependencyLoadError: If loading or validation fails.

    Example:
        provider = create_tool_dependency_provider(
            "victor/coding/tool_dependencies.yaml"
        )
        deps = provider.get_dependencies()
        sequences = provider.get_tool_sequences()
    """
    config = load_tool_dependency_yaml(yaml_path, canonicalize=canonicalize)
    return BaseToolDependencyProvider(config=config)


class YAMLToolDependencyProvider(BaseToolDependencyProvider):
    """Tool dependency provider that loads configuration from YAML.

    A convenience class that combines YAML loading with BaseToolDependencyProvider.
    Useful for verticals that want a single class to instantiate.

    Attributes:
        yaml_path: Path to the YAML configuration file.
        vertical: Name of the vertical from the YAML config.

    Example:
        provider = YAMLToolDependencyProvider(
            Path("victor/coding/tool_dependencies.yaml")
        )
        print(provider.vertical)  # "coding"
        print(provider.get_required_tools())  # {"read", "write", "edit", "ls", "grep"}
    """

    def __init__(
        self,
        yaml_path: Union[str, Path],
        canonicalize: bool = True,
        additional_dependencies: Optional[List[ToolDependency]] = None,
        additional_sequences: Optional[Dict[str, List[str]]] = None,
    ):
        """Initialize the provider from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.
            canonicalize: Whether to convert tool names to canonical form.
            additional_dependencies: Additional dependencies to merge.
            additional_sequences: Additional sequences to merge.

        Raises:
            ToolDependencyLoadError: If loading or validation fails.
        """
        self._yaml_path = Path(yaml_path) if isinstance(yaml_path, str) else yaml_path
        self._canonicalize = canonicalize

        # Load the base config
        config = load_tool_dependency_yaml(
            self._yaml_path,
            canonicalize=canonicalize,
        )

        # Merge additional dependencies if provided
        if additional_dependencies:
            config.dependencies.extend(additional_dependencies)

        # Merge additional sequences if provided
        if additional_sequences:
            config.sequences.update(additional_sequences)

        # Extract vertical name from the spec for reference
        loader = ToolDependencyLoader(canonicalize=False)
        spec = loader._load_and_validate(self._yaml_path)
        self._vertical = spec.vertical

        # Initialize parent
        super().__init__(config=config)

    @property
    def yaml_path(self) -> Path:
        """Get the path to the YAML configuration file."""
        return self._yaml_path

    @property
    def vertical(self) -> str:
        """Get the vertical name from the YAML config."""
        return self._vertical


# Mtime-aware provider cache for automatic cache invalidation
# Format: {path_str: (mtime_ns, provider_instance)}
_provider_cache: Dict[str, Tuple[int, BaseToolDependencyProvider]] = {}


def get_cached_provider(yaml_path: str) -> BaseToolDependencyProvider:
    """Get a cached tool dependency provider for a YAML file with mtime validation.

    This function caches provider instances with file modification time (mtime)
    validation. If the YAML file is modified, the cache is automatically invalidated
    and a fresh provider is created.

    Args:
        yaml_path: Path to the YAML configuration file (as string for caching).

    Returns:
        Cached BaseToolDependencyProvider instance.

    Example:
        # First call loads from disk
        provider1 = get_cached_provider("victor/coding/tool_dependencies.yaml")

        # Second call returns cached instance (if file unchanged)
        provider2 = get_cached_provider("victor/coding/tool_dependencies.yaml")
        assert provider1 is provider2

        # If file is modified, cache is invalidated and fresh instance returned
    """
    path = Path(yaml_path).resolve()
    path_str = str(path)

    # Check cache with mtime validation
    if path_str in _provider_cache:
        cached_mtime, provider = _provider_cache[path_str]
        try:
            current_mtime = path.stat().st_mtime_ns
            if current_mtime == cached_mtime:
                logger.debug(f"Cache HIT for {yaml_path} (mtime: {current_mtime})")
                return provider
            else:
                logger.debug(
                    f"Cache INVALIDATED for {yaml_path} "
                    f"(mtime changed: {cached_mtime} -> {current_mtime})"
                )
        except FileNotFoundError:
            logger.debug(f"Cache MISS (file not found): {yaml_path}")
            # File was deleted, invalidate cache entry
            del _provider_cache[path_str]

    # Create new provider (cache miss or invalidated)
    logger.debug(f"Cache MISS for {yaml_path}, creating new provider")
    provider = create_tool_dependency_provider(yaml_path)

    # Cache with current mtime
    try:
        mtime = path.stat().st_mtime_ns
        _provider_cache[path_str] = (mtime, provider)
    except FileNotFoundError:
        # File was deleted during load, don't cache
        pass

    return provider


def invalidate_provider_cache(yaml_path: Optional[str] = None) -> int:
    """Invalidate cached provider instances.

    Useful for testing or forcing a reload of YAML files.

    Args:
        yaml_path: Optional specific path to invalidate. If None, clears all.

    Returns:
        Number of cache entries invalidated.

    Example:
        # Invalidate all caches
        count = invalidate_provider_cache()

        # Invalidate specific file
        count = invalidate_provider_cache("victor/coding/tool_dependencies.yaml")
    """
    if yaml_path:
        path_str = str(Path(yaml_path).resolve())
        if path_str in _provider_cache:
            del _provider_cache[path_str]
            logger.debug(f"Invalidated cache for {yaml_path}")
            return 1
        return 0
    else:
        count = len(_provider_cache)
        _provider_cache.clear()
        logger.debug(f"Invalidated all {count} cached providers")
        return count


# Mapping of vertical names to their canonicalization settings
# Some verticals disable canonicalization to preserve distinct tool names
_VERTICAL_CANONICALIZE_SETTINGS: Dict[str, bool] = {
    "coding": True,
    "devops": False,  # Preserves distinct 'grep' vs 'code_search'
    "research": False,  # Preserves original tool names from ToolNames constants
    "rag": True,
    "data_analysis": False,  # Preserves 'code_search' as distinct from 'grep'
}


def create_vertical_tool_dependency_provider(
    vertical: str,
    canonicalize: Optional[bool] = None,
) -> Union["YAMLToolDependencyProvider", "EmptyToolDependencyProvider"]:
    """Factory function to create tool dependency providers for verticals.

    Consolidates vertical-specific tool dependency provider creation into
    a single factory function, reducing code duplication. This replaces the
    need for individual wrapper classes like CodingToolDependencyProvider,
    DevOpsToolDependencyProvider, etc.

    Args:
        vertical: Vertical name (coding, devops, research, rag, dataanalysis)
        canonicalize: Whether to canonicalize tool names. If None, uses the
            vertical's default setting (some verticals disable canonicalization
            to preserve distinct tool names like 'grep' vs 'code_search').

    Returns:
        Configured YAMLToolDependencyProvider for the vertical

    Raises:
        ValueError: If vertical is not recognized

    Example:
        # Create provider for coding vertical
        provider = create_vertical_tool_dependency_provider("coding")
        deps = provider.get_dependencies()

        # Create provider with explicit canonicalization setting
        provider = create_vertical_tool_dependency_provider("devops", canonicalize=False)

        # Get recommended sequence for edit task
        sequence = provider.get_recommended_sequence("edit")

    Note:
        This factory is the preferred way to create vertical providers for new code.
        The individual wrapper classes (e.g., CodingToolDependencyProvider) are
        maintained for backward compatibility but delegate to this factory internally.
    """
    from victor.core.verticals.naming import get_vertical_module_name, normalize_vertical_name

    vertical = normalize_vertical_name(vertical)

    # Map verticals to their tool dependency YAML files
    module_name = get_vertical_module_name("data_analysis")
    data_analysis_path = Path(__file__).parent.parent / module_name / "tool_dependencies.yaml"
    yaml_paths = {
        "coding": Path(__file__).parent.parent / "coding" / "tool_dependencies.yaml",
        "devops": Path(__file__).parent.parent / "devops" / "tool_dependencies.yaml",
        "research": Path(__file__).parent.parent / "research" / "tool_dependencies.yaml",
        "rag": Path(__file__).parent.parent / "rag" / "tool_dependencies.yaml",
        "data_analysis": data_analysis_path,
        "dataanalysis": data_analysis_path,
    }

    if vertical not in yaml_paths:
        available = ", ".join(sorted(yaml_paths.keys()))
        raise ValueError(f"Unknown vertical '{vertical}'. Available: {available}")

    yaml_path = yaml_paths[vertical]

    # Check if YAML exists, fall back to empty provider if not
    if not yaml_path.exists():
        logger.warning(f"Tool dependencies YAML not found for vertical '{vertical}': {yaml_path}")
        # Return an LSP-compliant empty provider (Null Object pattern)
        # This allows the system to function even without YAML files
        from victor.core.tool_types import EmptyToolDependencyProvider

        return EmptyToolDependencyProvider(vertical)

    # Determine canonicalization setting
    if canonicalize is None:
        canonicalize = _VERTICAL_CANONICALIZE_SETTINGS.get(vertical, True)

    return YAMLToolDependencyProvider(yaml_path, canonicalize=canonicalize)


__all__ = [
    "ToolDependencyLoader",
    "ToolDependencyLoadError",
    "YAMLToolDependencyProvider",
    "load_tool_dependency_yaml",
    "create_tool_dependency_provider",
    "get_cached_provider",
    "invalidate_provider_cache",
    "create_vertical_tool_dependency_provider",
]
