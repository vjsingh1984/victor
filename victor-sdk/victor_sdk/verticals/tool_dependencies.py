"""Base tool dependency implementations for vertical plugins.

This module provides the base classes and utilities that external verticals
use to define tool dependency relationships. By placing these in the SDK,
external verticals can stop importing from victor.core.

Classes:
    BaseToolDependencyProvider: Base implementation with common algorithms
    ToolDependencyConfig: Configuration dataclass for dependency data
    EmptyToolDependencyProvider: Null object fallback for missing configs
    ToolDependencyLoadError: Exception for YAML loading failures
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from victor_sdk.constants import get_canonical_name
from victor_sdk.verticals.protocols.tools import (
    ToolDependency,
    ToolDependencyProviderProtocol,
)


class ToolDependencyLoadError(Exception):
    """Exception raised when loading tool dependency configuration fails.

    Attributes:
        path: Path to the configuration that failed to load.
        message: Detailed error message.
    """

    def __init__(self, path: str, message: str = ""):
        self.path = path
        self.message = message or f"Failed to load tool dependencies from {path}"
        super().__init__(self.message)


@dataclass
class ToolDependencyConfig:
    """Configuration for tool dependency provider.

    Attributes:
        dependencies: List of ToolDependency objects defining depends_on/enables
        transitions: Dict mapping tool -> [(next_tool, probability), ...]
        clusters: Dict mapping cluster_name -> set of tools
        sequences: Dict mapping task_type -> [tool_sequence]
        required_tools: Tools that are essential for this vertical
        optional_tools: Tools that enhance but aren't required
        default_sequence: Fallback sequence when task type is unknown
    """

    dependencies: List[ToolDependency] = field(default_factory=list)
    transitions: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)
    clusters: Dict[str, Set[str]] = field(default_factory=dict)
    sequences: Dict[str, List[str]] = field(default_factory=dict)
    required_tools: Set[str] = field(default_factory=set)
    optional_tools: Set[str] = field(default_factory=set)
    default_sequence: List[str] = field(default_factory=lambda: ["read", "edit"])

    def clone(self) -> ToolDependencyConfig:
        """Return a defensive copy for mutation-safe reuse across caches."""

        return ToolDependencyConfig(
            dependencies=[
                ToolDependency(
                    tool_name=dependency.tool_name,
                    depends_on=set(dependency.depends_on),
                    enables=set(dependency.enables),
                    weight=dependency.weight,
                )
                for dependency in self.dependencies
            ],
            transitions={
                tool_name: [(next_tool, weight) for next_tool, weight in transitions]
                for tool_name, transitions in self.transitions.items()
            },
            clusters={cluster: set(tools) for cluster, tools in self.clusters.items()},
            sequences={name: list(sequence) for name, sequence in self.sequences.items()},
            required_tools=set(self.required_tools),
            optional_tools=set(self.optional_tools),
            default_sequence=list(self.default_sequence),
        )


class BaseToolDependencyProvider(ToolDependencyProviderProtocol):
    """Base implementation of tool dependency provider.

    Provides common algorithms for:
    - Computing transition weights
    - Suggesting next tools
    - Finding tool clusters
    - Recommending sequences

    Subclasses only need to provide configuration data.

    Example::

        class DevOpsToolDeps(BaseToolDependencyProvider):
            def __init__(self):
                super().__init__(ToolDependencyConfig(
                    dependencies=[
                        ToolDependency("shell", depends_on={"read"}),
                    ],
                    transitions={"read": [("shell", 0.4)]},
                    required_tools={"read", "write", "shell"},
                ))
    """

    def __init__(
        self,
        config: Optional[ToolDependencyConfig] = None,
        *,
        dependencies: Optional[List[ToolDependency]] = None,
        transitions: Optional[Dict[str, List[Tuple[str, float]]]] = None,
        clusters: Optional[Dict[str, Set[str]]] = None,
        sequences: Optional[Dict[str, List[str]]] = None,
        required_tools: Optional[Set[str]] = None,
        optional_tools: Optional[Set[str]] = None,
        default_sequence: Optional[List[str]] = None,
    ):
        if config:
            self._config = config
        else:
            self._config = ToolDependencyConfig(
                dependencies=dependencies or [],
                transitions=transitions or {},
                clusters=clusters or {},
                sequences=sequences or {},
                required_tools=required_tools or set(),
                optional_tools=optional_tools or set(),
                default_sequence=default_sequence or ["read", "edit"],
            )
        self._dependency_map: Dict[str, ToolDependency] = {
            d.tool_name: d for d in self._config.dependencies
        }

    # === ToolDependencyProviderProtocol ===

    def get_dependencies(self) -> List[ToolDependency]:
        return self._config.dependencies.copy()

    def get_tool_sequences(self) -> List[List[str]]:
        return [list(seq) for seq in self._config.sequences.values()]

    # === Extended methods ===

    def get_tool_transitions(self) -> Dict[str, List[Tuple[str, float]]]:
        return self._config.transitions.copy()

    def get_tool_clusters(self) -> Dict[str, Set[str]]:
        return {k: v.copy() for k, v in self._config.clusters.items()}

    def get_recommended_sequence(self, task_type: str) -> List[str]:
        return list(self._config.sequences.get(task_type, self._config.default_sequence))

    def get_required_tools(self) -> Set[str]:
        return self._config.required_tools.copy()

    def get_optional_tools(self) -> Set[str]:
        return self._config.optional_tools.copy()

    def get_transition_weight(self, from_tool: str, to_tool: str) -> float:
        if to_tool in self._dependency_map:
            dep = self._dependency_map[to_tool]
            if from_tool in dep.depends_on:
                return dep.weight

        if from_tool in self._dependency_map:
            dep = self._dependency_map[from_tool]
            if to_tool in dep.enables:
                return dep.weight * 0.8

        transitions = self._config.transitions.get(from_tool, [])
        for tool, prob in transitions:
            if tool == to_tool:
                return prob

        for seq in self._config.sequences.values():
            for i in range(len(seq) - 1):
                if seq[i] == from_tool and seq[i + 1] == to_tool:
                    return 0.6

        return 0.3

    def suggest_next_tool(
        self,
        current_tool: str,
        used_tools: Optional[List[str]] = None,
    ) -> str:
        used_tools = used_tools or []
        transitions = self._config.transitions.get(current_tool, [])

        if not transitions:
            if current_tool in self._dependency_map:
                dep = self._dependency_map[current_tool]
                if dep.enables:
                    return next(iter(dep.enables))
            return self._config.default_sequence[0] if self._config.default_sequence else "read"

        recent = set(used_tools[-3:]) if len(used_tools) >= 3 else set(used_tools)
        for tool, _prob in sorted(transitions, key=lambda x: x[1], reverse=True):
            if tool not in recent:
                return tool

        return transitions[0][0]

    def find_cluster(self, tool: str) -> Optional[str]:
        for cluster_name, tools in self._config.clusters.items():
            if tool in tools:
                return cluster_name
        return None

    def get_cluster_tools(self, cluster_name: str) -> Set[str]:
        return self._config.clusters.get(cluster_name, set()).copy()

    def is_valid_transition(self, from_tool: str, to_tool: str) -> bool:
        if to_tool in self._dependency_map:
            dep = self._dependency_map[to_tool]
            if from_tool in dep.depends_on:
                return True
        if from_tool in self._dependency_map:
            dep = self._dependency_map[from_tool]
            if to_tool in dep.enables:
                return True
        transitions = self._config.transitions.get(from_tool, [])
        return any(tool == to_tool for tool, _ in transitions)


class EmptyToolDependencyProvider:
    """Null object fallback when tool dependency config is missing.

    Provides minimal valid defaults so the system degrades gracefully.
    """

    def __init__(self, vertical: str = "unknown"):
        self._vertical = vertical

    @property
    def vertical(self) -> str:
        return self._vertical

    def get_dependencies(self) -> List[ToolDependency]:
        return []

    def get_tool_sequences(self) -> List[List[str]]:
        return [["read"]]

    def get_tool_transitions(self) -> Dict[str, List[Tuple[str, float]]]:
        return {}

    def get_tool_clusters(self) -> Dict[str, Set[str]]:
        return {}

    def get_recommended_sequence(self, task_type: str) -> List[str]:
        return ["read"]

    def get_required_tools(self) -> Set[str]:
        return {"read"}

    def get_optional_tools(self) -> Set[str]:
        return set()

    def get_transition_weight(self, from_tool: str, to_tool: str) -> float:
        return 0.3

    def suggest_next_tool(
        self,
        current_tool: str,
        used_tools: Optional[List[str]] = None,
    ) -> str:
        return "read"

    def find_cluster(self, tool: str) -> Optional[str]:
        return None

    def get_cluster_tools(self, cluster_name: str) -> Set[str]:
        return set()

    def is_valid_transition(self, from_tool: str, to_tool: str) -> bool:
        return True


class ToolDependencyLoader:
    """Load YAML tool-dependency specs into SDK-native config objects."""

    def __init__(self, canonicalize: bool = True):
        self._canonicalize = canonicalize
        self._cache: Dict[Path, ToolDependencyConfig] = {}

    def load(
        self,
        yaml_path: Union[str, Path],
        use_cache: bool = True,
    ) -> ToolDependencyConfig:
        """Load a YAML config file from disk."""

        path = Path(yaml_path).expanduser().resolve()
        if use_cache and path in self._cache:
            return self._cache[path]

        data = self._load_yaml_data(path)
        config = self._convert_to_config(data, path=path)
        self._cache[path] = config
        return config

    def load_from_string(self, yaml_content: str) -> ToolDependencyConfig:
        """Load a YAML config from an in-memory string."""

        data = self._safe_load_yaml(yaml_content, path=Path("<string>"))
        return self._convert_to_config(data, path=Path("<string>"))

    def clear_cache(self) -> None:
        """Clear the loader-level config cache."""

        self._cache.clear()

    def _load_yaml_data(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise ToolDependencyLoadError(path, "File not found")
        try:
            return self._safe_load_yaml(path.read_text(encoding="utf-8"), path=path)
        except OSError as exc:
            raise ToolDependencyLoadError(path, f"Could not read file: {exc}") from exc

    def _safe_load_yaml(self, yaml_content: str, *, path: Path) -> Dict[str, Any]:
        try:
            import yaml
        except ImportError as exc:
            raise ToolDependencyLoadError(
                path,
                "PyYAML is required to load tool dependency YAML. "
                "Install PyYAML or victor-sdk[yaml].",
            ) from exc

        try:
            data = yaml.safe_load(yaml_content)
        except yaml.YAMLError as exc:
            raise ToolDependencyLoadError(path, f"YAML parsing error: {exc}") from exc

        if data is None:
            raise ToolDependencyLoadError(path, "YAML file is empty")
        if not isinstance(data, dict):
            raise ToolDependencyLoadError(path, "Top-level YAML value must be a mapping")
        return data

    def _convert_to_config(self, data: Dict[str, Any], *, path: Path) -> ToolDependencyConfig:
        vertical = data.get("vertical")
        if not isinstance(vertical, str) or not vertical:
            raise ToolDependencyLoadError(path, "Missing required 'vertical' field")

        return ToolDependencyConfig(
            dependencies=self._convert_dependencies(data.get("dependencies"), path=path),
            transitions=self._convert_transitions(data.get("transitions"), path=path),
            clusters=self._convert_named_tool_sets(data.get("clusters"), path=path),
            sequences=self._convert_named_tool_lists(data.get("sequences"), path=path),
            required_tools=self._convert_tool_set(data.get("required_tools")),
            optional_tools=self._convert_tool_set(data.get("optional_tools")),
            default_sequence=self._convert_tool_list(data.get("default_sequence") or ["read"]),
        )

    def _convert_dependencies(
        self,
        raw_dependencies: Any,
        *,
        path: Path,
    ) -> List[ToolDependency]:
        if raw_dependencies is None:
            return []
        if not isinstance(raw_dependencies, list):
            raise ToolDependencyLoadError(path, "'dependencies' must be a list")

        dependencies: List[ToolDependency] = []
        for index, raw_dependency in enumerate(raw_dependencies):
            if not isinstance(raw_dependency, dict):
                raise ToolDependencyLoadError(
                    path,
                    f"'dependencies[{index}]' must be a mapping",
                )
            tool_name = raw_dependency.get("tool")
            if not isinstance(tool_name, str) or not tool_name:
                raise ToolDependencyLoadError(
                    path,
                    f"'dependencies[{index}].tool' must be a non-empty string",
                )
            weight = raw_dependency.get("weight", 1.0)
            if not isinstance(weight, (int, float)):
                raise ToolDependencyLoadError(
                    path,
                    f"'dependencies[{index}].weight' must be numeric",
                )
            dependencies.append(
                ToolDependency(
                    tool_name=self._canonicalize_name(tool_name),
                    depends_on=self._convert_tool_set(raw_dependency.get("depends_on")),
                    enables=self._convert_tool_set(raw_dependency.get("enables")),
                    weight=float(weight),
                )
            )
        return dependencies

    def _convert_transitions(
        self,
        raw_transitions: Any,
        *,
        path: Path,
    ) -> Dict[str, List[Tuple[str, float]]]:
        if raw_transitions is None:
            return {}
        if not isinstance(raw_transitions, dict):
            raise ToolDependencyLoadError(path, "'transitions' must be a mapping")

        transitions: Dict[str, List[Tuple[str, float]]] = {}
        for tool_name, entries in raw_transitions.items():
            if not isinstance(tool_name, str):
                raise ToolDependencyLoadError(path, "Transition tool names must be strings")
            if not isinstance(entries, list):
                raise ToolDependencyLoadError(
                    path,
                    f"'transitions.{tool_name}' must be a list",
                )

            parsed_entries: List[Tuple[str, float]] = []
            for index, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    raise ToolDependencyLoadError(
                        path,
                        f"'transitions.{tool_name}[{index}]' must be a mapping",
                    )
                next_tool = entry.get("tool")
                weight = entry.get("weight")
                if not isinstance(next_tool, str) or not next_tool:
                    raise ToolDependencyLoadError(
                        path,
                        f"'transitions.{tool_name}[{index}].tool' must be a non-empty string",
                    )
                if not isinstance(weight, (int, float)):
                    raise ToolDependencyLoadError(
                        path,
                        f"'transitions.{tool_name}[{index}].weight' must be numeric",
                    )
                parsed_entries.append((self._canonicalize_name(next_tool), float(weight)))
            transitions[self._canonicalize_name(tool_name)] = parsed_entries
        return transitions

    def _convert_named_tool_sets(
        self,
        raw_value: Any,
        *,
        path: Path,
    ) -> Dict[str, Set[str]]:
        if raw_value is None:
            return {}
        if not isinstance(raw_value, dict):
            raise ToolDependencyLoadError(path, "Expected a mapping of tool sets")
        return {
            str(name): self._convert_tool_set(tools, path=path)
            for name, tools in raw_value.items()
        }

    def _convert_named_tool_lists(
        self,
        raw_value: Any,
        *,
        path: Path,
    ) -> Dict[str, List[str]]:
        if raw_value is None:
            return {}
        if not isinstance(raw_value, dict):
            raise ToolDependencyLoadError(path, "Expected a mapping of tool sequences")
        return {
            str(name): self._convert_tool_list(sequence, path=path)
            for name, sequence in raw_value.items()
        }

    def _convert_tool_set(
        self,
        raw_tools: Any,
        *,
        path: Optional[Path] = None,
    ) -> Set[str]:
        if raw_tools is None:
            return set()
        if not isinstance(raw_tools, list):
            raise ToolDependencyLoadError(path or Path("<memory>"), "Expected a list of tool names")
        return {self._canonicalize_name(self._validate_tool_name(tool, path)) for tool in raw_tools}

    def _convert_tool_list(
        self,
        raw_tools: Any,
        *,
        path: Optional[Path] = None,
    ) -> List[str]:
        if raw_tools is None:
            return []
        if not isinstance(raw_tools, list):
            raise ToolDependencyLoadError(path or Path("<memory>"), "Expected a list of tool names")
        return [self._canonicalize_name(self._validate_tool_name(tool, path)) for tool in raw_tools]

    def _validate_tool_name(self, tool_name: Any, path: Optional[Path]) -> str:
        if not isinstance(tool_name, str) or not tool_name:
            raise ToolDependencyLoadError(
                path or Path("<memory>"),
                "Tool names must be non-empty strings",
            )
        return tool_name

    def _canonicalize_name(self, tool_name: str) -> str:
        if not self._canonicalize:
            return tool_name
        return get_canonical_name(tool_name)


_default_loader = ToolDependencyLoader(canonicalize=True)
_provider_cache: Dict[Tuple[str, bool], Tuple[int, BaseToolDependencyProvider]] = {}
_provider_cache_lock = threading.Lock()


def load_tool_dependency_yaml(
    yaml_path: Union[str, Path],
    canonicalize: bool = True,
    use_cache: bool = True,
) -> ToolDependencyConfig:
    """Load a YAML tool-dependency config using the shared default loader."""

    path = Path(yaml_path) if isinstance(yaml_path, str) else yaml_path
    if canonicalize != _default_loader._canonicalize:
        return ToolDependencyLoader(canonicalize=canonicalize).load(path, use_cache=False)
    return _default_loader.load(path, use_cache=use_cache)


def create_tool_dependency_provider(
    yaml_path: Union[str, Path],
    canonicalize: bool = True,
) -> BaseToolDependencyProvider:
    """Create a provider directly from a YAML config path."""

    return BaseToolDependencyProvider(
        config=load_tool_dependency_yaml(yaml_path, canonicalize=canonicalize)
    )


class YAMLToolDependencyProvider(BaseToolDependencyProvider):
    """Provider backed by a YAML config file on disk."""

    def __init__(
        self,
        yaml_path: Union[str, Path],
        canonicalize: bool = True,
        additional_dependencies: Optional[List[ToolDependency]] = None,
        additional_sequences: Optional[Dict[str, List[str]]] = None,
    ):
        self._yaml_path = Path(yaml_path) if isinstance(yaml_path, str) else yaml_path
        self._canonicalize = canonicalize
        self._vertical = self._read_vertical_name(self._yaml_path)

        config = load_tool_dependency_yaml(
            self._yaml_path,
            canonicalize=canonicalize,
        ).clone()
        if additional_dependencies:
            config.dependencies.extend(additional_dependencies)
        if additional_sequences:
            config.sequences.update(
                {
                    sequence_name: list(sequence)
                    for sequence_name, sequence in additional_sequences.items()
                }
            )
        super().__init__(config=config)

    @property
    def yaml_path(self) -> Path:
        return self._yaml_path

    @property
    def vertical(self) -> str:
        return self._vertical

    @staticmethod
    def _read_vertical_name(yaml_path: Path) -> str:
        data = ToolDependencyLoader(canonicalize=False)._load_yaml_data(yaml_path)
        vertical = data.get("vertical")
        if not isinstance(vertical, str) or not vertical:
            raise ToolDependencyLoadError(yaml_path, "Missing required 'vertical' field")
        return vertical


def get_cached_provider(yaml_path: str, canonicalize: bool = True) -> BaseToolDependencyProvider:
    """Return a provider cached by path and invalidated by file modification time."""

    path = Path(yaml_path).expanduser().resolve()
    cache_key = (str(path), canonicalize)
    current_mtime = path.stat().st_mtime_ns

    with _provider_cache_lock:
        cached = _provider_cache.get(cache_key)
        if cached and cached[0] == current_mtime:
            return cached[1]

        provider = YAMLToolDependencyProvider(path, canonicalize=canonicalize)
        _provider_cache[cache_key] = (current_mtime, provider)
        return provider


def invalidate_provider_cache(yaml_path: Optional[str] = None) -> None:
    """Invalidate all cached providers or a single path-specific entry."""

    with _provider_cache_lock:
        if yaml_path is None:
            _provider_cache.clear()
            return

        path = str(Path(yaml_path).expanduser().resolve())
        for cache_key in [key for key in _provider_cache if key[0] == path]:
            _provider_cache.pop(cache_key, None)


# Factory function placeholder — the full YAML-based factory lives in
# victor.core.tool_dependency_loader since it depends on core runtime
# utilities. This SDK version provides a simple config-based factory.
def create_vertical_tool_dependency_provider(
    vertical_name: str,
    config: Optional[ToolDependencyConfig] = None,
) -> BaseToolDependencyProvider:
    """Create a tool dependency provider for a vertical.

    Args:
        vertical_name: Name of the vertical
        config: Optional configuration (returns EmptyToolDependencyProvider if None)

    Returns:
        A tool dependency provider instance
    """
    if config is None:
        return EmptyToolDependencyProvider(vertical_name)
    return BaseToolDependencyProvider(config)


__all__ = [
    "BaseToolDependencyProvider",
    "EmptyToolDependencyProvider",
    "ToolDependencyLoader",
    "ToolDependencyConfig",
    "ToolDependencyLoadError",
    "YAMLToolDependencyProvider",
    "load_tool_dependency_yaml",
    "create_tool_dependency_provider",
    "get_cached_provider",
    "invalidate_provider_cache",
    "create_vertical_tool_dependency_provider",
]
