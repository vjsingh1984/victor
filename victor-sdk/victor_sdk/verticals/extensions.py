"""Lazy-loading container for vertical extension implementations.

VerticalExtensions aggregates all extension protocols for a vertical.
Fields are lazily resolved on first access when constructed with
factory callables, eliminating import storms at activation time.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union


class VerticalExtensions:
    """Container for all vertical extension implementations.

    Supports both eager (concrete values) and lazy (factory callables)
    construction. When constructed with callables, each extension is
    loaded on first access only.

    Eager construction (backward compatible)::

        ext = VerticalExtensions(
            middleware=[my_middleware],
            safety_extensions=[my_safety],
        )

    Lazy construction::

        ext = VerticalExtensions(
            middleware=lambda: load_middleware(),
            safety_extensions=lambda: [load_safety()],
        )
    """

    _FIELDS = [
        ("middleware", True),
        ("safety_extensions", True),
        ("prompt_contributors", True),
        ("mode_config_provider", False),
        ("tool_dependency_provider", False),
        ("workflow_provider", False),
        ("service_provider", False),
        ("rl_config_provider", False),
        ("team_spec_provider", False),
        ("enrichment_strategy", False),
        ("tool_selection_strategy", False),
        ("tiered_tool_config", False),
    ]

    def __init__(
        self,
        middleware: Union[List[Any], Callable, None] = None,
        safety_extensions: Union[List[Any], Callable, None] = None,
        prompt_contributors: Union[List[Any], Callable, None] = None,
        mode_config_provider: Union[Any, Callable, None] = None,
        tool_dependency_provider: Union[Any, Callable, None] = None,
        workflow_provider: Union[Any, Callable, None] = None,
        service_provider: Union[Any, Callable, None] = None,
        rl_config_provider: Union[Any, Callable, None] = None,
        team_spec_provider: Union[Any, Callable, None] = None,
        enrichment_strategy: Union[Any, Callable, None] = None,
        tool_selection_strategy: Union[Any, Callable, None] = None,
        tiered_tool_config: Union[Any, Callable, None] = None,
    ) -> None:
        self._resolved: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}

        args = {
            "middleware": middleware,
            "safety_extensions": safety_extensions,
            "prompt_contributors": prompt_contributors,
            "mode_config_provider": mode_config_provider,
            "tool_dependency_provider": tool_dependency_provider,
            "workflow_provider": workflow_provider,
            "service_provider": service_provider,
            "rl_config_provider": rl_config_provider,
            "team_spec_provider": team_spec_provider,
            "enrichment_strategy": enrichment_strategy,
            "tool_selection_strategy": tool_selection_strategy,
            "tiered_tool_config": tiered_tool_config,
        }

        for name, is_list in self._FIELDS:
            val = args.get(name)
            if val is None:
                self._resolved[name] = [] if is_list else None
            elif callable(val) and not isinstance(val, list):
                self._factories[name] = val
            else:
                self._resolved[name] = val

    def _resolve(self, name: str) -> Any:
        if name in self._resolved:
            return self._resolved[name]
        factory = self._factories.pop(name, None)
        if factory is not None:
            try:
                value = factory()
            except Exception:
                is_list = any(n == name and il for n, il in self._FIELDS)
                value = [] if is_list else None
            self._resolved[name] = value
            return value
        is_list = any(n == name and il for n, il in self._FIELDS)
        val = [] if is_list else None
        self._resolved[name] = val
        return val

    # --- Properties ---

    @property
    def middleware(self) -> List[Any]:
        return self._resolve("middleware")

    @middleware.setter
    def middleware(self, value: Any) -> None:
        self._factories.pop("middleware", None)
        self._resolved["middleware"] = value

    @property
    def safety_extensions(self) -> List[Any]:
        return self._resolve("safety_extensions")

    @safety_extensions.setter
    def safety_extensions(self, value: Any) -> None:
        self._factories.pop("safety_extensions", None)
        self._resolved["safety_extensions"] = value

    @property
    def prompt_contributors(self) -> List[Any]:
        return self._resolve("prompt_contributors")

    @prompt_contributors.setter
    def prompt_contributors(self, value: Any) -> None:
        self._factories.pop("prompt_contributors", None)
        self._resolved["prompt_contributors"] = value

    @property
    def mode_config_provider(self) -> Optional[Any]:
        return self._resolve("mode_config_provider")

    @mode_config_provider.setter
    def mode_config_provider(self, value: Any) -> None:
        self._factories.pop("mode_config_provider", None)
        self._resolved["mode_config_provider"] = value

    @property
    def tool_dependency_provider(self) -> Optional[Any]:
        return self._resolve("tool_dependency_provider")

    @tool_dependency_provider.setter
    def tool_dependency_provider(self, value: Any) -> None:
        self._factories.pop("tool_dependency_provider", None)
        self._resolved["tool_dependency_provider"] = value

    @property
    def workflow_provider(self) -> Optional[Any]:
        return self._resolve("workflow_provider")

    @workflow_provider.setter
    def workflow_provider(self, value: Any) -> None:
        self._factories.pop("workflow_provider", None)
        self._resolved["workflow_provider"] = value

    @property
    def service_provider(self) -> Optional[Any]:
        return self._resolve("service_provider")

    @service_provider.setter
    def service_provider(self, value: Any) -> None:
        self._factories.pop("service_provider", None)
        self._resolved["service_provider"] = value

    @property
    def rl_config_provider(self) -> Optional[Any]:
        return self._resolve("rl_config_provider")

    @rl_config_provider.setter
    def rl_config_provider(self, value: Any) -> None:
        self._factories.pop("rl_config_provider", None)
        self._resolved["rl_config_provider"] = value

    @property
    def team_spec_provider(self) -> Optional[Any]:
        return self._resolve("team_spec_provider")

    @team_spec_provider.setter
    def team_spec_provider(self, value: Any) -> None:
        self._factories.pop("team_spec_provider", None)
        self._resolved["team_spec_provider"] = value

    @property
    def enrichment_strategy(self) -> Optional[Any]:
        return self._resolve("enrichment_strategy")

    @enrichment_strategy.setter
    def enrichment_strategy(self, value: Any) -> None:
        self._factories.pop("enrichment_strategy", None)
        self._resolved["enrichment_strategy"] = value

    @property
    def tool_selection_strategy(self) -> Optional[Any]:
        return self._resolve("tool_selection_strategy")

    @tool_selection_strategy.setter
    def tool_selection_strategy(self, value: Any) -> None:
        self._factories.pop("tool_selection_strategy", None)
        self._resolved["tool_selection_strategy"] = value

    @property
    def tiered_tool_config(self) -> Optional[Any]:
        return self._resolve("tiered_tool_config")

    @tiered_tool_config.setter
    def tiered_tool_config(self, value: Any) -> None:
        self._factories.pop("tiered_tool_config", None)
        self._resolved["tiered_tool_config"] = value

    # --- Convenience methods ---

    def get_all_task_hints(self) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        for contributor in sorted(self.prompt_contributors, key=lambda c: c.get_priority()):
            merged.update(contributor.get_task_type_hints())
        return merged

    def get_all_safety_patterns(self) -> List[Any]:
        patterns: List[Any] = []
        for ext in self.safety_extensions:
            patterns.extend(ext.get_bash_patterns())
            patterns.extend(ext.get_file_patterns())
        return patterns

    def get_all_mode_configs(self) -> Dict[str, Any]:
        if self.mode_config_provider:
            return self.mode_config_provider.get_mode_configs()
        return {}

    @property
    def pending_factories(self) -> int:
        return len(self._factories)
