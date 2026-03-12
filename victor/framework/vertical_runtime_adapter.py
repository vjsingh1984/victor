"""Host-owned adapter for translating vertical definitions into runtime config.

.. deprecated::
    This module is deprecated and will be removed in Victor 2.0.
    Verticals now implement the runtime VerticalBase protocol directly.

Migration Path:
    **OLD (deprecated):**
        from victor.framework.vertical_runtime_adapter import VerticalRuntimeAdapter

        adapter = VerticalRuntimeAdapter()
        binding = adapter.build_runtime_binding(vertical)

    **NEW (direct implementation):**
        from victor.core.verticals.base import VerticalBase

        # Verticals directly implement VerticalBase protocol
        config = vertical.get_config()

For migration examples, see: ``docs/MIGRATION_GUIDE.md``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, Union

from victor.framework.tools import ToolSet
from victor_sdk.core.types import VerticalDefinition

if TYPE_CHECKING:
    from victor.core.vertical_types import StageDefinition as RuntimeStageDefinition
    from victor.core.verticals.base import VerticalConfig as RuntimeVerticalConfig
    from victor_sdk.verticals.protocols.base import VerticalBase as SdkVerticalBase

VerticalDefinitionSource = Union[Type["SdkVerticalBase"], VerticalDefinition]


@dataclass(frozen=True)
class VerticalRuntimeBinding:
    """Resolved runtime representation for a vertical definition source."""

    definition: VerticalDefinition
    runtime_config: "RuntimeVerticalConfig"

    def to_agent_kwargs(self) -> Dict[str, Any]:
        """Return keyword arguments suitable for ``Agent.create()``."""

        return self.runtime_config.to_agent_kwargs()


@dataclass(frozen=True)
class DefinitionBackedTeamSpecProvider:
    """Runtime team provider synthesized from SDK team metadata."""

    team_specs: Dict[str, Any]
    default_team: Optional[str] = None

    def get_team_specs(self) -> Dict[str, Any]:
        """Return runtime team specs keyed by team identifier."""

        return dict(self.team_specs)

    def get_default_team(self) -> Optional[str]:
        """Return the default team identifier if one is declared."""

        return self.default_team


class VerticalRuntimeAdapter:
    """Translate definition-layer vertical contracts into runtime configuration.

    .. deprecated::
        ``VerticalRuntimeAdapter`` is deprecated and will be removed in Victor 2.0.
        Verticals should implement ``VerticalBase`` protocol directly.

    Migration Path:
        **OLD (deprecated):**
            from victor.framework.vertical_runtime_adapter import VerticalRuntimeAdapter

            binding = VerticalRuntimeAdapter.build_runtime_binding(vertical)

        **NEW (direct protocol):**
            from victor.core.verticals.base import VerticalBase

            config = vertical.get_config()
    """

    _legacy_runtime_shims: Dict[Any, type] = {}
    _FORWARDED_SDK_HOOKS = (
        "get_tool_requirements",
        "get_capability_requirements",
        "get_prompt_templates",
        "get_task_type_hints",
        "get_prompt_metadata",
        "get_team_declarations",
        "get_default_team",
        "get_team_metadata",
        "get_initial_stage",
        "get_workflow_spec",
        "get_workflow_metadata",
        "get_metadata",
        "get_tiered_tool_config",
    )

    @classmethod
    def resolve_definition(cls, source: VerticalDefinitionSource) -> VerticalDefinition:
        """Resolve a vertical source into a normalized SDK definition.

        .. deprecated::
            Use ``vertical.get_definition()`` directly on VerticalBase implementations.
        """
        import warnings

        warnings.warn(
            "VerticalRuntimeAdapter.resolve_definition() is deprecated. "
            "Use vertical.get_definition() directly on VerticalBase implementations. "
            "See docs/MIGRATION_GUIDE.md for migration examples.",
            DeprecationWarning,
            stacklevel=2,
        )

        if isinstance(source, VerticalDefinition):
            return source

        definition_factory = getattr(source, "get_definition", None)
        if callable(definition_factory):
            resolved = definition_factory()
            if isinstance(resolved, VerticalDefinition):
                return resolved
            if isinstance(resolved, dict):
                return VerticalDefinition.from_dict(resolved)

        config_factory = getattr(source, "get_config", None)
        if callable(config_factory):
            return cls._definition_from_legacy_config(source, config_factory())

        raise TypeError(
            "Vertical runtime sources must provide get_definition() or get_config(), "
            f"got {source!r}"
        )

    @classmethod
    def build_runtime_binding(
        cls,
        source: VerticalDefinitionSource,
    ) -> VerticalRuntimeBinding:
        """Build the runtime binding for a vertical source.

        .. deprecated::
            Use ``vertical.get_config()`` directly on VerticalBase implementations.
        """
        import warnings

        warnings.warn(
            "VerticalRuntimeAdapter.build_runtime_binding() is deprecated. "
            "Use vertical.get_config() directly on VerticalBase implementations. "
            "See docs/MIGRATION_GUIDE.md for migration examples.",
            DeprecationWarning,
            stacklevel=2,
        )

        definition = cls.resolve_definition(source)
        runtime_config = cls.definition_to_runtime_config(definition)
        return VerticalRuntimeBinding(definition=definition, runtime_config=runtime_config)

    @classmethod
    def build_team_spec_provider(
        cls,
        source: VerticalDefinitionSource,
    ) -> Optional[DefinitionBackedTeamSpecProvider]:
        """Build a runtime team-spec provider from a definition source."""

        definition = cls.resolve_definition(source)
        return cls._build_definition_team_provider(definition)

    @classmethod
    def as_runtime_vertical_class(cls, source: Any) -> Any:
        """Return a runtime-compatible vertical class, shimming legacy shapes when needed."""

        if isinstance(source, VerticalDefinition):
            return source

        if cls._is_runtime_vertical_class(source):
            return source

        shim = cls._legacy_runtime_shims.get(source)
        if shim is not None:
            return shim

        binding = cls.build_runtime_binding(source)
        shim = cls._build_legacy_runtime_shim(source, binding)
        cls._legacy_runtime_shims[source] = shim
        return shim

    @classmethod
    def definition_to_runtime_config(
        cls,
        definition: VerticalDefinition,
    ) -> "RuntimeVerticalConfig":
        """Translate a serializable SDK definition into the runtime config shape."""

        from victor.core.verticals.base import VerticalConfig as RuntimeVerticalConfig

        metadata = dict(definition.metadata)
        metadata.setdefault("vertical_name", definition.name)
        metadata.setdefault("vertical_version", definition.version)
        metadata.setdefault("description", definition.description)
        metadata.setdefault("definition_version", definition.definition_version)
        if definition.tool_requirements:
            metadata.setdefault(
                "tool_requirements",
                [requirement.to_dict() for requirement in definition.tool_requirements],
            )
        if definition.capability_requirements:
            metadata.setdefault(
                "capability_requirements",
                [requirement.to_dict() for requirement in definition.capability_requirements],
            )
        if definition.prompt_metadata.templates or definition.prompt_metadata.task_type_hints:
            metadata.setdefault("prompt_metadata", definition.prompt_metadata.to_dict())
        if (
            definition.workflow_metadata.initial_stage is not None
            or definition.workflow_metadata.workflow_spec
            or definition.workflow_metadata.provider_hints
            or definition.workflow_metadata.evaluation_criteria
            or definition.workflow_metadata.metadata
        ):
            metadata.setdefault("workflow_metadata", definition.workflow_metadata.to_dict())
        if (
            definition.team_metadata.teams
            or definition.team_metadata.default_team is not None
            or definition.team_metadata.metadata
        ):
            metadata.setdefault("team_metadata", definition.team_metadata.to_dict())

        return RuntimeVerticalConfig(
            tools=ToolSet.from_tools(definition.get_tool_names()),
            system_prompt=definition.system_prompt,
            stages=cls._to_runtime_stages(definition),
            provider_hints=dict(definition.workflow_metadata.provider_hints),
            evaluation_criteria=list(definition.workflow_metadata.evaluation_criteria),
            metadata=metadata,
        )

    @classmethod
    async def create_agent(
        cls,
        source: VerticalDefinitionSource,
        *,
        provider: str = "anthropic",
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Create an agent using runtime-owned vertical translation.

        .. deprecated::
            Use ``Agent.create(vertical=...)`` directly with VerticalBase implementations.
        """
        import warnings

        warnings.warn(
            "VerticalRuntimeAdapter.create_agent() is deprecated. "
            "Use Agent.create(vertical=...) directly with VerticalBase implementations. "
            "See docs/MIGRATION_GUIDE.md for migration examples.",
            DeprecationWarning,
            stacklevel=2,
        )

        from victor.framework.agent import Agent

        binding = cls.build_runtime_binding(source)
        agent_kwargs = binding.to_agent_kwargs()
        agent_kwargs.update(kwargs)

        if not isinstance(source, VerticalDefinition):
            agent_kwargs.setdefault("vertical", source)

        return await Agent.create(
            provider=provider,
            model=model,
            **agent_kwargs,
        )

    @classmethod
    def _definition_from_legacy_config(
        cls,
        source: Type["SdkVerticalBase"],
        config: Any,
    ) -> VerticalDefinition:
        """Build a definition from the legacy config contract for compatibility."""

        raw_metadata = dict(getattr(config, "metadata", {}) or {})
        description = cls._resolve_string(
            raw_metadata.get("description"),
            getattr(config, "description", None),
            getattr(source, "description", None),
            default=cls._resolve_name(source),
        )
        version = cls._resolve_string(
            raw_metadata.get("vertical_version"),
            getattr(config, "version", None),
            getattr(source, "version", None),
            default="1.0.0",
        )
        metadata = {
            key: value
            for key, value in raw_metadata.items()
            if key not in {"description", "vertical_version", "vertical_name"}
        }

        return VerticalDefinition(
            name=cls._resolve_name(source, getattr(config, "name", None)),
            description=description,
            version=version,
            tools=cls._extract_tool_names(getattr(config, "tools", [])),
            system_prompt=str(getattr(config, "system_prompt", "")),
            stages=dict(getattr(config, "stages", {}) or {}),
            workflow_metadata={
                "provider_hints": dict(getattr(config, "provider_hints", {}) or {}),
                "evaluation_criteria": list(getattr(config, "evaluation_criteria", []) or []),
            },
            metadata=metadata,
        )

    @classmethod
    def _to_runtime_stages(
        cls,
        definition: VerticalDefinition,
    ) -> Dict[str, "RuntimeStageDefinition"]:
        """Convert SDK stage definitions into runtime stage objects."""

        from victor.core.vertical_types import StageDefinition as RuntimeStageDefinition

        runtime_stages: Dict[str, RuntimeStageDefinition] = {}
        for stage_name, stage_definition in definition.stages.items():
            runtime_stages[stage_name] = RuntimeStageDefinition(
                name=stage_definition.name,
                description=stage_definition.description,
                tools=set(stage_definition.required_tools + stage_definition.optional_tools),
            )
        return runtime_stages

    @classmethod
    def _build_legacy_runtime_shim(
        cls,
        source: Any,
        binding: VerticalRuntimeBinding,
    ) -> type:
        """Build a shim class exposing the runtime vertical hook surface."""

        from victor.core.verticals.base import VerticalBase as RuntimeVerticalBase

        runtime_config = binding.runtime_config
        definition = binding.definition
        shim_name = getattr(source, "__name__", definition.name.title())
        team_provider = cls._build_definition_team_provider(definition)
        source_team_provider = getattr(source, "get_team_spec_provider", None)
        source_team_specs = getattr(source, "get_team_specs", None)

        class LegacyRuntimeShim(RuntimeVerticalBase):
            name = definition.name
            description = definition.description
            version = definition.version

            @classmethod
            def get_name(cls) -> str:
                return definition.name

            @classmethod
            def get_description(cls) -> str:
                return definition.description

            @classmethod
            def get_tools(cls) -> list[str]:
                return list(runtime_config.tools.tools)

            @classmethod
            def get_system_prompt(cls) -> str:
                return runtime_config.system_prompt

            @classmethod
            def get_stages(cls) -> Dict[str, "RuntimeStageDefinition"]:
                return dict(runtime_config.stages)

            @classmethod
            def get_provider_hints(cls) -> Dict[str, Any]:
                return dict(runtime_config.provider_hints)

            @classmethod
            def get_evaluation_criteria(cls) -> list[str]:
                return list(runtime_config.evaluation_criteria)

            @classmethod
            def get_config(cls, *, use_cache: bool = True) -> "RuntimeVerticalConfig":
                return runtime_config

            @classmethod
            def get_definition(cls) -> VerticalDefinition:
                return definition

        LegacyRuntimeShim.__name__ = shim_name
        LegacyRuntimeShim.__qualname__ = shim_name
        LegacyRuntimeShim.__module__ = getattr(source, "__module__", __name__)
        LegacyRuntimeShim.__victor_runtime_shim__ = True
        LegacyRuntimeShim.__victor_sdk_source__ = source

        if team_provider is not None and not callable(source_team_provider):
            @classmethod
            def _get_team_spec_provider(
                cls,
            ) -> Optional[DefinitionBackedTeamSpecProvider]:
                return team_provider

            setattr(LegacyRuntimeShim, "get_team_spec_provider", _get_team_spec_provider)

        if team_provider is not None and not callable(source_team_specs):
            @classmethod
            def _get_team_specs(cls) -> Dict[str, Any]:
                return team_provider.get_team_specs()

            setattr(LegacyRuntimeShim, "get_team_specs", _get_team_specs)

        for hook_name in cls._FORWARDED_SDK_HOOKS:
            if hook_name in LegacyRuntimeShim.__dict__:
                continue

            hook = getattr(source, hook_name, None)
            if not callable(hook):
                continue

            setattr(LegacyRuntimeShim, hook_name, cls._build_source_delegate(source, hook_name))

        return LegacyRuntimeShim

    @classmethod
    def _build_definition_team_provider(
        cls,
        definition: VerticalDefinition,
    ) -> Optional[DefinitionBackedTeamSpecProvider]:
        """Build a runtime team-spec provider from SDK team metadata."""

        if not definition.team_metadata.teams:
            return None

        from victor.framework.team_schema import TeamSpec
        from victor.framework.teams import TeamMemberSpec
        from victor.teams.types import MemoryConfig, TeamFormation

        team_specs: Dict[str, TeamSpec] = {}
        for team in definition.team_metadata.teams:
            members = []
            for member in team.members:
                memory_config = (
                    MemoryConfig(**member.memory_config)
                    if member.memory_config
                    else None
                )
                members.append(
                    TeamMemberSpec(
                        role=member.role,
                        goal=member.goal,
                        name=member.name,
                        tool_budget=member.tool_budget,
                        allowed_tools=member.allowed_tools or None,
                        is_manager=member.is_manager,
                        priority=member.priority,
                        backstory=member.backstory,
                        expertise=member.expertise.copy(),
                        personality=member.personality,
                        max_delegation_depth=member.max_delegation_depth,
                        memory=member.memory,
                        memory_config=memory_config,
                        cache=member.cache,
                        verbose=member.verbose,
                        max_iterations=member.max_iterations,
                    )
                )

            team_specs[team.team_id] = TeamSpec(
                name=team.name,
                description=team.description,
                vertical=definition.name,
                formation=TeamFormation(team.formation),
                members=members,
                total_tool_budget=team.total_tool_budget,
                max_iterations=team.max_iterations,
                tags=team.tags.copy(),
                task_types=team.task_types.copy(),
                metadata=dict(team.metadata),
            )

        return DefinitionBackedTeamSpecProvider(
            team_specs=team_specs,
            default_team=definition.team_metadata.default_team,
        )

    @staticmethod
    def _is_runtime_vertical_class(source: Any) -> bool:
        """Return whether the object already exposes the runtime vertical hook surface."""

        if not isinstance(source, type):
            return False

        from victor.core.verticals.base import VerticalBase as RuntimeVerticalBase

        return issubclass(source, RuntimeVerticalBase)

    @staticmethod
    def _build_source_delegate(source: Any, hook_name: str) -> classmethod:
        """Build a classmethod that forwards optional hooks to the definition source."""

        @classmethod
        def _delegate(cls, *args: Any, **kwargs: Any) -> Any:
            return getattr(source, hook_name)(*args, **kwargs)

        return _delegate

    @staticmethod
    def _extract_tool_names(tools: Any) -> list[str]:
        """Normalize tool declarations from framework or SDK ToolSet variants."""

        if tools is None:
            return []
        if isinstance(tools, list):
            return [str(tool_name) for tool_name in tools]
        if hasattr(tools, "tools"):
            return [str(tool_name) for tool_name in tools.tools]
        if hasattr(tools, "names"):
            return [str(tool_name) for tool_name in tools.names]
        return [str(tool_name) for tool_name in tools]

    @staticmethod
    def _resolve_name(source: Any, fallback: Optional[str] = None) -> str:
        """Resolve a stable vertical name for compatibility conversion."""

        return VerticalRuntimeAdapter._resolve_string(
            fallback,
            getattr(source, "name", None),
            getattr(source, "__name__", None),
            default="vertical",
        )

    @staticmethod
    def _resolve_string(*candidates: Optional[Any], default: str) -> str:
        """Return the first non-empty string candidate."""

        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate
        return default


__all__ = [
    "VerticalRuntimeAdapter",
    "VerticalRuntimeBinding",
    "VerticalDefinitionSource",
]
