"""Host-owned adapter for translating vertical definitions into runtime config."""

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


class VerticalRuntimeAdapter:
    """Translate definition-layer vertical contracts into runtime configuration."""

    _legacy_runtime_shims: Dict[Any, type] = {}

    @classmethod
    def resolve_definition(cls, source: VerticalDefinitionSource) -> VerticalDefinition:
        """Resolve a vertical source into a normalized SDK definition."""

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
        """Build the runtime binding for a vertical source."""

        definition = cls.resolve_definition(source)
        runtime_config = cls.definition_to_runtime_config(definition)
        return VerticalRuntimeBinding(definition=definition, runtime_config=runtime_config)

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
        """Create an agent using runtime-owned vertical translation."""

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
        shim_name = f"{getattr(source, '__name__', definition.name.title())}RuntimeShim"

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
        return LegacyRuntimeShim

    @staticmethod
    def _is_runtime_vertical_class(source: Any) -> bool:
        """Return whether the object already exposes the runtime vertical hook surface."""

        return callable(getattr(source, "get_tools", None)) and callable(
            getattr(source, "get_system_prompt", None)
        )

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
