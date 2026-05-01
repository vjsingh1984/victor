"""Tests for SDK-owned capability runtime contracts."""

from __future__ import annotations

import pytest

from victor_sdk.capabilities import (
    BaseCapabilityProvider,
    CapabilityConfigService,
    CapabilityEntry,
    CapabilityMetadata,
    CapabilityType,
    OrchestratorCapability,
    build_capability_loader,
    capability,
    load_capability_config,
    register_capability_entries,
    resolve_capability_config_scope_key,
    resolve_capability_config_service,
    store_capability_config,
)


class _StubContainer:
    def __init__(self, service):
        self._service = service

    def get_optional(self, service_type):
        if isinstance(self._service, service_type):
            return self._service
        return self._service


class _ServiceBackedOrchestrator:
    def __init__(self, service):
        self._container = _StubContainer(service)

    def get_service_container(self):
        return self._container


class _ScopedServiceBackedOrchestrator(_ServiceBackedOrchestrator):
    def __init__(self, service, scope_key: str):
        super().__init__(service)
        self._scope_key = scope_key

    def get_capability_config_scope_key(self):
        return self._scope_key


class _FallbackOrchestrator:
    def __init__(self):
        self.public_config = {"enabled": True}


class _DemoCapabilityProvider(BaseCapabilityProvider[str]):
    def get_capabilities(self):
        return {"alpha": "A"}

    def get_capability_metadata(self):
        return {
            "alpha": CapabilityMetadata(
                name="alpha",
                description="Demo capability",
                tags=["demo"],
            )
        }


class _RecordingCapabilityLoader:
    def __init__(self) -> None:
        self.calls = []

    def _register_capability_internal(
        self,
        *,
        capability,
        handler=None,
        getter_handler=None,
        source_module=None,
    ):
        self.calls.append(
            {
                "capability": capability,
                "handler": handler,
                "getter_handler": getter_handler,
                "source_module": source_module,
            }
        )


def test_capability_entry_exposes_wrapped_capability_metadata() -> None:
    capability_def = OrchestratorCapability(
        name="demo",
        capability_type=CapabilityType.TOOL,
        setter="set_demo",
    )
    entry = CapabilityEntry(capability=capability_def)

    assert entry.name == "demo"
    assert entry.version == "1.0"
    assert entry.capability_type == CapabilityType.TOOL


def test_capability_decorator_attaches_sdk_metadata() -> None:
    @capability(name="demo", capability_type=CapabilityType.SAFETY)
    def apply_demo():
        return True

    assert apply_demo._capability_meta["name"] == "demo"  # type: ignore[attr-defined]
    assert apply_demo._capability_meta["capability_type"] == CapabilityType.SAFETY  # type: ignore[attr-defined]


def test_capability_config_helpers_use_sdk_service_first() -> None:
    service = CapabilityConfigService()
    orchestrator = _ServiceBackedOrchestrator(service)

    assert resolve_capability_config_service(orchestrator) is service
    assert store_capability_config(orchestrator, "alpha", {"x": 1}) is True
    assert load_capability_config(orchestrator, "alpha", {"x": 0}) == {"x": 1}


def test_capability_config_helpers_are_scope_isolated() -> None:
    service = CapabilityConfigService()
    orchestrator_a = _ScopedServiceBackedOrchestrator(service, "a")
    orchestrator_b = _ScopedServiceBackedOrchestrator(service, "b")

    store_capability_config(orchestrator_a, "alpha", {"x": 1})
    store_capability_config(orchestrator_b, "alpha", {"x": 9})

    assert resolve_capability_config_scope_key(orchestrator_a) == "a"
    assert load_capability_config(orchestrator_a, "alpha", {"x": 0}) == {"x": 1}
    assert load_capability_config(orchestrator_b, "alpha", {"x": 0}) == {"x": 9}


def test_private_fallback_only_blocked_in_strict_mode(monkeypatch) -> None:
    orchestrator = _FallbackOrchestrator()
    monkeypatch.delenv("VICTOR_STRICT_FRAMEWORK_PRIVATE_FALLBACKS", raising=False)
    assert load_capability_config(
        orchestrator,
        "private_name",
        {"enabled": False},
        fallback_attr="_private_name",
    ) == {"enabled": False}

    monkeypatch.setenv("VICTOR_STRICT_FRAMEWORK_PRIVATE_FALLBACKS", "1")
    with pytest.raises(RuntimeError):
        load_capability_config(
            orchestrator,
            "private_name",
            {"enabled": False},
            fallback_attr="_private_name",
        )


def test_base_capability_provider_exposes_metadata_and_metrics() -> None:
    provider = _DemoCapabilityProvider()
    provider.record_access("alpha")

    assert provider.has_capability("alpha") is True
    assert provider.get_capability("alpha") == "A"
    data = provider.get_observability_data()
    assert data["capabilities"] == ["alpha"]
    assert data["metadata"]["alpha"]["description"] == "Demo capability"
    assert data["metrics"]["access_count"] == 1


def test_register_capability_entries_populates_loader_protocol() -> None:
    loader = _RecordingCapabilityLoader()
    entry = CapabilityEntry(
        capability=OrchestratorCapability(
            name="demo",
            capability_type=CapabilityType.TOOL,
            setter="set_demo",
        ),
        handler=lambda orchestrator: orchestrator,
        source_module="demo.vertical",
    )

    returned_loader = register_capability_entries(loader, [entry])

    assert returned_loader is loader
    assert len(loader.calls) == 1
    assert loader.calls[0]["capability"].name == "demo"
    assert loader.calls[0]["source_module"] == "demo.vertical"


def test_build_capability_loader_uses_explicit_factory() -> None:
    entry = CapabilityEntry(
        capability=OrchestratorCapability(
            name="demo",
            capability_type=CapabilityType.MODE,
            setter="set_demo",
        ),
        source_module="demo.vertical",
    )

    built_loader = build_capability_loader(
        [entry],
        loader_factory=_RecordingCapabilityLoader,
        source_module="override.vertical",
    )

    assert isinstance(built_loader, _RecordingCapabilityLoader)
    assert len(built_loader.calls) == 1
    assert built_loader.calls[0]["source_module"] == "override.vertical"
