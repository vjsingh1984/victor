"""Helpers for declarative service registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Any

from victor.core.container import ServiceContainer, ServiceLifetime

Factory = Callable[[ServiceContainer], Any]


@dataclass(frozen=True)
class ServiceRegistrationSpec:
    """Declarative specification for container registrations."""

    protocol: type
    factory: Factory
    lifetime: ServiceLifetime = ServiceLifetime.SINGLETON


def register_services_from_specs(
    container: ServiceContainer, specs: Iterable[ServiceRegistrationSpec]
) -> None:
    """Register services defined by `specs` with the provided container."""

    for spec in specs:
        container.register(spec.protocol, spec.factory, spec.lifetime)


__all__ = ["ServiceRegistrationSpec", "register_services_from_specs"]
