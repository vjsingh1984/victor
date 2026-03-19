"""Service-related protocol definitions.

These protocols define how verticals provide service configurations.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable, Dict, Any, List, Optional, Type


@runtime_checkable
class ServiceProvider(Protocol):
    """Protocol for providing dependency injection services.

    Service providers enable verticals to register and access
    shared services through dependency injection.
    """

    def get_service_registrations(self) -> Dict[str, Type[Any]]:
        """Return service type registrations.

        Returns:
            Dictionary mapping service names to service types
        """
        ...

    def get_service_instances(self) -> Dict[str, Any]:
        """Return pre-configured service instances.

        Returns:
            Dictionary mapping service names to instances
        """
        ...

    def get_factory_functions(self) -> Dict[str, Callable[[], Any]]:
        """Return factory functions for lazy initialization.

        Returns:
            Dictionary mapping service names to factory functions
        """
        ...


@runtime_checkable
class ServiceLocator(Protocol):
    """Protocol for locating services.

    Service locators provide access to registered services.
    """

    def get_service(self, service_name: str, default: Optional[Any] = None) -> Optional[Any]:
        """Get a service by name.

        Args:
            service_name: Name of the service to retrieve
            default: Default value if service not found

        Returns:
            Service instance or default
        """
        ...

    def has_service(self, service_name: str) -> bool:
        """Check if a service is registered.

        Args:
            service_name: Name of the service to check

        Returns:
            True if service is registered
        """
        ...
