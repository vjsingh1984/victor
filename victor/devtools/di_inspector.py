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

"""DI Container Inspector - Tool to inspect DI container state.

This tool provides:
- Show all registered services
- Show service dependencies and lifecycle
- Identify singleton vs scoped services
- Help debug dependency resolution issues
- Visualize service dependency graph

Usage:
    python -m victor.devtools.di_inspector
    python -m victor.devtools.di_inspector --service ToolRegistry
    python -m victor.devtools.di_inspector --show-dependencies
    python -m victor.devtools.di_inspector --export container.json
"""

from __future__ import annotations

import argparse
import inspect
import logging
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


class ServiceLifetime(str, Enum):
    """Service lifetime."""

    SINGLETON = "singleton"
    SCOPED = "scoped"
    TRANSIENT = "transient"


@dataclass
class ServiceDescriptor:
    """Information about a registered service."""

    service_type: str
    lifetime: ServiceLifetime
    implementation_type: Optional[str] = None
    file_path: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    is_created: bool = False

    def __str__(self) -> str:
        """Return string representation."""
        lifetime_icon = {
            ServiceLifetime.SINGLETON: "🔷",
            ServiceLifetime.SCOPED: "🔸",
            ServiceLifetime.TRANSIENT: "🔹",
        }

        icon = lifetime_icon.get(self.lifetime, "")
        created = "✓" if self.is_created else "✗"

        return f"{icon} {self.service_type} [{self.lifetime.value}] {created}"


class DIInspector:
    """Inspector for Victor's DI container."""

    def __init__(self):
        """Initialize the inspector."""
        self.services: Dict[str, ServiceDescriptor] = {}
        self.dependency_graph: Dict[str, Set[str]] = {}
        self._container = None
        self._has_loaded = False

    def load_container(self) -> bool:
        """Load and inspect the DI container.

        Returns:
            True if successful, False otherwise
        """
        if self._has_loaded:
            return True

        try:
            from victor.core.container import ServiceContainer, ServiceLifetime as CoreLifetime

            # Try to get the global container
            try:
                from victor.core.container import get_container

                self._container = get_container()
            except Exception:
                # Create a new container and register services
                self._container = ServiceContainer()
                self._register_orchestrator_services()

            # Extract service registrations
            self._extract_services()

            self._has_loaded = True
            logger.info("Successfully loaded DI container")
            return True

        except ImportError as e:
            logger.error(f"Failed to import DI container: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load DI container: {e}")
            return False

    def _register_orchestrator_services(self) -> None:
        """Register orchestrator services for inspection."""
        try:
            from victor.config.settings import Settings
            from victor.agent.service_provider import configure_orchestrator_services

            # Create default settings
            settings = Settings()

            # Register services
            configure_orchestrator_services(self._container, settings)

        except Exception as e:
            logger.warning(f"Could not register orchestrator services: {e}")

    def _extract_services(self) -> None:
        """Extract service information from container."""
        if not self._container:
            return

        try:
            # Get registered types
            registered_types = self._container.get_registered_types()

            for service_type in registered_types:
                service_name = self._get_type_name(service_type)

                # Get descriptor if available
                descriptor = None
                if hasattr(self._container, "_descriptors"):
                    descriptor = self._container._descriptors.get(service_type)

                if descriptor:
                    lifetime = ServiceLifetime(descriptor.lifetime.value)
                    is_created = descriptor.instance is not None
                else:
                    lifetime = ServiceLifetime.SINGLETON
                    is_created = False

                # Get implementation type
                impl_type = None
                if descriptor and hasattr(descriptor, "factory"):
                    impl_type = self._infer_implementation_type(descriptor.factory)

                # Get file path
                file_path = None
                try:
                    module = inspect.getmodule(service_type)
                    if module and hasattr(module, "__file__"):
                        file_path = module.__file__
                except Exception:
                    pass

                # Get dependencies
                dependencies = self._extract_dependencies(service_type)

                service_desc = ServiceDescriptor(
                    service_type=service_name,
                    lifetime=lifetime,
                    implementation_type=impl_type,
                    file_path=file_path,
                    dependencies=dependencies,
                    is_created=is_created,
                )

                self.services[service_name] = service_desc
                self.dependency_graph[service_name] = set(dependencies)

        except Exception as e:
            logger.warning(f"Error extracting services: {e}")

    def _get_type_name(self, type_obj: Any) -> str:
        """Get type name."""
        try:
            if hasattr(type_obj, "__name__"):
                return type_obj.__name__
            return str(type_obj)
        except Exception:
            return str(type_obj)

    def _infer_implementation_type(self, factory: Any) -> Optional[str]:
        """Infer implementation type from factory."""
        try:
            # Try to call the factory and inspect the result
            # Note: This might actually create the service
            if callable(factory):
                # Check if factory has source code
                source = inspect.getsource(factory)
                # Look for class instantiation
                import re

                matches = re.findall(r"(\w+)\(", source)
                if matches:
                    return matches[0]
        except Exception:
            pass

        return None

    def _extract_dependencies(self, service_type: Any) -> List[str]:
        """Extract dependencies from service type."""
        dependencies = []

        try:
            # Check __init__ signature
            if hasattr(service_type, "__init__"):
                sig = inspect.signature(service_type.__init__)

                for param_name, param in sig.parameters.items():
                    if param_name == "self":
                        continue

                    # Get annotation
                    if param.annotation and param.annotation != inspect.Parameter.empty:
                        dep_name = self._get_type_name(param.annotation)
                        dependencies.append(dep_name)

        except Exception:
            pass

        return dependencies

    def list_services(
        self,
        lifetime: Optional[ServiceLifetime] = None,
        show_created_only: bool = False,
    ) -> List[ServiceDescriptor]:
        """List registered services.

        Args:
            lifetime: Filter by lifetime
            show_created_only: Only show instantiated services

        Returns:
            List of service descriptors
        """
        services = list(self.services.values())

        if lifetime:
            services = [s for s in services if s.lifetime == lifetime]

        if show_created_only:
            services = [s for s in services if s.is_created]

        return sorted(services, key=lambda s: s.service_type)

    def show_service_details(self, service_name: str) -> None:
        """Show details for a specific service.

        Args:
            service_name: Name of the service
        """
        if service_name not in self.services:
            print(f"Service '{service_name}' not found")
            return

        service = self.services[service_name]

        print(f"\n{'=' * 60}")
        print(f"Service: {service.service_type}")
        print(f"{'=' * 60}")
        print(f"Type: {service.service_type}")
        print(f"Implementation: {service.implementation_type or 'Unknown'}")
        print(f"Lifetime: {service.lifetime.value}")
        print(f"Status: {'Created' if service.is_created else 'Not created'}")

        if service.file_path:
            print(f"File: {service.file_path}")

        if service.dependencies:
            print(f"\nDependencies ({len(service.dependencies)}):")
            for dep in service.dependencies:
                status = "✓" if dep in self.services else "✗"
                print(f"  {status} {dep}")
        else:
            print("\nNo dependencies detected")

        # Show dependents (what depends on this service)
        dependents = self.find_dependents(service_name)
        if dependents:
            print(f"\nDepended on by ({len(dependents)}):")
            for dep in dependents:
                print(f"  - {dep}")

        print()

    def find_dependents(self, service_name: str) -> List[str]:
        """Find services that depend on this service.

        Args:
            service_name: Name of the service

        Returns:
            List of service names that depend on this service
        """
        dependents = []

        for name, service in self.services.items():
            if service_name in service.dependencies:
                dependents.append(name)

        return sorted(dependents)

    def show_dependency_graph(self, service_name: Optional[str] = None) -> None:
        """Show dependency graph.

        Args:
            service_name: Root service (None for all services)
        """
        if service_name:
            if service_name not in self.services:
                print(f"Service '{service_name}' not found")
                return

            print(f"\nDependency graph for {service_name}:")
            print("=" * 60)
            self._print_dependency_tree(service_name, depth=0)
        else:
            print("\nService Dependency Graph:")
            print("=" * 60)

            for name in sorted(self.services.keys()):
                service = self.services[name]
                if service.dependencies:
                    print(f"\n{name}:")
                    for dep in service.dependencies:
                        print(f"  -> {dep}")

    def _print_dependency_tree(self, service_name: str, depth: int = 0, visited: Optional[Set[str]] = None) -> None:
        """Print dependency tree recursively."""
        if visited is None:
            visited = set()

        if depth > 3 or service_name in visited:
            return

        visited.add(service_name)
        service = self.services.get(service_name)

        if not service:
            return

        indent = "  " * depth
        print(f"{indent}{service_name}")

        for dep in service.dependencies:
            self._print_dependency_tree(dep, depth + 1, visited.copy())

    def check_resolution_order(self) -> List[str]:
        """Check if services can be resolved in topological order.

        Returns:
            Ordered list of service names, or empty list if cycles detected
        """
        # Topological sort using Kahn's algorithm
        in_degree = {name: len(deps) for name, deps in self.dependency_graph.items()}

        # Add dependencies that aren't in our services
        for deps in self.dependency_graph.values():
            for dep in deps:
                if dep not in in_degree:
                    in_degree[dep] = 0

        # Start with nodes that have no dependencies
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            # Find nodes that depend on this one
            for name, deps in self.dependency_graph.items():
                if node in deps:
                    in_degree[name] -= 1
                    if in_degree[name] == 0:
                        queue.append(name)

        # Check for cycles
        if len(result) != len(in_degree):
            print("Warning: Circular dependencies detected!")
            remaining = [name for name in in_degree if name not in result]
            print(f"Services in cycles: {remaining}")
            return []

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get container statistics."""
        total = len(self.services)
        singletons = sum(1 for s in self.services.values() if s.lifetime == ServiceLifetime.SINGLETON)
        scoped = sum(1 for s in self.services.values() if s.lifetime == ServiceLifetime.SCOPED)
        transient = sum(1 for s in self.services.values() if s.lifetime == ServiceLifetime.TRANSIENT)

        created = sum(1 for s in self.services.values() if s.is_created)

        # Find most depended-upon services
        most_depended = []
        for name in self.services:
            dependents = self.find_dependents(name)
            most_depended.append((name, len(dependents)))

        most_depended.sort(key=lambda x: x[1], reverse=True)

        return {
            "total_services": total,
            "singletons": singletons,
            "scoped": scoped,
            "transient": transient,
            "created": created,
            "most_depended_upon": most_depended[:10],
        }

    def export_json(self, output_path: Path) -> None:
        """Export container state to JSON.

        Args:
            output_path: Path to output JSON file
        """
        import json

        data = {
            "services": {
                name: {
                    "type": service.service_type,
                    "lifetime": service.lifetime.value,
                    "implementation": service.implementation_type,
                    "file": service.file_path,
                    "dependencies": service.dependencies,
                    "is_created": service.is_created,
                }
                for name, service in self.services.items()
            },
            "statistics": self.get_statistics(),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"Exported to {output_path}")

    def print_summary(self) -> None:
        """Print container summary."""
        stats = self.get_statistics()

        print(f"\n{'=' * 60}")
        print("DI Container Summary")
        print(f"{'=' * 60}")
        print(f"Total Services: {stats['total_services']}")
        print(f"  Singletons: {stats['singletons']}")
        print(f"  Scoped: {stats['scoped']}")
        print(f"  Transient: {stats['transient']}")
        print(f"\nInstantiated: {stats['created']}")
        print(f"Not instantiated: {stats['total_services'] - stats['created']}")

        if stats['most_depended_upon']:
            print("\nTop 10 most depended-upon services:")
            for name, count in stats['most_depended_upon']:
                print(f"  {name}: {count}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Inspect Victor's DI container",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all services
  python -m victor.devtools.di_inspector --list

  # Show singletons only
  python -m victor.devtools.di_inspector --list --lifetime singleton

  # Show details for a specific service
  python -m victor.devtools.di_inspector --service ToolRegistry

  # Show dependency graph
  python -m victor.devtools.di_inspector --show-dependencies

  # Check resolution order
  python -m victor.devtools.di_inspector --check-resolution

  # Export to JSON
  python -m victor.devtools.di_inspector --export container.json
        """,
    )

    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List all registered services",
    )

    parser.add_argument(
        "--lifetime",
        type=str,
        choices=["singleton", "scoped", "transient"],
        metavar="TYPE",
        help="Filter by lifetime",
    )

    parser.add_argument(
        "--created-only",
        action="store_true",
        help="Show only instantiated services",
    )

    parser.add_argument(
        "-s",
        "--service",
        type=str,
        metavar="NAME",
        help="Show details for a specific service",
    )

    parser.add_argument(
        "-d",
        "--show-dependencies",
        action="store_true",
        help="Show dependency graph",
    )

    parser.add_argument(
        "--check-resolution",
        action="store_true",
        help="Check if services can be resolved (detect cycles)",
    )

    parser.add_argument(
        "--export",
        type=str,
        metavar="PATH",
        help="Export container state to JSON",
    )

    args = parser.parse_args()

    # Load container
    inspector = DIInspector()

    if not inspector.load_container():
        print("Failed to load DI container")
        return 1

    # Show summary by default
    if not any([args.list, args.service, args.show_dependencies, args.check_resolution, args.export]):
        inspector.print_summary()
        return 0

    # List services
    if args.list:
        lifetime = ServiceLifetime(args.lifetime) if args.lifetime else None

        services = inspector.list_services(
            lifetime=lifetime,
            show_created_only=args.created_only,
        )

        print(f"\nFound {len(services)} services:\n")

        for service in services:
            print(f"  {service}")

    # Show service details
    if args.service:
        inspector.show_service_details(args.service)

    # Show dependencies
    if args.show_dependencies:
        inspector.show_dependency_graph()

    # Check resolution
    if args.check_resolution:
        print("\nChecking service resolution order...")
        order = inspector.check_resolution_order()

        if order:
            print(f"\nValid resolution order ({len(order)} services):")
            for i, name in enumerate(order, 1):
                print(f"  {i}. {name}")
        else:
            print("\nCannot resolve services - circular dependencies detected!")

    # Export
    if args.export:
        inspector.export_json(Path(args.export))

    return 0


if __name__ == "__main__":
    sys.exit(main())
