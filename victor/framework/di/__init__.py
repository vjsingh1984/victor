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

"""Enhanced Dependency Injection Container with auto-resolution.

This module provides an advanced DI container that extends the current
ServiceContainer with additional capabilities for automatic dependency
resolution through constructor inspection.

Key Features:
- Auto-resolution of constructor dependencies via inspect
- Lifecycle management (singleton, transient, scoped)
- Lazy initialization
- Circular dependency detection
- Factory functions as fallback
- Type-safe registration and resolution

Example Usage:
    from victor.framework.di import DIContainer, ServiceLifetime

    # Create container
    container = DIContainer()

    # Auto-register with constructor injection
    container.register(Logger)
    container.register(Database, lifetime=ServiceLifetime.SINGLETON)
    container.register(UserService)

    # Resolve with auto-injected dependencies
    user_service = container.get(UserService)
    # Logger and Database are automatically injected into UserService

Migration Strategy:
    This container is designed to coexist with victor.core.container.ServiceContainer.
    Use this for new code or gradual migration of existing services.
"""

from victor.framework.di.container import (
    DIContainer,
    DIScope,
    ServiceLifetime,
    DIError,
    ServiceNotFoundError,
    CircularDependencyError,
    ServiceAlreadyRegisteredError,
    create_container,
)

__all__ = [
    "DIContainer",
    "DIScope",
    "ServiceLifetime",
    "DIError",
    "ServiceNotFoundError",
    "CircularDependencyError",
    "ServiceAlreadyRegisteredError",
    "create_container",
]
