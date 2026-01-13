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

"""Framework services module for lifecycle management.

This module provides universal service lifecycle management for use
across all verticals. It promotes common patterns from victor.workflows.services
to the framework layer with enhanced protocol-based interfaces.

Quick Start:
    from victor.framework.services import (
        ServiceRegistry,
        create_sqlite_service,
        create_http_service,
    )

    # Create and register services
    registry = ServiceRegistry()
    db = await create_sqlite_service("project_db", "./data/project.db")
    await registry.register(db)

    # Get service
    service = await registry.get("project_db")

YAML Configuration:
    services:
      - name: database
        type: sqlite
        db_path: ./data/project.db
        enable_wal: true

      - name: api_client
        type: http
        base_url: https://api.example.com
        timeout: 30
"""

from victor.framework.services.lifecycle import (
    # Protocols
    ServiceConfigurable,
    ServiceLifecycleProtocol,
    ServiceTypeHandler,
    # State enums
    HealthStatus,
    ServiceState,
    # Data classes
    BaseServiceConfig,
    DockerServiceConfig,
    ExternalServiceConfig,
    HealthCheckResult,
    HTTPServiceConfig,
    ServiceMetadata,
    SQLiteServiceConfig,
    # Exceptions
    ServiceStartError,
    ServiceStopError,
    # Base class
    BaseService,
    # Built-in services
    DockerServiceHandler,
    ExternalServiceHandler,
    HTTPServiceHandler,
    SQLiteServiceHandler,
    # Registry
    ServiceRegistry,
    # Manager
    ServiceManager,
    # Convenience functions
    create_http_service,
    create_sqlite_service,
)

__all__ = [
    # Protocols
    "ServiceLifecycleProtocol",
    "ServiceConfigurable",
    "ServiceTypeHandler",
    # State enums
    "ServiceState",
    "HealthStatus",
    # Data classes
    "ServiceMetadata",
    "HealthCheckResult",
    "BaseServiceConfig",
    "SQLiteServiceConfig",
    "DockerServiceConfig",
    "HTTPServiceConfig",
    "ExternalServiceConfig",
    # Exceptions
    "ServiceStartError",
    "ServiceStopError",
    # Base class
    "BaseService",
    # Built-in services
    "SQLiteServiceHandler",
    "DockerServiceHandler",
    "HTTPServiceHandler",
    "ExternalServiceHandler",
    # Registry
    "ServiceRegistry",
    # Manager
    "ServiceManager",
    # Convenience functions
    "create_sqlite_service",
    "create_http_service",
]
