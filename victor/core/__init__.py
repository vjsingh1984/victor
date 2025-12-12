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

"""Core infrastructure modules for Victor.

This package provides foundational infrastructure:
- Dependency injection container (ServiceContainer)
- Lazy initialization utilities (avoiding circular imports)
- Error handling utilities
- Retry mechanisms
"""

from victor.core.container import (
    ServiceContainer,
    ServiceScope,
    ServiceLifetime,
    get_container,
    set_container,
    reset_container,
)

from victor.core.lazy import (
    LazyProperty,
    deferred_import,
    SingletonFactory,
    CircularImportInfo,
    KNOWN_CIRCULAR_IMPORTS,
    get_circular_import_info,
    list_circular_imports,
)

from victor.core.protocols import (
    OrchestratorProtocol,
    TaskClassifierProtocol,
    IntentClassifierProtocol,
    EmbeddingServiceProtocol,
    ToolCallingAdapterProtocol,
    IntelligentPipelineProtocol,
    ProviderProtocol,
    CacheProtocol,
    TaskClassificationResultProtocol,
    IntentClassificationResultProtocol,
    ServiceFactory,
)

__all__ = [
    # Container
    "ServiceContainer",
    "ServiceScope",
    "ServiceLifetime",
    "get_container",
    "set_container",
    "reset_container",
    # Lazy initialization
    "LazyProperty",
    "deferred_import",
    "SingletonFactory",
    "CircularImportInfo",
    "KNOWN_CIRCULAR_IMPORTS",
    "get_circular_import_info",
    "list_circular_imports",
    # Protocols (for interface-based decoupling)
    "OrchestratorProtocol",
    "TaskClassifierProtocol",
    "IntentClassifierProtocol",
    "EmbeddingServiceProtocol",
    "ToolCallingAdapterProtocol",
    "IntelligentPipelineProtocol",
    "ProviderProtocol",
    "CacheProtocol",
    "TaskClassificationResultProtocol",
    "IntentClassificationResultProtocol",
    "ServiceFactory",
]
