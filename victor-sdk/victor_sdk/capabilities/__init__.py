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

"""Zero-dependency capability contracts for external vertical packages."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Set

from victor_sdk.capabilities.runtime import (
    BaseCapabilityProvider,
    CapabilityLoaderPortProtocol,
    CapabilityConfigMergePolicy,
    CapabilityConfigScopePortProtocol,
    CapabilityConfigService,
    CapabilityEntry,
    CapabilityMetadata,
    CapabilityType,
    DEFAULT_CAPABILITY_CONFIG_SCOPE_KEY,
    OrchestratorCapability,
    build_capability_loader,
    capability,
    load_capability_config,
    register_capability_entries,
    resolve_capability_config_scope_key,
    resolve_capability_config_service,
    store_capability_config,
    update_capability_config_section,
)


class FileOperationType(Enum):
    """Types of generic file operations exposed by a vertical."""

    READ = "read"
    WRITE = "write"
    EDIT = "edit"
    SEARCH = "search"


@dataclass(frozen=True)
class FileOperation:
    """Serializable file operation requirement."""

    operation: FileOperationType
    tool_name: str
    required: bool = True


@dataclass(frozen=True)
class PromptContribution:
    """Serializable prompt contribution contract (pure data, stays in SDK)."""

    name: str
    task_type: str
    hint: str
    tool_budget: int = 15
    priority: int = 50
    system_section: str = ""


# Runtime capability classes moved to victor.framework.capabilities.standard
# Re-exported here for backward compatibility.
try:
    from victor.framework.capabilities.standard import (
        FileOperationsCapability,
        PromptContributionCapability,
    )
except ImportError:
    # SDK-only mode: these require victor-ai runtime
    FileOperationsCapability = None  # type: ignore[assignment,misc]
    PromptContributionCapability = None  # type: ignore[assignment,misc]


__all__ = [
    "BaseCapabilityProvider",
    "CapabilityConfigMergePolicy",
    "CapabilityConfigScopePortProtocol",
    "CapabilityConfigService",
    "CapabilityEntry",
    "CapabilityLoaderPortProtocol",
    "CapabilityMetadata",
    "CapabilityType",
    "DEFAULT_CAPABILITY_CONFIG_SCOPE_KEY",
    "FileOperation",
    "FileOperationsCapability",
    "FileOperationType",
    "OrchestratorCapability",
    "PromptContribution",
    "PromptContributionCapability",
    "build_capability_loader",
    "capability",
    "load_capability_config",
    "register_capability_entries",
    "resolve_capability_config_scope_key",
    "resolve_capability_config_service",
    "store_capability_config",
    "update_capability_config_section",
]
