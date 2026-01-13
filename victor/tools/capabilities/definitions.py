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

"""Built-in tool capability definitions.

Provides pre-defined capability definitions for common tools.

Phase 2, Work Stream 2.2: Tool Capability Groups
"""

from victor.tools.capabilities.system import ToolCapability, CapabilityDefinition

# =============================================================================
# Built-in Capability Definitions
# =============================================================================

BUILTIN_CAPABILITIES = [
    # File Capabilities
    CapabilityDefinition(
        name=ToolCapability.FILE_READ,
        description="Read files and directories from the filesystem",
        tools=["read", "ls"],
        dependencies=[],
        conflicts=[],
    ),
    CapabilityDefinition(
        name=ToolCapability.FILE_WRITE,
        description="Write and edit files on the filesystem",
        tools=["write", "edit"],
        dependencies=[ToolCapability.FILE_READ],
        conflicts=[],
    ),
    CapabilityDefinition(
        name=ToolCapability.FILE_MANAGEMENT,
        description="Manage files (copy, move, delete, etc.)",
        tools=["file_editor"],
        dependencies=[ToolCapability.FILE_READ],
        conflicts=[],
    ),
    # Code Capabilities
    CapabilityDefinition(
        name=ToolCapability.CODE_ANALYSIS,
        description="Analyze code structure and patterns",
        tools=["code_intelligence"],
        dependencies=[ToolCapability.FILE_READ],
        conflicts=[],
    ),
    CapabilityDefinition(
        name=ToolCapability.CODE_SEARCH,
        description="Search codebase semantically and by pattern",
        tools=["code_search"],
        dependencies=[ToolCapability.FILE_READ],
        conflicts=[],
    ),
    CapabilityDefinition(
        name=ToolCapability.CODE_REVIEW,
        description="Review code quality and suggest improvements",
        tools=["code_review"],
        dependencies=[ToolCapability.FILE_READ, ToolCapability.CODE_ANALYSIS],
        conflicts=[],
    ),
    CapabilityDefinition(
        name=ToolCapability.CODE_INTELLIGENCE,
        description="Advanced code understanding (autocomplete, etc.)",
        tools=["code_intelligence"],
        dependencies=[ToolCapability.FILE_READ],
        conflicts=[],
    ),
    # Search Capabilities
    CapabilityDefinition(
        name=ToolCapability.WEB_SEARCH,
        description="Search the web for information",
        tools=["browser_tool"],
        dependencies=[],
        conflicts=[],
    ),
    # Infrastructure Capabilities
    CapabilityDefinition(
        name=ToolCapability.VERSION_CONTROL,
        description="Git and version control operations",
        tools=["bash"],  # Uses bash for git commands
        dependencies=[],
        conflicts=[],
    ),
    CapabilityDefinition(
        name=ToolCapability.DATABASE,
        description="Database operations and queries",
        tools=["database_tool"],
        dependencies=[],
        conflicts=[],
    ),
    CapabilityDefinition(
        name=ToolCapability.DOCKER,
        description="Docker container management",
        tools=["docker_tool"],
        dependencies=[],
        conflicts=[],
    ),
    CapabilityDefinition(
        name=ToolCapability.CI_CD,
        description="CI/CD pipeline operations",
        tools=["cicd_tool"],
        dependencies=[],
        conflicts=[],
    ),
    # Development Capabilities
    CapabilityDefinition(
        name=ToolCapability.TESTING,
        description="Test execution and management",
        tools=["bash"],  # Uses bash to run tests
        dependencies=[ToolCapability.FILE_READ],
        conflicts=[],
    ),
    CapabilityDefinition(
        name=ToolCapability.DOCUMENTATION,
        description="Documentation generation and analysis",
        tools=["documentation_tool"],
        dependencies=[ToolCapability.FILE_READ, ToolCapability.FILE_WRITE],
        conflicts=[],
    ),
    CapabilityDefinition(
        name=ToolCapability.DEPENDENCY,
        description="Dependency management and analysis",
        tools=["dependency_tool"],
        dependencies=[ToolCapability.FILE_READ],
        conflicts=[],
    ),
    # Execution Capabilities
    CapabilityDefinition(
        name=ToolCapability.BASH,
        description="Bash shell command execution",
        tools=["bash"],
        dependencies=[],
        conflicts=[],
    ),
    CapabilityDefinition(
        name=ToolCapability.BROWSER,
        description="Browser automation and control",
        tools=["browser_tool"],
        dependencies=[],
        conflicts=[],
    ),
    # Utility Capabilities
    CapabilityDefinition(
        name=ToolCapability.CACHE,
        description="Caching and memoization",
        tools=["cache_tool"],
        dependencies=[],
        conflicts=[],
    ),
    CapabilityDefinition(
        name=ToolCapability.BATCH,
        description="Batch processing of multiple operations",
        tools=["batch_processor"],
        dependencies=[],
        conflicts=[],
    ),
    CapabilityDefinition(
        name=ToolCapability.AUDIT,
        description="Security and compliance auditing",
        tools=["audit_tool"],
        dependencies=[ToolCapability.FILE_READ],
        conflicts=[],
    ),
]

__all__ = ["BUILTIN_CAPABILITIES"]
