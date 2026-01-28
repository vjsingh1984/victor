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
    # =============================================================================
    # FILE OPERATIONS
    # =============================================================================
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
        description="Manage files (copy, move, delete, archive, etc.)",
        tools=["file_editor"],
        dependencies=[ToolCapability.FILE_READ],
        conflicts=[],
    ),
    # =============================================================================
    # CODE ANALYSIS
    # =============================================================================
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
        description="Advanced code understanding (autocomplete, symbols, etc.)",
        tools=["code_intelligence"],
        dependencies=[ToolCapability.FILE_READ],
        conflicts=[],
    ),
    CapabilityDefinition(
        name=ToolCapability.CODE_REFACTORING,
        description="Automated code refactoring operations",
        tools=["refactor_tool"],
        dependencies=[
            ToolCapability.FILE_READ,
            ToolCapability.FILE_WRITE,
            ToolCapability.CODE_ANALYSIS,
        ],
        conflicts=[],
    ),
    # =============================================================================
    # SEARCH & DISCOVERY
    # =============================================================================
    CapabilityDefinition(
        name=ToolCapability.WEB_SEARCH,
        description="Search the web for information",
        tools=["browser_tool"],
        dependencies=[],
        conflicts=[],
    ),
    CapabilityDefinition(
        name=ToolCapability.SEMANTIC_SEARCH,
        description="Vector/RAG semantic search across documents and code",
        tools=["code_search"],
        dependencies=[ToolCapability.FILE_READ],
        conflicts=[],
    ),
    CapabilityDefinition(
        name=ToolCapability.KNOWLEDGE_BASE,
        description="Document ingestion, indexing, and querying",
        tools=["code_search"],  # Reuses semantic search infrastructure
        dependencies=[ToolCapability.FILE_READ],
        conflicts=[],
    ),
    # =============================================================================
    # VERSION CONTROL
    # =============================================================================
    CapabilityDefinition(
        name=ToolCapability.VERSION_CONTROL,
        description="Git and version control operations",
        tools=["bash"],  # Uses bash for git commands
        dependencies=[],
        conflicts=[],
    ),
    CapabilityDefinition(
        name=ToolCapability.CHANGE_MANAGEMENT,
        description="Merge management, conflict resolution, PR workflows",
        tools=["bash", "merge_tool"],
        dependencies=[ToolCapability.VERSION_CONTROL],
        conflicts=[],
    ),
    # =============================================================================
    # INFRASTRUCTURE
    # =============================================================================
    CapabilityDefinition(
        name=ToolCapability.CONTAINERIZATION,
        description="Docker, Kubernetes container management",
        tools=["docker_tool"],
        dependencies=[],
        conflicts=[],
    ),
    CapabilityDefinition(
        name=ToolCapability.CLOUD_INFRA,
        description="Terraform, CloudFormation, cloud infrastructure operations",
        tools=["iac_scanner_tool"],
        dependencies=[ToolCapability.FILE_READ],
        conflicts=[],
    ),
    CapabilityDefinition(
        name=ToolCapability.CI_CD,
        description="CI/CD pipeline operations (build, test, deploy)",
        tools=["cicd_tool"],
        dependencies=[],
        conflicts=[],
    ),
    CapabilityDefinition(
        name=ToolCapability.DATABASE,
        description="Database operations, queries, migrations",
        tools=["database_tool"],
        dependencies=[],
        conflicts=[],
    ),
    CapabilityDefinition(
        name=ToolCapability.MONITORING,
        description="Metrics collection, logging, alerting",
        tools=["metrics_tool"],
        dependencies=[],
        conflicts=[],
    ),
    # =============================================================================
    # DEVELOPMENT
    # =============================================================================
    CapabilityDefinition(
        name=ToolCapability.TESTING,
        description="Test execution, coverage, and management",
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
        name=ToolCapability.DEPENDENCY_MGMT,
        description="Package management and dependency analysis",
        tools=["dependency_tool"],
        dependencies=[ToolCapability.FILE_READ],
        conflicts=[],
    ),
    CapabilityDefinition(
        name=ToolCapability.SCAFFOLDING,
        description="Project scaffolding and template generation",
        tools=["scaffold_tool"],
        dependencies=[ToolCapability.FILE_WRITE],
        conflicts=[],
    ),
    # =============================================================================
    # EXECUTION
    # =============================================================================
    CapabilityDefinition(
        name=ToolCapability.CODE_EXECUTION,
        description="Execute code (Python, etc.) in sandboxed environments",
        tools=["code_executor"],
        dependencies=[],
        conflicts=[],
    ),
    CapabilityDefinition(
        name=ToolCapability.BROWSER_AUTOMATION,
        description="Browser control and web automation",
        tools=["browser_tool"],
        dependencies=[],
        conflicts=[],
    ),
    CapabilityDefinition(
        name=ToolCapability.BASH,
        description="Bash shell command execution",
        tools=["bash"],
        dependencies=[],
        conflicts=[],
    ),
    # =============================================================================
    # COMMUNICATION
    # =============================================================================
    CapabilityDefinition(
        name=ToolCapability.MESSAGING,
        description="Slack, Teams, and other messaging integrations",
        tools=["http"],  # Uses HTTP tool for webhooks
        dependencies=[],
        conflicts=[],
    ),
    CapabilityDefinition(
        name=ToolCapability.ISSUE_TRACKING,
        description="Jira, GitHub Issues, and project management integration",
        tools=["http"],  # Uses HTTP tool for API calls
        dependencies=[],
        conflicts=[],
    ),
    CapabilityDefinition(
        name=ToolCapability.NOTIFICATION,
        description="Alert and notification management",
        tools=["http"],  # Uses HTTP tool for notifications
        dependencies=[],
        conflicts=[],
    ),
    # =============================================================================
    # WORKFLOW & AUTOMATION
    # =============================================================================
    CapabilityDefinition(
        name=ToolCapability.WORKFLOW_ORCHESTRATION,
        description="StateGraph, DAG workflow orchestration",
        tools=["workflow_tool"],
        dependencies=[],
        conflicts=[],
    ),
    CapabilityDefinition(
        name=ToolCapability.AUTOMATION,
        description="Batch processing, scheduling, and task automation",
        tools=["batch_processor"],
        dependencies=[],
        conflicts=[],
    ),
    # =============================================================================
    # INTELLIGENCE
    # =============================================================================
    CapabilityDefinition(
        name=ToolCapability.LSP_INTEGRATION,
        description="Language Server Protocol integration for advanced IDE features",
        tools=["lsp_tool"],
        dependencies=[ToolCapability.FILE_READ],
        conflicts=[],
    ),
    CapabilityDefinition(
        name=ToolCapability.AI_ASSISTANCE,
        description="LLM-powered features and AI assistance",
        tools=[],  # Meta-capability, no specific tools
        dependencies=[],
        conflicts=[],
    ),
    # =============================================================================
    # SECURITY & COMPLIANCE
    # =============================================================================
    CapabilityDefinition(
        name=ToolCapability.SECURITY_SCANNING,
        description="Vulnerability scanning and security analysis",
        tools=["audit_tool"],
        dependencies=[ToolCapability.FILE_READ],
        conflicts=[],
    ),
    CapabilityDefinition(
        name=ToolCapability.COMPLIANCE_AUDIT,
        description="Policy checks and compliance auditing",
        tools=["audit_tool"],
        dependencies=[ToolCapability.FILE_READ],
        conflicts=[],
    ),
    CapabilityDefinition(
        name=ToolCapability.AUDIT,
        description="General security and compliance auditing",
        tools=["audit_tool"],
        dependencies=[ToolCapability.FILE_READ],
        conflicts=[],
    ),
    # =============================================================================
    # INTEGRATION
    # =============================================================================
    CapabilityDefinition(
        name=ToolCapability.API_INTEGRATION,
        description="External API integration, HTTP requests, webhooks",
        tools=["http"],
        dependencies=[],
        conflicts=[],
    ),
    # =============================================================================
    # UTILITIES
    # =============================================================================
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
        name=ToolCapability.GRAPH_ANALYSIS,
        description="Dependency graphs, call graphs, and code relationships",
        tools=["code_intelligence"],
        dependencies=[ToolCapability.FILE_READ, ToolCapability.CODE_ANALYSIS],
        conflicts=[],
    ),
]

__all__ = ["BUILTIN_CAPABILITIES"]
