"""Configuration protocol definitions for external verticals.

Promoted from victor.config.settings and victor.config.api_keys so
external verticals can access settings and secrets via protocols
instead of importing framework internals.

All types are plain dataclasses/protocols with zero dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class ProjectPathsData:
    """Project path configuration.

    Mirrors victor.config.settings.ProjectPaths as a plain dataclass.
    Derives standard subdirectory paths from project_root.

    Database Consolidation:
        conversation_db is now an alias for project.db (consolidated database)
        Returns path to project-specific database containing conversations, graph, entities

    Usage:
        from victor_sdk.verticals.protocols import ProjectPathsData

        paths = ProjectPathsData(project_root="/home/user/project")
        db_path = paths.conversation_db  # "/home/user/project/.victor/project.db"
    """

    project_root: str
    victor_dir_name: str = ".victor"
    context_file_name: str = "init.md"

    @property
    def victor_dir(self) -> str:
        """Get project-local .victor directory."""
        return f"{self.project_root}/{self.victor_dir_name}"

    @property
    def logs_dir(self) -> str:
        """Get project-local logs directory."""
        return f"{self.victor_dir}/logs"

    @property
    def embeddings_dir(self) -> str:
        """Get project-local embeddings directory."""
        return f"{self.victor_dir}/embeddings"

    @property
    def graph_dir(self) -> str:
        """Get project-local graph directory."""
        return f"{self.victor_dir}/graph"

    @property
    def sessions_dir(self) -> str:
        """Get project-local sessions directory."""
        return f"{self.victor_dir}/sessions"

    @property
    def backups_dir(self) -> str:
        """Get project-local backups directory."""
        return f"{self.victor_dir}/backups"

    @property
    def changes_dir(self) -> str:
        """Get project-local changes (undo/redo) directory."""
        return f"{self.victor_dir}/changes"

    @property
    def conversation_db(self) -> str:
        """Get project-local conversation database path.

        Database consolidation: Now returns path to project.db (consolidated database).
        This property is an alias for backward compatibility.
        """
        return f"{self.victor_dir}/project.db"

    @property
    def conversations_export_dir(self) -> str:
        """Get project-local conversations export directory."""
        return f"{self.victor_dir}/conversations"

    @property
    def index_metadata(self) -> str:
        """Get codebase index metadata file path."""
        return f"{self.victor_dir}/index_metadata.json"

    @property
    def mcp_config(self) -> str:
        """Get project-local MCP configuration file."""
        return f"{self.victor_dir}/mcp.yaml"

    @property
    def project_context_file(self) -> str:
        """Get project context file path (.victor/init.md by default)."""
        return f"{self.victor_dir}/{self.context_file_name}"


@runtime_checkable
class SettingsProviderProtocol(Protocol):
    """Protocol for accessing framework settings.

    External verticals should use this instead of importing
    victor.config.settings directly.
    """

    def get_project_paths(self) -> ProjectPathsData: ...

    def get_setting(self, key: str, default: Any = None) -> Any: ...


@runtime_checkable
class ApiKeyProviderProtocol(Protocol):
    """Protocol for accessing API keys and secrets.

    External verticals should use this instead of importing
    victor.config.api_keys directly.
    """

    def get_service_key(self, service: str) -> Optional[str]: ...


__all__ = [
    "ProjectPathsData",
    "SettingsProviderProtocol",
    "ApiKeyProviderProtocol",
]
