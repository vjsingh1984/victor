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
    """

    project_root: str
    victor_dir_name: str = ".victor"

    @property
    def victor_dir(self) -> str:
        return f"{self.project_root}/{self.victor_dir_name}"

    @property
    def logs_dir(self) -> str:
        return f"{self.victor_dir}/logs"

    @property
    def embeddings_dir(self) -> str:
        return f"{self.victor_dir}/embeddings"

    @property
    def graph_dir(self) -> str:
        return f"{self.victor_dir}/graph"

    @property
    def sessions_dir(self) -> str:
        return f"{self.victor_dir}/sessions"


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
