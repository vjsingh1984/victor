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

"""Checkpoint backend types for Victor AI.

This module defines the available checkpoint backend options for
persisting workflow and agent state.
"""

from enum import Enum


class CheckpointBackend(Enum):
    """Available checkpoint backend types.

    Attributes:
        MEMORY: In-memory checkpointing (ephemeral, lost on restart)
        SQLITE: SQLite database for persistent checkpointing
        JSON: JSON file-based checkpointing
        REDIS: Redis-backed distributed checkpointing
        POSTGRES: PostgreSQL database for distributed checkpointing
        FILESYSTEM: Filesystem-based checkpointing
    """

    MEMORY = "memory"
    SQLITE = "sqlite"
    JSON = "json"
    REDIS = "redis"
    POSTGRES = "postgres"
    FILESYSTEM = "filesystem"

    @classmethod
    def is_persistent(cls, backend: "CheckpointBackend") -> bool:
        """Check if a backend provides persistent storage.

        Args:
            backend: The backend type to check

        Returns:
            True if backend persists data across restarts
        """
        return backend not in [cls.MEMORY]

    @classmethod
    def is_distributed(cls, backend: "CheckpointBackend") -> bool:
        """Check if a backend supports distributed deployments.

        Args:
            backend: The backend type to check

        Returns:
            True if backend works across multiple processes/nodes
        """
        return backend in [cls.REDIS, cls.POSTGRES]

    def requires_external_service(self) -> bool:
        """Check if this backend requires an external service.

        Returns:
            True if backend needs Redis, PostgreSQL, etc.
        """
        return self in [self.REDIS, self.POSTGRES]
