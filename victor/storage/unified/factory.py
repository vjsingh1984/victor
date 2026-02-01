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

"""Factory for creating UnifiedSymbolStore instances.

Supports multiple backends via factory pattern (OCP - Open/Closed Principle):
- sqlite+lancedb: Default, local storage (SQLite graph + LanceDB vectors)
- postgres+pgvector: Cloud-ready single store
- lancedb: Single LanceDB for both vectors and metadata
- duckdb+lancedb: Analytical graph queries

Usage:
    # Default backend
    store = create_symbol_store(repo_root=Path("."))

    # Specific backend
    store = create_symbol_store(
        repo_root=Path("."),
        backend="postgres+pgvector",
        connection_string="postgresql://...",
    )
"""

from pathlib import Path
from typing import Any, Optional

from victor.storage.unified.protocol import UnifiedSymbolStoreProtocol


def create_symbol_store(
    repo_root: Optional[Path] = None,
    backend: str = "sqlite+lancedb",
    **kwargs: Any,
) -> UnifiedSymbolStoreProtocol:
    """Create a UnifiedSymbolStore with the specified backend.

    Args:
        repo_root: Root directory of the repository. If None, uses cwd().
        backend: Storage backend to use:
            - "sqlite+lancedb": Default, SQLite graph + LanceDB vectors (local/air-gapped)
            - "postgres+pgvector": PostgreSQL with pgvector extension (cloud-ready)
            - "lancedb": Single LanceDB store for vectors + metadata
            - "duckdb+lancedb": DuckDB for graph + LanceDB for vectors (analytical)
        **kwargs: Backend-specific configuration:
            - connection_string: For postgres backends
            - embedding_model: Override embedding model (default: sentence-transformers)
            - persist_directory: Override storage location
            - table_name: Override table names

    Returns:
        UnifiedSymbolStoreProtocol implementation

    Example:
        # Local development (air-gapped compatible)
        store = create_symbol_store(repo_root=Path("."))

        # Cloud deployment with PostgreSQL
        store = create_symbol_store(
            repo_root=Path("."),
            backend="postgres+pgvector",
            connection_string="postgresql://user:pass@host/db",
        )
    """
    if repo_root is None:
        repo_root = Path.cwd()

    if backend == "sqlite+lancedb":
        from victor.storage.unified.sqlite_lancedb import SqliteLanceDBStore

        return SqliteLanceDBStore(repo_root, **kwargs)

    elif backend == "postgres+pgvector":
        # Future implementation
        raise NotImplementedError(
            "postgres+pgvector backend not yet implemented. "
            "Use 'sqlite+lancedb' for local development."
        )

    elif backend == "lancedb":
        # Future implementation: LanceDB for both vectors and structured data
        raise NotImplementedError(
            "lancedb-only backend not yet implemented. "
            "Use 'sqlite+lancedb' for local development."
        )

    elif backend == "duckdb+lancedb":
        # Future implementation: DuckDB for analytical queries
        raise NotImplementedError(
            "duckdb+lancedb backend not yet implemented. "
            "Use 'sqlite+lancedb' for local development."
        )

    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Supported: sqlite+lancedb, postgres+pgvector, lancedb, duckdb+lancedb"
        )


__all__ = ["create_symbol_store"]
