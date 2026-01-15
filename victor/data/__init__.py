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

"""Data layer implementations for Victor.

This package contains concrete implementations of repository protocols,
providing separation between the UI layer and data access layer. This
enables Dependency Inversion Principle (DIP) compliance by allowing
UI components to depend on protocol abstractions while implementations
handle the details of database access.

Modules:
    session_repository: SQLite-based session repository implementation

Usage:
    from victor.data.session_repository import SQLiteSessionRepository

    # Create repository instance
    repo = SQLiteSessionRepository()

    # Use via protocol
    from victor.protocols import SessionRepositoryProtocol

    def my_function(session_repo: SessionRepositoryProtocol):
        sessions = await session_repo.list_sessions(limit=10)
        ...

    my_function(repo)
"""

from __future__ import annotations

from victor.data.session_repository import SQLiteSessionRepository

__all__ = [
    "SQLiteSessionRepository",
]
