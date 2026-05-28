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

"""Context utilities for Victor.

This module provides context-aware utilities for tracing, session management,
and other cross-cutting concerns using contextvars.
"""

from __future__ import annotations

from contextlib import contextmanager
import contextvars
import uuid
from typing import Iterator, Optional

# Trace ID for the current request/session
trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")

# Session ID for the current orchestrator session
session_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "session_id", default=""
)

# Vertical name for the current operation
active_vertical: contextvars.ContextVar[str] = contextvars.ContextVar(
    "active_vertical", default=""
)

# Vertical manifest version for the current operation
active_vertical_manifest_version: contextvars.ContextVar[str] = contextvars.ContextVar(
    "active_vertical_manifest_version",
    default="",
)

# Vertical plugin namespace for the current operation
active_vertical_namespace: contextvars.ContextVar[str] = contextvars.ContextVar(
    "active_vertical_namespace",
    default="",
)


def get_trace_id() -> str:
    """Get the current trace ID or generate a new one if not set."""
    tid = trace_id.get()
    if not tid:
        tid = str(uuid.uuid4())
        trace_id.set(tid)
    return tid


def set_trace_id(tid: str) -> contextvars.Token:
    """Set the current trace ID."""
    return trace_id.set(tid)


def get_session_id() -> str:
    """Get the current session ID."""
    return session_id.get()


def set_session_id(sid: str) -> contextvars.Token:
    """Set the current session ID."""
    return session_id.set(sid)


def get_active_vertical() -> str:
    """Get the current active vertical name."""
    return active_vertical.get()


def set_active_vertical(name: str) -> contextvars.Token:
    """Set the current active vertical name."""
    return active_vertical.set(name)


def get_active_vertical_manifest_version() -> str:
    """Get the current active vertical manifest version."""
    return active_vertical_manifest_version.get()


def set_active_vertical_manifest_version(version: str) -> contextvars.Token:
    """Set the current active vertical manifest version."""
    return active_vertical_manifest_version.set(version)


def get_active_vertical_namespace() -> str:
    """Get the current active vertical plugin namespace."""
    return active_vertical_namespace.get()


def set_active_vertical_namespace(namespace: str) -> contextvars.Token:
    """Set the current active vertical plugin namespace."""
    return active_vertical_namespace.set(namespace)


@contextmanager
def bind_active_vertical(
    name: str,
    *,
    manifest_version: str = "",
    namespace: str = "",
) -> Iterator[None]:
    """Bind vertical metadata to the current execution context."""

    name_token = active_vertical.set(name)
    version_token = active_vertical_manifest_version.set(manifest_version)
    namespace_token = active_vertical_namespace.set(namespace)
    try:
        yield
    finally:
        active_vertical.reset(name_token)
        active_vertical_manifest_version.reset(version_token)
        active_vertical_namespace.reset(namespace_token)
