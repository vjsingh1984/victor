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

import contextvars
import uuid
from typing import Optional

# Trace ID for the current request/session
trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")

# Session ID for the current orchestrator session
session_id: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="")

# Vertical name for the current operation
active_vertical: contextvars.ContextVar[str] = contextvars.ContextVar("active_vertical", default="")


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
