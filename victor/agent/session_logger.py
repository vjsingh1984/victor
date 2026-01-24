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

"""Session-aware logger setup.

Integrates session ID into logger names for better traceability.

This module provides convenience functions for getting session-aware loggers
that work with Victor's existing setup_logging infrastructure.

Usage:
    # Get session-aware logger
    logger = get_session_logger("myproj-9Kx7Z2")
    logger.info("Session started")  # Logs as [victor.myproj-9Kx7Z2] INFO: Session started

    # Or from agent
    logger = get_agent_logger(agent)
    logger.info("Agent initialized")  # Uses agent.active_session_id if available

Note: File logging automatically includes session_id when setup_logging() is
called with session_id parameter (done automatically in chat command).
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_session_logger(session_id: Optional[str] = None, component: str = "") -> logging.Logger:
    """Get a session-aware logger.

    Args:
        session_id: Optional session ID (format: projectroot-base62)
        component: Optional component name (e.g., "tools", "orchestrator")

    Returns:
        Logger instance with session ID in name

    Examples:
        >>> logger = get_session_logger("myproj-9Kx7Z2")
        >>> logger.name
        'victor.myproj-9Kx7Z2'

        >>> logger = get_session_logger("myproj-9Kx7Z2", "tools")
        >>> logger.name
        'victor.myproj-9Kx7Z2.tools'
    """
    if session_id:
        # Session-specific logger
        name = f"victor.{session_id}"
        if component:
            name = f"{name}.{component}"
    else:
        # Default logger
        name = "victor"
        if component:
            name = f"{name}.{component}"

    return logging.getLogger(name)


def get_agent_logger(agent: Any, component: str = "") -> logging.Logger:
    """Get logger for agent with session ID if available.

    Args:
        agent: Agent instance (should have active_session_id attribute)
        component: Optional component name

    Returns:
        Logger instance

    Examples:
        >>> logger = get_agent_logger(agent)
        >>> logger.name
        'victor.myproj-9Kx7Z2'  # if agent.active_session_id exists

        >>> logger = get_agent_logger(agent, "tools")
        >>> logger.name
        'victor.myproj-9Kx7Z2.tools'
    """
    session_id = getattr(agent, "active_session_id", None)
    return get_session_logger(session_id, component)


__all__ = [
    "get_session_logger",
    "get_agent_logger",
]
