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

Usage:
    # Get session-aware logger
    logger = get_session_logger("myproj-9Kx7Z2")
    logger.info("Session started")  # Logs as [victor.myproj-9Kx7Z2] INFO: Session started

    # Or from agent
    logger = get_agent_logger(agent)
    logger.info("Agent initialized")  # Uses agent.active_session_id if available
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
        name = f"victor"
        if component:
            name = f"{name}.{component}"

    return logging.getLogger(name)


def get_agent_logger(agent, component: str = "") -> logging.Logger:
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


def setup_session_logging(
    session_id: Optional[str] = None,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
) -> None:
    """Configure logging for a session.

    Args:
        session_id: Optional session ID
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path (will include session_id if provided)
    """
    # Get session logger
    logger = get_session_logger(session_id)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))

    # Format with session ID
    if session_id:
        format_str = f"[%(name)s] [%(asctime)s] [%(levelname)s] %(message)s"
    else:
        format_str = f"[%(name)s] [%(asctime)s] [%(levelname)s] %(message)s"

    formatter = logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        from pathlib import Path

        log_path = Path(log_file)
        if session_id and log_path.stem == "victor":
            # Inject session ID into log file name
            log_path = log_path.parent / f"victor_{session_id}.log"

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


__all__ = [
    "get_session_logger",
    "get_agent_logger",
    "setup_session_logging",
]
