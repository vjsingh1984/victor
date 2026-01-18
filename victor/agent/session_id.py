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

"""Session ID generation and utilities.

Generates unique session IDs in the format: {project_root}-{base62_timestamp}

Example: abc123-9Kx7Z2
- project_root: First 6 chars of project directory name (base62-encoded if needed)
- base62_timestamp: Millisecond timestamp in base62 (sortable, compact)

This provides:
- Project context in the session ID
- Lexicographically sortable by timestamp
- Compact representation (13 chars vs 15 for timestamp format)
- Collision-resistant across projects
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Base62 alphabet (0-9, A-Z, a-z)
BASE62_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
BASE62_LEN = len(BASE62_ALPHABET)


def encode_base62(num: int) -> str:
    """Encode a number to base62 string.

    Args:
        num: Integer to encode

    Returns:
        Base62 encoded string

    Example:
        >>> encode_base62(123456789)
        '8M0kX'
    """
    if num == 0:
        return BASE62_ALPHABET[0]

    encoded = []
    while num > 0:
        num, remainder = divmod(num, BASE62_LEN)
        encoded.append(BASE62_ALPHABET[remainder])

    return "".join(reversed(encoded))


def decode_base62(encoded: str) -> int:
    """Decode a base62 string to number.

    Args:
        encoded: Base62 encoded string

    Returns:
        Decoded integer

    Example:
        >>> decode_base62('8M0kX')
        123456789
    """
    num = 0
    for char in encoded:
        num = num * BASE62_LEN + BASE62_ALPHABET.index(char)
    return num


def get_project_root_hash(project_root: Path) -> str:
    """Generate 6-character project root identifier.

    Uses the first 6 characters of the directory name, encoded to base62
    if the name contains special characters or is too long.

    Args:
        project_root: Path to project root directory

    Returns:
        6-character identifier for the project

    Examples:
        >>> get_project_root_hash(Path("/home/user/myproject"))
        'myproj'
        >>> get_project_root_hash(Path("/home/user/victor-ai"))
        'victo'
    """
    # Get directory name
    dirname = project_root.name

    # If directory name is >= 6 chars and alphanumeric, use first 6
    if len(dirname) >= 6 and dirname.isalnum():
        return dirname[:6].lower()

    # Otherwise, hash the directory name and encode to base62
    # MD5 used for session ID generation, not security
    hash_bytes = hashlib.md5(dirname.encode(), usedforsecurity=False).digest()
    hash_num = int.from_bytes(hash_bytes[:4], byteorder="big")
    return encode_base62(hash_num).zfill(6)[:6].lower()


def generate_session_id(project_root: Optional[Path] = None) -> str:
    """Generate a unique session ID.

    Format: {project_root}-{base62_timestamp}

    Args:
        project_root: Path to project root (auto-detected if None)

    Returns:
        Session ID string (e.g., "abc123-9Kx7Z2")

    Examples:
        >>> generate_session_id(Path("/home/user/myproject"))
        'myproj-9Kx7Z2'

        >>> # Can be sorted chronologically
        >>> ids = [generate_session_id() for _ in range(3)]
        >>> sorted(ids) == ids  # base62 timestamps are sortable
        True
    """
    from victor.config.settings import get_project_paths

    # Auto-detect project root if not provided
    if project_root is None:
        project_root = get_project_paths().project_root

    # Get project root hash (6 chars)
    root_hash = get_project_root_hash(project_root)

    # Get current timestamp in milliseconds
    timestamp_ms = int(time.time() * 1000)

    # Encode timestamp to base62
    base62_timestamp = encode_base62(timestamp_ms)

    # Combine: {root_hash}-{base62_timestamp}
    session_id = f"{root_hash}-{base62_timestamp}"

    logger.debug(f"Generated session ID: {session_id} for project: {project_root}")
    return session_id


def parse_session_id(session_id: str) -> dict[str, str | int]:
    """Parse a session ID into components.

    Args:
        session_id: Session ID string (e.g., "abc123-9Kx7Z2")

    Returns:
        Dictionary with components:
        - project_root: 6-char project identifier
        - base62_timestamp: Base62 encoded timestamp
        - timestamp_ms: Decoded timestamp in milliseconds
        - timestamp_iso: ISO format timestamp

    Raises:
        ValueError: If session ID format is invalid

    Example:
        >>> parse = parse_session_id("abc123-9Kx7Z2")
        >>> parse['project_root']
        'abc123'
        >>> parse['base62_timestamp']
        '9Kx7Z2'
        >>> isinstance(parse['timestamp_ms'], int)
        True
    """
    correlation_id = str(uuid.uuid4())[:8]

    try:
        # Check if session_id is a string
        if not isinstance(session_id, str):
            raise ValueError(
                f"Invalid session ID format: expected string, got {type(session_id).__name__}\n"
                f"  Provided: {repr(session_id)}\n"
                f"  Expected format: <project_hash>-<base62_timestamp>\n"
                f"  Example: 'myproj-9Kx7Z2'\n"
                f"  Use 'victor sessions list' to see active sessions.\n"
                f"  Use 'victor sessions new' to create a new session.\n"
                f"  [Correlation ID: {correlation_id}]"
            )

        parts = session_id.split("-")

        if len(parts) != 2:
            # Provide helpful error message with examples
            error_parts = [
                f"Invalid session ID format: '{session_id}'",
                "  Expected format: <project_hash>-<base62_timestamp>",
                "  - project_hash: 6-character project identifier",
                "  - base62_timestamp: Base62 encoded timestamp",
                "",
                "  Examples:",
                "    - 'myproj-9Kx7Z2' (valid)",
                "    - 'victor-8M0kX' (valid)",
                "    - 'abc123-def456' (valid format, but may not exist)",
                "",
                f"  Your session ID has {len(parts)} parts, expected 2",
                f"  Parts found: {parts}",
                "",
                "  Recovery:",
                "    - Use 'victor sessions list' to see active sessions",
                "    - Use 'victor sessions new' to create a new session",
                "    - Check session ID for typos or missing dash",
                f"  [Correlation ID: {correlation_id}]",
            ]
            raise ValueError("\n".join(error_parts))

        project_root, base62_timestamp = parts

        # Validate project_root length
        if len(project_root) != 6:
            logger.warning(
                f"[{correlation_id}] Session ID '{session_id}' has non-standard project hash "
                f"length ({len(project_root)} chars, expected 6)"
            )

        # Validate base62_timestamp
        if not base62_timestamp or not all(c in BASE62_ALPHABET for c in base62_timestamp):
            raise ValueError(
                f"Invalid base62 timestamp in session ID: '{base62_timestamp}'\n"
                f"  Session ID: '{session_id}'\n"
                f"  Base62 timestamp should only contain: 0-9, A-Z, a-z\n"
                f"  [Correlation ID: {correlation_id}]"
            )

        # Decode timestamp
        timestamp_ms = decode_base62(base62_timestamp)

        # Convert to ISO format
        from datetime import datetime, timezone

        timestamp_iso = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()

        return {
            "project_root": project_root,
            "base62_timestamp": base62_timestamp,
            "timestamp_ms": timestamp_ms,
            "timestamp_iso": timestamp_iso,
        }
    except ValueError as e:
        # Re-raise ValueError with our detailed message
        logger.error(f"[{correlation_id}] Failed to parse session ID '{session_id}': {e}")
        raise
    except Exception as e:
        # Catch any other exceptions and wrap them
        logger.error(f"[{correlation_id}] Unexpected error parsing session ID '{session_id}': {e}")
        raise ValueError(
            f"Invalid session ID: '{session_id}'\n"
            f"  Error: {str(e)}\n"
            f"  Expected format: <project_hash>-<base62_timestamp>\n"
            f"  Example: 'myproj-9Kx7Z2'\n"
            f"  [Correlation ID: {correlation_id}]"
        ) from e


def validate_session_id(session_id: str) -> bool:
    """Validate session ID format.

    Args:
        session_id: Session ID string to validate

    Returns:
        True if valid format

    Example:
        >>> validate_session_id("abc123-9Kx7Z2")
        True
        >>> validate_session_id("invalid")
        False
    """
    try:
        parse_session_id(session_id)
        return True
    except (ValueError, IndexError):
        return False


__all__ = [
    "encode_base62",
    "decode_base62",
    "get_project_root_hash",
    "generate_session_id",
    "parse_session_id",
    "validate_session_id",
]
