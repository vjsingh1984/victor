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

"""Telemetry for deprecated chat coordinator shims.

This module provides telemetry functions for tracking access to deprecated
chat coordinator shims during the migration to the new chat service architecture.
"""

import logging

logger = logging.getLogger(__name__)


def record_deprecated_chat_shim_access(
    location: str,
    name: str,
    access_type: str,
) -> None:
    """Record access to a deprecated chat coordinator shim.

    Args:
        location: Where the access occurred (e.g., "coordinators_package")
        name: Name of the deprecated shim
        access_type: Type of access (e.g., "package_export", "attribute")
    """
    logger.debug(
        f"Deprecated chat shim access: {location} accessed {name} via {access_type}"
    )
    # In production, this would emit a telemetry event
    # For now, just log at debug level to avoid spam


__all__ = ["record_deprecated_chat_shim_access"]
