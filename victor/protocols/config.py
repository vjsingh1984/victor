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

"""Configuration provider protocol for dependency inversion.

This module defines the IConfigProvider protocol that enables
dependency injection for configuration sources, following the
Dependency Inversion Principle (DIP).

Design Principles:
    - DIP: High-level modules depend on this protocol, not concrete providers
    - OCP: New configuration sources can be added without modifying existing code
    - ISP: Protocol contains only configuration-related methods

Usage:
    class SettingsConfigProvider(IConfigProvider):
        async def get_config(self, session_id: str) -> Dict[str, Any]:
            return load_settings()

        def priority(self) -> int:
            return 100  # High priority
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class IConfigProvider(Protocol):
    """Protocol for configuration data providers.

    Implementations provide configuration from various sources:
    - Environment variables
    - Settings files
    - Database
    - Remote config service
    - CLI arguments

    Configuration from multiple providers is merged by priority,
    with higher priority values overriding lower ones.
    """

    async def get_config(self, session_id: str) -> Dict[str, Any]:
        """Get configuration for a session.

        Args:
            session_id: Unique session identifier

        Returns:
            Dictionary of configuration key-value pairs

        Example:
            {
                "provider": "anthropic",
                "model": "claude-sonnet-4-20250514",
                "temperature": 0.7,
                "max_tokens": 4096,
            }
        """
        ...

    def priority(self) -> int:
        """Priority for merging configuration.

        Higher priority values override lower priority values.
        When multiple providers return the same key, the value
        from the provider with the highest priority is used.

        Returns:
            Priority value (0-1000+, higher = more important)

        Example Priority Levels:
            - 0-99: Low priority (defaults, fallbacks)
            - 100-499: Medium priority (user preferences, profiles)
            - 500-999: High priority (environment variables, CLI args)
            - 1000+: Critical priority (explicit overrides, admin settings)
        """
        ...


__all__ = ["IConfigProvider"]
