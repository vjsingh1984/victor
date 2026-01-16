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

"""Base configuration class for agent coordinators.

This module provides the BaseCoordinatorConfig dataclass that defines
common configuration fields shared across all coordinators, reducing
duplication and ensuring consistent configuration patterns.

Design Patterns:
    - DRY: Single source of truth for common coordinator config fields
    - Inheritance: Subclasses extend with coordinator-specific fields
    - Default Values: Sensible defaults for all common fields

Usage:
    from dataclasses import dataclass
    from victor.agent.coordinators.base_config import BaseCoordinatorConfig

    @dataclass
    class MyCoordinatorConfig(BaseCoordinatorConfig):
        # Coordinator-specific fields
        custom_field: str = "default"

    # Inherits: enabled, timeout, max_retries from BaseCoordinatorConfig
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class BaseCoordinatorConfig:
    """Base configuration class for all coordinator configurations.

    This dataclass provides common configuration fields that are shared
    across multiple coordinators, reducing duplication and ensuring
    consistent defaults.

    Attributes:
        enabled: Whether the coordinator is enabled. When False, the
            coordinator may skip operations or behave in a pass-through
            manner depending on implementation.
        timeout: Default timeout in seconds for operations performed by
            the coordinator. Individual operations may override this.
        max_retries: Maximum number of retry attempts for failed operations
            that support retry logic. Set to 0 to disable retries.
        retry_enabled: Whether retry logic is enabled for operations that
            support it. Some coordinators may have more granular retry config.
        log_level: Logging level for coordinator-specific messages.
            Common values: "DEBUG", "INFO", "WARNING", "ERROR".
        enable_metrics: Whether to collect metrics for coordinator operations.
            When True, performance counters and statistics are tracked.

    Example:
        @dataclass
        class ToolCoordinatorConfig(BaseCoordinatorConfig):
            budget: int = 25

        config = ToolCoordinatorConfig(
            enabled=True,
            timeout=30.0,
            max_retries=3,
            budget=50
        )
    """

    enabled: bool = True
    timeout: float = 30.0
    max_retries: int = 3
    retry_enabled: bool = True
    log_level: str = "INFO"
    enable_metrics: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration including
            all fields from this class and any subclasses.
        """
        result: Dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                result[key] = value
        return result

    def validate(self) -> list[str]:
        """Validate configuration values.

        Returns:
            List of validation error messages. Empty list if valid.
        """
        errors: list[str] = []

        if self.timeout < 0:
            errors.append("timeout must be non-negative")

        if self.max_retries < 0:
            errors.append("max_retries must be non-negative")

        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            errors.append(f"Invalid log_level: {self.log_level}")

        return errors

    def is_valid(self) -> bool:
        """Check if configuration is valid.

        Returns:
            True if configuration passes validation, False otherwise.
        """
        return len(self.validate()) == 0


__all__ = [
    "BaseCoordinatorConfig",
]
