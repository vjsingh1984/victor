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

"""Common validators for use across all verticals.

Phase 2.3: Tool Validation Unification (SOLID Refactoring)

This package provides reusable validation components that eliminate
duplicate validation logic across verticals.

Main exports:
- ToolAvailabilityValidator: Check if tools exist in registry
- ToolBudgetValidator: Validate tool budget values
- CombinedToolValidator: Validate both availability and budget
- ValidationResult: Common validation result dataclass
"""

from victor.tools.validators.common import (
    CombinedToolValidator,
    ToolAvailabilityValidator,
    ToolBudgetValidator,
    ToolRegistryProtocol,
    ValidationResult,
    validate_budget,
    validate_tools,
)

__all__ = [
    # Protocols
    "ToolRegistryProtocol",
    # Results
    "ValidationResult",
    # Validators
    "ToolAvailabilityValidator",
    "ToolBudgetValidator",
    "CombinedToolValidator",
    # Convenience functions
    "validate_tools",
    "validate_budget",
]
