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

"""Common middleware implementations.

This package provides reusable middleware implementations for common
cross-cutting concerns like validation, logging, safety checks,
caching, and metrics collection.

All middleware implementations are consolidated in framework.py and
re-exported here for convenience.

Common safety rule factory functions are available in common_middleware.py
for use across all verticals.
"""

# Export all middleware from framework.py (consolidated implementation)
from victor.framework.middleware.framework import (
    # Middleware implementations
    CacheMiddleware,
    GitSafetyMiddleware,
    LoggingMiddleware,
    MetricsMiddleware,
    OutputValidationMiddleware,
    RateLimitMiddleware,
    SafetyCheckMiddleware,
    SecretMaskingMiddleware,
    ValidationMiddleware,
    # Validation types
    ContentValidationResult,
    FixableValidatorProtocol,
    ToolMetrics,
    ValidationIssue,
    ValidationSeverity,
    ValidatorProtocol,
    # Result types
    CacheResult,
    RateLimitResult,
)
from victor.framework.middleware.builder import (
    MiddlewareBuilder,
    MiddlewareChain,
    MiddlewareProtocol,
)

# Export common safety rule factory functions
from victor.framework.middleware.common_middleware import (
    # Git safety
    create_git_safety_rules,
    # File operations
    create_file_operation_safety_rules,
    # DevOps infrastructure
    create_deployment_safety_rules,
    create_container_safety_rules,
    create_infrastructure_safety_rules,
    # Data privacy
    create_pii_safety_rules,
    # Research
    create_source_credibility_safety_rules,
    create_content_quality_safety_rules,
    # RAG
    create_bulk_operation_safety_rules,
    create_ingestion_safety_rules,
    # Data analysis
    create_data_export_safety_rules,
    # Convenience
    create_all_common_safety_rules,
)

__all__ = [
    # Middleware implementations (all from framework.py)
    "ValidationMiddleware",
    "LoggingMiddleware",
    "SafetyCheckMiddleware",
    "GitSafetyMiddleware",
    "SecretMaskingMiddleware",
    "MetricsMiddleware",
    "OutputValidationMiddleware",
    "CacheMiddleware",
    "RateLimitMiddleware",
    # Validation types
    "ValidationSeverity",
    "ValidationIssue",
    "ContentValidationResult",
    "ValidatorProtocol",
    "FixableValidatorProtocol",
    "ToolMetrics",
    # Result types
    "CacheResult",
    "RateLimitResult",
    # Builder
    "MiddlewareBuilder",
    "MiddlewareChain",
    "MiddlewareProtocol",
    # Common safety rule factory functions
    "create_git_safety_rules",
    "create_file_operation_safety_rules",
    "create_deployment_safety_rules",
    "create_container_safety_rules",
    "create_infrastructure_safety_rules",
    "create_pii_safety_rules",
    "create_source_credibility_safety_rules",
    "create_content_quality_safety_rules",
    "create_bulk_operation_safety_rules",
    "create_ingestion_safety_rules",
    "create_data_export_safety_rules",
    "create_all_common_safety_rules",
]
