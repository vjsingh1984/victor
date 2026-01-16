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
]
