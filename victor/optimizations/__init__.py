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

"""Performance optimization modules for Victor.

This package provides comprehensive performance optimizations across multiple domains:
- Database query optimization
- Memory optimization and pooling
- Concurrency and async optimization
- Network optimization
- Algorithm and data structure optimization

Usage:
    from victor.optimizations import (
        DatabaseOptimizer,
        MemoryOptimizer,
        ConcurrencyOptimizer,
        NetworkOptimizer,
        apply_all_optimizations,
    )

    # Apply all optimizations
    apply_all_optimizations()

    # Use specific optimizers
    db_opt = DatabaseOptimizer()
    await db_opt.optimize_queries()

    mem_opt = MemoryOptimizer()
    mem_opt.enable_gc_tuning()
"""

from victor.optimizations.database import DatabaseOptimizer
from victor.optimizations.memory import MemoryOptimizer
from victor.optimizations.concurrency import ConcurrencyOptimizer
from victor.optimizations.network import NetworkOptimizer
from victor.optimizations.algorithms import AlgorithmOptimizer

__all__ = [
    "DatabaseOptimizer",
    "MemoryOptimizer",
    "ConcurrencyOptimizer",
    "NetworkOptimizer",
    "AlgorithmOptimizer",
    "apply_all_optimizations",
]


def apply_all_optimizations() -> None:
    """Apply all performance optimizations.

    This function enables all optimization modules with sensible defaults.
    It's recommended to call this during application initialization.
    """
    MemoryOptimizer.enable_gc_tuning()
    ConcurrencyOptimizer.configure_default_thread_pools()
    # Database and Network optimizers require async context
    # They should be initialized separately
