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

"""This module has moved to victor.observability.profiler.

This module is kept for backward compatibility. Please update imports to use:
    from victor.observability.profiler import ...
"""

# Re-export from new location for backward compatibility
from victor.observability.profiler.protocol import (
    ProfilerType,
    MetricType,
    FunctionStats,
    LineStats,
    FileProfile,
    MemorySnapshot,
    MemoryAllocation,
    HotSpot,
    CallGraphNode,
    FlameGraphData,
    ProfileResult,
    ProfileConfig,
    Benchmark,
    BenchmarkSuite,
)
from victor.observability.profiler.profilers import (
    BaseProfiler,
    CPUProfiler,
    LineProfiler,
    MemoryProfiler,
    TimingProfiler,
)
from victor.observability.profiler.manager import (
    ProfilerManager,
    get_profiler_manager,
)

__all__ = [
    # Protocol types
    "ProfilerType",
    "MetricType",
    "FunctionStats",
    "LineStats",
    "FileProfile",
    "MemorySnapshot",
    "MemoryAllocation",
    "HotSpot",
    "CallGraphNode",
    "FlameGraphData",
    "ProfileResult",
    "ProfileConfig",
    "Benchmark",
    "BenchmarkSuite",
    # Profilers
    "BaseProfiler",
    "CPUProfiler",
    "LineProfiler",
    "MemoryProfiler",
    "TimingProfiler",
    # Manager
    "ProfilerManager",
    "get_profiler_manager",
]
