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

"""SOLID Remediation Monitoring Framework.

This package provides comprehensive monitoring and metrics collection
for the SOLID remediation changes.

Modules:
    solid_metrics: Core metrics collection framework
"""

from victor.monitoring.solid_metrics import (
    SolidMetricsCollector,
    StartupMetrics,
    CacheMetrics,
    FeatureFlagMetrics,
    MemoryMetrics,
    ErrorMetrics,
    get_metrics_collector,
    collect_feature_flags,
    measure_startup_time,
    print_metrics_summary,
)

__all__ = [
    "SolidMetricsCollector",
    "StartupMetrics",
    "CacheMetrics",
    "FeatureFlagMetrics",
    "MemoryMetrics",
    "ErrorMetrics",
    "get_metrics_collector",
    "collect_feature_flags",
    "measure_startup_time",
    "print_metrics_summary",
]
