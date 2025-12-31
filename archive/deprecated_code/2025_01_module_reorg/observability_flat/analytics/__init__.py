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

"""This module has moved to victor.observability.analytics.

This module is kept for backward compatibility. Please update imports to use:
    from victor.observability.analytics import ...
"""

# Re-export from new location for backward compatibility
from victor.observability.analytics.logger import UsageLogger
from victor.observability.analytics.enhanced_logger import (
    EnhancedUsageLogger,
    PIIScrubber,
    LogRotator,
    LogEncryptor,
    create_usage_logger,
)
from victor.observability.analytics.streaming_metrics import (
    MetricType,
    StreamMetrics,
    MetricsSummary,
    StreamingMetricsCollector,
    MetricsStreamWrapper,
)

__all__ = [
    "UsageLogger",
    "EnhancedUsageLogger",
    "PIIScrubber",
    "LogRotator",
    "LogEncryptor",
    "create_usage_logger",
    "MetricType",
    "StreamMetrics",
    "MetricsSummary",
    "StreamingMetricsCollector",
    "MetricsStreamWrapper",
]
