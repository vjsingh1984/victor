# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""This module has moved to victor.processing.serialization.

This module is maintained for backward compatibility only.
Please update your imports to use the new location:

    # OLD:
    from victor.serialization import SerializationFormat, AdaptiveSerializer

    # NEW (preferred):
    from victor.processing.serialization import SerializationFormat, AdaptiveSerializer
"""

# Re-export everything from the new location for backward compatibility
from victor.processing.serialization import (
    # Strategy types
    SerializationFormat,
    SerializationConfig,
    SerializationResult,
    DataCharacteristics,
    DataStructureType,
    FORMAT_DESCRIPTIONS,
    # Analyzer
    DataAnalyzer,
    get_data_analyzer,
    # Capabilities
    ModelSerializationCapabilities,
    CapabilityRegistry,
    get_capability_registry,
    reset_capability_registry,
    config_from_settings,
    is_serialization_enabled,
    # Adaptive
    AdaptiveSerializer,
    SerializationContext,
    SerializationMetrics,
    get_adaptive_serializer,
    reset_adaptive_serializer,
    serialize,
    # Tool config
    ToolOutputType,
    ToolSerializationConfig,
    ToolSerializationRegistry,
    get_tool_serialization_registry,
    reset_tool_serialization_registry,
    # Metrics
    SerializationMetricRecord,
    SerializationMetricsCollector,
    get_metrics_collector,
    reset_metrics_collector,
    record_serialization_metrics,
    # Intelligent selector
    IntelligentFormatSelector,
    SelectionContext,
    FormatScore,
    get_intelligent_selector,
    reset_intelligent_selector,
)

__all__ = [
    # Strategy types
    "SerializationFormat",
    "SerializationConfig",
    "SerializationResult",
    "DataCharacteristics",
    "DataStructureType",
    "FORMAT_DESCRIPTIONS",
    # Analyzer
    "DataAnalyzer",
    "get_data_analyzer",
    # Capabilities
    "ModelSerializationCapabilities",
    "CapabilityRegistry",
    "get_capability_registry",
    "reset_capability_registry",
    "config_from_settings",
    "is_serialization_enabled",
    # Adaptive
    "AdaptiveSerializer",
    "SerializationContext",
    "SerializationMetrics",
    "get_adaptive_serializer",
    "reset_adaptive_serializer",
    "serialize",
    # Tool config
    "ToolOutputType",
    "ToolSerializationConfig",
    "ToolSerializationRegistry",
    "get_tool_serialization_registry",
    "reset_tool_serialization_registry",
    # Metrics
    "SerializationMetricRecord",
    "SerializationMetricsCollector",
    "get_metrics_collector",
    "reset_metrics_collector",
    "record_serialization_metrics",
    # Intelligent selector
    "IntelligentFormatSelector",
    "SelectionContext",
    "FormatScore",
    "get_intelligent_selector",
    "reset_intelligent_selector",
]
