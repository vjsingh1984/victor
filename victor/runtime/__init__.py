"""Victor runtime module — execution context, lifecycle, cache, and tracing."""

from victor.runtime.cache_registry import CacheCategory, CacheRegistry
from victor.runtime.context import (
    ExecutionContext,  # Backward compatibility alias
    ResolvedRuntimeServices,
    RuntimeExecutionContext,
    ServiceAccessor,
    resolve_execution_context,
    resolve_runtime_services,
)
from victor.runtime.trace_context import TraceContext, current_trace, get_correlation_id

__all__ = [
    "CacheCategory",
    "CacheRegistry",
    "RuntimeExecutionContext",
    "ExecutionContext",  # Backward compatibility alias
    "ResolvedRuntimeServices",
    "ServiceAccessor",
    "resolve_execution_context",
    "resolve_runtime_services",
    "TraceContext",
    "current_trace",
    "get_correlation_id",
]
