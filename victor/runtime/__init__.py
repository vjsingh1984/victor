"""Victor runtime module — execution context, lifecycle, cache, and tracing."""

from victor.runtime.cache_registry import CacheCategory, CacheRegistry
from victor.runtime.context import (
    ExecutionContext,  # Backward compatibility alias
    RuntimeExecutionContext,
    ServiceAccessor,
)
from victor.runtime.trace_context import TraceContext, current_trace, get_correlation_id

__all__ = [
    "CacheCategory",
    "CacheRegistry",
    "RuntimeExecutionContext",
    "ExecutionContext",  # Backward compatibility alias
    "ServiceAccessor",
    "TraceContext",
    "current_trace",
    "get_correlation_id",
]
