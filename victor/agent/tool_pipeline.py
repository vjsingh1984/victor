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

"""Tool Execution Pipeline - Coordinates tool execution flow.

This module extracts tool execution coordination from AgentOrchestrator:
- Tool call validation
- Argument normalization
- Execution via ToolExecutor
- Result tracking and analytics
- Failed signature tracking to avoid loops

Design Principles:
- Single Responsibility: Coordinates tool execution only
- Composable: Works with existing ToolExecutor and ToolRegistry
- Observable: Emits events for monitoring
- Configurable: Budget, retry, and caching settings
"""

from __future__ import annotations

import asyncio
import ast
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from victor.agent.argument_normalizer import ArgumentNormalizer, NormalizationStrategy
from victor.agent.tool_executor import ToolExecutor, ToolExecutionResult
from victor.agent.parallel_executor import (
    ParallelToolExecutor,
    ParallelExecutionConfig,
)
from victor.agent.parameter_enforcer import (
    get_enforcer_for_tool,
    ParameterInferenceError,
    ParameterValidationError,
)
from victor.agent.output_aggregator import (
    OutputAggregator,
    AggregationState,
)
from victor.agent.synthesis_checkpoint import (
    CompositeSynthesisCheckpoint,
    get_checkpoint_for_complexity,
)
from victor.config.tool_selection_defaults import (
    ToolPipelineDefaults,
    IdempotentTools,
)

# Import native compute_signature for 10-20x faster signature generation
try:
    from victor.processing.native import compute_signature as native_compute_signature

    _NATIVE_SIGNATURE_AVAILABLE = True
except ImportError:
    _NATIVE_SIGNATURE_AVAILABLE = False
    native_compute_signature = None  # type: ignore

if TYPE_CHECKING:
    from victor.agent.search_router import SearchRouter
    from victor.tools.base import ToolRegistry
    from victor.storage.cache.tool_cache import ToolCache
    from victor.agent.code_correction_middleware import CodeCorrectionMiddleware
    from victor.agent.signature_store import SignatureStore
    from victor.agent.middleware_chain import MiddlewareChain
    from victor.agent.tool_result_cache import ToolResultCache
    from victor.security.permissions import PermissionPolicy

logger = logging.getLogger(__name__)


@dataclass
class ToolPipelineConfig:
    """Configuration for tool pipeline.

    Uses centralized defaults from ToolPipelineDefaults.
    """

    tool_budget: int = ToolPipelineDefaults.TOOL_BUDGET
    enable_caching: bool = True
    enable_analytics: bool = True
    enable_failed_signature_tracking: bool = True
    max_tool_name_length: int = 64

    # Code correction settings
    enable_code_correction: bool = False
    code_correction_auto_fix: bool = True

    # Per-tool-call timeout for serial execution path
    per_tool_timeout_seconds: float = 30.0

    # Parallel execution settings
    enable_parallel_execution: bool = True
    max_concurrent_tools: int = ToolPipelineDefaults.MAX_CONCURRENT_TOOLS
    parallel_batch_size: int = ToolPipelineDefaults.PARALLEL_BATCH_SIZE
    parallel_timeout_per_tool: float = ToolPipelineDefaults.PARALLEL_TIMEOUT_PER_TOOL

    # Session-level result caching for idempotent tools
    # This prevents re-reading same files during a session (helps DeepSeek, Ollama)
    enable_idempotent_caching: bool = True
    idempotent_cache_max_size: int = ToolPipelineDefaults.IDEMPOTENT_CACHE_MAX_SIZE

    # Batch-level deduplication for providers that send duplicate tool calls
    # (helps DeepSeek, Ollama which sometimes issue identical calls in same batch)
    enable_batch_deduplication: bool = True

    # Output aggregation and synthesis checkpoint settings
    enable_output_aggregation: bool = True
    enable_synthesis_checkpoints: bool = True
    synthesis_checkpoint_complexity: str = "medium"  # simple, medium, complex

    # Semantic result caching with FAISS (mtime-based invalidation)
    # Enables intelligent caching based on semantic similarity, reducing redundant tool calls
    enable_semantic_caching: bool = True


# Tools that are safe to cache (read-only, deterministic for same arguments)
# Include both lowercase and capitalized variants for tool name normalization
# Extended from centralized IdempotentTools with additional pipeline-specific variants
IDEMPOTENT_TOOLS = IdempotentTools.IDEMPOTENT_TOOLS | frozenset(
    {
        # Additional capitalized variants and aliases for tool name normalization
        "Read",
        "Grep",
        "Graph",
        "Glob",
        "read_file",  # Alternative name for 'read'
        "list_directory",  # Alternative name for 'ls'
        "code_search",  # Semantic code search alias
        "semantic_code_search",
        "refs",  # Reference lookup
    }
)

GRAPH_TOOL_ARGUMENTS = frozenset(
    {
        "mode",
        "node",
        "source",
        "target",
        "query",
        "file",
        "node_type",
        "edge_types",
        "depth",
        "top_k",
        "direction",
        "expand",
        "exclude_paths",
        "only_runtime",
        "runtime_weighted",
        "modules_only",
        "include_callsites",
        "max_callsites",
        "structured",
        "include_symbols",
        "include_modules",
        "include_calls",
        "include_refs",
    }
)


class ToolRateLimiter:
    """Token bucket rate limiter for tool execution.

    Implements a token bucket algorithm to limit the rate of tool executions.
    This prevents overwhelming external services or local resources.

    Attributes:
        rate: Tokens per second to add (refill rate)
        burst: Maximum tokens that can accumulate (bucket size)
        tokens: Current available tokens
        last_update: Timestamp of last token refill

    Example:
        limiter = ToolRateLimiter(rate=10.0, burst=5)

        # Synchronous check
        if limiter.acquire():
            execute_tool()

        # Asynchronous wait
        await limiter.wait()
        execute_tool()
    """

    def __init__(self, rate: float = 10.0, burst: int = 5):
        """Initialize the rate limiter.

        Args:
            rate: Tokens per second (refill rate)
            burst: Maximum tokens (bucket capacity)
        """
        self.rate = rate
        self.burst = burst
        self.tokens = float(burst)  # Start with full bucket
        self.last_update = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        """Try to acquire a token without blocking.

        Returns:
            True if token acquired, False if no tokens available.
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    async def wait(self) -> None:
        """Wait until a token is available.

        This method blocks asynchronously until a token can be acquired.
        """
        while not self.acquire():
            # Sleep for time needed to get one token
            await asyncio.sleep(1.0 / self.rate)


class LRUToolCache:
    """LRU (Least Recently Used) cache with TTL for idempotent tool results.

    Provides efficient caching with LRU eviction policy and time-based expiry.
    This prevents stale results from being returned after files change and
    bounds memory growth for long-running sessions.

    Attributes:
        _cache: OrderedDict maintaining insertion/access order
        _timestamps: Dict mapping keys to creation timestamps
        _max_size: Maximum number of entries to cache (default: 50)
        _ttl_seconds: Time-to-live in seconds (default: 300s = 5 minutes)

    Example:
        cache = LRUToolCache(max_size=50, ttl_seconds=300)
        cache.set("key", result)
        result = cache.get("key")  # Returns cached result if not expired
    """

    def __init__(
        self,
        max_size: int = ToolPipelineDefaults.IDEMPOTENT_CACHE_MAX_SIZE,
        ttl_seconds: float = ToolPipelineDefaults.IDEMPOTENT_CACHE_TTL,
    ):
        """Initialize the LRU cache with TTL.

        Args:
            max_size: Maximum number of entries to cache
            ttl_seconds: Time-to-live in seconds
        """
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds

    def _is_expired(self, key: str) -> bool:
        """Check if a cache entry has expired.

        Args:
            key: Cache key to check

        Returns:
            True if entry is expired or timestamp missing
        """
        if key not in self._timestamps:
            return True
        age = time.time() - self._timestamps[key]
        return age > self._ttl_seconds

    def _evict_expired(self) -> int:
        """Remove all expired entries from the cache.

        Returns:
            Number of entries evicted
        """
        expired_keys = [k for k in self._cache if self._is_expired(k)]
        for key in expired_keys:
            del self._cache[key]
            self._timestamps.pop(key, None)
        return len(expired_keys)

    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache.

        If found and not expired, the entry is moved to the end (most recently used).
        Expired entries are automatically removed.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
        """
        if key in self._cache:
            # Check TTL expiration
            if self._is_expired(key):
                # Remove expired entry
                del self._cache[key]
                self._timestamps.pop(key, None)
                return None
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Set a value in the cache.

        If the key exists, it's updated and moved to the end with refreshed TTL.
        If cache is full, expired entries are evicted first, then LRU eviction.

        Args:
            key: Cache key
            value: Value to cache
        """
        current_time = time.time()

        if key in self._cache:
            # Update existing and move to end
            self._cache.move_to_end(key)
        self._cache[key] = value
        self._timestamps[key] = current_time

        # Evict expired entries first (cheaper than LRU for memory)
        if len(self._cache) > self._max_size:
            self._evict_expired()

        # Still over capacity? LRU eviction
        while len(self._cache) > self._max_size:
            oldest_key, _ = self._cache.popitem(last=False)
            self._timestamps.pop(oldest_key, None)

    def clear(self) -> None:
        """Clear all entries from the cache."""
        self._cache.clear()
        self._timestamps.clear()

    def __len__(self) -> int:
        """Return the number of entries in the cache."""
        return len(self._cache)

    def items(self):
        """Return all non-expired items in the cache."""
        # Filter out expired entries
        return [(k, v) for k, v in self._cache.items() if not self._is_expired(k)]

    def remove(self, key: str) -> bool:
        """Remove a specific key from the cache.

        Args:
            key: Cache key to remove

        Returns:
            True if key was found and removed, False otherwise
        """
        if key in self._cache:
            del self._cache[key]
            self._timestamps.pop(key, None)
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring.

        Returns:
            Dict with size, max_size, ttl_seconds, expired_count
        """
        expired_count = sum(1 for k in self._cache if self._is_expired(k))
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl_seconds,
            "expired_count": expired_count,
            "active_count": len(self._cache) - expired_count,
        }


@dataclass
class ToolCallResult:
    """Result of a single tool call through the pipeline."""

    tool_name: str
    arguments: Dict[str, Any]
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    cached: bool = False
    normalization_applied: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None

    # Code correction tracking
    code_corrected: bool = False
    code_validation_errors: Optional[List[str]] = None


@dataclass
class PipelineExecutionResult:
    """Result of executing multiple tool calls."""

    results: List[ToolCallResult] = field(default_factory=list)
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    skipped_calls: int = 0
    total_time_ms: float = 0.0
    budget_exhausted: bool = False
    # Parallel execution metrics
    parallel_execution_used: bool = False
    parallel_speedup: float = 1.0
    # Output aggregation and synthesis
    synthesis_recommended: bool = False
    synthesis_reason: Optional[str] = None
    synthesis_prompt: Optional[str] = None
    aggregation_state: Optional[str] = None


class ToolPipeline:
    """Coordinates tool execution flow.

    This class handles the complete lifecycle of tool calls:
    1. Validation (name format, tool exists, budget check)
    2. Argument normalization
    3. Deduplication (skip repeated failures)
    4. Execution via ToolExecutor
    5. Result tracking and analytics

    Example:
        pipeline = ToolPipeline(
            tool_registry=registry,
            tool_executor=executor,
            config=ToolPipelineConfig(tool_budget=20),
        )

        result = await pipeline.execute_tool_calls(tool_calls, context)
        for call_result in result.results:
            if call_result.success:
                print(f"{call_result.tool_name}: {call_result.result}")
    """

    # Regex pattern for valid tool names
    VALID_TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

    def __init__(
        self,
        tool_registry: "ToolRegistry",
        tool_executor: ToolExecutor,
        config: Optional[ToolPipelineConfig] = None,
        tool_cache: Optional["ToolCache"] = None,
        argument_normalizer: Optional[ArgumentNormalizer] = None,
        code_correction_middleware: Optional["CodeCorrectionMiddleware"] = None,
        signature_store: Optional["SignatureStore"] = None,
        on_tool_start: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_tool_complete: Optional[Callable[[ToolCallResult], None]] = None,
        on_tool_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        deduplication_tracker: Optional[Any] = None,
        middleware_chain: Optional["MiddlewareChain"] = None,
        semantic_cache: Optional["ToolResultCache"] = None,
        search_router: Optional["SearchRouter"] = None,
        permission_policy: Optional["PermissionPolicy"] = None,
    ):
        """Initialize tool pipeline.

        Args:
            tool_registry: Registry of available tools
            tool_executor: Executor for running tools
            config: Pipeline configuration
            tool_cache: Optional cache for tool results
            argument_normalizer: Optional argument normalizer
            code_correction_middleware: Optional middleware for code validation/fixing
            signature_store: Optional persistent storage for failed
                signatures (cross-session learning)
            on_tool_start: Callback when tool execution starts
            on_tool_complete: Callback when tool execution completes
            deduplication_tracker: Optional tracker for detecting redundant tool calls
            middleware_chain: Optional middleware chain for processing tool calls
            semantic_cache: Optional FAISS-based semantic cache for tool results
            search_router: Optional search router for enriching search tool calls
        """
        self.tools = tool_registry
        self.executor = tool_executor
        self.config = config or ToolPipelineConfig()
        self.tool_cache = tool_cache
        self.normalizer = argument_normalizer or ArgumentNormalizer()
        self.code_correction_middleware = code_correction_middleware
        self.signature_store = signature_store
        self.deduplication_tracker = deduplication_tracker
        self.middleware_chain = middleware_chain
        self.semantic_cache = semantic_cache
        self.search_router = search_router
        self._permission_policy = permission_policy

        # Callbacks
        self.on_tool_start = on_tool_start
        self.on_tool_complete = on_tool_complete
        self.on_tool_event = on_tool_event

        # Spin detection: track whether last batch had all calls blocked
        self.last_batch_all_skipped: bool = False

        # Output aggregation and synthesis checkpoints
        self._output_aggregator: Optional[OutputAggregator] = None
        self._synthesis_checkpoint: Optional[CompositeSynthesisCheckpoint] = None
        if self.config.enable_output_aggregation:
            self._output_aggregator = OutputAggregator(
                max_results=self.config.tool_budget,
                stale_threshold_seconds=60.0,
            )
        if self.config.enable_synthesis_checkpoints:
            self._synthesis_checkpoint = get_checkpoint_for_complexity(
                self.config.synthesis_checkpoint_complexity
            )

        # State tracking
        self._calls_used = 0
        # In-memory failed signatures for session-only tracking (backward compatible)
        # When signature_store is provided, it's used for persistent cross-session tracking
        self._failed_signatures: Set[tuple] = set()
        self._executed_tools: List[str] = []

        # Analytics
        self._tool_stats: Dict[str, Dict[str, Any]] = {}

        # Parallel executor (lazy initialized)
        self._parallel_executor: Optional[ParallelToolExecutor] = None

        # Session-level idempotent tool result cache with LRU eviction (Workstream E fix)
        # Prevents DeepSeek/Ollama from re-reading same files multiple times
        # Uses LRU instead of FIFO for more optimal cache behavior
        self._idempotent_cache: LRUToolCache = LRUToolCache(
            max_size=self.config.idempotent_cache_max_size
        )
        self._cache_hits = 0
        self._cache_misses = 0

        # Batch-level deduplication tracking
        self._batch_dedup_count = 0  # Total duplicates skipped

        # File read timestamp tracking for deduplication (prompting loop fix)
        # Prevents re-reading identical files within TTL window
        self._read_file_timestamps: Dict[str, float] = {}

        # Observability: pre-execution intent logging (LogAct-inspired)
        self._observability_bus: Optional[Any] = None
        self._trace_enricher: Optional[Any] = None

    @property
    def calls_used(self) -> int:
        """Number of tool calls used."""
        return self._calls_used

    @property
    def calls_remaining(self) -> int:
        """Number of tool calls remaining in budget."""
        return max(0, self.config.tool_budget - self._calls_used)

    @property
    def executed_tools(self) -> List[str]:
        """List of executed tool names."""
        return list(self._executed_tools)

    @property
    def parallel_executor(self) -> ParallelToolExecutor:
        """Get or create the parallel executor (lazy initialization)."""
        if self._parallel_executor is None:
            parallel_config = ParallelExecutionConfig(
                max_concurrent=self.config.max_concurrent_tools,
                enable_parallel=self.config.enable_parallel_execution,
                timeout_per_tool=self.config.parallel_timeout_per_tool,
            )
            self._parallel_executor = ParallelToolExecutor(
                tool_executor=self.executor,
                config=parallel_config,
                progress_callback=self._parallel_progress_callback,
            )
        return self._parallel_executor

    def _parallel_progress_callback(self, tool_name: str, status: str, success: bool) -> None:
        """Progress callback for parallel execution."""
        if status == "started" and self.on_tool_start:
            try:
                self.on_tool_start(tool_name, {})
            except Exception as e:
                logger.warning(f"on_tool_start callback failed: {e}")

    def reset(self) -> None:
        """Reset pipeline state for new conversation."""
        self._calls_used = 0
        self._failed_signatures.clear()
        self._executed_tools.clear()
        # Clear idempotent cache (new session may have different file contents)
        self._idempotent_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._batch_dedup_count = 0
        # Reset output aggregator
        if self._output_aggregator:
            self._output_aggregator.reset()
        # Clear semantic cache
        if self.semantic_cache is not None:
            self.semantic_cache.clear()

    def set_semantic_cache(self, cache: "ToolResultCache") -> None:
        """Set the semantic cache after initialization.

        Useful when embedding service isn't available at construction time.

        Args:
            cache: ToolResultCache instance to use for semantic caching
        """
        self.semantic_cache = cache
        self.config.enable_semantic_caching = cache is not None
        logger.debug("Semantic cache set on tool pipeline")

    def is_valid_tool_name(self, name: str) -> bool:
        """Check if tool name is valid format.

        Args:
            name: Tool name to validate

        Returns:
            True if valid format
        """
        if not name or not isinstance(name, str):
            return False
        if len(name) > self.config.max_tool_name_length:
            return False
        return bool(self.VALID_TOOL_NAME_PATTERN.match(name))

    def _normalize_arguments(
        self, tool_name: str, arguments: Any, context: Optional[Dict[str, Any]] = None
    ) -> tuple[Dict[str, Any], NormalizationStrategy]:
        """Normalize tool arguments with parameter enforcement.

        Args:
            tool_name: Name of the tool
            arguments: Raw arguments (may be string, dict, or None)
            context: Optional context for parameter inference

        Returns:
            Tuple of (normalized_args, strategy_used)
        """
        # Handle string arguments (from streaming)
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                try:
                    arguments = ast.literal_eval(arguments)
                except Exception:
                    arguments = {"value": arguments}
        elif arguments is None:
            arguments = {}

        # Apply basic normalizer first
        normalized_args, strategy = self.normalizer.normalize_arguments(arguments, tool_name)

        # Apply parameter enforcement if available for this tool
        # This handles missing required parameters and type coercion (GAP-9)
        enforcer = get_enforcer_for_tool(tool_name)
        if enforcer is not None:
            try:
                enforced_args = enforcer.enforce(normalized_args, context=context or {})
                if enforced_args != normalized_args:
                    logger.debug(
                        f"Parameter enforcer modified args for {tool_name}: "
                        f"{set(enforced_args.keys()) - set(normalized_args.keys())} added"
                    )
                    normalized_args = enforced_args
                    # Update strategy to indicate enforcement was applied
                    if strategy == NormalizationStrategy.DIRECT:
                        strategy = NormalizationStrategy.MANUAL_REPAIR
            except ParameterInferenceError as e:
                # Could not infer required parameter - log but don't fail here
                # The tool execution will fail with a more meaningful error
                logger.warning(f"Could not infer required parameter for {tool_name}: {e}")
            except ParameterValidationError as e:
                # Invalid parameter value - log warning
                logger.warning(f"Parameter validation failed for {tool_name}: {e}")

        return normalized_args, strategy

    def _apply_search_routing(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any], bool]:
        """Apply search router hints to search-tool calls when beneficial."""
        if self.search_router is None:
            return tool_name, arguments, False

        if tool_name not in ("code_search", "semantic_code_search"):
            return tool_name, arguments, False

        if arguments.get("mode"):
            return tool_name, arguments, False

        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return tool_name, arguments, False

        try:
            route = self.search_router.route(query)
        except Exception as exc:
            logger.debug("Search router enrichment failed for %s: %s", tool_name, exc)
            return tool_name, arguments, False

        if not route.tool_name or not route.tool_arguments:
            return tool_name, arguments, False

        routed_tool_name = route.tool_name
        if routed_tool_name == "graph" and routed_tool_name != tool_name:
            routed_args = {
                key: value for key, value in arguments.items() if key in GRAPH_TOOL_ARGUMENTS
            }
            routed_args.update(route.tool_arguments)
        else:
            routed_args = dict(route.tool_arguments)
            routed_args.update(arguments)
        changed = routed_tool_name != tool_name or routed_args != arguments

        if changed:
            logger.debug(
                "Applied search route to %s query: tool=%s extra_args=%s",
                tool_name,
                routed_tool_name,
                route.tool_arguments,
            )

        return routed_tool_name, routed_args, changed

    def _normalize_tool_call(
        self, tool_name: str, arguments: Any, context: Optional[Dict[str, Any]] = None
    ) -> tuple[str, Dict[str, Any], NormalizationStrategy]:
        """Normalize a full tool call, including search-routing adjustments."""
        normalized_args, strategy = self._normalize_arguments(tool_name, arguments, context=context)
        normalized_tool_name, normalized_args, changed = self._apply_search_routing(
            tool_name, normalized_args
        )
        if changed and strategy == NormalizationStrategy.DIRECT:
            strategy = NormalizationStrategy.MANUAL_REPAIR
        return normalized_tool_name, normalized_args, strategy

    def _get_call_signature(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Generate signature for deduplication.

        Uses native xxHash3-based signature computation when available for
        10-20x speedup over JSON serialization + MD5.

        Args:
            tool_name: Tool name
            args: Tool arguments

        Returns:
            Hashable signature string (16-char hex with native, or tool:args with fallback)
        """
        # Use native signature computation if available (10-20x faster)
        if _NATIVE_SIGNATURE_AVAILABLE and native_compute_signature is not None:
            try:
                return native_compute_signature(tool_name, args)
            except Exception:
                pass  # Fall through to Python implementation

        # Pure Python fallback
        try:
            args_str = json.dumps(args, sort_keys=True, default=str)
        except Exception:
            args_str = str(args)
        return f"{tool_name}:{args_str}"

    def is_known_failure(self, tool_name: str, args: Dict[str, Any]) -> bool:
        """Check if a tool call is known to fail.

        Uses persistent SignatureStore when available for cross-session learning,
        falls back to in-memory set for session-only tracking.

        Args:
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            True if this call has failed before
        """
        if not self.config.enable_failed_signature_tracking:
            return False

        # Check persistent store first (cross-session)
        if self.signature_store is not None:
            try:
                if self.signature_store.is_known_failure(tool_name, args):
                    logger.debug(f"Tool call is known failure (persistent): {tool_name}")
                    return True
            except (OSError, IOError) as e:
                logger.warning(f"Signature store check failed (I/O error): {e}")
            except (KeyError, ValueError) as e:
                logger.debug(f"Signature store check failed (data error): {e}")

        # Fall back to in-memory check (session-only)
        signature = self._get_call_signature(tool_name, args)
        return signature in self._failed_signatures

    def record_failure(self, tool_name: str, args: Dict[str, Any], error_message: str) -> None:
        """Record a failed tool call.

        Uses persistent SignatureStore when available for cross-session learning,
        also records in-memory for current session.

        Args:
            tool_name: Name of the tool
            args: Tool arguments
            error_message: Error message from the failure
        """
        if not self.config.enable_failed_signature_tracking:
            return

        # Record in persistent store (cross-session)
        if self.signature_store is not None:
            try:
                self.signature_store.record_failure(tool_name, args, error_message)
                logger.debug(f"Recorded failure to persistent store: {tool_name}")
            except (OSError, IOError) as e:
                logger.warning(f"Failed to record to signature store (I/O error): {e}")
            except (ValueError, TypeError) as e:
                logger.debug(f"Failed to record to signature store (data error): {e}")

        # Also record in-memory for current session
        signature = self._get_call_signature(tool_name, args)
        self._failed_signatures.add(signature)

    def clear_failure(self, tool_name: str, args: Dict[str, Any]) -> bool:
        """Clear a specific failure signature (e.g., after a fix).

        Args:
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            True if signature was found and cleared
        """
        cleared = False

        # Clear from persistent store
        if self.signature_store is not None:
            try:
                if self.signature_store.clear_signature(tool_name, args):
                    cleared = True
                    logger.debug(f"Cleared failure from persistent store: {tool_name}")
            except (OSError, IOError) as e:
                logger.warning(f"Failed to clear from signature store (I/O error): {e}")
            except (KeyError, ValueError) as e:
                logger.debug(f"Failed to clear from signature store (data error): {e}")

        # Clear from in-memory
        signature = self._get_call_signature(tool_name, args)
        if signature in self._failed_signatures:
            self._failed_signatures.discard(signature)
            cleared = True

        return cleared

    def is_idempotent_tool(self, tool_name: str) -> bool:
        """Check if a tool is idempotent (safe to cache results).

        Idempotent tools are read-only and produce the same output for
        the same input arguments. Results can be safely cached within a session.

        Args:
            tool_name: Name of the tool

        Returns:
            True if tool is idempotent
        """
        return tool_name in IDEMPOTENT_TOOLS

    def get_cached_result(self, tool_name: str, args: Dict[str, Any]) -> Optional[ToolCallResult]:
        """Get cached result for an idempotent tool call.

        Args:
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            Cached ToolCallResult if found, None otherwise
        """
        if not self.config.enable_idempotent_caching:
            return None

        if not self.is_idempotent_tool(tool_name):
            return None

        signature = self._get_call_signature(tool_name, args)
        cached = self._idempotent_cache.get(signature)
        if cached is not None:
            self._cache_hits += 1
            logger.debug(f"Cache hit for {tool_name}: {signature[:100]}...")
        return cached

    def cache_result(self, tool_name: str, args: Dict[str, Any], result: ToolCallResult) -> None:
        """Cache result for an idempotent tool call.

        Only successful results from idempotent tools are cached.

        Args:
            tool_name: Name of the tool
            args: Tool arguments
            result: Tool call result to cache
        """
        if not self.config.enable_idempotent_caching:
            return

        if not self.is_idempotent_tool(tool_name):
            return

        if not result.success:
            return  # Don't cache failures

        signature = self._get_call_signature(tool_name, args)
        # Mark as cached in the result
        cached_result = ToolCallResult(
            tool_name=result.tool_name,
            arguments=result.arguments,
            success=result.success,
            result=result.result,
            error=result.error,
            execution_time_ms=0.0,  # Cached results have ~0 execution time
            cached=True,  # Mark as from cache
            normalization_applied=result.normalization_applied,
        )
        # LRU cache handles eviction automatically (Workstream E fix)
        self._idempotent_cache.set(signature, cached_result)
        self._cache_misses += 1
        logger.debug(f"Cached result for {tool_name} (cache size: {len(self._idempotent_cache)})")

    def invalidate_file_cache(self, file_path: str) -> int:
        """Invalidate cached results that involve a specific file path.

        Call this after a file is modified to ensure subsequent reads
        get fresh content.

        Args:
            file_path: Path of the modified file

        Returns:
            Number of cache entries invalidated
        """
        if not self.config.enable_idempotent_caching:
            return 0

        # Find and remove all cache entries involving this file
        to_remove = []
        for sig, result in self._idempotent_cache.items():
            # Check if file path is in arguments
            if file_path in str(result.arguments):
                to_remove.append(sig)

        for sig in to_remove:
            self._idempotent_cache.remove(sig)

        if to_remove:
            logger.debug(f"Invalidated {len(to_remove)} cache entries for {file_path}")

        return len(to_remove)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with idempotent cache stats and semantic cache stats
        """
        stats = {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_size": len(self._idempotent_cache),
            "hit_rate": (
                self._cache_hits / (self._cache_hits + self._cache_misses)
                if (self._cache_hits + self._cache_misses) > 0
                else 0.0
            ),
            "batch_dedup_count": self._batch_dedup_count,
        }
        # Include semantic cache stats if available
        if self.semantic_cache is not None:
            stats["semantic_cache"] = self.semantic_cache.get_stats()
        return stats

    def _is_duplicate_read(self, file_path: str, max_age_seconds: float = 300.0) -> bool:
        """Check if file was recently read (within TTL window).

        Used to prevent re-reading identical files during prompting loops.

        Args:
            file_path: Path of the file
            max_age_seconds: TTL for file read cache (default 5 minutes)

        Returns:
            True if file was read recently and should not be re-read
        """
        if file_path not in self._read_file_timestamps:
            return False
        age = time.monotonic() - self._read_file_timestamps[file_path]
        return age < max_age_seconds

    def record_file_read(self, file_path: str) -> None:
        """Record that a file was read.

        Updates the timestamp for deduplication tracking.

        Args:
            file_path: Path of the file that was read
        """
        # Evict oldest entries if at capacity (prevent unbounded growth)
        if len(self._read_file_timestamps) >= 2000:
            oldest_key = next(iter(self._read_file_timestamps))
            del self._read_file_timestamps[oldest_key]
        self._read_file_timestamps[file_path] = time.monotonic()
        logger.debug(f"Recorded file read: {file_path}")

    def deduplicate_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[tuple[int, tuple]]]:
        """Deduplicate identical tool calls within a batch.

        Some providers (DeepSeek, Ollama) sometimes issue duplicate tool calls
        in the same response. This method detects and removes duplicates,
        returning deduplicated results for the duplicates that use the first
        occurrence's result.

        Args:
            tool_calls: List of tool call requests

        Returns:
            Tuple of (unique_calls, duplicate_info) where duplicate_info
            contains (original_index, signature) tuples for duplicates
        """
        if not self.config.enable_batch_deduplication:
            return tool_calls, []

        seen_signatures: Dict[tuple, int] = {}  # signature -> index in unique_calls
        unique_calls: List[Dict[str, Any]] = []
        duplicate_indices: List[tuple[int, tuple]] = []  # (original_index, signature)

        for i, tc in enumerate(tool_calls):
            # Skip invalid structures - let execute_tool_calls handle them
            if not isinstance(tc, dict):
                unique_calls.append(tc)
                continue

            tool_name = tc.get("name", "")
            raw_args = tc.get("arguments", {})

            # Normalize arguments for signature comparison
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except Exception:
                    raw_args = {"value": raw_args}
            elif raw_args is None:
                raw_args = {}

            signature = self._get_call_signature(tool_name, raw_args)

            if signature in seen_signatures:
                # Duplicate found
                duplicate_indices.append((i, signature))
                self._batch_dedup_count += 1
                logger.info(
                    f"[Pipeline] Deduplicating {tool_name} call "
                    f"(duplicate #{self._batch_dedup_count})"
                )
            else:
                # First occurrence
                seen_signatures[signature] = len(unique_calls)
                unique_calls.append(tc)

        return unique_calls, duplicate_indices

    async def execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> PipelineExecutionResult:
        """Execute multiple tool calls.

        Args:
            tool_calls: List of tool call requests
            context: Execution context passed to tools

        Returns:
            PipelineExecutionResult with all results
        """
        context = context or {}
        result = PipelineExecutionResult(total_calls=len(tool_calls))
        start_time = time.monotonic()

        # Batch-level deduplication to handle providers that send duplicate calls
        unique_calls, duplicate_info = self.deduplicate_tool_calls(tool_calls)

        # Track results by signature for duplicate resolution
        results_by_signature: Dict[tuple, ToolCallResult] = {}

        # Build tool history for synthesis checkpoint
        tool_history: List[Dict[str, Any]] = []

        for tool_call in unique_calls:
            call_result = await self._execute_single_call(tool_call, context)
            result.results.append(call_result)

            # Store result by signature for duplicate resolution (only for dict items)
            tool_name = ""
            raw_args: Dict[str, Any] = {}
            if isinstance(tool_call, dict):
                tool_name = tool_call.get("name", "")
                raw_args = tool_call.get("arguments", {})
                if isinstance(raw_args, str):
                    try:
                        raw_args = json.loads(raw_args)
                    except Exception:
                        raw_args = {"value": raw_args}
                elif raw_args is None:
                    raw_args = {}
                signature = self._get_call_signature(tool_name, raw_args)
                results_by_signature[signature] = call_result

            # Add to output aggregator
            if self._output_aggregator and tool_name:
                self._output_aggregator.add_result(
                    tool_name=tool_name,
                    result=call_result.result,
                    success=call_result.success,
                    metadata={"args": raw_args},
                )

            # Track for synthesis checkpoint
            tool_history.append(
                {
                    "tool": tool_name,
                    "args": raw_args,
                    "success": call_result.success,
                    "result": (str(call_result.result)[:500] if call_result.result else ""),
                    "error": call_result.error,
                }
            )

            if call_result.skipped:
                result.skipped_calls += 1
            elif call_result.success:
                result.successful_calls += 1
            else:
                result.failed_calls += 1

            # Check if budget exhausted
            if self._calls_used >= self.config.tool_budget:
                result.budget_exhausted = True
                # Skip remaining calls
                break

        # Add skipped results for duplicates (reuse first occurrence's result)
        for _original_idx, signature in duplicate_info:
            if signature in results_by_signature:
                original_result = results_by_signature[signature]
                # Create a copy marked as from deduplication
                dup_result = ToolCallResult(
                    tool_name=original_result.tool_name,
                    arguments=original_result.arguments,
                    success=original_result.success,
                    result=original_result.result,
                    error=original_result.error,
                    execution_time_ms=0.0,  # Deduplicated, no execution time
                    cached=True,  # Mark as effectively cached
                    skipped=True,
                    skip_reason="Deduplicated (duplicate call in batch)",
                )
                result.results.append(dup_result)
                result.skipped_calls += 1

        # Track whether ALL tool calls in this batch were skipped/blocked
        # Used by spin detection in the streaming pipeline
        self.last_batch_all_skipped = (
            len(result.results) > 0
            and all(getattr(r, "skipped", False) for r in result.results)
        )
        if self.last_batch_all_skipped:
            logger.info(
                f"[pipeline] All {len(result.results)} tool calls in batch were "
                f"skipped/blocked (dedup or session filter)"
            )

        # Check synthesis checkpoint
        if self._synthesis_checkpoint and tool_history:
            task_context = {
                "elapsed_time": context.get("elapsed_time", 0),
                "timeout": context.get("timeout", 180),
                "task_type": context.get("task_type", "unknown"),
            }
            checkpoint_result = self._synthesis_checkpoint.check(tool_history, task_context)
            if checkpoint_result.should_synthesize:
                result.synthesis_recommended = True
                result.synthesis_reason = checkpoint_result.reason
                result.synthesis_prompt = checkpoint_result.suggested_prompt
                logger.info(f"Synthesis checkpoint triggered: {checkpoint_result.reason}")

        # Set aggregation state
        if self._output_aggregator:
            result.aggregation_state = self._output_aggregator.state.value
            # If aggregator is in a synthesis-ready state, also recommend synthesis
            if self._output_aggregator.state in (
                AggregationState.READY_TO_SYNTHESIZE,
                AggregationState.LOOPING,
                AggregationState.STUCK,
            ):
                if not result.synthesis_recommended:
                    result.synthesis_recommended = True
                    result.synthesis_reason = (
                        f"Aggregation state: {self._output_aggregator.state.value}"
                    )
                    result.synthesis_prompt = self._output_aggregator.get_synthesis_prompt()

        result.total_time_ms = (time.monotonic() - start_time) * 1000
        return result

    async def execute_tool_calls_parallel(
        self,
        tool_calls: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        force_parallel: bool = False,
    ) -> PipelineExecutionResult:
        """Execute tool calls with parallelization when beneficial.

        Automatically decides whether to use parallel execution based on:
        - Number of tool calls (>1 for parallel)
        - Tool categories (read-only tools can parallelize)
        - Configuration settings

        Args:
            tool_calls: List of tool call requests
            context: Execution context passed to tools
            force_parallel: Override automatic decision (for testing)

        Returns:
            PipelineExecutionResult with parallel metrics
        """
        context = context or {}
        start_time = time.monotonic()

        # Check if parallelization is worthwhile
        should_parallelize = self.config.enable_parallel_execution and (
            force_parallel or len(tool_calls) > 1
        )

        if not should_parallelize:
            # Fall back to sequential execution
            return await self.execute_tool_calls(tool_calls, context)

        # Pre-validate and normalize tool calls
        validated_calls = []
        skipped_results = []

        for tc in tool_calls:
            tool_name = tc.get("name", "")

            # Quick validation checks
            if not tool_name or not self.is_valid_tool_name(tool_name):
                skipped_results.append(
                    ToolCallResult(
                        tool_name=tool_name or "unknown",
                        arguments={},
                        success=False,
                        skipped=True,
                        skip_reason=f"Invalid tool name: {tool_name}",
                    )
                )
                continue

            raw_args = tc.get("arguments", {})
            tool_name, normalized_args, _ = self._normalize_tool_call(
                tool_name, raw_args, context=context
            )

            if not self.tools.is_tool_enabled(tool_name):
                skipped_results.append(
                    ToolCallResult(
                        tool_name=tool_name,
                        arguments=normalized_args,
                        success=False,
                        skipped=True,
                        skip_reason=f"Unknown or disabled tool: {tool_name}",
                    )
                )
                continue

            # Check for repeated failures
            if self.config.enable_failed_signature_tracking:
                signature = self._get_call_signature(tool_name, normalized_args)
                if signature in self._failed_signatures:
                    skipped_results.append(
                        ToolCallResult(
                            tool_name=tool_name,
                            arguments=normalized_args,
                            success=False,
                            skipped=True,
                            skip_reason="Repeated failing call with same arguments",
                        )
                    )
                    continue

            validated_calls.append({"name": tool_name, "arguments": normalized_args})

        # Execute validated calls in parallel
        parallel_result = await self.parallel_executor.execute_parallel(validated_calls, context)

        # Build pipeline result
        result = PipelineExecutionResult(
            total_calls=len(tool_calls),
            parallel_execution_used=True,
            parallel_speedup=parallel_result.parallel_speedup,
        )

        # Add skipped results first
        for skipped in skipped_results:
            result.results.append(skipped)
            result.skipped_calls += 1

        # Convert parallel results to pipeline results
        for exec_result in parallel_result.results:
            call_result = ToolCallResult(
                tool_name=exec_result.tool_name,
                arguments={},  # Arguments already logged
                success=exec_result.success,
                result=exec_result.result,
                error=exec_result.error,
                execution_time_ms=exec_result.execution_time * 1000,
            )
            result.results.append(call_result)

            if exec_result.success:
                result.successful_calls += 1
                self._executed_tools.append(exec_result.tool_name)
            else:
                result.failed_calls += 1
                # Track failed signature
                if self.config.enable_failed_signature_tracking:
                    # We need to get arguments back - use tool name + error as key
                    self._failed_signatures.add((exec_result.tool_name, exec_result.error or ""))

            # Update call count
            self._calls_used += 1

            # Check budget
            if self._calls_used >= self.config.tool_budget:
                result.budget_exhausted = True
                break

        result.total_time_ms = (time.monotonic() - start_time) * 1000

        logger.info(
            f"Parallel pipeline: {len(validated_calls)} tools, "
            f"speedup={parallel_result.parallel_speedup:.2f}x, "
            f"time={result.total_time_ms:.1f}ms"
        )

        return result

    def _emit_tool_intent(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Emit pre-execution intent event (LogAct-inspired).

        Records tool intent BEFORE execution, enabling future voting/gating.
        Uses ObservabilityBus for event emission.
        """
        bus = getattr(self, "_observability_bus", None)
        if bus is None:
            return

        import hashlib

        args_str = str(sorted(arguments.items())) if arguments else ""
        args_hash = hashlib.md5(args_str.encode()).hexdigest()[:12]

        reasoning = ""
        enricher = getattr(self, "_trace_enricher", None)
        if enricher:
            reasoning = getattr(enricher, "_pending_reasoning", "") or ""

        try:
            bus.emit_sync(
                "tool.intent",
                {
                    "tool_name": tool_name,
                    "arguments_hash": args_hash,
                    "reasoning_before": reasoning[:500],
                    "timestamp": time.time(),
                },
            )
        except Exception:
            pass  # Intent logging is non-critical

    async def _execute_single_call(
        self,
        tool_call: Dict[str, Any],
        context: Dict[str, Any],
    ) -> ToolCallResult:
        """Execute a single tool call.

        Args:
            tool_call: Tool call request
            context: Execution context

        Returns:
            ToolCallResult
        """
        # Validate structure
        if not isinstance(tool_call, dict):
            return ToolCallResult(
                tool_name="unknown",
                arguments={},
                success=False,
                skipped=True,
                skip_reason="Invalid tool call structure (not a dict)",
            )

        tool_name = tool_call.get("name", "")
        raw_args = tool_call.get("arguments", {})

        # Validate tool name
        if not tool_name:
            return ToolCallResult(
                tool_name="unknown",
                arguments={},
                success=False,
                skipped=True,
                skip_reason="Tool call missing name",
            )

        if not self.is_valid_tool_name(tool_name):
            return ToolCallResult(
                tool_name=tool_name,
                arguments={},
                success=False,
                skipped=True,
                skip_reason=f"Invalid tool name format: {tool_name}",
            )

        # Check if tool exists
        tool_name, normalized_args, strategy = self._normalize_tool_call(
            tool_name, raw_args, context=context
        )

        if not self.tools.is_tool_enabled(tool_name):
            return ToolCallResult(
                tool_name=tool_name,
                arguments=normalized_args,
                success=False,
                skipped=True,
                skip_reason=f"Unknown or disabled tool: {tool_name}",
            )

        # Check permission policy (if configured)
        if self._permission_policy is not None:
            decision = self._permission_policy.authorize(tool_name, normalized_args)
            if not decision.allowed and not decision.needs_prompt:
                return ToolCallResult(
                    tool_name=tool_name,
                    arguments=normalized_args,
                    success=False,
                    skipped=True,
                    skip_reason=f"Permission denied: {decision.reason}",
                )
            # needs_prompt tools are allowed through — the UI layer
            # handles escalation prompting before reaching the pipeline.

        # Check budget
        if self._calls_used >= self.config.tool_budget:
            return ToolCallResult(
                tool_name=tool_name,
                arguments=normalized_args,
                success=False,
                skipped=True,
                skip_reason="Tool budget exhausted",
            )

        normalization_applied = None if strategy == NormalizationStrategy.DIRECT else strategy.value

        # Check idempotent cache for read-only tools
        # This prevents DeepSeek/Ollama from re-reading the same file multiple times
        cached_result = self.get_cached_result(tool_name, normalized_args)
        if cached_result is not None:
            logger.info(f"[Pipeline] Returning cached result for {tool_name}")
            # Still count as a tool call for tracking, but don't execute
            self._executed_tools.append(tool_name)
            # Notify callbacks about cached result
            if self.on_tool_start:
                try:
                    self.on_tool_start(tool_name, normalized_args)
                except Exception as e:
                    logger.warning(f"on_tool_start callback failed: {e}")
            if self.on_tool_complete:
                try:
                    self.on_tool_complete(cached_result)
                except Exception as e:
                    logger.warning(f"on_tool_complete callback failed: {e}")
            return cached_result

        # Check semantic cache (FAISS-based with mtime invalidation)
        # This catches similar-but-not-identical queries that would return same results
        if self.config.enable_semantic_caching and self.semantic_cache is not None:
            try:
                semantic_result = await self.semantic_cache.get(tool_name, normalized_args)
                if semantic_result is not None:
                    logger.info(f"[Pipeline] Semantic cache hit for {tool_name}")
                    self._executed_tools.append(tool_name)
                    # Build result from cached data
                    sem_cached_result = ToolCallResult(
                        tool_name=tool_name,
                        arguments=normalized_args,
                        success=True,
                        result=semantic_result,
                        cached=True,
                        normalization_applied=normalization_applied,
                    )
                    if self.on_tool_start:
                        try:
                            self.on_tool_start(tool_name, normalized_args)
                        except Exception as e:
                            logger.warning(f"on_tool_start callback failed: {e}")
                    if self.on_tool_complete:
                        try:
                            self.on_tool_complete(sem_cached_result)
                        except Exception as e:
                            logger.warning(f"on_tool_complete callback failed: {e}")
                    return sem_cached_result
            except (TimeoutError, asyncio.TimeoutError):
                logger.debug("Semantic cache timeout, skipping")
            except (OSError, ValueError) as e:
                logger.warning(f"Semantic cache error (may indicate corruption): {e}")

        # Session-level file read dedup - prevents re-reading files even if cache was cleared
        # This is a fallback for the prompting loop fix when idempotent cache doesn't match
        if tool_name.lower() in ("read", "read_file"):
            file_path = normalized_args.get("path") or normalized_args.get("file_path")
            if file_path and self._is_duplicate_read(file_path):
                logger.info(
                    f"[Pipeline] Skipping duplicate read of {file_path} "
                    "(file was read recently in this session)"
                )
                return ToolCallResult(
                    tool_name=tool_name,
                    arguments=normalized_args,
                    success=True,  # Mark as success but indicate it was cached
                    result=f"[File '{file_path}' was already read in this session - "
                    "content unchanged. See previous read result.]",
                    skipped=True,
                    skip_reason="Duplicate file read within session",
                    normalization_applied=normalization_applied,
                )

        # Code correction middleware - validate and fix code arguments
        code_corrected = False
        code_validation_errors: Optional[List[str]] = None

        if (
            self.config.enable_code_correction
            and self.code_correction_middleware is not None
            and self.code_correction_middleware.should_validate(tool_name)
        ):
            try:
                correction_result = self.code_correction_middleware.validate_and_fix(
                    tool_name, normalized_args
                )

                if correction_result.was_corrected:
                    # Apply the correction
                    normalized_args = self.code_correction_middleware.apply_correction(
                        normalized_args, correction_result
                    )
                    code_corrected = True
                    logger.info(
                        f"Code auto-corrected for tool '{tool_name}': "
                        f"{len(correction_result.validation.errors)} issues fixed"
                    )

                if not correction_result.validation.valid:
                    # Collect validation errors for feedback
                    code_validation_errors = list(correction_result.validation.errors)
                    logger.warning(
                        f"Code validation errors for tool '{tool_name}': "
                        f"{code_validation_errors}"
                    )
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Code correction middleware failed (data error): {e}")
            except AttributeError as e:
                logger.debug(f"Code correction middleware not configured: {e}")

        # Check for repeated failures
        if self.config.enable_failed_signature_tracking:
            signature = self._get_call_signature(tool_name, normalized_args)
            if signature in self._failed_signatures:
                return ToolCallResult(
                    tool_name=tool_name,
                    arguments=normalized_args,
                    success=False,
                    skipped=True,
                    skip_reason="Repeated failing call with same arguments",
                    normalization_applied=normalization_applied,
                )

        # Check for redundant tool calls using deduplication tracker
        if self.deduplication_tracker is not None:
            try:
                if self.deduplication_tracker.is_redundant(tool_name, normalized_args):
                    logger.info(
                        f"[Pipeline] Skipping redundant tool call: {tool_name} "
                        f"(semantic overlap with recent calls)"
                    )
                    return ToolCallResult(
                        tool_name=tool_name,
                        arguments=normalized_args,
                        success=False,
                        skipped=True,
                        skip_reason="Redundant call (semantic overlap with recent operations)",
                        normalization_applied=normalization_applied,
                    )
            except (ValueError, TypeError) as e:
                logger.warning(f"Deduplication tracker check failed (data error): {e}")
            except AttributeError as e:
                logger.debug(f"Deduplication tracker not properly initialized: {e}")

        # Process through middleware chain (before execution)
        if self.middleware_chain is not None:
            try:
                before_result = await self.middleware_chain.process_before(
                    tool_name, normalized_args
                )
                if not before_result.proceed:
                    logger.info(
                        f"[Pipeline] Tool '{tool_name}' blocked by middleware: "
                        f"{before_result.error_message}"
                    )
                    return ToolCallResult(
                        tool_name=tool_name,
                        arguments=normalized_args,
                        success=False,
                        skipped=True,
                        skip_reason=f"Blocked by middleware: {before_result.error_message}",
                        normalization_applied=normalization_applied,
                    )
                # Apply any argument modifications from middleware
                if before_result.modified_arguments:
                    normalized_args = before_result.modified_arguments
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Middleware chain process_before failed (data error): {e}")
            except AttributeError as e:
                logger.debug(f"Middleware chain not properly configured: {e}")

        # Emit pre-execution intent event (LogAct-inspired)
        self._emit_tool_intent(tool_name, normalized_args)

        # Notify start
        if self.on_tool_start:
            try:
                self.on_tool_start(tool_name, normalized_args)
            except Exception as e:
                logger.warning(f"on_tool_start callback failed: {e}")

        # Execute with per-tool timeout
        # Log tool call sent to LLM
        logger.debug(
            "[ToolCall→LLM] Executing tool=%s args=%s",
            tool_name,
            json.dumps(normalized_args, default=str)[:500],
        )
        start_time = time.monotonic()
        try:
            exec_result = await asyncio.wait_for(
                self.executor.execute(
                    tool_name=tool_name,
                    arguments=normalized_args,
                    context=context,
                ),
                timeout=self._get_tool_timeout(tool_name),
            )
        except asyncio.TimeoutError:
            effective_timeout = self._get_tool_timeout(tool_name)
            logger.warning(
                f"[Pipeline] Tool '{tool_name}' timed out after " f"{effective_timeout}s"
            )
            exec_result = ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Tool execution timed out after {effective_timeout}s",
            )
        execution_time_ms = (time.monotonic() - start_time) * 1000

        # Update state
        self._calls_used += 1
        self._executed_tools.append(tool_name)

        # Build result
        call_result = ToolCallResult(
            tool_name=tool_name,
            arguments=normalized_args,
            success=exec_result.success,
            result=exec_result.result,
            error=exec_result.error,
            execution_time_ms=execution_time_ms,
            normalization_applied=normalization_applied,
            code_corrected=code_corrected,
            code_validation_errors=code_validation_errors,
        )

        # Log tool result returned to LLM
        result_preview = str(exec_result.result)[:500] if exec_result.result else "(empty)"
        logger.debug(
            "[ToolResult→LLM] tool=%s success=%s time=%.0fms result_preview=%s",
            tool_name,
            exec_result.success,
            execution_time_ms,
            result_preview,
        )

        # Attempt error recovery fallback on failure
        if not exec_result.success and exec_result.error:
            try:
                from victor.agent.error_recovery import (
                    recover_from_error,
                    RecoveryAction,
                )

                recovery = recover_from_error(
                    Exception(exec_result.error), tool_name, normalized_args
                )
                if recovery.action == RecoveryAction.FALLBACK_TOOL and recovery.fallback_tool:
                    fallback_name = recovery.fallback_tool
                    if self.tools.is_tool_enabled(fallback_name):
                        logger.info(f"[Pipeline] Falling back from {tool_name} to {fallback_name}")
                        try:
                            fb_result = await asyncio.wait_for(
                                self.executor.execute(
                                    tool_name=fallback_name,
                                    arguments=normalized_args,
                                    context=context,
                                ),
                                timeout=self.config.per_tool_timeout_seconds,
                            )
                        except asyncio.TimeoutError:
                            fb_result = None
                        if fb_result is not None and fb_result.success:
                            call_result = ToolCallResult(
                                tool_name=fallback_name,
                                arguments=normalized_args,
                                success=fb_result.success,
                                result=fb_result.result,
                                error=fb_result.error,
                                execution_time_ms=((time.monotonic() - start_time) * 1000),
                            )
            except Exception as e:
                logger.debug(f"[Pipeline] Error recovery fallback failed: {e}")

        if self.on_tool_event:
            try:
                payload = {
                    "tool_name": tool_name,
                    "success": exec_result.success,
                    "execution_time_ms": execution_time_ms,
                    "arguments": normalized_args,
                }
                if exec_result.error:
                    payload["error"] = exec_result.error
                if exec_result.result is not None:
                    payload["result"] = exec_result.result
                self.on_tool_event("tool.raw_result", payload)
            except Exception as e:
                logger.debug(f"on_tool_event callback failed for raw result: {e}")

        # Process through middleware chain (after execution)
        if self.middleware_chain is not None:
            try:
                modified_result = await self.middleware_chain.process_after(
                    tool_name, normalized_args, call_result.result, call_result.success
                )
                if modified_result is not None and modified_result != call_result.result:
                    call_result = ToolCallResult(
                        tool_name=call_result.tool_name,
                        arguments=call_result.arguments,
                        success=call_result.success,
                        result=modified_result,
                        error=call_result.error,
                        execution_time_ms=call_result.execution_time_ms,
                        normalization_applied=call_result.normalization_applied,
                        code_corrected=call_result.code_corrected,
                        code_validation_errors=call_result.code_validation_errors,
                    )
                    if self.on_tool_event:
                        try:
                            self.on_tool_event(
                                "tool.middleware_adjusted",
                                {
                                    "tool_name": call_result.tool_name,
                                    "description": "Result modified by middleware chain",
                                },
                            )
                        except Exception as e:
                            logger.debug(f"on_tool_event middleware notification failed: {e}")
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Middleware chain process_after failed (data error): {e}")
            except AttributeError as e:
                logger.debug(f"Middleware chain not properly configured: {e}")

        # Record file read for deduplication (prompting loop fix)
        # Include capitalized variants for tool name normalization
        if call_result.success and tool_name.lower() in ("read", "read_file"):
            file_path = normalized_args.get("path") or normalized_args.get("file_path")
            if file_path:
                self.record_file_read(file_path)

        # Track failed signatures
        if not exec_result.success and self.config.enable_failed_signature_tracking:
            signature = self._get_call_signature(tool_name, normalized_args)
            self._failed_signatures.add(signature)

        # Cache successful results for idempotent tools
        if call_result.success:
            self.cache_result(tool_name, normalized_args, call_result)

            # Store in semantic cache (FAISS-based with mtime tracking)
            if self.config.enable_semantic_caching and self.semantic_cache is not None:
                try:
                    # Get file path for mtime tracking
                    file_path = normalized_args.get("path") or normalized_args.get("file_path")
                    await self.semantic_cache.put(
                        tool_name,
                        normalized_args,
                        call_result.result,
                        file_path=file_path,
                    )
                except (OSError, IOError) as e:
                    logger.debug(f"Semantic cache store failed (I/O error): {e}")
                except (ValueError, TypeError) as e:
                    logger.debug(f"Semantic cache store failed (data error): {e}")

        # Invalidate caches when files are modified (write/edit tools)
        if call_result.success and tool_name.lower() in ("write", "edit"):
            file_path = normalized_args.get("path") or normalized_args.get("file_path")
            if file_path:
                # Invalidate idempotent cache
                self.invalidate_file_cache(file_path)
                # Invalidate semantic cache
                if self.semantic_cache is not None:
                    self.semantic_cache.invalidate(file_path)
                logger.debug(f"Invalidated caches for modified file: {file_path}")

        # Record successful tool call in deduplication tracker
        if call_result.success and self.deduplication_tracker is not None:
            try:
                self.deduplication_tracker.add_call(tool_name, normalized_args)
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to record call in deduplication tracker (data error): {e}")
            except AttributeError as e:
                logger.debug(f"Deduplication tracker not properly initialized: {e}")

        # Update analytics
        if self.config.enable_analytics:
            self._update_analytics(call_result)

        # Notify complete
        if self.on_tool_complete:
            try:
                self.on_tool_complete(call_result)
            except Exception as e:
                logger.warning(f"on_tool_complete callback failed: {e}")
        elif self.on_tool_event:
            # Emit a simple fallback event so UI can still react
            try:
                self.on_tool_event(
                    "tool.complete",
                    {
                        "tool_name": call_result.tool_name,
                        "success": call_result.success,
                        "execution_time_ms": call_result.execution_time_ms,
                    },
                )
            except Exception:
                logger.debug("on_tool_event fallback emission failed", exc_info=True)

        return call_result

    def _update_analytics(self, result: ToolCallResult) -> None:
        """Update tool analytics.

        Args:
            result: Tool call result
        """
        tool_name = result.tool_name
        if tool_name not in self._tool_stats:
            self._tool_stats[tool_name] = {
                "calls": 0,
                "successes": 0,
                "failures": 0,
                "total_time_ms": 0.0,
            }

        stats = self._tool_stats[tool_name]
        stats["calls"] += 1
        if result.success:
            stats["successes"] += 1
        else:
            stats["failures"] += 1
        stats["total_time_ms"] += result.execution_time_ms

    def get_analytics(self) -> Dict[str, Any]:
        """Get tool execution analytics.

        Returns:
            Dictionary with analytics data
        """
        return {
            "total_calls": self._calls_used,
            "budget": self.config.tool_budget,
            "remaining": self.calls_remaining,
            "tools": dict(self._tool_stats),
        }

    def _get_tool_timeout(self, tool_name: str) -> float:
        """Get effective timeout for a tool, preferring per-tool override.

        Resolution order:
        1. Per-tool timeout from @tool(timeout=N) decorator
        2. Pipeline default (config.per_tool_timeout_seconds)
        """
        # Check if the tool has a per-tool timeout via the registry
        try:
            tool_func = self.executor.get_tool_function(tool_name)
            if tool_func and hasattr(tool_func, "_tool_timeout"):
                per_tool = tool_func._tool_timeout
                if isinstance(per_tool, (int, float)) and per_tool > 0:
                    return float(per_tool)
        except Exception:
            pass
        return self.config.per_tool_timeout_seconds

    def clear_failed_signatures(self) -> None:
        """Clear the failed signature cache."""
        self._failed_signatures.clear()
