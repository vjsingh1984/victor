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

"""Production metrics collector for comprehensive monitoring.

This module provides comprehensive metrics collection for production monitoring,
including performance, functional, business, agentic AI, vertical-specific, and
security metrics. Integrates with Prometheus for metrics exposition.

Key Features:
- Automatic metric collection from system events
- Prometheus metric exposition at /metrics
- Custom metric registration
- Integration with EventBus
- Periodic metric reporting

Example:
    from victor.observability.metrics_collector import ProductionMetricsCollector

    # Initialize collector
    collector = ProductionMetricsCollector()

    # Start metrics server
    collector.start(port=9091)

    # Record metrics
    collector.record_tool_execution("read_file", "coding", "success", duration=0.5)
    collector.record_provider_request("anthropic", "claude-sonnet-4-5", "success", duration=2.3)
"""

from __future__ import annotations

import logging
import os
import psutil
import threading
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable
from functools import wraps
from contextlib import contextmanager

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Production Metrics Definition
# =============================================================================


@dataclass
class MetricConfig:
    """Configuration for a metric."""

    name: str
    description: str
    labels: List[str] = field(default_factory=list)
    metric_type: str = "counter"  # counter, gauge, histogram


# =============================================================================
# Production Metrics Collector
# =============================================================================


class ProductionMetricsCollector:
    """Comprehensive production metrics collector.

    Collects and exports metrics for production monitoring, including:
    - Performance metrics (response time, memory, CPU)
    - Functional metrics (tool execution, provider requests)
    - Business metrics (requests, users, sessions)
    - Agentic AI metrics (planning, memory, skills)
    - Vertical-specific metrics (coding, RAG, DevOps, etc.)
    - Security metrics (authorization, vulnerabilities)

    Attributes:
        registry: Prometheus CollectorRegistry
        enabled: Whether metrics collection is enabled
    """

    _instance: Optional["ProductionMetricsCollector"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "ProductionMetricsCollector":
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize metrics collector."""
        if self._initialized:
            return

        self._initialized = True
        self.enabled = os.getenv("VICTOR_PROMETHEUS_ENABLED", "true").lower() == "true"
        self.registry = CollectorRegistry()
        self._start_time = time.time()

        # Initialize all metrics
        self._init_performance_metrics()
        self._init_functional_metrics()
        self._init_business_metrics()
        self._init_agentic_metrics()
        self._init_vertical_metrics()
        self._init_security_metrics()

        # Track current state
        self._active_users: Dict[str, float] = {}
        self._session_start_times: Dict[str, float] = {}
        self._memory_baseline: Optional[float] = None

        logger.info("ProductionMetricsCollector initialized")

    # =========================================================================
    # Performance Metrics
    # =========================================================================

    def _init_performance_metrics(self) -> None:
        """Initialize performance-related metrics."""
        # Request duration
        self.request_duration = Histogram(
            "victor_request_duration_seconds",
            "Request response time in seconds",
            ["endpoint", "status", "method"],
            buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry,
        )

        # Chat request duration
        self.chat_duration = Histogram(
            "victor_chat_request_duration_seconds",
            "Chat request duration in seconds",
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 45.0, 60.0],
            registry=self.registry,
        )

        # Tool execution duration
        self.tool_execution_duration = Histogram(
            "victor_tool_execution_duration_seconds",
            "Tool execution time in seconds",
            ["tool", "vertical", "status"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0],
            registry=self.registry,
        )

        # Provider latency
        self.provider_latency = Histogram(
            "victor_provider_latency_seconds",
            "Provider API latency in seconds",
            ["provider", "model"],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
            registry=self.registry,
        )

        # Initialization duration
        self.init_duration = Histogram(
            "victor_initialization_duration_seconds",
            "System initialization time in seconds",
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry,
        )

        # Memory usage
        self.memory_usage = Gauge(
            "victor_memory_usage_bytes",
            "Current memory usage in bytes",
            ["component"],
            registry=self.registry,
        )

        self.memory_limit = Gauge(
            "victor_memory_limit_bytes",
            "Memory limit in bytes",
            registry=self.registry,
        )

        # CPU usage
        self.cpu_usage = Gauge(
            "victor_cpu_usage_percent",
            "Current CPU usage percentage",
            ["component"],
            registry=self.registry,
        )

        # Request rate
        self.request_rate = Gauge(
            "victor_request_rate",
            "Request rate per second",
            registry=self.registry,
        )

    # =========================================================================
    # Functional Metrics
    # =========================================================================

    def _init_functional_metrics(self) -> None:
        """Initialize functional metrics."""
        # Tool executions
        self.tool_executions = Counter(
            "victor_tool_executions_total",
            "Total number of tool executions",
            ["tool", "vertical", "status", "mode"],
            registry=self.registry,
        )

        # Tool success rate
        self.tool_success_rate = Gauge(
            "victor_tool_success_rate",
            "Tool success rate",
            ["tool", "vertical"],
            registry=self.registry,
        )

        # Provider requests
        self.provider_requests = Counter(
            "victor_provider_requests_total",
            "Total provider API requests",
            ["provider", "model", "status", "tool_calls"],
            registry=self.registry,
        )

        # Provider success rate
        self.provider_success_rate = Gauge(
            "victor_provider_success_rate",
            "Provider success rate",
            ["provider", "model"],
            registry=self.registry,
        )

        # Vertical usage
        self.vertical_usage = Counter(
            "victor_vertical_usage_total",
            "Vertical usage count",
            ["vertical", "mode"],
            registry=self.registry,
        )

        # Workflow executions
        self.workflow_executions = Counter(
            "victor_workflow_executions_total",
            "Workflow executions",
            ["workflow", "status"],
            registry=self.registry,
        )

        # Feature usage
        self.feature_usage = Counter(
            "victor_feature_usage_total",
            "Feature usage",
            ["feature"],
            registry=self.registry,
        )

        # Cache metrics
        self.cache_hits = Counter(
            "victor_cache_hits_total",
            "Cache hits",
            ["cache_name"],
            registry=self.registry,
        )

        self.cache_misses = Counter(
            "victor_cache_misses_total",
            "Cache misses",
            ["cache_name"],
            registry=self.registry,
        )

        self.cache_evictions = Counter(
            "victor_cache_evictions_total",
            "Cache evictions",
            ["cache_name"],
            registry=self.registry,
        )

    # =========================================================================
    # Business Metrics
    # =========================================================================

    def _init_business_metrics(self) -> None:
        """Initialize business metrics."""
        # Total requests
        self.total_requests = Counter(
            "victor_total_requests",
            "Total requests served",
            registry=self.registry,
        )

        # Active users
        self.active_users_gauge = Gauge(
            "victor_active_users",
            "Current number of active users",
            ["interface"],
            registry=self.registry,
        )

        # Session duration
        self.session_duration = Histogram(
            "victor_session_duration_seconds",
            "User session duration in seconds",
            ["interface"],
            buckets=[60, 300, 600, 1800, 3600, 7200, 14400],
            registry=self.registry,
        )

        # Requests per user
        self.requests_per_user = Histogram(
            "victor_requests_per_user",
            "Requests per user",
            buckets=[1, 5, 10, 25, 50, 100, 250, 500],
            registry=self.registry,
        )

    # =========================================================================
    # Agentic AI Metrics
    # =========================================================================

    def _init_agentic_metrics(self) -> None:
        """Initialize agentic AI metrics."""
        # Planning operations
        self.planning_operations = Counter(
            "victor_planning_operations_total",
            "Planning operations",
            ["vertical", "status"],
            registry=self.registry,
        )

        self.planning_success_rate = Gauge(
            "victor_planning_success_rate",
            "Planning success rate",
            ["vertical"],
            registry=self.registry,
        )

        # Memory operations
        self.memory_operations = Counter(
            "victor_memory_operations_total",
            "Memory operations",
            ["memory_type", "operation", "status"],
            registry=self.registry,
        )

        self.memory_recall_accuracy = Gauge(
            "victor_memory_recall_accuracy",
            "Memory recall accuracy",
            ["memory_type"],
            registry=self.registry,
        )

        # Skill discovery
        self.skill_discovery = Counter(
            "victor_skill_discovery_total",
            "Skills discovered",
            ["skill_type"],
            registry=self.registry,
        )

        # Proficiency score
        self.proficiency_score = Gauge(
            "victor_proficiency_score",
            "Current proficiency score",
            ["skill"],
            registry=self.registry,
        )

        # Self-improvement loops
        self.self_improvement_loops = Counter(
            "victor_self_improvement_loops_total",
            "Self-improvement loops",
            ["outcome"],
            registry=self.registry,
        )

    # =========================================================================
    # Vertical-Specific Metrics
    # =========================================================================

    def _init_vertical_metrics(self) -> None:
        """Initialize vertical-specific metrics."""
        # Coding
        self.coding_files_analyzed = Counter(
            "victor_coding_files_analyzed_total",
            "Files analyzed",
            registry=self.registry,
        )

        self.coding_loc_reviewed = Counter(
            "victor_coding_loc_reviewed_total",
            "Lines of code reviewed",
            registry=self.registry,
        )

        self.coding_issues_found = Counter(
            "victor_coding_issues_found_total",
            "Issues found",
            registry=self.registry,
        )

        self.coding_tests_generated = Counter(
            "victor_coding_tests_generated_total",
            "Tests generated",
            ["status"],
            registry=self.registry,
        )

        self.coding_pending_analysis = Gauge(
            "victor_coding_pending_analysis_files",
            "Files pending analysis",
            registry=self.registry,
        )

        # RAG
        self.rag_documents_ingested = Counter(
            "victor_rag_documents_ingested_total",
            "Documents ingested",
            registry=self.registry,
        )

        self.rag_search_accuracy = Gauge(
            "victor_rag_search_accuracy",
            "Search accuracy",
            registry=self.registry,
        )

        self.rag_retrieval_latency = Histogram(
            "victor_rag_retrieval_latency_seconds",
            "Retrieval latency",
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0],
            registry=self.registry,
        )

        self.rag_index_size = Gauge(
            "victor_rag_index_size_bytes",
            "Index size in bytes",
            registry=self.registry,
        )

        # DevOps
        self.devops_deployments = Counter(
            "victor_devops_deployments_total",
            "Deployments performed",
            ["status"],
            registry=self.registry,
        )

        self.devops_containers_managed = Gauge(
            "victor_devops_containers_managed",
            "Containers managed",
            ["environment"],
            registry=self.registry,
        )

        self.devops_ci_pipelines = Counter(
            "victor_devops_ci_pipelines_executed_total",
            "CI pipelines executed",
            registry=self.registry,
        )

        # DataAnalysis
        self.dataanalysis_queries = Counter(
            "victor_dataanalysis_queries_total",
            "Queries executed",
            registry=self.registry,
        )

        self.dataanalysis_visualizations = Counter(
            "victor_dataanalysis_visualizations_total",
            "Visualizations created",
            registry=self.registry,
        )

        self.dataanalysis_query_duration = Histogram(
            "victor_dataanalysis_query_duration_seconds",
            "Query duration",
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
            registry=self.registry,
        )

        # Research
        self.research_searches = Counter(
            "victor_research_searches_total",
            "Searches performed",
            registry=self.registry,
        )

        self.research_citations = Counter(
            "victor_research_citations_generated_total",
            "Citations generated",
            registry=self.registry,
        )

        self.research_synthesis_duration = Histogram(
            "victor_research_synthesis_duration_seconds",
            "Synthesis time",
            buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
            registry=self.registry,
        )

    # =========================================================================
    # Security Metrics
    # =========================================================================

    def _init_security_metrics(self) -> None:
        """Initialize security metrics."""
        # Authorization
        self.security_authorizations = Counter(
            "victor_security_authorizations_total",
            "Authorization attempts",
            ["status", "reason"],
            registry=self.registry,
        )

        self.security_authorization_success_rate = Gauge(
            "victor_security_authorization_success_rate",
            "Authorization success rate",
            registry=self.registry,
        )

        # Security tests
        self.security_tests = Counter(
            "victor_security_tests_total",
            "Security tests",
            ["status", "test_type"],
            registry=self.registry,
        )

        self.security_test_pass_rate = Gauge(
            "victor_security_test_pass_rate",
            "Security test pass rate",
            registry=self.registry,
        )

        # Vulnerabilities
        self.security_vulnerabilities = Counter(
            "victor_security_vulnerabilities_found_total",
            "Vulnerabilities found",
            ["severity"],
            registry=self.registry,
        )

        self.security_scan_duration = Histogram(
            "victor_security_scan_duration_seconds",
            "Security scan duration",
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry,
        )

    # =========================================================================
    # Metric Recording Methods
    # =========================================================================

    def record_tool_execution(
        self,
        tool: str,
        vertical: str,
        status: str,
        duration: float,
        mode: str = "build",
    ) -> None:
        """Record tool execution metric.

        Args:
            tool: Tool name
            vertical: Vertical name
            status: Execution status (success, failure, timeout)
            duration: Execution duration in seconds
            mode: Agent mode (build, plan, explore)
        """
        if not self.enabled:
            return

        self.tool_executions.labels(tool=tool, vertical=vertical, status=status, mode=mode).inc()
        self.tool_execution_duration.labels(tool=tool, vertical=vertical, status=status).observe(duration)

    def record_provider_request(
        self,
        provider: str,
        model: str,
        status: str,
        duration: float,
        tool_calls: bool = False,
    ) -> None:
        """Record provider request metric.

        Args:
            provider: Provider name
            model: Model name
            status: Request status (success, failure, timeout, rate_limited)
            duration: Request duration in seconds
            tool_calls: Whether tool calls were used
        """
        if not self.enabled:
            return

        self.provider_requests.labels(
            provider=provider,
            model=model,
            status=status,
            tool_calls=str(tool_calls).lower(),
        ).inc()
        self.provider_latency.labels(provider=provider, model=model).observe(duration)

    def record_chat_request(self, duration: float) -> None:
        """Record chat request metric.

        Args:
            duration: Request duration in seconds
        """
        if not self.enabled:
            return

        self.chat_duration.observe(duration)

    def record_request(
        self,
        endpoint: str,
        status: str,
        method: str,
        duration: float,
    ) -> None:
        """Record request metric.

        Args:
            endpoint: API endpoint
            status: HTTP status code
            method: HTTP method
            duration: Request duration in seconds
        """
        if not self.enabled:
            return

        self.request_duration.labels(endpoint=endpoint, status=status, method=method).observe(duration)
        self.total_requests.inc()

    def record_cache_operation(
        self,
        cache_name: str,
        hit: bool,
    ) -> None:
        """Record cache operation.

        Args:
            cache_name: Cache name
            hit: Whether it was a cache hit
        """
        if not self.enabled:
            return

        if hit:
            self.cache_hits.labels(cache_name=cache_name).inc()
        else:
            self.cache_misses.labels(cache_name=cache_name).inc()

    def update_system_metrics(self) -> None:
        """Update system resource metrics."""
        if not self.enabled:
            return

        process = psutil.Process()

        # Memory usage
        memory_info = process.memory_info()
        self.memory_usage.labels(component="orchestrator").set(memory_info.rss)

        # Set memory limit if not set
        if self._memory_baseline is None:
            self._memory_baseline = memory_info.rss

        # CPU usage
        cpu_percent = process.cpu_percent(interval=0.1)
        self.cpu_usage.labels(component="orchestrator").set(cpu_percent)

    def update_active_users(self, user_id: str, interface: str = "cli") -> None:
        """Update active users.

        Args:
            user_id: User identifier
            interface: Interface type (cli, tui, api, mcp)
        """
        if not self.enabled:
            return

        now = time.time()
        self._active_users[user_id] = now

        # Track session start if new user
        if user_id not in self._session_start_times:
            self._session_start_times[user_id] = now

        # Count active users (active in last 5 minutes)
        cutoff = now - 300
        active_count = sum(1 for t in self._active_users.values() if t > cutoff)
        self.active_users_gauge.labels(interface=interface).set(active_count)

    def record_session_end(self, user_id: str, interface: str = "cli") -> None:
        """Record session end.

        Args:
            user_id: User identifier
            interface: Interface type
        """
        if not self.enabled:
            return

        if user_id in self._session_start_times:
            duration = time.time() - self._session_start_times[user_id]
            self.session_duration.labels(interface=interface).observe(duration)
            del self._session_start_times[user_id]

        if user_id in self._active_users:
            del self._active_users[user_id]

    def record_vertical_usage(self, vertical: str, mode: str = "build") -> None:
        """Record vertical usage.

        Args:
            vertical: Vertical name
            mode: Agent mode
        """
        if not self.enabled:
            return

        self.vertical_usage.labels(vertical=vertical, mode=mode).inc()

    # =========================================================================
    # Metrics Export
    # =========================================================================

    def get_metrics_text(self) -> bytes:
        """Get metrics in Prometheus text format.

        Returns:
            Metrics in Prometheus exposition format
        """
        # Update system metrics before export
        self.update_system_metrics()

        return generate_latest(self.registry)

    def get_content_type(self) -> str:
        """Get content type for metrics endpoint.

        Returns:
            Content type string
        """
        return CONTENT_TYPE_LATEST

    def start(self, port: int = 9091, host: str = "0.0.0.0") -> None:
        """Start Prometheus metrics HTTP server.

        Args:
            port: Port to listen on
            host: Host to bind to
        """
        if not self.enabled:
            logger.warning("Metrics collection is disabled")
            return

        try:
            from prometheus_client import start_http_server

            start_http_server(port, host, registry=self.registry)
            logger.info(f"Prometheus metrics server started on http://{host}:{port}/metrics")
        except ImportError:
            logger.error("prometheus_client is required for metrics server")

    # =========================================================================
    # Decorators and Context Managers
    # =========================================================================

    def track_request(self, endpoint: str, method: str = "GET"):
        """Decorator to track requests.

        Args:
            endpoint: API endpoint
            method: HTTP method

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.time()
                status = "200"
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    status = "500"
                    logger.error(f"Request error: {e}")
                    raise
                finally:
                    duration = time.time() - start
                    self.record_request(endpoint, status, method, duration)

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.time()
                status = "200"
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    status = "500"
                    logger.error(f"Request error: {e}")
                    raise
                finally:
                    duration = time.time() - start
                    self.record_request(endpoint, status, method, duration)

            import asyncio

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    @contextmanager
    def track_tool_execution(
        self,
        tool: str,
        vertical: str,
        mode: str = "build",
    ):
        """Context manager to track tool execution.

        Args:
            tool: Tool name
            vertical: Vertical name
            mode: Agent mode

        Yields:
            None
        """
        start = time.time()
        status = "success"
        try:
            yield
        except Exception as e:
            status = "failure"
            logger.error(f"Tool execution error: {e}\n{traceback.format_exc()}")
            raise
        finally:
            duration = time.time() - start
            self.record_tool_execution(tool, vertical, status, duration, mode)

    @contextmanager
    def track_provider_request(
        self,
        provider: str,
        model: str,
        tool_calls: bool = False,
    ):
        """Context manager to track provider request.

        Args:
            provider: Provider name
            model: Model name
            tool_calls: Whether tool calls are used

        Yields:
            None
        """
        start = time.time()
        status = "success"
        try:
            yield
        except Exception as e:
            status = "failure"
            logger.error(f"Provider request error: {e}")
            raise
        finally:
            duration = time.time() - start
            self.record_provider_request(provider, model, status, duration, tool_calls)


# =============================================================================
# Singleton Accessor
# =============================================================================


def get_metrics_collector() -> ProductionMetricsCollector:
    """Get the singleton metrics collector instance.

    Returns:
        ProductionMetricsCollector instance
    """
    return ProductionMetricsCollector()
