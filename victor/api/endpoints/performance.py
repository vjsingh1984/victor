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

"""Performance monitoring API endpoints.

Provides REST API for accessing performance metrics:
- GET /api/performance/summary - Overall metrics
- GET /api/performance/cache - Cache metrics
- GET /api/performance/providers - Provider metrics
- GET /api/performance/tools - Tool metrics
- GET /api/performance/prometheus - Prometheus format export

Example:
    from fastapi import FastAPI
    from victor.api.endpoints.performance import register_performance_routes

    app = FastAPI()
    register_performance_routes(app)
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from victor.observability.performance_collector import (
    PerformanceMetricsCollector,
    get_performance_collector,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/performance", tags=["performance"])


# =============================================================================
# Performance Endpoints
# =============================================================================


@router.get("/summary")
async def get_performance_summary() -> Dict[str, Any]:
    """Get overall performance metrics summary.

    Returns:
        Dictionary containing all performance metrics.
    """
    try:
        collector = get_performance_collector()
        metrics = collector.get_all_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Failed to get performance summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache")
async def get_cache_metrics() -> Dict[str, Any]:
    """Get cache performance metrics.

    Returns:
        Dictionary with cache metrics including:
        - Hit rates by namespace
        - Entry counts
        - Memory usage
        - Operations counts
    """
    try:
        collector = get_performance_collector()
        metrics = collector.get_tool_selection_metrics()
        return metrics.to_dict()
    except Exception as e:
        logger.error(f"Failed to get cache metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/providers")
async def get_provider_metrics() -> Dict[str, Any]:
    """Get provider pool performance metrics.

    Returns:
        Dictionary with provider metrics including:
        - Pool health (active/unhealthy providers)
        - Request metrics (success/failure rates)
        - Latency metrics (p50, p95, p99)
        - Per-provider health status
    """
    try:
        collector = get_performance_collector()
        metrics = collector.get_provider_pool_metrics()
        return metrics.to_dict()
    except Exception as e:
        logger.error(f"Failed to get provider metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tools")
async def get_tool_metrics() -> Dict[str, Any]:
    """Get tool execution performance metrics.

    Returns:
        Dictionary with tool metrics including:
        - Execution counts (total, successful, failed)
        - Duration metrics (avg, p50, p95, p99)
        - Error rate
        - Top tools by usage
        - Per-tool metrics
    """
    try:
        collector = get_performance_collector()
        metrics = collector.get_tool_execution_metrics()
        return metrics.to_dict()
    except Exception as e:
        logger.error(f"Failed to get tool metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system")
async def get_system_metrics() -> Dict[str, Any]:
    """Get system resource metrics.

    Returns:
        Dictionary with system metrics including:
        - Memory usage (bytes, MB, GB)
        - CPU usage (percent, count)
        - Uptime (seconds, minutes, hours)
        - Active threads
    """
    try:
        collector = get_performance_collector()
        metrics = collector.get_system_metrics()
        return metrics.to_dict()
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/bootstrap")
async def get_bootstrap_metrics() -> Dict[str, Any]:
    """Get bootstrap and startup performance metrics.

    Returns:
        Dictionary with bootstrap metrics including:
        - Total startup time
        - Phase timings (container, services, providers, tools)
        - Lazy loading metrics (count, time, overhead)
    """
    try:
        collector = get_performance_collector()
        metrics = collector.get_bootstrap_metrics()
        return metrics.to_dict()
    except Exception as e:
        logger.error(f"Failed to get bootstrap metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/prometheus", response_class=PlainTextResponse)
async def get_prometheus_metrics() -> str:
    """Get performance metrics in Prometheus text format.

    Returns:
        Prometheus format metrics string for scraping.
    """
    try:
        collector = get_performance_collector()
        metrics_text = collector.export_prometheus()
        return metrics_text
    except Exception as e:
        logger.error(f"Failed to export Prometheus metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Registration Function
# =============================================================================


def register_performance_routes(app: Any) -> None:
    """Register performance monitoring routes with FastAPI app.

    Args:
        app: FastAPI application instance.

    Example:
        from fastapi import FastAPI
        from victor.api.endpoints.performance import register_performance_routes

        app = FastAPI()
        register_performance_routes(app)
    """
    app.include_router(router)
    logger.info("Registered performance monitoring routes")


__all__ = [
    "router",
    "register_performance_routes",
]
