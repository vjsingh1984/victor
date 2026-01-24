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

"""HTTP endpoints for Prometheus metrics and health checks.

This module provides FastAPI endpoints for:
- /metrics - Prometheus metrics scraping
- /health - Liveness probe (Kubernetes)
- /ready - Readiness probe (Kubernetes)
- /health/detailed - Detailed health information

Usage:
    from fastapi import FastAPI
    from victor.observability.metrics_endpoint import create_observability_app

    # Create observability app
    app = create_observability_app()

    # Or mount on existing app
    from victor.observability.metrics_endpoint import mount_observability_routes
    mount_observability_routes(app)

    # Run with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# FastAPI is optional
try:
    from fastapi import FastAPI, Response, status
    from fastapi.responses import PlainTextResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.debug("FastAPI not available. Install with: pip install fastapi uvicorn")


def create_observability_app(
    metrics_path: str = "/metrics",
    health_path: str = "/health",
    ready_path: str = "/ready",
    enable_cors: bool = False,
) -> "FastAPI":
    """Create a standalone FastAPI app for observability endpoints.

    Args:
        metrics_path: Path for metrics endpoint.
        health_path: Path for health endpoint.
        ready_path: Path for readiness endpoint.
        enable_cors: Enable CORS for all origins.

    Returns:
        FastAPI application instance.

    Raises:
        ImportError: If FastAPI is not installed.

    Example:
        app = create_observability_app()
        uvicorn.run(app, host="0.0.0.0", port=9090)
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is required for observability endpoints. "
            "Install with: pip install victor-ai[api]"
        )

    app = FastAPI(
        title="Victor Observability",
        description="Prometheus metrics and health check endpoints",
        version="0.5.1",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add CORS if enabled
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Mount routes
    mount_observability_routes(
        app,
        metrics_path=metrics_path,
        health_path=health_path,
        ready_path=ready_path,
    )

    return app


def mount_observability_routes(
    app: "FastAPI",
    metrics_path: str = "/metrics",
    health_path: str = "/health",
    ready_path: str = "/ready",
) -> None:
    """Mount observability routes on existing FastAPI app.

    Args:
        app: FastAPI application instance.
        metrics_path: Path for metrics endpoint.
        health_path: Path for health endpoint.
        ready_path: Path for readiness endpoint.

    Example:
        from fastapi import FastAPI
        from victor.observability.metrics_endpoint import mount_observability_routes

        app = FastAPI()
        mount_observability_routes(app)
    """
    if not FASTAPI_AVAILABLE:
        logger.warning("FastAPI not available, skipping observability routes")
        return

    @app.get(metrics_path, response_class=PlainTextResponse)
    async def get_metrics():
        """Prometheus metrics endpoint."""
        from victor.observability.prometheus_metrics import get_prometheus_exporter

        try:
            exporter = get_prometheus_exporter()
            metrics_text = exporter.export_metrics()
            return Response(
                content=metrics_text,
                media_type="text/plain",
                headers={"Content-Type": "text/plain; version=0.0.4"},
            )
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return Response(
                content="# Metrics export failed\n",
                media_type="text/plain",
                status_code=500,
            )

    @app.get(health_path)
    async def liveness_probe():
        """Kubernetes liveness probe - is the service running?"""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "service": "victor-observability",
        }

    @app.get(ready_path)
    async def readiness_probe():
        """Kubernetes readiness probe - is the service ready to serve traffic?"""
        from victor.core.health import HealthStatus

        try:
            from victor.observability.production_health import get_production_health_checker

            health_checker = get_production_health_checker()
            is_ready = await health_checker.check_readiness()

            if is_ready:
                return {
                    "status": "ready",
                    "timestamp": time.time(),
                }
            else:
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "not_ready",
                        "timestamp": time.time(),
                    },
                )

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "error": str(e),
                    "timestamp": time.time(),
                },
            )

    @app.get("/health/detailed")
    async def detailed_health():
        """Detailed health check with component status."""
        try:
            from victor.observability.production_health import get_production_health_checker

            health_checker = get_production_health_checker()
            report = await health_checker.check_health()
            return report.to_dict()

        except Exception as e:
            logger.error(f"Detailed health check failed: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "error": str(e),
                    "timestamp": time.time(),
                },
            )


def start_metrics_server(
    host: str = "0.0.0.0",
    port: int = 9090,
    metrics_path: str = "/metrics",
    health_path: str = "/health",
    ready_path: str = "/ready",
    log_level: str = "info",
):
    """Start a standalone metrics server.

    This is a convenience function for running a dedicated metrics server.

    Args:
        host: Server host.
        port: Server port.
        metrics_path: Path for metrics endpoint.
        health_path: Path for health endpoint.
        ready_path: Path for readiness endpoint.
        log_level: Log level for uvicorn.

    Example:
        start_metrics_server(port=9090)
        # Metrics available at http://localhost:9090/metrics
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI and uvicorn are required for metrics server. "
            "Install with: pip install victor-ai[api]"
        )

    app = create_observability_app(
        metrics_path=metrics_path,
        health_path=health_path,
        ready_path=ready_path,
    )

    import uvicorn

    logger.info(f"Starting metrics server on {host}:{port}")
    logger.info(f"Metrics endpoint: http://{host}:{port}{metrics_path}")
    logger.info(f"Health endpoint: http://{host}:{port}{health_path}")
    logger.info(f"Readiness endpoint: http://{host}:{port}{ready_path}")

    uvicorn.run(app, host=host, port=port, log_level=log_level)


# =============================================================================
# Standalone Server (for development)
# =============================================================================

if __name__ == "__main__":
    import sys

    # Parse command line args
    port = int(os.getenv("VICTOR_METRICS_PORT", "9090"))
    host = os.getenv("VICTOR_METRICS_HOST", "0.0.0.0")

    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}")
            sys.exit(1)

    start_metrics_server(host=host, port=port)
