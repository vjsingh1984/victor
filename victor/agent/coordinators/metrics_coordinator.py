# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Deprecated coordinator-path shim for MetricsCoordinator."""

from victor.agent.services.metrics_service import MetricsCoordinator, create_metrics_coordinator

__all__ = ["MetricsCoordinator", "create_metrics_coordinator"]
