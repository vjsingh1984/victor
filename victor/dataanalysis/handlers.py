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

"""DataAnalysis vertical compute handlers.

Domain-specific handlers for data analysis workflows:
- stats_compute: Statistical computations on datasets
- ml_training: Model training orchestration

Usage:
    # Handlers are auto-registered when DataAnalysis vertical is loaded
    from victor.dataanalysis import handlers
    handlers.register_handlers()

    # Or in YAML workflow:
    - id: compute_stats
      type: compute
      handler: stats_compute
      inputs:
        data: $ctx.raw_data
        operations: [describe, correlation]
      output: statistics
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from victor.tools.registry import ToolRegistry
    from victor.workflows.definition import ComputeNode
    from victor.workflows.executor import NodeResult, NodeStatus, WorkflowContext

logger = logging.getLogger(__name__)


@dataclass
class StatsComputeHandler:
    """Compute statistical measures on datasets.

    Runs statistical computations without LLM involvement.
    Supports common operations like mean, median, std, correlation.

    Example YAML:
        - id: compute_stats
          type: compute
          handler: stats_compute
          inputs:
            data: $ctx.raw_data
            operations: [describe, correlation, skewness]
          output: statistics
    """

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        from victor.workflows.executor import NodeResult, NodeStatus

        start_time = time.time()

        try:
            data = None
            operations = []

            for key, value in node.input_mapping.items():
                if key == "data":
                    data = context.get(value) if isinstance(value, str) else value
                elif key == "operations":
                    operations = value if isinstance(value, list) else [value]

            if data is None:
                return NodeResult(
                    node_id=node.id,
                    status=NodeStatus.FAILED,
                    error="No 'data' input provided",
                    duration_seconds=time.time() - start_time,
                )

            results = {}
            for op in operations:
                results[op] = self._compute_stat(data, op)

            output_key = node.output_key or node.id
            context.set(output_key, results)

            return NodeResult(
                node_id=node.id,
                status=NodeStatus.COMPLETED,
                output=results,
                duration_seconds=time.time() - start_time,
            )

        except Exception as e:
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

    def _compute_stat(self, data: Any, operation: str) -> Any:
        """Compute a single statistic."""
        if isinstance(data, list) and data:
            numeric = [x for x in data if isinstance(x, (int, float))]
            if not numeric:
                return None

            if operation == "mean":
                return sum(numeric) / len(numeric)
            elif operation == "median":
                sorted_data = sorted(numeric)
                n = len(sorted_data)
                mid = n // 2
                return (sorted_data[mid] + sorted_data[~mid]) / 2
            elif operation == "min":
                return min(numeric)
            elif operation == "max":
                return max(numeric)
            elif operation == "sum":
                return sum(numeric)
            elif operation == "count":
                return len(numeric)
            elif operation == "std":
                mean = sum(numeric) / len(numeric)
                variance = sum((x - mean) ** 2 for x in numeric) / len(numeric)
                return variance ** 0.5
            elif operation == "describe":
                mean = sum(numeric) / len(numeric)
                return {
                    "count": len(numeric),
                    "mean": mean,
                    "min": min(numeric),
                    "max": max(numeric),
                    "sum": sum(numeric),
                }
        return None


@dataclass
class MLTrainingHandler:
    """Orchestrate ML model training.

    Manages training workflow including data split, training,
    and evaluation without LLM involvement.

    Example YAML:
        - id: train_model
          type: compute
          handler: ml_training
          inputs:
            features: $ctx.features
            target: $ctx.target
            model_type: random_forest
          output: trained_model
    """

    async def __call__(
        self,
        node: "ComputeNode",
        context: "WorkflowContext",
        tool_registry: "ToolRegistry",
    ) -> "NodeResult":
        from victor.workflows.executor import NodeResult, NodeStatus

        start_time = time.time()

        model_type = node.input_mapping.get("model_type", "linear")
        features_key = node.input_mapping.get("features")
        target_key = node.input_mapping.get("target")

        try:
            train_cmd = f"python -m victor.ml.train --model {model_type}"
            result = await tool_registry.execute("shell", command=train_cmd)

            output = {
                "model_type": model_type,
                "status": "trained" if result.success else "failed",
                "output": result.output,
            }

            output_key = node.output_key or node.id
            context.set(output_key, output)

            return NodeResult(
                node_id=node.id,
                status=NodeStatus.COMPLETED if result.success else NodeStatus.FAILED,
                output=output,
                duration_seconds=time.time() - start_time,
                tool_calls_used=1,
            )
        except Exception as e:
            return NodeResult(
                node_id=node.id,
                status=NodeStatus.FAILED,
                error=str(e),
                duration_seconds=time.time() - start_time,
            )


# Handler instances
HANDLERS = {
    "stats_compute": StatsComputeHandler(),
    "ml_training": MLTrainingHandler(),
}


def register_handlers() -> None:
    """Register DataAnalysis handlers with the workflow executor."""
    from victor.workflows.executor import register_compute_handler

    for name, handler in HANDLERS.items():
        register_compute_handler(name, handler)
        logger.debug(f"Registered DataAnalysis handler: {name}")


__all__ = [
    "StatsComputeHandler",
    "MLTrainingHandler",
    "HANDLERS",
    "register_handlers",
]
