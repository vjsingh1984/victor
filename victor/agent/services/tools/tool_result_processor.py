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

"""Tool result processor implementation.

Handles result processing, formatting, and aggregation.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from victor.tools.base import ToolResult

logger = logging.getLogger(__name__)


class ToolResultProcessorConfig:
    """Configuration for ToolResultProcessor.

    Attributes:
        enable_insights: Enable insight extraction
        max_result_length: Maximum result length for formatting
        truncate_long_results: Truncate long results in formatting
    """

    def __init__(
        self,
        enable_insights: bool = True,
        max_result_length: int = 10000,
        truncate_long_results: bool = True,
    ):
        self.enable_insights = enable_insights
        self.max_result_length = max_result_length
        self.truncate_long_results = truncate_long_results


class ToolResultProcessor:
    """Service for processing tool results.

    Responsible for:
    - Result processing and normalization
    - Result formatting for LLM consumption
    - Result aggregation
    - Insight extraction

    This service does NOT handle:
    - Tool selection (delegated to ToolSelectorService)
    - Tool execution (delegated to ToolExecutorService)
    - Budget tracking (delegated to ToolTrackerService)
    - Execution planning (delegated to ToolPlannerService)

    Example:
        config = ToolResultProcessorConfig()
        processor = ToolResultProcessor(config=config)

        # Process single result
        processed = processor.process_result(result)

        # Format for LLM
        formatted = processor.format_result_for_llm(result)

        # Aggregate multiple results
        aggregated = processor.aggregate_results(results)
    """

    def __init__(self, config: ToolResultProcessorConfig):
        """Initialize ToolResultProcessor.

        Args:
            config: Service configuration
        """
        self.config = config

        # Health tracking
        self._healthy = True

    def process_result(self, result: "ToolResult") -> Dict[str, Any]:
        """Process a single tool result.

        Normalizes and enriches the result with metadata.

        Args:
            result: ToolResult to process

        Returns:
            Processed result dictionary
        """
        processed = {
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "has_output": result.output is not None,
            "has_error": result.error is not None,
        }

        # Add metadata
        if hasattr(result, "metadata") and result.metadata:
            processed["metadata"] = result.metadata

        # Add output type
        if result.output is not None:
            processed["output_type"] = type(result.output).__name__

        return processed

    def format_result_for_llm(self, result: "ToolResult") -> str:
        """Format a tool result for LLM consumption.

        Creates a clean, readable format suitable for LLM input.

        Args:
            result: ToolResult to format

        Returns:
            Formatted result string
        """
        if result.success:
            output_str = self._format_output(result.output)

            # Truncate if needed
            if (
                self.config.truncate_long_results
                and len(output_str) > self.config.max_result_length
            ):
                output_str = output_str[: self.config.max_result_length] + "... (truncated)"

            return f"✓ Success: {output_str}"
        else:
            return f"✗ Error: {result.error or 'Unknown error'}"

    def _format_output(self, output: Any) -> str:
        """Format output value to string.

        Args:
            output: Output value to format

        Returns:
            Formatted string
        """
        if output is None:
            return "No output"

        if isinstance(output, str):
            return output

        if isinstance(output, (dict, list)):
            return json.dumps(output, indent=2)

        if isinstance(output, (int, float, bool)):
            return str(output)

        # For other types, try string representation
        return str(output)

    def aggregate_results(self, results: List["ToolResult"]) -> Dict[str, Any]:
        """Aggregate multiple tool results.

        Combines multiple results into a single aggregated view.

        Args:
            results: List of ToolResults to aggregate

        Returns:
            Aggregated results dictionary
        """
        if not results:
            return {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "results": [],
            }

        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        aggregated = {
            "total": len(results),
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / len(results)) * 100 if results else 0,
            "results": [self.process_result(r) for r in results],
        }

        return aggregated

    def extract_insights(self, results: List["ToolResult"]) -> List[str]:
        """Extract insights from tool results.

        Analyzes results to extract key insights and patterns.

        Args:
            results: List of ToolResults to analyze

        Returns:
            List of insight strings
        """
        if not self.config.enable_insights:
            return []

        insights = []

        # Success rate insight
        successful = sum(1 for r in results if r.success)
        if results:
            rate = (successful / len(results)) * 100
            if rate == 100:
                insights.append("All tool executions succeeded")
            elif rate >= 80:
                insights.append(f"Most tool executions succeeded ({rate:.0f}%)")
            elif rate >= 50:
                insights.append(f"Many tool executions failed ({100 - rate:.0f}% failure rate)")
            else:
                insights.append(f"Most tool executions failed ({100 - rate:.0f}% failure rate)")

        # Error pattern insights
        errors = [r.error for r in results if not r.success and r.error]
        if errors:
            # Check for common error patterns
            if any("timeout" in str(e).lower() for e in errors):
                insights.append("Some tools timed out - consider increasing timeouts")

            if any("permission" in str(e).lower() for e in errors):
                insights.append("Some tools failed due to permission issues")

            if any("not found" in str(e).lower() for e in errors):
                insights.append("Some resources were not found")

        # Output size insights
        large_outputs = [r for r in results if r.success and r.output and len(str(r.output)) > 1000]
        if large_outputs:
            insights.append(f"{len(large_outputs)} tool(s) produced large outputs (>1000 chars)")

        return insights

    def summarize_results(self, results: List["ToolResult"]) -> str:
        """Create a human-readable summary of results.

        Args:
            results: List of ToolResults to summarize

        Returns:
            Summary string
        """
        if not results:
            return "No results to summarize"

        aggregated = self.aggregate_results(results)
        insights = self.extract_insights(results)

        lines = [
            f"Executed {aggregated['total']} tool(s)",
            f"✓ Successful: {aggregated['successful']}",
            f"✗ Failed: {aggregated['failed']}",
            f"Success rate: {aggregated['success_rate']:.0f}%",
        ]

        if insights:
            lines.append("\nInsights:")
            lines.extend(f"  • {insight}" for insight in insights)

        return "\n".join(lines)

    def filter_results(
        self, results: List["ToolResult"], successful_only: bool = False
    ) -> List["ToolResult"]:
        """Filter results based on criteria.

        Args:
            results: List of ToolResults to filter
            successful_only: If True, only return successful results

        Returns:
            Filtered list of ToolResults
        """
        if successful_only:
            return [r for r in results if r.success]
        return results

    def is_healthy(self) -> bool:
        """Check if service is healthy.

        Returns:
            True if healthy, False otherwise
        """
        return self._healthy
