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

"""Escape hatches for DataAnalysis YAML workflows.

Complex conditions and transforms that cannot be expressed in YAML.
These are registered with the YAML workflow loader for use in condition nodes.

Example YAML usage:
    - id: check_quality
      type: condition
      condition: "quality_threshold"  # References escape hatch
      branches:
        "high_quality": proceed
        "acceptable": proceed_with_warning
        "needs_cleanup": cleanup
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


# =============================================================================
# Condition Functions
# =============================================================================


def should_retry_cleaning(ctx: Dict[str, Any]) -> str:
    """Determine if data cleaning should be retried.

    Multi-factor decision based on validation status and iteration count.

    Args:
        ctx: Workflow context with keys:
            - validation_passed (bool): Whether validation succeeded
            - iteration (int): Current iteration count
            - max_iterations (int): Maximum allowed iterations

    Returns:
        "retry" if should retry, "done" if should proceed
    """
    if ctx.get("validation_passed", False):
        return "done"

    iteration = ctx.get("iteration", 0)
    max_iter = ctx.get("max_iterations", 3)

    if iteration >= max_iter:
        logger.warning(f"Max cleaning iterations ({max_iter}) reached, proceeding anyway")
        return "done"

    return "retry"


def should_tune_more(ctx: Dict[str, Any]) -> str:
    """Determine if more model tuning is needed.

    Checks if model performance meets threshold or max iterations reached.

    Args:
        ctx: Workflow context with keys:
            - metrics (dict): Model metrics with 'primary_metric' key
            - iteration (int): Current iteration count
            - max_iterations (int): Maximum allowed iterations

    Returns:
        "done" if performance acceptable, "tune" if more tuning needed
    """
    metrics = ctx.get("metrics", {})
    iteration = ctx.get("iteration", 0)
    max_iter = ctx.get("max_iterations", 3)

    score = metrics.get("primary_metric", 0)
    threshold = ctx.get("performance_threshold", 0.9)

    if score >= threshold:
        logger.info(f"Performance threshold met: {score:.3f} >= {threshold}")
        return "done"

    if iteration >= max_iter:
        logger.warning(f"Max tuning iterations ({max_iter}) reached")
        return "done"

    return "tune"


def quality_threshold(ctx: Dict[str, Any]) -> str:
    """Multi-level quality assessment for data.

    Evaluates data quality across multiple dimensions.

    Args:
        ctx: Workflow context with keys:
            - quality_score (float): Overall quality score (0-1)
            - missing_pct (float): Percentage of missing values
            - outlier_count (int): Number of detected outliers

    Returns:
        "high_quality", "acceptable", or "needs_cleanup"
    """
    quality_score = ctx.get("quality_score", 0)
    missing_pct = ctx.get("missing_pct", 100)
    outlier_count = ctx.get("outlier_count", 0)

    # High quality: score >= 0.9, < 5% missing, < 10 outliers
    if quality_score >= 0.9 and missing_pct < 5 and outlier_count < 10:
        return "high_quality"

    # Acceptable: score >= 0.7, < 15% missing
    if quality_score >= 0.7 and missing_pct < 15:
        return "acceptable"

    return "needs_cleanup"


def model_selection_criteria(ctx: Dict[str, Any]) -> str:
    """Select best model based on multiple criteria.

    Balances accuracy, speed, and interpretability.

    Args:
        ctx: Workflow context with keys:
            - evaluation_results (list): List of model evaluation dicts
            - prioritize (str): "accuracy", "speed", or "interpretability"

    Returns:
        Model selection recommendation
    """
    results = ctx.get("evaluation_results", [])
    # priority = ctx.get("prioritize", "accuracy")  # For future use

    if not results:
        return "no_models"

    best_score = max(r.get("score", 0) for r in results)

    if best_score >= 0.95:
        return "excellent"
    elif best_score >= 0.85:
        return "good"
    elif best_score >= 0.7:
        return "acceptable"
    else:
        return "needs_improvement"


def analysis_confidence(ctx: Dict[str, Any]) -> str:
    """Assess confidence level of analysis results.

    Used to determine if human review is needed.

    Args:
        ctx: Workflow context with keys:
            - insights (dict): Analysis insights
            - uncertainty_areas (list): Areas with low confidence
            - sample_size (int): Number of data points analyzed

    Returns:
        "high", "medium", or "low" confidence level
    """
    uncertainty_areas = ctx.get("uncertainty_areas", [])
    sample_size = ctx.get("sample_size", 0)
    confidence_score = ctx.get("confidence_score", 0.5)

    # Low confidence if many uncertain areas or small sample
    if len(uncertainty_areas) > 5 or sample_size < 100:
        return "low"

    # High confidence if score > 0.8 and few uncertainties
    if confidence_score > 0.8 and len(uncertainty_areas) <= 2:
        return "high"

    return "medium"


# =============================================================================
# Transform Functions
# =============================================================================


def merge_parallel_stats(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Merge results from parallel statistical computations.

    Combines statistics, correlations, distributions, and anomalies.

    Args:
        ctx: Workflow context with parallel computation results

    Returns:
        Merged analysis results dict
    """
    return {
        "statistics": ctx.get("statistics", {}),
        "correlation_matrix": ctx.get("correlation_matrix", {}),
        "distribution_data": ctx.get("distribution_data", {}),
        "anomalies": ctx.get("anomalies", []),
        "merged": True,
    }


def aggregate_model_results(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate results from multiple model training runs.

    Args:
        ctx: Workflow context with model results

    Returns:
        Aggregated results with best model selection
    """
    models = ["rf_model", "xgb_model", "lgb_model", "nn_model"]
    results = []

    for model_key in models:
        if model_key in ctx:
            results.append(
                {
                    "name": model_key,
                    "metrics": ctx[model_key].get("metrics", {}),
                    "status": ctx[model_key].get("status", "unknown"),
                }
            )

    best_model = max(results, key=lambda x: x["metrics"].get("accuracy", 0)) if results else None

    return {
        "all_models": results,
        "best_model": best_model,
        "best_model_name": best_model["name"] if best_model else None,
        "best_model_score": best_model["metrics"].get("accuracy", 0) if best_model else 0,
    }


# =============================================================================
# Chat Workflow Escape Hatches
# =============================================================================


def analysis_type_check(ctx: Dict[str, Any]) -> str:
    """Determine analysis type for workflow routing.

    Args:
        ctx: Workflow context with keys:
            - analysis_type (str): Type from understand_request
            - user_message (str): Original user message

    Returns:
        "exploratory", "statistical", "visualization", or "ml_insights"
    """
    analysis_type = ctx.get("analysis_type", "").lower()

    if analysis_type in ["exploratory", "explore", "eda"]:
        return "exploratory"

    if analysis_type in ["statistical", "statistics", "test", "hypothesis"]:
        return "statistical"

    if analysis_type in ["visualization", "visualize", "chart", "plot", "graph"]:
        return "visualization"

    if analysis_type in ["ml", "machine learning", "predict", "forecast"]:
        return "ml_insights"

    return "exploratory"


def can_continue_analysis_iteration(ctx: Dict[str, Any]) -> str:
    """Check if workflow can continue with another iteration.

    Args:
        ctx: Workflow context with keys:
            - iteration_count (int): Current iteration count
            - max_iterations (int): Maximum allowed iterations

    Returns:
        "continue" or "max_reached"
    """
    iteration = ctx.get("iteration_count", 0)
    max_iter = ctx.get("max_iterations", 50)

    if iteration < max_iter:
        return "continue"
    return "max_reached"


def has_pending_tool_calls(ctx: Dict[str, Any]) -> str:
    """Check if there are pending tool calls to execute.

    Args:
        ctx: Workflow context with keys:
            - tool_calls (list): Tool calls from agent response
            - pending_tool_calls (list): Pending tool calls

    Returns:
        "has_tools" or "no_tools"
    """
    tool_calls = ctx.get("tool_calls", [])
    pending = ctx.get("pending_tool_calls", [])

    if tool_calls or pending:
        return "has_tools"
    return "no_tools"


def update_analysis_conversation(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Update conversation with tool results from execution.

    Args:
        ctx: Workflow context with tool execution results

    Returns:
        Updated conversation state
    """
    conversation_history = ctx.get("conversation_history", [])
    content = ctx.get("content", "")
    tool_calls = ctx.get("tool_calls", [])
    tool_results = ctx.get("tool_results", [])

    # Add assistant message with tool calls
    if tool_calls:
        conversation_history.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            }
        )

        # Add tool results
        for result in tool_results:
            conversation_history.append(
                {
                    "role": "tool",
                    "content": str(result),
                }
            )

    # Increment iteration count
    iteration_count = ctx.get("iteration_count", 0) + 1

    return {
        "conversation_history": conversation_history,
        "iteration_count": iteration_count,
        "last_content": content,
    }


def finalize_analysis_chat(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Format final data analysis chat response for user.

    Args:
        ctx: Workflow context with all conversation data

    Returns:
        Formatted final response
    """
    content = ctx.get("content", "")
    iteration_count = ctx.get("iteration_count", 0)
    insights = ctx.get("insights", [])
    recommendations = ctx.get("recommendations", [])

    final_response = content
    if not final_response:
        final_response = "Data analysis completed."

    return {
        "final_response": final_response,
        "status": "completed",
        "iterations": iteration_count,
        "analysis_summary": {
            "insights_count": len(insights),
            "recommendations_count": len(recommendations),
        },
        "insights": insights,
        "recommendations": recommendations,
    }


# =============================================================================
# Registry Exports
# =============================================================================

# Conditions available in YAML workflows
CONDITIONS = {
    # Original conditions
    "should_retry_cleaning": should_retry_cleaning,
    "should_tune_more": should_tune_more,
    "quality_threshold": quality_threshold,
    "model_selection_criteria": model_selection_criteria,
    "analysis_confidence": analysis_confidence,
    # Chat workflow conditions
    "analysis_type_check": analysis_type_check,
    "can_continue_analysis_iteration": can_continue_analysis_iteration,
    "has_pending_tool_calls": has_pending_tool_calls,
}

# Transforms available in YAML workflows
TRANSFORMS = {
    # Original transforms
    "merge_parallel_stats": merge_parallel_stats,
    "aggregate_model_results": aggregate_model_results,
    # Chat workflow transforms
    "update_analysis_conversation": update_analysis_conversation,
    "finalize_analysis_chat": finalize_analysis_chat,
}

__all__ = [
    # Original Conditions
    "should_retry_cleaning",
    "should_tune_more",
    "quality_threshold",
    "model_selection_criteria",
    "analysis_confidence",
    # Chat workflow conditions
    "analysis_type_check",
    "can_continue_analysis_iteration",
    "has_pending_tool_calls",
    # Original Transforms
    "merge_parallel_stats",
    "aggregate_model_results",
    # Chat workflow transforms
    "update_analysis_conversation",
    "finalize_analysis_chat",
    # Registries
    "CONDITIONS",
    "TRANSFORMS",
]
