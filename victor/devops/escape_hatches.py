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

"""Escape hatches for DevOps YAML workflows.

Complex conditions and transforms that cannot be expressed in YAML.
These are registered with the YAML workflow loader for use in condition nodes.

Example YAML usage:
    - id: check_deployment
      type: condition
      condition: "deployment_ready"  # References escape hatch
      branches:
        "ready": deploy
        "blocked": wait
        "failed": abort
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# =============================================================================
# Condition Functions
# =============================================================================


def deployment_ready(ctx: Dict[str, Any]) -> str:
    """Check if deployment is ready to proceed.

    Multi-factor check including config, dependencies, and approvals.

    Args:
        ctx: Workflow context with keys:
            - config_valid (bool): Configuration validation status
            - dependencies_met (bool): All dependencies satisfied
            - approval_status (str): Approval state
            - environment (str): Target environment

    Returns:
        "ready", "blocked", or "failed"
    """
    config_valid = ctx.get("config_valid", False)
    dependencies_met = ctx.get("dependencies_met", False)
    approval_status = ctx.get("approval_status", "pending")
    environment = ctx.get("environment", "development")

    if not config_valid:
        logger.warning("Deployment blocked: configuration invalid")
        return "failed"

    if not dependencies_met:
        logger.info("Deployment blocked: waiting for dependencies")
        return "blocked"

    # Production requires explicit approval
    if environment == "production" and approval_status != "approved":
        return "blocked"

    if approval_status == "rejected":
        return "failed"

    return "ready"


def health_check_status(ctx: Dict[str, Any]) -> str:
    """Evaluate health check results.

    Args:
        ctx: Workflow context with keys:
            - health_results (dict): Health check results per endpoint
            - min_healthy_pct (float): Minimum percentage of healthy endpoints

    Returns:
        "healthy", "degraded", or "unhealthy"
    """
    results = ctx.get("health_results", {})
    min_healthy = ctx.get("min_healthy_pct", 0.8)

    if not results:
        return "unhealthy"

    total = len(results)
    healthy = sum(1 for r in results.values() if r.get("status") == "healthy")
    healthy_pct = healthy / total if total > 0 else 0

    if healthy_pct >= 1.0:
        return "healthy"
    elif healthy_pct >= min_healthy:
        return "degraded"
    else:
        return "unhealthy"


def rollback_needed(ctx: Dict[str, Any]) -> str:
    """Determine if rollback is needed based on deployment status.

    Args:
        ctx: Workflow context with keys:
            - deploy_result (dict): Deployment result
            - health_status (str): Health check status
            - error_rate (float): Current error rate

    Returns:
        "rollback", "monitor", or "stable"
    """
    deploy_result = ctx.get("deploy_result", {})
    health_status = ctx.get("health_status", "unknown")
    error_rate = ctx.get("error_rate", 0)

    # Immediate rollback conditions
    if not deploy_result.get("success", False):
        return "rollback"

    if health_status == "unhealthy":
        return "rollback"

    if error_rate > 0.1:  # More than 10% error rate
        return "rollback"

    # Degraded but acceptable - monitor
    if health_status == "degraded" or error_rate > 0.01:
        return "monitor"

    return "stable"


def container_build_status(ctx: Dict[str, Any]) -> str:
    """Check container build status.

    Args:
        ctx: Workflow context with keys:
            - build_result (dict): Build result
            - image_size (int): Image size in MB
            - max_size (int): Maximum allowed size

    Returns:
        "success", "warning", or "failed"
    """
    build_result = ctx.get("build_result", {})
    image_size = ctx.get("image_size", 0)
    max_size = ctx.get("max_size", 2000)  # 2GB default

    if not build_result.get("success", False):
        return "failed"

    if image_size > max_size:
        logger.warning(f"Image size {image_size}MB exceeds limit {max_size}MB")
        return "warning"

    return "success"


def infrastructure_drift(ctx: Dict[str, Any]) -> str:
    """Detect infrastructure drift from desired state.

    Args:
        ctx: Workflow context with keys:
            - plan_changes (dict): Terraform/IaC plan changes
            - auto_remediate (bool): Whether to auto-fix drift

    Returns:
        "no_drift", "minor_drift", "major_drift", or "destructive"
    """
    changes = ctx.get("plan_changes", {})

    if not changes:
        return "no_drift"

    create_count = changes.get("create", 0)
    update_count = changes.get("update", 0)
    destroy_count = changes.get("destroy", 0)

    # Destructive changes require approval
    if destroy_count > 0:
        return "destructive"

    # Major drift: many changes
    if create_count + update_count > 10:
        return "major_drift"

    # Minor drift: few changes
    if create_count + update_count > 0:
        return "minor_drift"

    return "no_drift"


def security_scan_verdict(ctx: Dict[str, Any]) -> str:
    """Evaluate security scan results.

    Args:
        ctx: Workflow context with keys:
            - scan_results (dict): Security scan findings
            - severity_threshold (str): Minimum severity to block

    Returns:
        "pass", "warn", or "fail"
    """
    results = ctx.get("scan_results", {})
    threshold = ctx.get("severity_threshold", "high")

    severity_levels = ["info", "low", "medium", "high", "critical"]
    threshold_idx = severity_levels.index(threshold) if threshold in severity_levels else 3

    critical = results.get("critical", 0)
    high = results.get("high", 0)
    medium = results.get("medium", 0)

    if critical > 0:
        return "fail"

    if high > 0 and threshold_idx <= severity_levels.index("high"):
        return "fail"

    if medium > 0 and threshold_idx <= severity_levels.index("medium"):
        return "warn"

    return "pass"


def pipeline_stage_gate(ctx: Dict[str, Any]) -> str:
    """Evaluate whether to proceed to next pipeline stage.

    Args:
        ctx: Workflow context with keys:
            - stage_results (dict): Current stage results
            - required_coverage (float): Minimum code coverage
            - allow_failures (bool): Allow test failures

    Returns:
        "proceed", "retry", or "abort"
    """
    results = ctx.get("stage_results", {})
    required_coverage = ctx.get("required_coverage", 0.8)
    allow_failures = ctx.get("allow_failures", False)

    tests_passed = results.get("tests_passed", False)
    coverage = results.get("coverage", 0)

    if not tests_passed and not allow_failures:
        return "abort"

    if coverage < required_coverage:
        logger.warning(f"Coverage {coverage:.1%} below required {required_coverage:.1%}")
        return "abort"

    return "proceed"


# =============================================================================
# Transform Functions
# =============================================================================


def merge_deployment_results(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Merge results from parallel deployment tasks.

    Args:
        ctx: Workflow context with parallel deployment results

    Returns:
        Merged deployment summary
    """
    monitoring_result = ctx.get("monitoring_result", {})
    notification_result = ctx.get("notification_result", {})
    docs_result = ctx.get("docs_result", {})

    all_success = all(
        [
            monitoring_result.get("success", False),
            notification_result.get("success", False),
            docs_result.get("success", True),  # Docs update is optional
        ]
    )

    return {
        "all_tasks_complete": True,
        "all_tasks_success": all_success,
        "monitoring_updated": monitoring_result.get("success", False),
        "notification_sent": notification_result.get("success", False),
        "docs_updated": docs_result.get("success", False),
    }


def generate_deployment_summary(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Generate deployment summary for notifications.

    Args:
        ctx: Workflow context with deployment data

    Returns:
        Deployment summary dict
    """
    return {
        "environment": ctx.get("target_env", "unknown"),
        "version": ctx.get("deploy_version", "unknown"),
        "status": ctx.get("status", "unknown"),
        "duration_seconds": ctx.get("duration", 0),
        "rollback_performed": ctx.get("rollback_performed", False),
        "changes": ctx.get("change_summary", "No changes recorded"),
    }


# =============================================================================
# Chat Workflow Escape Hatches (Phase 2)
# =============================================================================


def operation_type_check(ctx: Dict[str, Any]) -> str:
    """Determine operation type for DevOps chat workflow routing.

    Args:
        ctx: Workflow context with keys:
            - operation_type (str): Type from extract_requirements
            - user_message (str): Original user message

    Returns:
        "deployment" or "other"
    """
    operation_type = ctx.get("operation_type", "").lower()

    if operation_type in ["deployment", "deploy", "upgrade", "rollback"]:
        return "deployment"

    return "other"


def deployment_success_check(ctx: Dict[str, Any]) -> str:
    """Check if deployment was successful.

    Args:
        ctx: Workflow context with keys:
            - deployment_status (str): Status from deployment execution
            - error (str): Error message if any

    Returns:
        "success" or "failed"
    """
    deployment_status = ctx.get("deployment_status", "").lower()
    error = ctx.get("error", "")

    if deployment_status == "success" or (not error and deployment_status == "completed"):
        return "success"

    return "failed"


def can_continue_devops_iteration(ctx: Dict[str, Any]) -> str:
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


def update_devops_conversation(ctx: Dict[str, Any]) -> Dict[str, Any]:
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
        conversation_history.append({
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        })

        # Add tool results
        for result in tool_results:
            conversation_history.append({
                "role": "tool",
                "content": str(result),
            })

    # Increment iteration count
    iteration_count = ctx.get("iteration_count", 0) + 1

    return {
        "conversation_history": conversation_history,
        "iteration_count": iteration_count,
        "last_content": content,
    }


def finalize_devops_chat(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Format final DevOps chat response for user.

    Args:
        ctx: Workflow context with all conversation data

    Returns:
        Formatted final response
    """
    content = ctx.get("content", "")
    iteration_count = ctx.get("iteration_count", 0)
    conversation_history = ctx.get("conversation_history", [])

    # Extract deployment details from context
    deployment_target = ctx.get("deployment_target", "unknown")
    environment = ctx.get("environment", "unknown")
    operation_summary = ctx.get("operation_summary", "")

    final_response = content
    if not final_response:
        final_response = "DevOps operation completed."

    # Build deployment details
    deployment_details = {
        "target": deployment_target,
        "environment": environment,
        "summary": operation_summary,
    }

    return {
        "final_response": final_response,
        "status": "completed",
        "iterations": iteration_count,
        "deployment_details": deployment_details,
        "operation_summary": operation_summary,
    }


# =============================================================================
# Registry Exports
# =============================================================================

# Conditions available in YAML workflows
CONDITIONS = {
    # Original conditions
    "deployment_ready": deployment_ready,
    "health_check_status": health_check_status,
    "rollback_needed": rollback_needed,
    "container_build_status": container_build_status,
    "infrastructure_drift": infrastructure_drift,
    "security_scan_verdict": security_scan_verdict,
    "pipeline_stage_gate": pipeline_stage_gate,
    # Chat workflow conditions (Phase 2)
    "operation_type_check": operation_type_check,
    "deployment_success_check": deployment_success_check,
    "can_continue_devops_iteration": can_continue_devops_iteration,
}

# Transforms available in YAML workflows
TRANSFORMS = {
    # Original transforms
    "merge_deployment_results": merge_deployment_results,
    "generate_deployment_summary": generate_deployment_summary,
    # Chat workflow transforms (Phase 2)
    "update_devops_conversation": update_devops_conversation,
    "finalize_devops_chat": finalize_devops_chat,
}

__all__ = [
    # Original Conditions
    "deployment_ready",
    "health_check_status",
    "rollback_needed",
    "container_build_status",
    "infrastructure_drift",
    "security_scan_verdict",
    "pipeline_stage_gate",
    # Chat workflow conditions (Phase 2)
    "operation_type_check",
    "deployment_success_check",
    "can_continue_devops_iteration",
    # Original Transforms
    "merge_deployment_results",
    "generate_deployment_summary",
    # Chat workflow transforms (Phase 2)
    "update_devops_conversation",
    "finalize_devops_chat",
    # Registries
    "CONDITIONS",
    "TRANSFORMS",
]
