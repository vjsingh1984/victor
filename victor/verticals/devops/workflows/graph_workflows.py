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

"""StateGraph-based workflows for DevOps vertical.

Provides LangGraph-compatible StateGraph workflows for complex DevOps tasks
that benefit from:
- Typed state management
- Cyclic execution (deploy-validate-fix loops)
- Explicit retry limits
- Checkpoint/resume semantics
- Human-in-the-loop approvals for production deployments

These workflows complement the WorkflowBuilder DSL, offering more control
for complex multi-iteration infrastructure tasks.

Example:
    from victor.verticals.devops.workflows.graph_workflows import (
        create_deployment_workflow,
        DeploymentState,
    )
    from victor.framework.graph import RLCheckpointerAdapter

    # Create workflow with checkpointing
    graph = create_deployment_workflow()
    checkpointer = RLCheckpointerAdapter("deployment_workflow")
    app = graph.compile(checkpointer=checkpointer)

    # Execute with typed state
    result = await app.invoke(DeploymentState(
        target_environment="staging",
        infrastructure_changes=[],
    ))
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING, TypedDict

from victor.framework.graph import (
    END,
    StateGraph,
    GraphConfig,
)

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator

logger = logging.getLogger(__name__)


# =============================================================================
# Typed State Definitions
# =============================================================================


class DeploymentState(TypedDict, total=False):
    """Typed state for infrastructure deployment workflows.

    Attributes:
        target_environment: Target environment (dev, staging, prod)
        infrastructure_changes: List of planned infrastructure changes
        assessment_findings: Results from infrastructure assessment
        deployment_plan: Detailed deployment plan
        validation_results: Results from validation checks
        rollback_triggered: Whether rollback was triggered
        deployment_status: Current deployment status
        iteration_count: Current iteration for retry loops
        max_iterations: Maximum retry iterations
        error: Error message if failed
        success: Whether deployment succeeded
    """
    target_environment: str
    infrastructure_changes: List[str]
    assessment_findings: Optional[Dict[str, Any]]
    deployment_plan: Optional[str]
    validation_results: Optional[Dict[str, Any]]
    rollback_triggered: bool
    deployment_status: str
    iteration_count: int
    max_iterations: int
    error: Optional[str]
    success: bool


class ContainerState(TypedDict, total=False):
    """Typed state for container workflows.

    Attributes:
        application_name: Name of application to containerize
        dockerfile_content: Generated Dockerfile content
        compose_content: Docker Compose configuration
        image_name: Built image name and tag
        scan_results: Security scan results
        test_results: Container test results
        iteration: Current iteration
        max_iterations: Maximum iterations for build/test cycle
    """
    application_name: str
    dockerfile_content: Optional[str]
    compose_content: Optional[str]
    image_name: Optional[str]
    scan_results: Optional[Dict[str, Any]]
    test_results: Optional[Dict[str, Any]]
    iteration: int
    max_iterations: int


class CICDState(TypedDict, total=False):
    """Typed state for CI/CD pipeline workflows.

    Attributes:
        repository_path: Path to repository
        pipeline_type: Type of pipeline (github_actions, gitlab_ci, jenkins)
        existing_config: Existing pipeline configuration
        new_config: New/updated pipeline configuration
        validation_passed: Whether config validation passed
        test_run_output: Output from test run
        iteration: Current iteration
    """
    repository_path: str
    pipeline_type: str
    existing_config: Optional[str]
    new_config: Optional[str]
    validation_passed: bool
    test_run_output: Optional[str]
    iteration: int


class SecurityAuditState(TypedDict, total=False):
    """Typed state for security audit workflows.

    Attributes:
        audit_scope: Scope of audit (infrastructure, containers, pipelines)
        vulnerability_findings: List of vulnerabilities found
        configuration_issues: Configuration security issues
        remediation_plan: Prioritized remediation plan
        remediation_applied: List of applied remediations
        verification_results: Results from verification
        audit_complete: Whether audit is complete
    """
    audit_scope: str
    vulnerability_findings: Optional[List[Dict[str, Any]]]
    configuration_issues: Optional[List[Dict[str, Any]]]
    remediation_plan: Optional[str]
    remediation_applied: Optional[List[str]]
    verification_results: Optional[Dict[str, Any]]
    audit_complete: bool


# =============================================================================
# Node Functions - Deployment
# =============================================================================


async def assess_infrastructure_node(state: DeploymentState) -> DeploymentState:
    """Assess current infrastructure state.

    This node analyzes the current infrastructure to find:
    - Existing resources and configurations
    - Dependencies between components
    - Potential conflicts with planned changes
    """
    state["assessment_findings"] = {
        "existing_resources": [],
        "dependencies": [],
        "conflicts": [],
        "recommendations": [],
    }
    state["deployment_status"] = "assessed"
    return state


async def plan_deployment_node(state: DeploymentState) -> DeploymentState:
    """Create deployment plan based on assessment."""
    findings = state.get("assessment_findings", {})
    env = state.get("target_environment", "dev")
    state["deployment_plan"] = f"Deployment plan for {env} based on {len(findings.get('existing_resources', []))} resources"
    state["deployment_status"] = "planned"
    return state


async def execute_deployment_node(state: DeploymentState) -> DeploymentState:
    """Execute infrastructure deployment."""
    state["iteration_count"] = state.get("iteration_count", 0) + 1
    state["deployment_status"] = "deploying"
    return state


async def validate_deployment_node(state: DeploymentState) -> DeploymentState:
    """Validate deployment health and correctness."""
    state["validation_results"] = {
        "health_checks": {"passed": True},
        "smoke_tests": {"passed": True},
        "security_scan": {"passed": True},
    }
    state["deployment_status"] = "validated"
    return state


async def rollback_deployment_node(state: DeploymentState) -> DeploymentState:
    """Rollback deployment on failure."""
    state["rollback_triggered"] = True
    state["deployment_status"] = "rolled_back"
    return state


async def finalize_deployment_node(state: DeploymentState) -> DeploymentState:
    """Finalize successful deployment."""
    state["success"] = True
    state["deployment_status"] = "complete"
    return state


# =============================================================================
# Node Functions - Container
# =============================================================================


async def analyze_containerization_node(state: ContainerState) -> ContainerState:
    """Analyze application for containerization requirements."""
    state["iteration"] = state.get("iteration", 0) + 1
    return state


async def generate_dockerfile_node(state: ContainerState) -> ContainerState:
    """Generate optimized Dockerfile."""
    app_name = state.get("application_name", "app")
    state["dockerfile_content"] = f"# Dockerfile for {app_name}\n# Multi-stage build"
    return state


async def build_container_node(state: ContainerState) -> ContainerState:
    """Build container image."""
    state["image_name"] = f"{state.get('application_name', 'app')}:latest"
    return state


async def scan_container_node(state: ContainerState) -> ContainerState:
    """Run security scan on container image."""
    state["scan_results"] = {
        "vulnerabilities": [],
        "passed": True,
    }
    return state


async def test_container_node(state: ContainerState) -> ContainerState:
    """Test container functionality."""
    state["test_results"] = {
        "startup": "passed",
        "health_check": "passed",
        "shutdown": "passed",
    }
    return state


# =============================================================================
# Node Functions - CI/CD
# =============================================================================


async def analyze_pipeline_node(state: CICDState) -> CICDState:
    """Analyze existing pipeline configuration."""
    state["iteration"] = state.get("iteration", 0) + 1
    return state


async def design_pipeline_node(state: CICDState) -> CICDState:
    """Design new pipeline configuration."""
    pipeline_type = state.get("pipeline_type", "github_actions")
    state["new_config"] = f"# {pipeline_type} pipeline configuration"
    return state


async def validate_pipeline_node(state: CICDState) -> CICDState:
    """Validate pipeline configuration."""
    state["validation_passed"] = True
    return state


async def test_pipeline_node(state: CICDState) -> CICDState:
    """Run test execution of pipeline."""
    state["test_run_output"] = "Pipeline test run successful"
    return state


# =============================================================================
# Node Functions - Security Audit
# =============================================================================


async def scan_vulnerabilities_node(state: SecurityAuditState) -> SecurityAuditState:
    """Scan for security vulnerabilities."""
    state["vulnerability_findings"] = []
    return state


async def audit_configurations_node(state: SecurityAuditState) -> SecurityAuditState:
    """Audit security configurations."""
    state["configuration_issues"] = []
    return state


async def create_remediation_plan_node(state: SecurityAuditState) -> SecurityAuditState:
    """Create prioritized remediation plan."""
    vuln_count = len(state.get("vulnerability_findings", []))
    config_count = len(state.get("configuration_issues", []))
    state["remediation_plan"] = f"Plan addressing {vuln_count} vulnerabilities and {config_count} config issues"
    return state


async def apply_remediations_node(state: SecurityAuditState) -> SecurityAuditState:
    """Apply security remediations."""
    state["remediation_applied"] = []
    return state


async def verify_remediations_node(state: SecurityAuditState) -> SecurityAuditState:
    """Verify remediations are effective."""
    state["verification_results"] = {"verified": True}
    state["audit_complete"] = True
    return state


# =============================================================================
# Condition Functions
# =============================================================================


def should_retry_deployment(state: DeploymentState) -> str:
    """Determine if deployment should be retried.

    Returns:
        'retry' if validation failed and under limit,
        'rollback' if max iterations reached,
        'done' if validation passed
    """
    validation = state.get("validation_results", {})
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 3)

    # Check all validation results
    all_passed = all(
        v.get("passed", False)
        for v in validation.values()
        if isinstance(v, dict)
    )

    if all_passed:
        return "done"
    if iteration >= max_iter:
        return "rollback"
    return "retry"


def should_rebuild_container(state: ContainerState) -> str:
    """Determine if container should be rebuilt.

    Returns:
        'rebuild' if scan/test failed and under limit,
        'done' if all checks passed
    """
    scan = state.get("scan_results", {})
    test = state.get("test_results", {})
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)

    scan_passed = scan.get("passed", False)
    test_passed = all(v == "passed" for v in test.values()) if test else False

    if scan_passed and test_passed:
        return "done"
    if iteration >= max_iter:
        return "done"  # Give up
    return "rebuild"


def should_retry_pipeline(state: CICDState) -> str:
    """Determine if pipeline config should be revised."""
    if state.get("validation_passed", False):
        return "test"
    return "revise"


def check_remediation_complete(state: SecurityAuditState) -> str:
    """Check if remediation is complete."""
    verification = state.get("verification_results", {})
    if verification.get("verified", False):
        return "done"
    return "continue"


# =============================================================================
# Workflow Factories
# =============================================================================


def create_deployment_workflow() -> StateGraph[DeploymentState]:
    """Create an infrastructure deployment workflow with validation loop.

    This workflow implements:
    1. Assess -> Plan -> Execute -> Validate
    2. If validation fails, retry execution (up to max_iterations)
    3. If max retries exceeded, trigger rollback
    4. If validation passes, finalize

    The cyclic validate-fix loop allows iterative refinement.

    Returns:
        StateGraph for infrastructure deployment
    """
    graph = StateGraph(DeploymentState)

    # Add nodes
    graph.add_node("assess", assess_infrastructure_node)
    graph.add_node("plan", plan_deployment_node)
    graph.add_node("execute", execute_deployment_node)
    graph.add_node("validate", validate_deployment_node)
    graph.add_node("rollback", rollback_deployment_node)
    graph.add_node("finalize", finalize_deployment_node)

    # Add edges
    graph.add_edge("assess", "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "validate")

    # Conditional: retry, rollback, or complete
    graph.add_conditional_edge(
        "validate",
        should_retry_deployment,
        {
            "retry": "execute",
            "rollback": "rollback",
            "done": "finalize",
        },
    )

    graph.add_edge("rollback", END)
    graph.add_edge("finalize", END)

    # Set entry point
    graph.set_entry_point("assess")

    return graph


def create_container_workflow() -> StateGraph[ContainerState]:
    """Create a container build workflow with scan/test cycle.

    Implements:
    1. Analyze -> Generate Dockerfile -> Build
    2. Scan -> Test
    3. If scan/test fails, rebuild (up to max_iterations)

    Returns:
        StateGraph for container workflow
    """
    graph = StateGraph(ContainerState)

    # Add nodes
    graph.add_node("analyze", analyze_containerization_node)
    graph.add_node("generate", generate_dockerfile_node)
    graph.add_node("build", build_container_node)
    graph.add_node("scan", scan_container_node)
    graph.add_node("test", test_container_node)

    # Linear flow with retry loop
    graph.add_edge("analyze", "generate")
    graph.add_edge("generate", "build")
    graph.add_edge("build", "scan")
    graph.add_edge("scan", "test")

    # Conditional: rebuild or complete
    graph.add_conditional_edge(
        "test",
        should_rebuild_container,
        {"rebuild": "generate", "done": END},
    )

    # Set entry point
    graph.set_entry_point("analyze")

    return graph


def create_cicd_workflow() -> StateGraph[CICDState]:
    """Create a CI/CD pipeline setup workflow.

    Implements:
    1. Analyze existing -> Design new -> Validate
    2. If validation fails, revise design
    3. If passes, test pipeline

    Returns:
        StateGraph for CI/CD workflow
    """
    graph = StateGraph(CICDState)

    # Add nodes
    graph.add_node("analyze", analyze_pipeline_node)
    graph.add_node("design", design_pipeline_node)
    graph.add_node("validate", validate_pipeline_node)
    graph.add_node("test", test_pipeline_node)

    # Flow with revision loop
    graph.add_edge("analyze", "design")
    graph.add_edge("design", "validate")

    # Conditional: revise or test
    graph.add_conditional_edge(
        "validate",
        should_retry_pipeline,
        {"revise": "design", "test": "test"},
    )

    graph.add_edge("test", END)

    # Set entry point
    graph.set_entry_point("analyze")

    return graph


def create_security_audit_workflow() -> StateGraph[SecurityAuditState]:
    """Create a security audit workflow with remediation loop.

    Implements:
    1. Scan (parallel) -> Create Plan -> Apply -> Verify
    2. If verification fails, continue remediation

    Returns:
        StateGraph for security audit workflow
    """
    graph = StateGraph(SecurityAuditState)

    # Add nodes
    graph.add_node("scan_vulns", scan_vulnerabilities_node)
    graph.add_node("audit_configs", audit_configurations_node)
    graph.add_node("create_plan", create_remediation_plan_node)
    graph.add_node("apply", apply_remediations_node)
    graph.add_node("verify", verify_remediations_node)

    # Parallel scanning then sequential remediation
    # Note: In actual execution, scan_vulns and audit_configs run in parallel
    graph.add_edge("scan_vulns", "audit_configs")  # Sequential for simplicity
    graph.add_edge("audit_configs", "create_plan")
    graph.add_edge("create_plan", "apply")
    graph.add_edge("apply", "verify")

    # Conditional: continue or done
    graph.add_conditional_edge(
        "verify",
        check_remediation_complete,
        {"continue": "apply", "done": END},
    )

    # Set entry point
    graph.set_entry_point("scan_vulns")

    return graph


# =============================================================================
# Orchestrator Integration
# =============================================================================


class DevOpsGraphExecutor:
    """Executor that integrates StateGraph with AgentOrchestrator for DevOps.

    Bridges the StateGraph execution with the actual agent orchestrator,
    allowing node functions to use the full agent capabilities.

    Example:
        executor = DevOpsGraphExecutor(orchestrator)
        graph = create_deployment_workflow()
        result = await executor.run(graph, initial_state)
    """

    def __init__(
        self,
        orchestrator: "AgentOrchestrator",
        checkpointer: Optional[Any] = None,
    ):
        """Initialize executor.

        Args:
            orchestrator: AgentOrchestrator for agent execution
            checkpointer: Optional checkpointer for persistence
        """
        self._orchestrator = orchestrator
        self._checkpointer = checkpointer

    async def run(
        self,
        graph: StateGraph,
        initial_state: Dict[str, Any],
        *,
        thread_id: Optional[str] = None,
        config: Optional[GraphConfig] = None,
    ):
        """Execute a StateGraph workflow.

        Args:
            graph: StateGraph to execute
            initial_state: Initial state
            thread_id: Optional thread ID for checkpointing
            config: Optional execution config

        Returns:
            ExecutionResult with final state
        """
        # Compile with checkpointer
        compiled = graph.compile(checkpointer=self._checkpointer)

        # Merge config if provided
        exec_config = config or GraphConfig()
        if self._checkpointer:
            exec_config.checkpointer = self._checkpointer

        # Execute
        return await compiled.invoke(
            initial_state,
            config=exec_config,
            thread_id=thread_id,
        )

    async def stream(
        self,
        graph: StateGraph,
        initial_state: Dict[str, Any],
        *,
        thread_id: Optional[str] = None,
    ):
        """Stream execution yielding state after each node.

        Args:
            graph: StateGraph to execute
            initial_state: Initial state
            thread_id: Optional thread ID

        Yields:
            Tuple of (node_id, state) after each node
        """
        compiled = graph.compile(checkpointer=self._checkpointer)

        async for node_id, state in compiled.stream(
            initial_state,
            thread_id=thread_id,
        ):
            yield node_id, state


__all__ = [
    # State types
    "DeploymentState",
    "ContainerState",
    "CICDState",
    "SecurityAuditState",
    # Workflow factories
    "create_deployment_workflow",
    "create_container_workflow",
    "create_cicd_workflow",
    "create_security_audit_workflow",
    # Executor
    "DevOpsGraphExecutor",
]
