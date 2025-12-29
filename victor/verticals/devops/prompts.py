"""DevOps Prompt Contributor - Task hints and system prompt extensions for infrastructure."""

from typing import Dict, Optional

from victor.verticals.protocols import PromptContributorProtocol, TaskTypeHint


# DevOps-specific task type hints
# Keys align with TaskTypeClassifier task types (infrastructure, ci_cd)
# Also includes granular hints for specific technologies
DEVOPS_TASK_TYPE_HINTS: Dict[str, TaskTypeHint] = {
    # Classifier task types (matched by TaskTypeClassifier)
    "infrastructure": TaskTypeHint(
        task_type="infrastructure",
        hint="""[INFRASTRUCTURE] Deploy infrastructure (Kubernetes, Terraform, Docker, cloud):
1. Use Infrastructure as Code (Terraform, CloudFormation, Pulumi)
2. Implement multi-stage Docker builds for smaller images
3. Define resource limits and requests for Kubernetes
4. Use ConfigMaps/Secrets for configuration management
5. Tag all resources for cost tracking and organization""",
        tool_budget=15,
        priority_tools=["shell", "read", "write", "edit"],
    ),
    "ci_cd": TaskTypeHint(
        task_type="ci_cd",
        hint="""[CI/CD] Configure continuous integration/deployment:
1. Define clear stages: lint, test, build, deploy
2. Cache dependencies for faster builds
3. Use matrix builds for cross-platform testing
4. Implement proper secret management (GitHub Secrets, Vault)
5. Add manual approval for production deployments""",
        tool_budget=12,
        priority_tools=["read", "write", "edit", "shell"],
    ),
    # Granular hints for specific technologies (context_hints)
    "dockerfile": TaskTypeHint(
        task_type="dockerfile",
        hint="""[DOCKER] Create or optimize Dockerfile:
1. Use official base images with specific tags
2. Implement multi-stage builds for smaller images
3. Order layers for cache optimization
4. Add health checks and proper signal handling
5. Run as non-root user when possible""",
        tool_budget=8,
        priority_tools=["read", "write", "edit", "shell"],
    ),
    "docker_compose": TaskTypeHint(
        task_type="docker_compose",
        hint="""[COMPOSE] Create Docker Compose configuration:
1. Define all services with explicit dependencies
2. Use named volumes for persistent data
3. Configure proper network isolation
4. Add health checks for service readiness
5. Use environment files for secrets (never hardcode)""",
        tool_budget=10,
        priority_tools=["read", "write", "edit", "ls"],
    ),
    # Note: ci_cd_pipeline removed as duplicate - use ci_cd key which aligns with TaskType.CI_CD
    "kubernetes": TaskTypeHint(
        task_type="kubernetes",
        hint="""[K8S] Create Kubernetes manifests:
1. Use Deployments for stateless apps, StatefulSets for stateful
2. Define resource requests and limits
3. Add liveness and readiness probes
4. Use ConfigMaps for config, Secrets for sensitive data
5. Implement NetworkPolicies for security""",
        tool_budget=12,
        priority_tools=["read", "write", "edit", "shell"],
    ),
    "terraform": TaskTypeHint(
        task_type="terraform",
        hint="""[TERRAFORM] Write Infrastructure as Code:
1. Organize into modules for reusability
2. Use remote state with locking
3. Implement proper variable typing and validation
4. Tag all resources for cost tracking
5. Use data sources instead of hardcoded IDs""",
        tool_budget=15,
        priority_tools=["read", "write", "edit", "ls", "shell"],
    ),
    "monitoring": TaskTypeHint(
        task_type="monitoring",
        hint="""[MONITOR] Set up observability:
1. Define key metrics and SLIs/SLOs
2. Configure alerting with appropriate thresholds
3. Set up distributed tracing for microservices
4. Implement structured logging
5. Create dashboards for visibility""",
        tool_budget=12,
        priority_tools=["read", "write", "edit", "shell"],
    ),
    # Default fallback for 'general' task type
    "general": TaskTypeHint(
        task_type="general",
        hint="""[GENERAL DEVOPS] For general infrastructure queries:
1. Read existing configuration files first (ls, read)
2. Check for Dockerfiles, compose files, k8s manifests
3. Use shell for quick inspections
4. Follow security best practices
5. Document any changes made""",
        tool_budget=10,
        priority_tools=["read", "ls", "shell", "grep"],
    ),
}


class DevOpsPromptContributor(PromptContributorProtocol):
    """Contributes DevOps-specific prompts and task hints."""

    def get_task_type_hints(self) -> Dict[str, TaskTypeHint]:
        """Return DevOps-specific task type hints.

        Returns:
            Dict mapping task types to TaskTypeHint objects
        """
        return DEVOPS_TASK_TYPE_HINTS.copy()

    def get_system_prompt_section(self) -> str:
        """Return additional system prompt content for DevOps context.

        Returns:
            System prompt section for DevOps tasks
        """
        return """
## Security Checklist

Before finalizing any infrastructure configuration:
- [ ] No hardcoded secrets, passwords, or API keys
- [ ] Using least-privilege IAM/RBAC policies
- [ ] Network traffic encrypted in transit
- [ ] Data encrypted at rest
- [ ] Container running as non-root
- [ ] Resource limits defined
- [ ] Logging and audit trails enabled

## Common Pitfalls to Avoid

1. **Docker**: Using `latest` tag, running as root, missing health checks
2. **Kubernetes**: No resource limits, missing probes, using default namespace
3. **Terraform**: Local state, no locking, hardcoded values
4. **CI/CD**: Secrets in logs, no artifact versioning, missing rollback
5. **Monitoring**: Alert fatigue, missing business metrics, no runbooks
""".strip()

    def get_grounding_rules(self) -> str:
        """Get DevOps-specific grounding rules.

        Returns:
            Grounding rules text for DevOps tasks
        """
        return """GROUNDING: Base ALL responses on tool output only. Never invent file paths or content.
Verify configuration syntax before suggesting. Always check existing resources first.""".strip()

    def get_priority(self) -> int:
        """Get priority for prompt section ordering.

        Returns:
            Priority value (DevOps is specialized, so medium priority)
        """
        return 5

    def get_context_hints(self, task_type: Optional[str] = None) -> Optional[str]:
        """Return contextual hints based on detected task type."""
        if task_type and task_type in DEVOPS_TASK_TYPE_HINTS:
            return DEVOPS_TASK_TYPE_HINTS[task_type].hint
        return None
