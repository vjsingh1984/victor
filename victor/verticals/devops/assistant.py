"""DevOps Assistant - Complete vertical for infrastructure and deployment.

Competitive positioning: Docker Desktop AI, Terraform Assistant, Pulumi AI, K8s GPT.
"""

from typing import List, Optional

from victor.verticals.base import StageDefinition, VerticalBase, VerticalConfig
from victor.verticals.protocols import (
    ModeConfigProviderProtocol,
    PromptContributorProtocol,
    SafetyExtensionProtocol,
    ToolDependencyProviderProtocol,
)


class DevOpsAssistant(VerticalBase):
    """DevOps assistant for infrastructure, deployment, and CI/CD automation.

    Competitive with: Docker Desktop AI, Terraform Assistant, Pulumi AI.
    """

    name = "devops"
    description = "Infrastructure automation, container management, CI/CD, and deployment"
    version = "1.0.0"

    @classmethod
    def get_tools(cls) -> List[str]:
        """Get the list of tools for DevOps tasks."""
        return [
            # Core filesystem
            "read_file",
            "write_file",
            "edit_files",
            "list_directory",
            # Shell for infrastructure commands
            "bash",
            # Git for version control
            "git_status",
            "git_diff",
            "git_commit",
            # Code search for configurations
            "code_search",
            "semantic_code_search",
            "codebase_overview",
            # Web for documentation
            "web_search",
            "web_fetch",
        ]

    @classmethod
    def get_system_prompt(cls) -> str:
        """Get the system prompt for DevOps tasks."""
        return cls._get_system_prompt()

    @classmethod
    def get_config(cls) -> VerticalConfig:
        return VerticalConfig(
            name="devops",
            description="DevOps assistant for infrastructure and deployment automation",
            tools=[
                # Core filesystem
                "read_file",
                "write_file",
                "edit_files",
                "list_directory",
                # Shell for infrastructure commands
                "bash",
                # Git for version control
                "git_status",
                "git_diff",
                "git_commit",
                # Code search for configurations
                "code_search",
                "semantic_code_search",
                "codebase_overview",
                # Web for documentation
                "web_search",
                "web_fetch",
            ],
            stages=cls._get_stages(),
            system_prompt=cls._get_system_prompt(),
            provider_hints={
                "preferred_providers": ["anthropic", "openai"],
                "min_context_window": 100000,
                "features": ["tool_calling", "large_context"],
                "requires_tool_calling": True,
            },
            evaluation_criteria=[
                "configuration_correctness",
                "security_best_practices",
                "idempotency",
                "documentation_completeness",
                "resource_efficiency",
                "disaster_recovery",
                "monitoring_coverage",
            ],
        )

    @classmethod
    def _get_stages(cls) -> List[StageDefinition]:
        return [
            StageDefinition(
                name="INITIAL",
                description="Understanding the infrastructure request",
                allowed_tools=["read_file", "list_directory", "codebase_overview"],
                next_stages=["ASSESSMENT", "PLANNING"],
            ),
            StageDefinition(
                name="ASSESSMENT",
                description="Assessing current infrastructure state",
                allowed_tools=["read_file", "bash", "code_search", "git_status"],
                next_stages=["PLANNING", "IMPLEMENTATION"],
            ),
            StageDefinition(
                name="PLANNING",
                description="Planning infrastructure changes",
                allowed_tools=["read_file", "code_search", "web_search", "web_fetch"],
                next_stages=["IMPLEMENTATION"],
            ),
            StageDefinition(
                name="IMPLEMENTATION",
                description="Implementing infrastructure changes",
                allowed_tools=["write_file", "edit_files", "bash"],
                next_stages=["VALIDATION", "DEPLOYMENT"],
            ),
            StageDefinition(
                name="VALIDATION",
                description="Validating configurations and testing",
                allowed_tools=["bash", "read_file"],
                next_stages=["DEPLOYMENT", "IMPLEMENTATION"],
            ),
            StageDefinition(
                name="DEPLOYMENT",
                description="Deploying to target environment",
                allowed_tools=["bash", "git_commit", "git_status"],
                next_stages=["MONITORING", "COMPLETION"],
            ),
            StageDefinition(
                name="MONITORING",
                description="Setting up monitoring and observability",
                allowed_tools=["write_file", "edit_files", "bash"],
                next_stages=["COMPLETION"],
            ),
            StageDefinition(
                name="COMPLETION",
                description="Documenting and finalizing",
                allowed_tools=["write_file", "git_commit"],
                next_stages=[],
            ),
        ]

    @classmethod
    def _get_system_prompt(cls) -> str:
        return """You are a DevOps engineer assistant specializing in infrastructure automation.

## Core Capabilities

1. **Containerization**: Docker, Docker Compose, container best practices
2. **CI/CD**: GitHub Actions, GitLab CI, Jenkins, CircleCI
3. **Infrastructure as Code**: Terraform, Ansible, CloudFormation, Pulumi
4. **Orchestration**: Kubernetes, Helm, ArgoCD
5. **Monitoring**: Prometheus, Grafana, ELK, Datadog

## Security-First Principles

1. **Never commit secrets**: Use environment variables, secrets managers
2. **Least privilege**: Minimize permissions in all configurations
3. **Encrypted at rest**: Enable encryption for data storage
4. **Network isolation**: Use proper network segmentation

## DevOps Workflow

1. **ASSESS**: Understand current infrastructure state
2. **PLAN**: Design solution with security and scalability in mind
3. **IMPLEMENT**: Write declarative configurations
4. **VALIDATE**: Test in staging before production
5. **DEPLOY**: Use blue-green or canary deployments when possible
6. **MONITOR**: Set up metrics, logs, and alerts

## Configuration Best Practices

- Always use multi-stage builds in Dockerfiles
- Pin versions in all dependencies
- Include health checks in container configs
- Use resource limits in Kubernetes
- Document all configuration decisions
- Keep infrastructure code DRY with modules/templates

## Output Format

When creating configurations:
1. Provide complete, runnable configurations
2. Include inline comments explaining key decisions
3. Note any prerequisites or dependencies
4. Suggest validation commands to verify correctness
"""

    @classmethod
    def get_prompt_contributor(cls) -> Optional[PromptContributorProtocol]:
        from victor.verticals.devops.prompts import DevOpsPromptContributor
        return DevOpsPromptContributor()

    @classmethod
    def get_mode_config_provider(cls) -> Optional[ModeConfigProviderProtocol]:
        from victor.verticals.devops.mode_config import DevOpsModeConfigProvider
        return DevOpsModeConfigProvider()

    @classmethod
    def get_safety_extension(cls) -> Optional[SafetyExtensionProtocol]:
        from victor.verticals.devops.safety import DevOpsSafetyExtension
        return DevOpsSafetyExtension()

    @classmethod
    def get_tool_dependency_provider(cls) -> Optional[ToolDependencyProviderProtocol]:
        from victor.verticals.devops.tool_dependencies import DevOpsToolDependencyProvider
        return DevOpsToolDependencyProvider()
