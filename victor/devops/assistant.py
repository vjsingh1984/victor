"""DevOps Assistant - Complete vertical for infrastructure and deployment.

Competitive positioning: Docker Desktop AI, Terraform Assistant, Pulumi AI, K8s GPT.
"""

from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from victor.core.verticals.base import StageDefinition, VerticalBase
from victor.core.verticals.protocols import (
    MiddlewareProtocol,
    ModeConfigProviderProtocol,
    PromptContributorProtocol,
    SafetyExtensionProtocol,
    TieredToolConfig,
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
        """Get the list of tools for DevOps tasks.

        Uses canonical tool names from victor.tools.tool_names.
        """
        from victor.tools.tool_names import ToolNames

        return [
            # Core filesystem
            ToolNames.READ,  # read_file → read
            ToolNames.WRITE,  # write_file → write
            ToolNames.EDIT,  # edit_files → edit
            ToolNames.LS,  # list_directory → ls
            # Shell for infrastructure commands
            ToolNames.SHELL,  # bash → shell
            # Git for version control
            ToolNames.GIT,  # Git operations
            # Container and infrastructure tools
            ToolNames.DOCKER,  # Docker operations - essential for DevOps
            ToolNames.TEST,  # Run tests - essential for validation
            # Code search for configurations
            ToolNames.GREP,  # Keyword search
            ToolNames.CODE_SEARCH,  # Semantic code search
            ToolNames.OVERVIEW,  # codebase_overview → overview
            # Web for documentation
            ToolNames.WEB_SEARCH,  # Web search (internet search)
            ToolNames.WEB_FETCH,  # Fetch URL content
        ]

    @classmethod
    def get_system_prompt(cls) -> str:
        """Get the system prompt for DevOps tasks."""
        return cls._get_system_prompt()

    @classmethod
    def get_stages(cls) -> Dict[str, StageDefinition]:
        """Get DevOps-specific stage definitions.

        Uses canonical tool names from victor.tools.tool_names.
        """
        from victor.tools.tool_names import ToolNames

        return {
            "INITIAL": StageDefinition(
                name="INITIAL",
                description="Understanding the infrastructure request",
                tools={ToolNames.READ, ToolNames.LS, ToolNames.OVERVIEW},
                keywords=["what", "how", "explain", "infrastructure", "setup"],
                next_stages={"ASSESSMENT", "PLANNING"},
            ),
            "ASSESSMENT": StageDefinition(
                name="ASSESSMENT",
                description="Assessing current infrastructure state",
                tools={ToolNames.READ, ToolNames.SHELL, ToolNames.GREP, ToolNames.GIT},
                keywords=["check", "status", "current", "existing", "review"],
                next_stages={"PLANNING", "IMPLEMENTATION"},
            ),
            "PLANNING": StageDefinition(
                name="PLANNING",
                description="Planning infrastructure changes",
                tools={ToolNames.READ, ToolNames.GREP, ToolNames.WEB_SEARCH, ToolNames.WEB_FETCH},
                keywords=["plan", "design", "architecture", "strategy"],
                next_stages={"IMPLEMENTATION"},
            ),
            "IMPLEMENTATION": StageDefinition(
                name="IMPLEMENTATION",
                description="Implementing infrastructure changes",
                tools={ToolNames.WRITE, ToolNames.EDIT, ToolNames.SHELL, ToolNames.DOCKER},
                keywords=["create", "build", "configure", "implement", "deploy"],
                next_stages={"VALIDATION", "DEPLOYMENT"},
            ),
            "VALIDATION": StageDefinition(
                name="VALIDATION",
                description="Validating configurations and testing",
                tools={ToolNames.SHELL, ToolNames.READ, ToolNames.TEST},
                keywords=["test", "validate", "verify", "check"],
                next_stages={"DEPLOYMENT", "IMPLEMENTATION"},
            ),
            "DEPLOYMENT": StageDefinition(
                name="DEPLOYMENT",
                description="Deploying to target environment",
                tools={ToolNames.SHELL, ToolNames.GIT, ToolNames.DOCKER},
                keywords=["deploy", "push", "release", "launch"],
                next_stages={"MONITORING", "COMPLETION"},
            ),
            "MONITORING": StageDefinition(
                name="MONITORING",
                description="Setting up monitoring and observability",
                tools={ToolNames.WRITE, ToolNames.EDIT, ToolNames.SHELL},
                keywords=["monitor", "observe", "alert", "metrics"],
                next_stages={"COMPLETION"},
            ),
            "COMPLETION": StageDefinition(
                name="COMPLETION",
                description="Documenting and finalizing",
                tools={ToolNames.WRITE, ToolNames.GIT},
                keywords=["done", "complete", "document", "finish"],
                next_stages=set(),
            ),
        }

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
    def get_middleware(cls) -> List[MiddlewareProtocol]:
        """Get DevOps-specific middleware.

        Uses MiddlewareComposer for consistent middleware composition:
        - GitSafetyMiddleware: Block dangerous git operations (strict for infrastructure)
        - SecretMaskingMiddleware: Mask secrets in tool results
        - LoggingMiddleware: Audit logging for tool calls

        DevOps has stricter git safety since infrastructure changes are critical.

        Returns:
            List of middleware implementations
        """
        from victor.framework.middleware import MiddlewareComposer

        return (
            MiddlewareComposer()
            .git_safety(
                block_dangerous=True,  # Strict for infrastructure
                warn_on_risky=True,
                protected_branches={"production", "staging"},  # Additional protected branches
            )
            .secret_masking(
                replacement="[REDACTED]",
                mask_in_arguments=True,  # Also mask secrets in inputs
            )
            .logging(
                include_arguments=True,
                include_results=True,
            )
            .build()
        )

    # =========================================================================
    # New Framework Integrations (Workflows, RL, Teams)
    # =========================================================================