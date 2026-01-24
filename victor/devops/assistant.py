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

# Import ISP-compliant provider protocols
from victor.core.verticals.protocols.providers import (
    HandlerProvider,
    MiddlewareProvider,
    ModeConfigProvider,
    PromptContributorProvider,
    SafetyProvider,
    TieredToolConfigProvider,
    ToolDependencyProvider,
    ToolProvider,
)

# Phase 2.1: Protocol auto-registration decorator
from victor.core.verticals.protocol_decorators import register_protocols


@register_protocols
class DevOpsAssistant(VerticalBase):
    """DevOps assistant for infrastructure, deployment, and CI/CD automation.

    Competitive with: Docker Desktop AI, Terraform Assistant, Pulumi AI, K8s GPT.

    ISP Compliance:
        This vertical explicitly declares which protocols it implements through
        protocol registration, rather than inheriting from all possible protocol
        interfaces. This follows the Interface Segregation Principle (ISP) by
        implementing only needed protocols.

        Implemented Protocols:
        - ToolProvider: Provides tools optimized for DevOps tasks
        - PromptContributorProvider: Provides DevOps-specific task hints
        - MiddlewareProvider: Provides git safety, secret masking, and logging middleware
        - ToolDependencyProvider: Provides tool dependency patterns
        - HandlerProvider: Provides workflow compute handlers
        - ModeConfigProvider: Provides mode configurations
        - SafetyProvider: Provides DevOps safety patterns
        - TieredToolConfigProvider: Provides tiered tool configuration
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

    # =========================================================================
    # Extension Protocol Methods
    # =========================================================================
    # Most extension getters are auto-generated by VerticalExtensionLoaderMeta
    # to eliminate ~800 lines of duplication. Only override for custom logic.

    @classmethod
    def get_middleware(cls) -> List[MiddlewareProtocol]:
        """Get DevOps-specific middleware.

        Custom implementation for DevOps vertical with framework-level middleware.
        Auto-generated getter would return empty list.

        Uses framework-level middleware for common functionality:
        - GitSafetyMiddleware: Block dangerous git operations (force push, hard reset)
        - SecretMaskingMiddleware: Mask secrets in tool results
        - LoggingMiddleware: Audit logging for tool calls

        DevOps has stricter git safety since infrastructure changes are critical.

        Returns:
            List of middleware implementations
        """
        from victor.framework.middleware import (
            GitSafetyMiddleware,
            LoggingMiddleware,
            SecretMaskingMiddleware,
        )

        return [
            # Git safety is critical for DevOps - block dangerous operations
            GitSafetyMiddleware(
                block_dangerous=True,  # Strict for infrastructure
                warn_on_risky=True,
                protected_branches={"production", "staging"},  # Additional protected branches
            ),
            # Always mask secrets in infrastructure output
            SecretMaskingMiddleware(
                replacement="[REDACTED]",
                mask_in_arguments=True,  # Also mask secrets in inputs
            ),
            # Audit logging for compliance
            LoggingMiddleware(
                include_arguments=True,
                sanitize_arguments=True,
            ),
        ]

    @classmethod
    def get_tool_dependency_provider(cls) -> Optional[ToolDependencyProviderProtocol]:
        """Get DevOps tool dependency provider (cached).

        Custom implementation using create_vertical_tool_dependency_provider.
        Auto-generated getter would try to import from victor.devops.tool_dependencies.

        Returns:
            Tool dependency provider
        """

        def _create():
            from victor.core.tool_dependency_loader import create_vertical_tool_dependency_provider

            return create_vertical_tool_dependency_provider("devops")

        return cls._get_cached_extension("tool_dependency_provider", _create)

    @classmethod
    def get_handlers(cls) -> Dict[str, Any]:
        """Get compute handlers for DevOps workflows.

        Returns handlers from victor.devops.handlers for workflow execution.
        This replaces the previous import-side-effect registration pattern.

        Returns:
            Dict mapping handler names to handler instances
        """
        from victor.devops.handlers import HANDLERS

        return HANDLERS

    # NOTE: The following getters are auto-generated by VerticalExtensionLoaderMeta:
    # - get_safety_extension()
    # - get_prompt_contributor()
    # - get_mode_config_provider()
    # - get_tiered_tools()
    # - get_workflow_provider()
    # - get_rl_config_provider()
    # - get_rl_hooks()
    # - get_team_spec_provider()
    # - get_capability_provider()
    #
    # get_extensions() is inherited from VerticalBase with full caching support.
    # To clear all caches, use cls.clear_config_cache().


# Protocol registration is now handled by @register_protocols decorator
# which auto-detects implemented protocols:
# - ToolProvider (get_tools)
# - PromptContributorProvider (get_prompt_contributor)
# - MiddlewareProvider (get_middleware)
# - ToolDependencyProvider (get_tool_dependency_provider)
# - HandlerProvider (get_handlers)
# - ModeConfigProvider (get_mode_config_provider)
# - SafetyProvider (get_safety_extension)
# - TieredToolConfigProvider (get_tiered_tools)
#
# ISP Compliance Note:
# This vertical implements only the protocols it needs. The @register_protocols
# decorator auto-detects and registers these protocols at class decoration time.
