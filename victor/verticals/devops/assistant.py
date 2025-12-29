"""DevOps Assistant - Complete vertical for infrastructure and deployment.

Competitive positioning: Docker Desktop AI, Terraform Assistant, Pulumi AI, K8s GPT.
"""

from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from victor.verticals.base import StageDefinition, VerticalBase, VerticalConfig
from victor.verticals.protocols import (
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
    def get_config(cls) -> VerticalConfig:
        """Get the complete configuration for DevOps vertical.

        Uses base class implementation with DevOps-specific customizations.
        """
        from victor.framework.tools import ToolSet

        return VerticalConfig(
            tools=ToolSet.from_tools(cls.get_tools()),
            system_prompt=cls._get_system_prompt(),
            stages=cls._get_stages(),
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
            metadata={
                "vertical_name": cls.name,
                "vertical_version": cls.version,
                "description": cls.description,
            },
        )

    @classmethod
    def _get_stages(cls) -> Dict[str, StageDefinition]:
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

    @classmethod
    def get_tiered_tools(cls) -> Optional[TieredToolConfig]:
        """Get tiered tool configuration for DevOps.

        Simplified configuration using consolidated tool metadata:
        - Mandatory: Core tools always included for any task
        - Vertical Core: Essential tools for DevOps tasks
        - semantic_pool: Derived from ToolMetadataRegistry.get_all_tool_names()
        - stage_tools: Derived from @tool(stages=[...]) decorator metadata

        Returns:
            TieredToolConfig for DevOps vertical
        """
        from victor.tools.tool_names import ToolNames

        return TieredToolConfig(
            # Tier 1: Mandatory - always included for any task
            mandatory={
                ToolNames.READ,  # Read files - essential
                ToolNames.LS,  # List directory - essential
                ToolNames.GREP,  # Search code/configs - essential for DevOps
            },
            # Tier 2: Vertical Core - essential for DevOps tasks
            vertical_core={
                ToolNames.SHELL,  # Shell commands - core for infrastructure
                ToolNames.GIT,  # Git operations - core for version control
                ToolNames.DOCKER,  # Docker - core for containers
                ToolNames.OVERVIEW,  # Codebase overview - core for understanding
            },
            # semantic_pool and stage_tools are now derived from @tool decorator metadata
            # Use get_effective_semantic_pool() and get_tools_for_stage_from_registry()
            # DevOps often needs write/execute tools even for analysis queries
            readonly_only_for_analysis=False,
        )

    # =========================================================================
    # New Framework Integrations (Workflows, RL, Teams)
    # =========================================================================

    @classmethod
    def get_workflow_provider(cls) -> Optional[Any]:
        """Get DevOps-specific workflow provider.

        Provides workflows for:
        - deploy_infrastructure: Infrastructure deployment
        - container_setup: Docker container configuration
        - cicd_pipeline: CI/CD pipeline setup

        Returns:
            DevOpsWorkflowProvider instance
        """
        from victor.verticals.devops.workflows import DevOpsWorkflowProvider

        return DevOpsWorkflowProvider()

    @classmethod
    def get_rl_config(cls) -> Optional[Any]:
        """Get RL configuration for DevOps vertical.

        Returns:
            DevOpsRLConfig instance
        """
        from victor.verticals.devops.rl import DevOpsRLConfig

        return DevOpsRLConfig()

    @classmethod
    def get_rl_hooks(cls) -> Optional[Any]:
        """Get RL hooks for DevOps vertical.

        Returns:
            DevOpsRLHooks instance
        """
        from victor.verticals.devops.rl import DevOpsRLHooks

        return DevOpsRLHooks()

    @classmethod
    def get_team_specs(cls) -> Dict[str, Any]:
        """Get team specifications for DevOps tasks.

        Provides pre-configured team specifications for:
        - deployment_team: Infrastructure deployment
        - container_team: Container management
        - monitoring_team: Observability setup

        Returns:
            Dict mapping team names to DevOpsTeamSpec instances
        """
        from victor.verticals.devops.teams import DEVOPS_TEAM_SPECS

        return DEVOPS_TEAM_SPECS
