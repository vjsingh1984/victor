"""DevOps Assistant - Complete vertical for infrastructure and deployment.

Competitive positioning: Docker Desktop AI, Terraform Assistant, Pulumi AI, K8s GPT.
"""

from __future__ import annotations

from typing import Any, Dict, List

from victor_sdk import (
    CapabilityIds,
    CapabilityRequirement,
    PromptMetadata,
    StageDefinition,
    ToolNames,
    VerticalBase,
)

from victor.verticals.contrib.devops.prompt_metadata import (
    DEVOPS_GROUNDING_RULES,
    DEVOPS_PROMPT_PRIORITY,
    DEVOPS_PROMPT_TEMPLATES,
    DEVOPS_SYSTEM_PROMPT_SECTION,
    DEVOPS_TASK_TYPE_HINTS,
)


class DevOpsAssistant(VerticalBase):
    """DevOps assistant for infrastructure, deployment, and CI/CD automation.

    Competitive with: Docker Desktop AI, Terraform Assistant, Pulumi AI.
    """

    name = "devops"
    description = "Infrastructure automation, container management, CI/CD, and deployment"
    version = "1.0.0"

    @classmethod
    def get_name(cls) -> str:
        """Return the stable identifier for this vertical."""

        return cls.name

    @classmethod
    def get_description(cls) -> str:
        """Return the human-readable vertical description."""

        return cls.description

    @classmethod
    def get_tools(cls) -> List[str]:
        """Get the list of tools for DevOps tasks.

        Uses SDK-owned canonical tool identifiers.
        """
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
    def get_prompt_templates(cls) -> Dict[str, str]:
        """Return serializable prompt templates for the DevOps definition."""

        return dict(DEVOPS_PROMPT_TEMPLATES)

    @classmethod
    def get_task_type_hints(cls) -> Dict[str, Dict[str, Any]]:
        """Return serializable task-type hints for the DevOps definition."""

        return {task_type: dict(config) for task_type, config in DEVOPS_TASK_TYPE_HINTS.items()}

    @classmethod
    def get_prompt_metadata(cls) -> PromptMetadata:
        """Return full prompt metadata, including runtime adapter hints."""

        metadata = super().get_prompt_metadata()
        return PromptMetadata(
            templates=metadata.templates,
            task_type_hints=metadata.task_type_hints,
            metadata={
                "system_prompt_section": DEVOPS_SYSTEM_PROMPT_SECTION,
                "grounding_rules": DEVOPS_GROUNDING_RULES,
                "priority": DEVOPS_PROMPT_PRIORITY,
            },
        )

    @classmethod
    def get_capability_requirements(cls) -> List[CapabilityRequirement]:
        """Declare runtime capabilities required by the DevOps definition."""

        return [
            CapabilityRequirement(
                capability_id=CapabilityIds.FILE_OPS,
                purpose="Inspect and modify infrastructure configuration files and repository assets.",
            ),
            CapabilityRequirement(
                capability_id=CapabilityIds.SHELL_ACCESS,
                purpose="Run infrastructure, deployment, and validation commands from the shell.",
            ),
            CapabilityRequirement(
                capability_id=CapabilityIds.GIT,
                purpose="Inspect version control history and prepare safe infrastructure changes.",
            ),
            CapabilityRequirement(
                capability_id=CapabilityIds.CONTAINER_RUNTIME,
                purpose="Build, inspect, and validate container images and runtime settings.",
            ),
            CapabilityRequirement(
                capability_id=CapabilityIds.VALIDATION,
                purpose="Validate configurations and run tests before deployment changes are finalized.",
            ),
            CapabilityRequirement(
                capability_id=CapabilityIds.WEB_ACCESS,
                optional=True,
                purpose="Fetch remote documentation and platform references when local sources are insufficient.",
            ),
        ]

    @classmethod
    def get_stages(cls) -> Dict[str, StageDefinition]:
        """Get DevOps-specific stage definitions.

        Uses SDK-owned canonical tool identifiers.
        """
        return {
            "INITIAL": StageDefinition(
                name="INITIAL",
                description="Understanding the infrastructure request",
                optional_tools=[ToolNames.READ, ToolNames.LS, ToolNames.OVERVIEW],
                keywords=["what", "how", "explain", "infrastructure", "setup"],
                next_stages={"ASSESSMENT", "PLANNING"},
            ),
            "ASSESSMENT": StageDefinition(
                name="ASSESSMENT",
                description="Assessing current infrastructure state",
                optional_tools=[ToolNames.READ, ToolNames.SHELL, ToolNames.GREP, ToolNames.GIT],
                keywords=["check", "status", "current", "existing", "review"],
                next_stages={"PLANNING", "IMPLEMENTATION"},
            ),
            "PLANNING": StageDefinition(
                name="PLANNING",
                description="Planning infrastructure changes",
                optional_tools=[
                    ToolNames.READ,
                    ToolNames.GREP,
                    ToolNames.WEB_SEARCH,
                    ToolNames.WEB_FETCH,
                ],
                keywords=["plan", "design", "architecture", "strategy"],
                next_stages={"IMPLEMENTATION"},
            ),
            "IMPLEMENTATION": StageDefinition(
                name="IMPLEMENTATION",
                description="Implementing infrastructure changes",
                optional_tools=[ToolNames.WRITE, ToolNames.EDIT, ToolNames.SHELL, ToolNames.DOCKER],
                keywords=["create", "build", "configure", "implement", "deploy"],
                next_stages={"VALIDATION", "DEPLOYMENT"},
            ),
            "VALIDATION": StageDefinition(
                name="VALIDATION",
                description="Validating configurations and testing",
                optional_tools=[ToolNames.SHELL, ToolNames.READ, ToolNames.TEST],
                keywords=["test", "validate", "verify", "check"],
                next_stages={"DEPLOYMENT", "IMPLEMENTATION"},
            ),
            "DEPLOYMENT": StageDefinition(
                name="DEPLOYMENT",
                description="Deploying to target environment",
                optional_tools=[ToolNames.SHELL, ToolNames.GIT, ToolNames.DOCKER],
                keywords=["deploy", "push", "release", "launch"],
                next_stages={"MONITORING", "COMPLETION"},
            ),
            "MONITORING": StageDefinition(
                name="MONITORING",
                description="Setting up monitoring and observability",
                optional_tools=[ToolNames.WRITE, ToolNames.EDIT, ToolNames.SHELL],
                keywords=["monitor", "observe", "alert", "metrics"],
                next_stages={"COMPLETION"},
            ),
            "COMPLETION": StageDefinition(
                name="COMPLETION",
                description="Documenting and finalizing",
                optional_tools=[ToolNames.WRITE, ToolNames.GIT],
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
