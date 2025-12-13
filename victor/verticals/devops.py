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

"""DevOpsAssistant - Vertical for infrastructure and deployment tasks.

This vertical is optimized for:
- Docker and container management
- CI/CD pipeline configuration
- Infrastructure as Code (IaC)
- Deployment automation
- Monitoring and observability setup

Example:
    from victor.verticals import DevOpsAssistant

    config = DevOpsAssistant.get_config()
    agent = await Agent.create(tools=config.tools)
    result = await agent.run("Set up a CI/CD pipeline for this project")
"""

from __future__ import annotations

from typing import Any, Dict, List

from victor.verticals.base import StageDefinition, VerticalBase, VerticalConfig


class DevOpsAssistant(VerticalBase):
    """DevOps and infrastructure assistant vertical.

    Optimized for:
    - Docker container management
    - Kubernetes deployments
    - CI/CD pipeline configuration
    - Infrastructure as Code (Terraform, Ansible)
    - Monitoring and logging setup

    Example:
        from victor.verticals import DevOpsAssistant

        agent = await DevOpsAssistant.create_agent()
        result = await agent.run("Create a Dockerfile for this Python project")
    """

    name = "devops"
    description = "DevOps assistant for infrastructure, deployment, and CI/CD automation"
    version = "1.0.0"

    @classmethod
    def get_tools(cls) -> List[str]:
        """Get tools optimized for DevOps tasks.

        Returns:
            List of tool names for infrastructure and deployment.
        """
        return [
            # Core filesystem
            "read",
            "write",
            "edit",
            "ls",
            "overview",
            # Shell for infrastructure commands
            "shell",
            "bash",
            # Docker
            "docker",
            "docker_compose",
            # Git for version control
            "git",
            "git_status",
            "git_diff",
            "git_commit",
            # Search for configuration
            "search",
            "code_search",
            # Web for documentation
            "web_search",
            "web_fetch",
        ]

    @classmethod
    def get_system_prompt(cls) -> str:
        """Get DevOps-focused system prompt.

        Returns:
            System prompt optimized for infrastructure tasks.
        """
        return """You are a DevOps engineer assistant specializing in infrastructure automation.

Your capabilities:
- Docker container creation and management
- CI/CD pipeline configuration (GitHub Actions, GitLab CI, Jenkins)
- Infrastructure as Code (Terraform, Ansible, CloudFormation)
- Kubernetes deployment and configuration
- Monitoring and observability setup (Prometheus, Grafana, ELK)
- Security best practices and hardening

Guidelines:
1. **Security First**: Always follow security best practices
   - Never commit secrets or credentials
   - Use environment variables for sensitive data
   - Follow principle of least privilege
2. **Infrastructure as Code**: Prefer declarative configurations
3. **Idempotency**: Ensure operations are idempotent when possible
4. **Documentation**: Comment complex configurations
5. **Testing**: Include health checks and validation steps

DevOps Workflow:
1. ASSESSMENT: Understand current infrastructure
2. PLANNING: Design the solution architecture
3. IMPLEMENTATION: Write configurations and scripts
4. VALIDATION: Test deployments in staging
5. DEPLOYMENT: Roll out to production
6. MONITORING: Set up observability

Common Tasks:
- Dockerizing applications: Create optimized Dockerfiles
- CI/CD: Configure automated testing and deployment
- Container orchestration: Write Kubernetes manifests
- Infrastructure: Create Terraform/Ansible configurations
- Monitoring: Set up metrics, logs, and alerts

You have access to Docker, shell, and file tools. Use them to help automate infrastructure tasks."""

    @classmethod
    def get_stages(cls) -> Dict[str, StageDefinition]:
        """Get DevOps-specific stage definitions.

        Returns:
            Stage definitions optimized for infrastructure workflows.
        """
        return {
            "INITIAL": StageDefinition(
                name="INITIAL",
                description="Understanding the infrastructure request",
                tools={"read", "ls", "overview", "search"},
                keywords=["what", "how", "explain", "infrastructure", "deploy"],
                next_stages={"ASSESSMENT", "PLANNING"},
            ),
            "ASSESSMENT": StageDefinition(
                name="ASSESSMENT",
                description="Assessing current infrastructure state",
                tools={"read", "docker", "shell", "search"},
                keywords=["check", "status", "current", "existing", "assess"],
                next_stages={"PLANNING", "IMPLEMENTATION"},
            ),
            "PLANNING": StageDefinition(
                name="PLANNING",
                description="Planning the infrastructure changes",
                tools={"read", "search", "web_search"},
                keywords=["plan", "design", "architecture", "approach"],
                next_stages={"IMPLEMENTATION"},
            ),
            "IMPLEMENTATION": StageDefinition(
                name="IMPLEMENTATION",
                description="Implementing infrastructure changes",
                tools={"write", "edit", "shell", "docker", "docker_compose"},
                keywords=[
                    "create",
                    "write",
                    "configure",
                    "implement",
                    "build",
                    "setup",
                ],
                next_stages={"VALIDATION", "DEPLOYMENT"},
            ),
            "VALIDATION": StageDefinition(
                name="VALIDATION",
                description="Validating configurations and testing",
                tools={"shell", "docker", "read"},
                keywords=["test", "validate", "verify", "check", "lint"],
                next_stages={"DEPLOYMENT", "IMPLEMENTATION"},
            ),
            "DEPLOYMENT": StageDefinition(
                name="DEPLOYMENT",
                description="Deploying to target environment",
                tools={"shell", "docker", "docker_compose", "git"},
                keywords=["deploy", "push", "release", "rollout", "apply"],
                next_stages={"MONITORING", "COMPLETION"},
            ),
            "MONITORING": StageDefinition(
                name="MONITORING",
                description="Setting up monitoring and observability",
                tools={"write", "edit", "shell"},
                keywords=["monitor", "observe", "logs", "metrics", "alerts"],
                next_stages={"COMPLETION"},
            ),
            "COMPLETION": StageDefinition(
                name="COMPLETION",
                description="Documenting and finalizing",
                tools={"write", "git_commit"},
                keywords=["done", "complete", "document", "commit"],
                next_stages=set(),
            ),
        }

    @classmethod
    def get_provider_hints(cls) -> Dict[str, Any]:
        """Get provider hints for DevOps tasks.

        Returns:
            Provider preferences for infrastructure work.
        """
        return {
            "preferred_providers": ["anthropic", "openai"],
            "preferred_models": [
                "claude-sonnet-4-20250514",
                "gpt-4-turbo",
            ],
            "min_context_window": 100000,
            "requires_tool_calling": True,
            "prefers_extended_thinking": True,  # Infrastructure decisions benefit from reasoning
        }

    @classmethod
    def get_evaluation_criteria(cls) -> List[str]:
        """Get evaluation criteria for DevOps tasks.

        Returns:
            Criteria for evaluating infrastructure quality.
        """
        return [
            "Configuration correctness and validity",
            "Security best practices adherence",
            "Idempotency of operations",
            "Documentation completeness",
            "Resource efficiency",
            "Disaster recovery considerations",
            "Monitoring coverage",
        ]

    @classmethod
    def customize_config(cls, config: VerticalConfig) -> VerticalConfig:
        """Add DevOps-specific configuration.

        Args:
            config: Base configuration.

        Returns:
            Customized configuration.
        """
        config.metadata["supports_docker"] = True
        config.metadata["supports_kubernetes"] = True
        config.metadata["requires_shell"] = True
        config.metadata["infrastructure_types"] = [
            "docker",
            "kubernetes",
            "terraform",
            "ansible",
            "github_actions",
            "gitlab_ci",
        ]
        config.metadata["security_checklist"] = [
            "no_hardcoded_secrets",
            "least_privilege",
            "encrypted_at_rest",
            "network_isolation",
        ]
        return config
