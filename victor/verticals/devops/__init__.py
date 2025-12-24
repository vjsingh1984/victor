"""DevOps Vertical Package - Complete implementation with extensions.

Competitive use case: Replaces/augments Docker Desktop AI, Terraform Assistant, Pulumi AI.

This vertical provides:
- Docker and container management
- CI/CD pipeline configuration
- Infrastructure as Code (IaC) generation
- Kubernetes manifest creation
- Monitoring and observability setup
"""

from victor.verticals.devops.assistant import DevOpsAssistant
from victor.verticals.devops.prompts import DevOpsPromptContributor
from victor.verticals.devops.mode_config import DevOpsModeConfigProvider
from victor.verticals.devops.safety import DevOpsSafetyExtension
from victor.verticals.devops.tool_dependencies import DevOpsToolDependencyProvider

__all__ = [
    "DevOpsAssistant",
    "DevOpsPromptContributor",
    "DevOpsModeConfigProvider",
    "DevOpsSafetyExtension",
    "DevOpsToolDependencyProvider",
]
