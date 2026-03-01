"""DevOps Vertical Package - Complete implementation with extensions.

Competitive use case: Replaces/augments Docker Desktop AI, Terraform Assistant, Pulumi AI.

This vertical provides:
- Docker and container management
- CI/CD pipeline configuration
- Infrastructure as Code (IaC) generation
- Kubernetes manifest creation
- Monitoring and observability setup

Enhanced Features:
- Enhanced safety with SafetyCoordinator (safety_enhanced.py)
- Enhanced conversation management with ConversationCoordinator (conversation_enhanced.py)
"""

from victor.verticals.contrib.devops.assistant import DevOpsAssistant
from victor.verticals.contrib.devops.prompts import DevOpsPromptContributor
from victor.verticals.contrib.devops.mode_config import DevOpsModeConfigProvider
from victor.verticals.contrib.devops.safety import DevOpsSafetyExtension
from victor.verticals.contrib.devops.safety_enhanced import (
    DevOpsSafetyRules,
    EnhancedDevOpsSafetyExtension,
)
from victor.verticals.contrib.devops.conversation_enhanced import (
    DevOpsContext,
    EnhancedDevOpsConversationManager,
)
from victor.verticals.contrib.devops.tool_dependencies import DevOpsToolDependencyProvider
from victor.verticals.contrib.devops.capabilities import DevOpsCapabilityProvider

__all__ = [
    "DevOpsAssistant",
    "DevOpsPromptContributor",
    "DevOpsModeConfigProvider",
    "DevOpsSafetyExtension",
    # Enhanced Extensions (with new coordinators)
    "EnhancedDevOpsSafetyExtension",
    "EnhancedDevOpsConversationManager",
    "DevOpsSafetyRules",
    "DevOpsContext",
    # Other extensions
    "DevOpsToolDependencyProvider",
    "DevOpsCapabilityProvider",
]
