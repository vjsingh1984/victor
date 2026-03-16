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

import warnings

warnings.warn(
    "victor.verticals.contrib.devops is deprecated and will be removed in v0.7.0. Install the victor-devops package instead.",
    DeprecationWarning,
    stacklevel=2,
)

from victor.framework.vertical_runtime_adapter import VerticalRuntimeAdapter
from victor.verticals.contrib.devops.assistant import DevOpsAssistant as DevOpsAssistantDefinition
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
from victor.verticals.contrib.devops.tool_dependencies import get_provider
from victor.verticals.contrib.devops.capabilities import DevOpsCapabilityProvider

DevOpsAssistant = VerticalRuntimeAdapter.as_runtime_vertical_class(DevOpsAssistantDefinition)

__all__ = [
    "DevOpsAssistant",
    "DevOpsAssistantDefinition",
    "DevOpsPromptContributor",
    "DevOpsModeConfigProvider",
    "DevOpsSafetyExtension",
    # Enhanced Extensions (with new coordinators)
    "EnhancedDevOpsSafetyExtension",
    "EnhancedDevOpsConversationManager",
    "DevOpsSafetyRules",
    "DevOpsContext",
    # Tool dependency provider factory
    "get_provider",
    "DevOpsCapabilityProvider",
]
