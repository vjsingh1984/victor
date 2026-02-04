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

"""Human-in-the-Loop (HITL) Framework for agent workflows.

This package provides a comprehensive framework for human-in-the-loop
interactions within agent workflows. It supports multiple interaction
patterns, session management, and YAML workflow integration.

Components:
    - gates: Predefined interaction patterns (ApprovalGate, TextInputGate, etc.)
    - templates: Template system for common HITL prompts
    - session: Session management across workflows
    - protocols: Protocol-based interfaces for extensibility

Example Usage:
    from victor.framework.hitl import (
        ApprovalGate,
        TextInputGate,
        ChoiceInput,
        ConfirmationDialog,
        ReviewGate,
        HITLSession,
        get_prompt_template,
    )

    # Create an approval gate
    gate = ApprovalGate(
        title="Deploy to Production",
        description="Deploy version 1.2.3 to production",
        timeout_seconds=300,
    )

    # Execute gate
    result = await gate.execute(context={"version": "1.2.3"})
    if result.is_approved:
        # Proceed with deployment
        pass

    # Create a text input gate
    input_gate = TextInputGate(
        title="Enter Deployment Notes",
        prompt="Please provide deployment notes",
        required=True,
    )

    result = await input_gate.execute()
    notes = result.value

    # Use session for multi-step workflows
    session = HITLSession(workflow_id="deployment_workflow")

    gate1 = ApprovalGate(title="Step 1: Review")
    result1 = await session.execute_gate(gate1)

    gate2 = TextInputGate(title="Step 2: Comments")
    result2 = await session.execute_gate(gate2)
"""

# Import the old HITL module types for backward compatibility
from victor.framework.hitl._orig import (
    ApprovalHandler,
    ApprovalRequest,
    ApprovalStatus,
    HITLCheckpoint,
    HITLController,
)

# Import new HITL framework components
from victor.framework.hitl.gates import (
    ApprovalGate,
    ChoiceInputGate,
    ConfirmationDialogGate,
    ReviewGate,
    TextInputGate,
)

# Convenience aliases
ChoiceInput = ChoiceInputGate
ConfirmationDialog = ConfirmationDialogGate

from victor.framework.hitl.protocols import (
    FallbackBehavior,
    FallbackStrategy,
    HITLGateProtocol,
    HITLResponseProtocol,
    InputValidationProtocol,
)
from victor.framework.hitl.session import (
    HITLSession,
    HITLSessionConfig,
    HITLSessionManager,
    SessionState,
    get_global_session_manager,
)
from victor.framework.hitl.templates import (
    PromptTemplate,
    PromptTemplateRegistry,
    TemplateContext,
    get_prompt_template,
    list_templates,
    register_template,
    render_template,
)

__all__ = [
    # Old HITL module (backward compatibility)
    "ApprovalHandler",
    "ApprovalRequest",
    "ApprovalStatus",
    "HITLCheckpoint",
    "HITLController",
    # Gates
    "ApprovalGate",
    "TextInputGate",
    "ChoiceInput",
    "ConfirmationDialog",
    "ReviewGate",
    # Protocols
    "HITLGateProtocol",
    "HITLResponseProtocol",
    "InputValidationProtocol",
    "FallbackBehavior",
    "FallbackStrategy",
    # Session
    "HITLSession",
    "HITLSessionConfig",
    "HITLSessionManager",
    "SessionState",
    "get_global_session_manager",
    # Templates
    "PromptTemplate",
    "PromptTemplateRegistry",
    "TemplateContext",
    "get_prompt_template",
    "register_template",
    "render_template",
    "list_templates",
]
