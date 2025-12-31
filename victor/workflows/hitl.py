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

"""Human-in-the-Loop (HITL) workflow nodes.

Provides LangGraph-style interrupt points for human approval, review,
and intervention during workflow execution.

Node Types:
- APPROVAL: Binary approve/reject decision
- REVIEW: Review and optionally modify intermediate results
- CHOICE: Select from predefined options
- INPUT: Provide freeform text input
- CONFIRMATION: Confirm to proceed (with default behavior on timeout)

Example:
    workflow = (
        WorkflowBuilder("code_change")
        .add_agent("analyze", "researcher", "Find code to change")
        .add_hitl_approval(
            "approve_changes",
            prompt="Proceed with the following changes?",
            context_keys=["files_to_modify", "change_description"],
            timeout=300.0,
        )
        .add_agent("implement", "executor", "Make the changes")
        .build()
    )
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


from typing import Any, Callable, Dict, List, Optional, Protocol, Union

from victor.workflows.definition import NodeType, WorkflowNode


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


logger = logging.getLogger(__name__)


class HITLNodeType(str, Enum):
    """Types of human-in-the-loop interactions."""

    APPROVAL = "approval"  # Binary approve/reject
    REVIEW = "review"  # Review and modify
    CHOICE = "choice"  # Select from options
    INPUT = "input"  # Freeform text input
    CONFIRMATION = "confirmation"  # Simple confirmation


class HITLFallback(str, Enum):
    """Fallback behavior when HITL times out."""

    ABORT = "abort"  # Stop workflow execution
    CONTINUE = "continue"  # Continue with default value
    SKIP = "skip"  # Skip this node entirely
    RETRY = "retry"  # Retry the request


class HITLStatus(str, Enum):
    """Status of a HITL request."""

    PENDING = "pending"  # Waiting for human response
    APPROVED = "approved"  # Approved to continue
    REJECTED = "rejected"  # Rejected/aborted
    MODIFIED = "modified"  # Modified by human
    TIMEOUT = "timeout"  # Timed out
    SKIPPED = "skipped"  # Skipped by fallback


@dataclass
class HITLRequest:
    """A request for human intervention.

    Attributes:
        request_id: Unique identifier for this request
        node_id: ID of the HITL node
        hitl_type: Type of interaction requested
        prompt: Message to display to human
        context: Contextual data for display
        choices: Available choices (for CHOICE type)
        default_value: Default value if timeout
        timeout: Timeout in seconds
        fallback: Behavior on timeout
        created_at: When request was created
    """

    request_id: str
    node_id: str
    hitl_type: HITLNodeType
    prompt: str
    context: Dict[str, Any] = field(default_factory=dict)
    choices: Optional[List[str]] = None
    default_value: Optional[Any] = None
    timeout: float = 300.0
    fallback: HITLFallback = HITLFallback.ABORT
    created_at: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for transmission/display."""
        return {
            "request_id": self.request_id,
            "node_id": self.node_id,
            "hitl_type": self.hitl_type.value,
            "prompt": self.prompt,
            "context": self.context,
            "choices": self.choices,
            "default_value": self.default_value,
            "timeout": self.timeout,
            "fallback": self.fallback.value,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class HITLResponse:
    """Response from human for HITL request.

    Attributes:
        request_id: ID of the request being responded to
        status: Result status
        approved: Whether action was approved (for APPROVAL/CONFIRMATION)
        value: Selected/input value (for CHOICE/INPUT/REVIEW)
        modifications: Modified data (for REVIEW)
        reason: Optional explanation for decision
        responded_at: When response was provided
    """

    request_id: str
    status: HITLStatus
    approved: bool = True
    value: Optional[Any] = None
    modifications: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None
    responded_at: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize response."""
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "approved": self.approved,
            "value": self.value,
            "modifications": self.modifications,
            "reason": self.reason,
            "responded_at": self.responded_at.isoformat(),
        }


class HITLHandler(Protocol):
    """Protocol for handling HITL interactions.

    Implementations provide UI-specific handling for HITL requests
    (CLI, TUI, web API, etc.).
    """

    async def request_human_input(self, request: HITLRequest) -> HITLResponse:
        """Request input from human.

        Args:
            request: The HITL request

        Returns:
            HITLResponse with human's decision
        """
        ...


@dataclass
class HITLNode(WorkflowNode):
    """Workflow node that pauses for human intervention.

    Attributes:
        hitl_type: Type of interaction
        prompt: Message to display
        context_keys: Keys from workflow context to include in display
        choices: Available choices (for CHOICE type)
        default_value: Default value if timeout
        timeout: Timeout in seconds (default: 5 minutes)
        fallback: Behavior on timeout
        validator: Optional function to validate input
    """

    hitl_type: HITLNodeType = HITLNodeType.APPROVAL
    prompt: str = "Proceed?"
    context_keys: List[str] = field(default_factory=list)
    choices: Optional[List[str]] = None
    default_value: Optional[Any] = None
    timeout: float = 300.0
    fallback: HITLFallback = HITLFallback.ABORT
    validator: Optional[Callable[[Any], bool]] = None

    @property
    def node_type(self) -> NodeType:
        """HITL nodes have their own node type."""
        return NodeType.HITL

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node."""
        d = super().to_dict()
        d.update(
            {
                "hitl_type": self.hitl_type.value,
                "prompt": self.prompt,
                "context_keys": self.context_keys,
                "choices": self.choices,
                "default_value": self.default_value,
                "timeout": self.timeout,
                "fallback": self.fallback.value,
                # Note: validator function is not serialized
            }
        )
        # Override type to indicate HITL
        d["type"] = "hitl"
        d["_hitl_node"] = True
        return d

    def create_request(
        self,
        workflow_context: Dict[str, Any],
    ) -> HITLRequest:
        """Create a HITL request from workflow context.

        Args:
            workflow_context: Current workflow execution context

        Returns:
            HITLRequest ready for handler
        """
        # Extract relevant context
        context = {}
        for key in self.context_keys:
            if key in workflow_context:
                context[key] = workflow_context[key]

        return HITLRequest(
            request_id=f"hitl_{uuid.uuid4().hex[:12]}",
            node_id=self.id,
            hitl_type=self.hitl_type,
            prompt=self.prompt,
            context=context,
            choices=self.choices,
            default_value=self.default_value,
            timeout=self.timeout,
            fallback=self.fallback,
        )

    def validate_response(self, response: HITLResponse) -> bool:
        """Validate a HITL response.

        Args:
            response: The response to validate

        Returns:
            True if valid, False otherwise
        """
        if self.validator and response.value is not None:
            try:
                return self.validator(response.value)
            except Exception as e:
                logger.warning(f"HITL validation failed: {e}")
                return False

        return True


class HITLExecutor:
    """Executes HITL nodes during workflow execution.

    Handles the async communication between workflow executor and
    human interface (CLI, TUI, API).
    """

    def __init__(self, handler: HITLHandler):
        """Initialize with a UI handler.

        Args:
            handler: Handler for human interactions
        """
        self.handler = handler
        self._pending_requests: Dict[str, asyncio.Event] = {}
        self._responses: Dict[str, HITLResponse] = {}

    async def execute_hitl_node(
        self,
        node: HITLNode,
        context: Dict[str, Any],
    ) -> HITLResponse:
        """Execute a HITL node, waiting for human response.

        Args:
            node: The HITL node to execute
            context: Current workflow context

        Returns:
            HITLResponse with result

        Raises:
            asyncio.TimeoutError: If timeout and fallback is ABORT
        """
        request = node.create_request(context)

        logger.info(f"HITL request {request.request_id}: {node.hitl_type.value} - {node.prompt}")

        try:
            # Request human input with timeout
            response = await asyncio.wait_for(
                self.handler.request_human_input(request),
                timeout=node.timeout,
            )

            # Validate response
            if not node.validate_response(response):
                logger.warning(f"HITL response validation failed for {request.request_id}")
                response = HITLResponse(
                    request_id=request.request_id,
                    status=HITLStatus.REJECTED,
                    approved=False,
                    reason="Response validation failed",
                )

            return response

        except asyncio.TimeoutError:
            logger.warning(f"HITL request {request.request_id} timed out after {node.timeout}s")
            return self._handle_timeout(request, node)

    def _handle_timeout(
        self,
        request: HITLRequest,
        node: HITLNode,
    ) -> HITLResponse:
        """Handle timeout based on fallback strategy.

        Args:
            request: The timed-out request
            node: The HITL node

        Returns:
            HITLResponse based on fallback behavior
        """
        if node.fallback == HITLFallback.CONTINUE:
            return HITLResponse(
                request_id=request.request_id,
                status=HITLStatus.TIMEOUT,
                approved=True,
                value=node.default_value,
                reason="Timed out, continuing with default",
            )

        elif node.fallback == HITLFallback.SKIP:
            return HITLResponse(
                request_id=request.request_id,
                status=HITLStatus.SKIPPED,
                approved=True,
                reason="Timed out, skipping",
            )

        else:  # ABORT or RETRY
            return HITLResponse(
                request_id=request.request_id,
                status=HITLStatus.TIMEOUT,
                approved=False,
                reason=f"Timed out after {node.timeout}s",
            )


class DefaultHITLHandler:
    """Default CLI handler for HITL requests.

    Uses simple input() for approval/choice prompts.
    Suitable for basic CLI usage.
    """

    async def request_human_input(self, request: HITLRequest) -> HITLResponse:
        """Handle HITL request via CLI.

        Args:
            request: The HITL request

        Returns:
            HITLResponse from user input
        """
        print(f"\n{'='*60}")
        print(f"Human Input Required: {request.hitl_type.value.upper()}")
        print(f"{'='*60}")
        print(f"\n{request.prompt}\n")

        # Show context if available
        if request.context:
            print("Context:")
            for key, value in request.context.items():
                if isinstance(value, str) and len(value) > 200:
                    value = value[:200] + "..."
                print(f"  {key}: {value}")
            print()

        if request.hitl_type == HITLNodeType.APPROVAL:
            return await self._handle_approval(request)
        elif request.hitl_type == HITLNodeType.CHOICE:
            return await self._handle_choice(request)
        elif request.hitl_type == HITLNodeType.INPUT:
            return await self._handle_input(request)
        elif request.hitl_type == HITLNodeType.CONFIRMATION:
            return await self._handle_confirmation(request)
        elif request.hitl_type == HITLNodeType.REVIEW:
            return await self._handle_review(request)
        else:
            return HITLResponse(
                request_id=request.request_id,
                status=HITLStatus.REJECTED,
                approved=False,
                reason=f"Unknown HITL type: {request.hitl_type}",
            )

    async def _handle_approval(self, request: HITLRequest) -> HITLResponse:
        """Handle approval request."""
        while True:
            response = await asyncio.get_event_loop().run_in_executor(
                None, input, "Approve? [y/n]: "
            )
            response = response.strip().lower()

            if response in ("y", "yes"):
                return HITLResponse(
                    request_id=request.request_id,
                    status=HITLStatus.APPROVED,
                    approved=True,
                )
            elif response in ("n", "no"):
                reason = await asyncio.get_event_loop().run_in_executor(
                    None, input, "Reason (optional): "
                )
                return HITLResponse(
                    request_id=request.request_id,
                    status=HITLStatus.REJECTED,
                    approved=False,
                    reason=reason.strip() or None,
                )
            else:
                print("Please enter 'y' or 'n'")

    async def _handle_choice(self, request: HITLRequest) -> HITLResponse:
        """Handle choice selection."""
        if not request.choices:
            return HITLResponse(
                request_id=request.request_id,
                status=HITLStatus.REJECTED,
                approved=False,
                reason="No choices provided",
            )

        print("Options:")
        for i, choice in enumerate(request.choices, 1):
            print(f"  [{i}] {choice}")

        while True:
            response = await asyncio.get_event_loop().run_in_executor(
                None, input, "Select option number: "
            )
            try:
                idx = int(response.strip()) - 1
                if 0 <= idx < len(request.choices):
                    return HITLResponse(
                        request_id=request.request_id,
                        status=HITLStatus.APPROVED,
                        approved=True,
                        value=request.choices[idx],
                    )
                else:
                    print(f"Please enter a number between 1 and {len(request.choices)}")
            except ValueError:
                print("Please enter a valid number")

    async def _handle_input(self, request: HITLRequest) -> HITLResponse:
        """Handle freeform input."""
        response = await asyncio.get_event_loop().run_in_executor(None, input, "Your input: ")
        return HITLResponse(
            request_id=request.request_id,
            status=HITLStatus.APPROVED,
            approved=True,
            value=response.strip(),
        )

    async def _handle_confirmation(self, request: HITLRequest) -> HITLResponse:
        """Handle simple confirmation."""
        response = await asyncio.get_event_loop().run_in_executor(
            None, input, "Press Enter to continue or 'q' to abort: "
        )
        if response.strip().lower() == "q":
            return HITLResponse(
                request_id=request.request_id,
                status=HITLStatus.REJECTED,
                approved=False,
                reason="User aborted",
            )
        return HITLResponse(
            request_id=request.request_id,
            status=HITLStatus.APPROVED,
            approved=True,
        )

    async def _handle_review(self, request: HITLRequest) -> HITLResponse:
        """Handle review with modification."""
        print("Review the context above.")
        response = await asyncio.get_event_loop().run_in_executor(
            None, input, "Approve as-is [a], modify [m], or reject [r]: "
        )
        response = response.strip().lower()

        if response == "a":
            return HITLResponse(
                request_id=request.request_id,
                status=HITLStatus.APPROVED,
                approved=True,
            )
        elif response == "r":
            reason = await asyncio.get_event_loop().run_in_executor(None, input, "Reason: ")
            return HITLResponse(
                request_id=request.request_id,
                status=HITLStatus.REJECTED,
                approved=False,
                reason=reason.strip(),
            )
        else:
            # Simple modification - just get text input
            print("Enter modifications (one per line, empty line to finish):")
            modifications = {}
            while True:
                line = await asyncio.get_event_loop().run_in_executor(None, input, "")
                if not line:
                    break
                if "=" in line:
                    key, value = line.split("=", 1)
                    modifications[key.strip()] = value.strip()

            return HITLResponse(
                request_id=request.request_id,
                status=HITLStatus.MODIFIED,
                approved=True,
                modifications=modifications,
            )


# Type alias for cleaner imports
HITLCallback = Callable[[HITLRequest], HITLResponse]


__all__ = [
    "HITLNodeType",
    "HITLFallback",
    "HITLStatus",
    "HITLRequest",
    "HITLResponse",
    "HITLHandler",
    "HITLNode",
    "HITLExecutor",
    "DefaultHITLHandler",
    "HITLCallback",
]
