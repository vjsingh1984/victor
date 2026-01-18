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

"""Common HITL gate patterns for human interactions.

This module provides ready-to-use gate implementations for common
human-in-the-loop scenarios in agent workflows.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
    Union,
)

from victor.framework.hitl.protocols import (
    BaseHITLGate,
    BaseHITLResponse,
    FallbackBehavior,
    FallbackStrategy,
    HITLGateProtocol,
    HITLResponseProtocol,
    InputValidationProtocol,
)

T = TypeVar("T")


# =============================================================================
# Response Types
# =============================================================================


class ApprovalResponse:
    """Response from an approval gate.

    Attributes:
        approved: Whether the request was approved
        reason: Optional explanation for the decision
    """

    def __init__(
        self,
        gate_id: str,
        approved: bool,
        reason: Optional[str] = None,
        responder: Optional[str] = None,
        created_at: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize approval response."""
        self.gate_id = gate_id
        self.approved = approved
        self.reason = reason
        self.responder = responder
        self.created_at = created_at or time.time()
        self.metadata = metadata or {}
        self.value = None

    @property
    def is_approved(self) -> bool:
        """Whether the interaction was approved."""
        return self.approved

    @property
    def is_rejected(self) -> bool:
        """Whether the interaction was rejected."""
        return not self.approved

    @property
    def is_timeout(self) -> bool:
        """Whether the interaction timed out."""
        return self.metadata.get("timed_out", False)

    @property
    def is_skipped(self) -> bool:
        """Whether the interaction was skipped."""
        return self.metadata.get("skipped", False)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "gate_id": self.gate_id,
            "approved": self.approved,
            "value": self.value,
            "reason": self.reason,
            "responder": self.responder,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


class TextResponse:
    """Response from a text input gate.

    Attributes:
        text: The text input provided by the human
    """

    def __init__(
        self,
        gate_id: str,
        text: str,
        approved: bool = True,
        reason: Optional[str] = None,
        responder: Optional[str] = None,
        created_at: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize text response."""
        self.gate_id = gate_id
        self.approved = approved
        self.reason = reason
        self.responder = responder
        self.created_at = created_at or time.time()
        self.metadata = metadata or {}
        self.text = text

    @property
    def value(self) -> str:
        """Return the text value."""
        return self.text

    @property
    def is_approved(self) -> bool:
        """Whether the interaction was approved."""
        return self.approved

    @property
    def is_rejected(self) -> bool:
        """Whether the interaction was rejected."""
        return not self.approved

    @property
    def is_timeout(self) -> bool:
        """Whether the interaction timed out."""
        return self.metadata.get("timed_out", False)

    @property
    def is_skipped(self) -> bool:
        """Whether the interaction was skipped."""
        return self.metadata.get("skipped", False)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "gate_id": self.gate_id,
            "approved": self.approved,
            "value": self.text,
            "reason": self.reason,
            "responder": self.responder,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


class ChoiceResponse:
    """Response from a choice input gate.

    Attributes:
        selected: The selected choice
        index: Index of the selected choice
    """

    def __init__(
        self,
        gate_id: str,
        selected: str,
        index: int,
        approved: bool = True,
        reason: Optional[str] = None,
        responder: Optional[str] = None,
        created_at: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize choice response."""
        self.gate_id = gate_id
        self.approved = approved
        self.reason = reason
        self.responder = responder
        self.created_at = created_at or time.time()
        self.metadata = metadata or {}
        self.selected = selected
        self.index = index

    @property
    def value(self) -> str:
        """Return the selected choice."""
        return self.selected

    @property
    def is_approved(self) -> bool:
        """Whether the interaction was approved."""
        return self.approved

    @property
    def is_rejected(self) -> bool:
        """Whether the interaction was rejected."""
        return not self.approved

    @property
    def is_timeout(self) -> bool:
        """Whether the interaction timed out."""
        return self.metadata.get("timed_out", False)

    @property
    def is_skipped(self) -> bool:
        """Whether the interaction was skipped."""
        return self.metadata.get("skipped", False)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "gate_id": self.gate_id,
            "approved": self.approved,
            "value": self.selected,
            "reason": self.reason,
            "responder": self.responder,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


class ReviewResponse:
    """Response from a review gate.

    Attributes:
        approved: Whether to proceed
        modifications: Any modifications made during review
        comments: Review comments
    """

    def __init__(
        self,
        gate_id: str,
        approved: bool,
        modifications: Optional[Dict[str, Any]] = None,
        comments: Optional[str] = None,
        responder: Optional[str] = None,
        created_at: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize review response."""
        self.gate_id = gate_id
        self.approved = approved
        self.modifications = modifications
        self.comments = comments
        self.responder = responder
        self.created_at = created_at or time.time()
        self.metadata = metadata or {}

    @property
    def value(self) -> Optional[Dict[str, Any]]:
        """Return the modifications value."""
        return self.modifications

    @property
    def is_approved(self) -> bool:
        """Whether the interaction was approved."""
        return self.approved

    @property
    def is_rejected(self) -> bool:
        """Whether the interaction was rejected."""
        return not self.approved

    @property
    def is_timeout(self) -> bool:
        """Whether the interaction timed out."""
        return self.metadata.get("timed_out", False)

    @property
    def is_skipped(self) -> bool:
        """Whether the interaction was skipped."""
        return self.metadata.get("skipped", False)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "gate_id": self.gate_id,
            "approved": self.approved,
            "value": self.modifications,
            "reason": self.comments,
            "responder": self.responder,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


# =============================================================================
# Input Validators
# =============================================================================


class Validator(InputValidationProtocol):
    """Base validator class."""

    async def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate input. Override in subclasses."""
        return True, None


class RequiredValidator(Validator):
    """Validator that requires non-empty input."""

    async def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """Check that value is not empty."""
        if value is None or (isinstance(value, str) and not value.strip()):
            return False, "This field is required"
        return True, None


class LengthValidator(Validator):
    """Validator for string length constraints."""

    def __init__(self, min_length: int = 0, max_length: int = 10000):
        """Initialize length validator.

        Args:
            min_length: Minimum allowed length
            max_length: Maximum allowed length
        """
        self.min_length = min_length
        self.max_length = max_length

    async def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate string length."""
        if not isinstance(value, str):
            return False, "Value must be a string"
        if len(value) < self.min_length:
            return False, f"Minimum length is {self.min_length}"
        if len(value) > self.max_length:
            return False, f"Maximum length is {self.max_length}"
        return True, None


class PatternValidator(Validator):
    """Validator for regex pattern matching."""

    def __init__(self, pattern: str, message: str = "Invalid format"):
        """Initialize pattern validator.

        Args:
            pattern: Regex pattern to match
            message: Error message for invalid input
        """
        import re

        self.pattern = re.compile(pattern)
        self.message = message

    async def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate against pattern."""
        if not isinstance(value, str):
            return False, "Value must be a string"
        if not self.pattern.match(value):
            return False, self.message
        return True, None


class ChoiceValidator(Validator):
    """Validator for choice selection."""

    def __init__(self, choices: List[str]):
        """Initialize choice validator.

        Args:
            choices: Valid choices
        """
        self.choices = choices

    async def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate choice is in options."""
        if value not in self.choices:
            return False, f"Must be one of: {', '.join(self.choices)}"
        return True, None


# =============================================================================
# Gate Implementations
# =============================================================================


class ApprovalGate(BaseHITLGate):
    """Gate for binary yes/no approval decisions.

    Example:
        gate = ApprovalGate(
            title="Deploy to Production",
            description="This will deploy to production servers",
        )
        result = await gate.execute()
        if result.approved:
            # Proceed with deployment
    """

    def __init__(
        self,
        title: str,
        description: str,
        context: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 300.0,
        fallback_strategy: Optional[FallbackStrategy] = None,
        show_details: bool = False,
        require_reason_on_reject: bool = False,
    ):
        """Initialize approval gate.

        Args:
            title: Short title for the approval
            description: Detailed description of what needs approval
            context: Additional context for the approval
            timeout_seconds: Timeout in seconds
            fallback_strategy: Fallback strategy
            show_details: Whether to show detailed context
            require_reason_on_reject: Require reason when rejecting
        """
        super().__init__(
            _gate_id=f"approval_{uuid.uuid4().hex[:12]}",
            gate_type="approval",
            title=title,
            prompt=description,
            timeout_seconds=timeout_seconds,
            fallback_strategy=fallback_strategy or FallbackStrategy.abort(),
            required=True,
            context=context or {},
        )
        self._description = description  # Store for copying
        self.show_details = show_details
        self.require_reason_on_reject = require_reason_on_reject

    def with_context(self, context: Dict[str, Any]) -> "ApprovalGate":
        """Create a new gate with merged context."""
        return ApprovalGate(
            title=self.title,
            description=self._description,
            context={**self.context, **context},
            timeout_seconds=self.timeout_seconds,
            fallback_strategy=self.fallback_strategy,
            show_details=self.show_details,
            require_reason_on_reject=self.require_reason_on_reject,
        )

    def with_timeout(self, timeout: float) -> "ApprovalGate":
        """Create a new gate with custom timeout."""
        return ApprovalGate(
            title=self.title,
            description=self._description,
            context=self.context,
            timeout_seconds=timeout,
            fallback_strategy=self.fallback_strategy,
            show_details=self.show_details,
            require_reason_on_reject=self.require_reason_on_reject,
        )

    def with_fallback(self, strategy: FallbackStrategy) -> "ApprovalGate":
        """Create a new gate with custom fallback."""
        return ApprovalGate(
            title=self.title,
            description=self._description,
            context=self.context,
            timeout_seconds=self.timeout_seconds,
            fallback_strategy=strategy,
            show_details=self.show_details,
            require_reason_on_reject=self.require_reason_on_reject,
        )

    async def execute(
        self,
        context: Optional[Dict[str, Any]] = None,
        handler: Optional[Callable[..., Awaitable[HITLResponseProtocol]]] = None,
    ) -> ApprovalResponse:
        """Execute the approval gate.

        Args:
            context: Additional context for rendering
            handler: Custom handler for the interaction

        Returns:
            ApprovalResponse with the decision
        """
        prompt = self._render_prompt(context)

        if handler:
            response = await handler(
                gate_id=self.gate_id,
                gate_type=self.gate_type,
                title=self.title,
                prompt=prompt,
                context={**self.context, **(context or {})},
            )
            return ApprovalResponse(
                gate_id=self.gate_id,
                approved=getattr(response, "approved", False),
                reason=getattr(response, "reason", None),
                responder=getattr(response, "responder", None),
            )

        # Default: auto-approve for testing
        return ApprovalResponse(
            gate_id=self.gate_id,
            approved=True,
            reason="Auto-approved (no handler provided)",
            responder="system",
        )


class TextInputGate(BaseHITLGate):
    """Gate for free-form text input.

    Example:
        gate = TextInputGate(
            title="Enter Deployment Notes",
            prompt="Please provide deployment notes",
            required=True,
            min_length=10,
        )
        result = await gate.execute()
        notes = result.text
    """

    def __init__(
        self,
        title: str,
        prompt: str,
        placeholder: str = "",
        default_value: str = "",
        required: bool = True,
        min_length: int = 0,
        max_length: int = 10000,
        context: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 300.0,
        fallback_strategy: Optional[FallbackStrategy] = None,
    ):
        """Initialize text input gate.

        Args:
            title: Short title
            prompt: Prompt message
            placeholder: Placeholder text for input field
            default_value: Default value if not required
            required: Whether input is required
            min_length: Minimum string length
            max_length: Maximum string length
            context: Additional context
            timeout_seconds: Timeout in seconds
            fallback_strategy: Fallback strategy
        """
        validators: List[InputValidationProtocol] = []
        if required:
            validators.append(RequiredValidator())
        if min_length > 0 or max_length < 10000:
            validators.append(LengthValidator(min_length, max_length))

        super().__init__(
            _gate_id=f"text_input_{uuid.uuid4().hex[:12]}",
            gate_type="text_input",
            title=title,
            prompt=prompt,
            timeout_seconds=timeout_seconds,
            fallback_strategy=fallback_strategy
            or FallbackStrategy.continue_with_default(default_value),
            required=required,
            context=context or {},
        )
        self._prompt = prompt  # Store for copying
        self.placeholder = placeholder
        self.default_value = default_value
        self.validators = validators
        self._min_length = min_length
        self._max_length = max_length

    def with_context(self, context: Dict[str, Any]) -> "TextInputGate":
        """Create a new gate with merged context."""
        return TextInputGate(
            title=self.title,
            prompt=self._prompt,
            placeholder=self.placeholder,
            default_value=self.default_value,
            required=self.required,
            min_length=self._min_length,
            max_length=self._max_length,
            context={**self.context, **context},
            timeout_seconds=self.timeout_seconds,
            fallback_strategy=self.fallback_strategy,
        )

    def with_timeout(self, timeout: float) -> "TextInputGate":
        """Create a new gate with custom timeout."""
        return TextInputGate(
            title=self.title,
            prompt=self._prompt,
            placeholder=self.placeholder,
            default_value=self.default_value,
            required=self.required,
            min_length=self._min_length,
            max_length=self._max_length,
            context=self.context,
            timeout_seconds=timeout,
            fallback_strategy=self.fallback_strategy,
        )

    def with_fallback(self, strategy: FallbackStrategy) -> "TextInputGate":
        """Create a new gate with custom fallback."""
        return TextInputGate(
            title=self.title,
            prompt=self._prompt,
            placeholder=self.placeholder,
            default_value=self.default_value,
            required=self.required,
            min_length=self._min_length,
            max_length=self._max_length,
            context=self.context,
            timeout_seconds=self.timeout_seconds,
            fallback_strategy=strategy,
        )

    async def execute(
        self,
        context: Optional[Dict[str, Any]] = None,
        handler: Optional[Callable[..., Awaitable[HITLResponseProtocol]]] = None,
    ) -> TextResponse:
        """Execute the text input gate.

        Args:
            context: Additional context for rendering
            handler: Custom handler for the interaction

        Returns:
            TextResponse with the input text
        """
        prompt = self._render_prompt(context)

        if handler:
            response = await handler(
                gate_id=self.gate_id,
                gate_type=self.gate_type,
                title=self.title,
                prompt=prompt,
                placeholder=self.placeholder,
                context={**self.context, **(context or {})},
            )
            text = getattr(response, "value", getattr(response, "text", ""))
        else:
            # Default: return default value
            text = self.default_value

        # Validate
        if self.validators:
            for validator in self.validators:
                is_valid, error = await validator.validate(text)
                if not is_valid:
                    return TextResponse(
                        gate_id=self.gate_id,
                        approved=False,
                        text=text,
                        reason=error or "Validation failed",
                    )

        return TextResponse(
            gate_id=self.gate_id,
            approved=True,
            text=text,
            responder="user",
        )


class ChoiceInputGate(BaseHITLGate):
    """Gate for selecting from predefined choices.

    Example:
        gate = ChoiceInputGate(
            title="Select Deployment Environment",
            prompt="Choose the target environment",
            choices=["development", "staging", "production"],
        )
        result = await gate.execute()
        env = result.selected
    """

    def __init__(
        self,
        title: str,
        prompt: str,
        choices: List[str],
        default_index: int = 0,
        allow_multiple: bool = False,
        context: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 300.0,
        fallback_strategy: Optional[FallbackStrategy] = None,
    ):
        """Initialize choice input gate.

        Args:
            title: Short title
            prompt: Prompt message
            choices: Available choices
            default_index: Default selected index
            allow_multiple: Allow multiple selections
            context: Additional context
            timeout_seconds: Timeout in seconds
            fallback_strategy: Fallback strategy
        """
        super().__init__(
            _gate_id=f"choice_input_{uuid.uuid4().hex[:12]}",
            gate_type="choice_input",
            title=title,
            prompt=prompt,
            timeout_seconds=timeout_seconds,
            fallback_strategy=fallback_strategy
            or FallbackStrategy.continue_with_default(choices[default_index]),
            required=True,
            context=context or {},
            validator=ChoiceValidator(choices),
        )
        self._prompt = prompt  # Store for copying
        self.choices = choices
        self.default_index = default_index
        self.allow_multiple = allow_multiple

    def with_context(self, context: Dict[str, Any]) -> "ChoiceInputGate":
        """Create a new gate with merged context."""
        return ChoiceInputGate(
            title=self.title,
            prompt=self._prompt,
            choices=self.choices,
            default_index=self.default_index,
            allow_multiple=self.allow_multiple,
            context={**self.context, **context},
            timeout_seconds=self.timeout_seconds,
            fallback_strategy=None,  # Will be recreated
        )

    def with_timeout(self, timeout: float) -> "ChoiceInputGate":
        """Create a new gate with custom timeout."""
        return ChoiceInputGate(
            title=self.title,
            prompt=self._prompt,
            choices=self.choices,
            default_index=self.default_index,
            allow_multiple=self.allow_multiple,
            context=self.context,
            timeout_seconds=timeout,
            fallback_strategy=None,  # Will be recreated
        )

    def with_fallback(self, strategy: FallbackStrategy) -> "ChoiceInputGate":
        """Create a new gate with custom fallback."""
        return ChoiceInputGate(
            title=self.title,
            prompt=self._prompt,
            choices=self.choices,
            default_index=self.default_index,
            allow_multiple=self.allow_multiple,
            context=self.context,
            timeout_seconds=self.timeout_seconds,
            fallback_strategy=strategy,
        )

    async def execute(
        self,
        context: Optional[Dict[str, Any]] = None,
        handler: Optional[Callable[..., Awaitable[HITLResponseProtocol]]] = None,
    ) -> ChoiceResponse:
        """Execute the choice input gate.

        Args:
            context: Additional context for rendering
            handler: Custom handler for the interaction

        Returns:
            ChoiceResponse with the selected choice
        """
        prompt = self._render_prompt(context)

        if handler:
            response = await handler(
                gate_id=self.gate_id,
                gate_type=self.gate_type,
                title=self.title,
                prompt=prompt,
                choices=self.choices,
                context={**self.context, **(context or {})},
            )
            selected = getattr(response, "value", getattr(response, "selected", ""))
        else:
            # Default: use default choice
            selected = self.choices[self.default_index]

        # Validate
        is_valid, error = await self.validator.validate(selected)  # type: ignore
        if not is_valid:
            return ChoiceResponse(
                gate_id=self.gate_id,
                approved=False,
                selected=selected,
                index=self.choices.index(selected) if selected in self.choices else -1,
                reason=error,
            )

        return ChoiceResponse(
            gate_id=self.gate_id,
            approved=True,
            selected=selected,
            index=self.choices.index(selected),
            responder="user",
        )


class ConfirmationDialogGate(BaseHITLGate):
    """Gate for simple yes/no confirmation with timeout default.

    Example:
        gate = ConfirmationDialogGate(
            title="Confirm Deployment",
            prompt="Deploy to production?",
            default_approved=True,  # Auto-approve on timeout
        )
        result = await gate.execute()
        if result.approved:
            # Proceed
    """

    def __init__(
        self,
        title: str,
        prompt: str,
        default_approved: bool = False,
        context: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 60.0,
    ):
        """Initialize confirmation dialog.

        Args:
            title: Short title
            prompt: Prompt message
            default_approved: Default on timeout
            context: Additional context
            timeout_seconds: Timeout in seconds
        """
        super().__init__(
            _gate_id=f"confirmation_{uuid.uuid4().hex[:12]}",
            gate_type="confirmation",
            title=title,
            prompt=prompt,
            timeout_seconds=timeout_seconds,
            fallback_strategy=FallbackStrategy.continue_with_default(default_approved),
            required=True,
            context=context or {},
        )
        self.default_approved = default_approved

    async def execute(
        self,
        context: Optional[Dict[str, Any]] = None,
        handler: Optional[Callable[..., Awaitable[HITLResponseProtocol]]] = None,
    ) -> ApprovalResponse:
        """Execute the confirmation dialog.

        Args:
            context: Additional context for rendering
            handler: Custom handler for the interaction

        Returns:
            ApprovalResponse with the decision
        """
        prompt = self._render_prompt(context)

        if handler:
            response = await handler(
                gate_id=self.gate_id,
                gate_type=self.gate_type,
                title=self.title,
                prompt=prompt,
                context={**self.context, **(context or {})},
            )
            return ApprovalResponse(
                gate_id=self.gate_id,
                approved=getattr(response, "approved", self.default_approved),
                reason=getattr(response, "reason", None),
                responder=getattr(response, "responder", None),
            )

        # Default: use default_approved
        return ApprovalResponse(
            gate_id=self.gate_id,
            approved=self.default_approved,
            reason="Auto-confirmed (no handler)",
            responder="system",
        )


class ReviewGate(BaseHITLGate):
    """Gate for reviewing and optionally modifying content.

    Example:
        gate = ReviewGate(
            title="Review Code Changes",
            content=code_diff,
            allow_modifications=True,
        )
        result = await gate.execute()
        if result.modifications:
            # Apply modifications
    """

    def __init__(
        self,
        title: str,
        content: str,
        prompt: str = "Review the content below",
        allow_modifications: bool = True,
        context: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 600.0,
        fallback_strategy: Optional[FallbackStrategy] = None,
    ):
        """Initialize review gate.

        Args:
            title: Short title
            content: Content to review
            prompt: Prompt message
            allow_modifications: Allow modifications during review
            context: Additional context
            timeout_seconds: Timeout in seconds
            fallback_strategy: Fallback strategy
        """
        super().__init__(
            _gate_id=f"review_{uuid.uuid4().hex[:12]}",
            gate_type="review",
            title=title,
            prompt=prompt,
            timeout_seconds=timeout_seconds,
            fallback_strategy=fallback_strategy or FallbackStrategy.skip(),
            required=True,
            context=context or {},
        )
        self.content = content
        self.allow_modifications = allow_modifications

    async def execute(
        self,
        context: Optional[Dict[str, Any]] = None,
        handler: Optional[Callable[..., Awaitable[HITLResponseProtocol]]] = None,
    ) -> ReviewResponse:
        """Execute the review gate.

        Args:
            context: Additional context for rendering
            handler: Custom handler for the interaction

        Returns:
            ReviewResponse with the decision and modifications
        """
        prompt = self._render_prompt(context)

        if handler:
            response = await handler(
                gate_id=self.gate_id,
                gate_type=self.gate_type,
                title=self.title,
                prompt=prompt,
                content=self.content,
                allow_modifications=self.allow_modifications,
                context={**self.context, **(context or {})},
            )
            return ReviewResponse(
                gate_id=self.gate_id,
                approved=getattr(response, "approved", True),
                modifications=getattr(response, "modifications", None),
                comments=getattr(response, "reason", None),
                responder=getattr(response, "responder", None),
            )

        # Default: auto-approve without modifications
        return ReviewResponse(
            gate_id=self.gate_id,
            approved=True,
            modifications=None,
            comments="Auto-approved (no handler)",
            responder="system",
        )


# =============================================================================
# Convenience Aliases
# =============================================================================

# Alias for backward compatibility
TextInput = TextInputGate

# Export additional convenience names
ChoiceInput = ChoiceInputGate
ConfirmationDialog = ConfirmationDialogGate
