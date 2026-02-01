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

"""Protocol definitions for HITL framework.

This module defines the core protocol interfaces that enable
extensibility and type safety for the HITL framework.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)
from collections.abc import Awaitable, Callable


class FallbackBehavior(str, Enum):
    """Behavior to apply when a HITL gate times out.

    Attributes:
        ABORT: Stop workflow execution with an error
        CONTINUE: Continue with the default value
        SKIP: Skip the current gate entirely
        RETRY: Retry the request with a new timeout
    """

    ABORT = "abort"
    CONTINUE = "continue"
    SKIP = "skip"
    RETRY = "retry"


@dataclass
class FallbackStrategy:
    """Strategy for handling HITL timeout scenarios.

    Attributes:
        behavior: The fallback behavior to apply
        default_value: Default value for CONTINUE behavior
        max_retries: Maximum retries for RETRY behavior
        retry_delay: Delay between retries in seconds
    """

    behavior: FallbackBehavior = FallbackBehavior.ABORT
    default_value: Optional[Any] = None
    max_retries: int = 3
    retry_delay: float = 5.0

    @classmethod
    def abort(cls) -> "FallbackStrategy":
        """Create an abort strategy (default)."""
        return cls(behavior=FallbackBehavior.ABORT)

    @classmethod
    def continue_with_default(cls, default: Any) -> "FallbackStrategy":
        """Create a continue strategy with default value."""
        return cls(behavior=FallbackBehavior.CONTINUE, default_value=default)

    @classmethod
    def skip(cls) -> "FallbackStrategy":
        """Create a skip strategy."""
        return cls(behavior=FallbackBehavior.SKIP)

    @classmethod
    def retry(cls, max_retries: int = 3, delay: float = 5.0) -> "FallbackStrategy":
        """Create a retry strategy."""
        return cls(behavior=FallbackBehavior.RETRY, max_retries=max_retries, retry_delay=delay)


class HITLResponseProtocol(Protocol):
    """Protocol for HITL gate responses.

    Ensures consistent response structure across all gate types.
    """

    @property
    def is_approved(self) -> bool:
        """Whether the interaction was approved/proceed."""
        ...

    @property
    def value(self) -> Optional[Any]:
        """The value provided by the human."""
        ...

    @property
    def reason(self) -> Optional[str]:
        """Optional reason/explanation for the decision."""
        ...

    @property
    def responder(self) -> Optional[str]:
        """Identity of the responder."""
        ...

    @property
    def created_at(self) -> float:
        """Unix timestamp of when response was created."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize response to dictionary."""
        ...


@runtime_checkable
class InputValidationProtocol(Protocol):
    """Protocol for input validation.

    Validates human input before accepting it.
    """

    async def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate the input value.

        Args:
            value: The value to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        ...


@runtime_checkable
class HITLGateProtocol(Protocol):
    """Protocol for HITL gate implementations.

    All HITL gates must implement this protocol to ensure
    consistent behavior across the framework.
    """

    @property
    def gate_id(self) -> str:
        """Unique identifier for this gate."""
        ...

    @property
    def gate_type(self) -> str:
        """Type identifier for the gate."""
        ...

    @property
    def title(self) -> str:
        """Human-readable title."""
        ...

    @property
    def prompt(self) -> str:
        """Main prompt message."""
        ...

    @property
    def timeout_seconds(self) -> float:
        """Timeout in seconds."""
        ...

    @property
    def fallback_strategy(self) -> FallbackStrategy:
        """Strategy for timeout handling."""
        ...

    @property
    def is_required(self) -> bool:
        """Whether a response is required."""
        ...

    def with_context(self, context: dict[str, Any]) -> "HITLGateProtocol":
        """Create a new gate with additional context."""
        ...

    def with_timeout(self, timeout: float) -> "HITLGateProtocol":
        """Create a new gate with custom timeout."""
        ...

    def with_fallback(self, strategy: FallbackStrategy) -> "HITLGateProtocol":
        """Create a new gate with custom fallback."""
        ...

    async def execute(
        self,
        context: Optional[dict[str, Any]] = None,
        handler: Optional[Callable[..., Awaitable[HITLResponseProtocol]]] = None,
    ) -> HITLResponseProtocol:
        """Execute the gate and wait for response.

        Args:
            context: Optional context for template rendering
            handler: Optional custom handler for the interaction

        Returns:
            The response from the human interaction
        """
        ...


T = TypeVar("T", bound=HITLResponseProtocol)


@dataclass
class BaseHITLResponse:
    """Base implementation of HITL response protocol.

    Attributes:
        gate_id: ID of the gate that generated this response
        approved: Whether the interaction was approved
        value: The value provided
        reason: Optional explanation
        responder: Identity of responder
        created_at: Timestamp
        metadata: Additional metadata
    """

    gate_id: str
    approved: bool
    value: Optional[Any] = None
    reason: Optional[str] = None
    responder: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

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
        value = self.metadata.get("timed_out", False)
        assert isinstance(value, (bool, int))
        return bool(value)

    @property
    def is_skipped(self) -> bool:
        """Whether the interaction was skipped."""
        value = self.metadata.get("skipped", False)
        assert isinstance(value, (bool, int))
        return bool(value)

    def to_dict(self) -> dict[str, Any]:
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


@dataclass
class BaseHITLGate:
    """Base implementation providing common gate functionality.

    Attributes:
        _gate_id: Unique identifier
        gate_type: Type identifier
        title: Human-readable title
        prompt: Main prompt message
        timeout_seconds: Timeout
        fallback_strategy: Fallback strategy
        required: Whether response is required
        context: Additional context for templates
        validator: Optional input validator
    """

    _gate_id: str
    gate_type: str
    title: str
    prompt: str
    timeout_seconds: float = 300.0
    fallback_strategy: FallbackStrategy = field(default_factory=FallbackStrategy.abort)
    required: bool = True
    context: dict[str, Any] = field(default_factory=dict)
    validator: Optional[InputValidationProtocol] = None

    @property
    def gate_id(self) -> str:
        """Unique identifier for this gate."""
        return self._gate_id

    @property
    def is_required(self) -> bool:
        """Whether a response is required."""
        return self.required

    def with_context(self, context: dict[str, Any]) -> "BaseHITLGate":
        """Create a new gate with additional context."""
        new_context = {**self.context, **context}
        return self.__class__(**{**self.__dict__, "context": new_context})

    def with_timeout(self, timeout: float) -> "BaseHITLGate":
        """Create a new gate with custom timeout."""
        return self.__class__(**{**self.__dict__, "timeout_seconds": timeout})

    def with_fallback(self, strategy: FallbackStrategy) -> "BaseHITLGate":
        """Create a new gate with custom fallback."""
        return self.__class__(**{**self.__dict__, "fallback_strategy": strategy})

    def _render_prompt(self, additional_context: Optional[dict[str, Any]] = None) -> str:
        """Render prompt with context variables."""
        from string import Template

        merged_context = {**self.context}
        if additional_context:
            merged_context.update(additional_context)

        try:
            return Template(self.prompt).safe_substitute(merged_context)
        except (KeyError, ValueError):
            # If template rendering fails, return original
            return self.prompt


__all__ = [
    "FallbackBehavior",
    "FallbackStrategy",
    "HITLResponseProtocol",
    "InputValidationProtocol",
    "HITLGateProtocol",
    "BaseHITLResponse",
    "BaseHITLGate",
]
