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

"""Loop detection with context-aware signature generation.

This module provides enhanced loop detection that considers:
- Conversation stage (same operation in different stages is not a loop)
- Tool-specific volatile parameters (pattern variations are exploration)
- Progressive parameters (different queries are intentional exploration)

The key insight is that loops are about *intent*, not just parameters:
- Reading different parts of the same file = exploration
- Listing with different patterns = exploration
- Same operation at different stages = different intent
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from victor.agent.conversation_state_machine import ConversationStage

from victor.tools.tool_names import get_canonical_name

logger = logging.getLogger(__name__)


class OperationPurpose(Enum):
    """The purpose of a tool call for loop detection.

    Different purposes mean the same tool/args aren't loops:
    - EXPLORE: Initial discovery (ls, search, browse)
    - ANALYZE: Understanding specific content (read with offset)
    - MODIFY: Making changes (edit, write)
    - VERIFY: Checking results (read after edit)
    """

    EXPLORE = "explore"
    ANALYZE = "analyze"
    MODIFY = "modify"
    VERIFY = "verify"


@dataclass
class LoopContext:
    """Context for loop-aware signature generation.

    Attributes:
        stage: Current conversation stage
        purpose: Purpose of the operation (inferred if not provided)
        previous_milestones: Milestones achieved so far
    """

    stage: Optional[Any] = None
    purpose: Optional[OperationPurpose] = None
    previous_milestones: Set[str] = field(default_factory=set)

    @classmethod
    def from_stage(cls, stage: Any, milestones: Optional[Set[str]] = None) -> "LoopContext":
        """Create context from conversation stage.

        Args:
            stage: Conversation stage enum
            milestones: Set of achieved milestones

        Returns:
            LoopContext instance
        """
        return cls(stage=stage, previous_milestones=milestones or set())


class LoopSignature:
    """Generate context-aware signatures for loop detection.

    The signature generation is tool-specific to avoid false positives:
    - ls tool: pattern variations are exploration (excluded from signature)
    - read tool: offset/limit variations are exploration (excluded from signature)
    - search tool: different queries are exploration (only path matters)

    This allows intentional exploration while catching actual loops.
    """

    # Volatile parameters by tool (excluded from signatures)
    VOLATILE_BY_TOOL: Dict[str, Set[str]] = {
        "read": {"offset", "limit", "line_start", "line_end"},
        "ls": {"pattern", "recursive", "depth", "limit"},  # Pattern variations = exploration
        "search": {"offset", "limit"},
        "grep": {"offset", "limit"},
        "code_search": {"offset", "limit"},
        "glob": {"pattern"},  # Different patterns are intentional
    }

    @staticmethod
    def generate(
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[LoopContext] = None,
    ) -> str:
        """Generate a context-aware loop detection signature.

        Args:
            tool_name: Name of the tool being called
            arguments: Tool call arguments
            context: Optional loop context for additional awareness

        Returns:
            Signature string for loop detection (short, hash-like)
        """
        # Get stable arguments (exclude volatile params)
        stable_args = LoopSignature._get_stable_args(tool_name, arguments)

        # Build base signature
        sig_parts = [tool_name]

        # Add non-empty stable args
        for key, value in sorted(stable_args.items()):
            if value not in (None, "", [], {}):
                # Truncate long values
                if isinstance(value, str) and len(value) > 50:
                    value = value[:50]
                sig_parts.append(f"{key}={value}")

        base_sig = "|".join(sig_parts)

        # Add context qualifier if available
        if context:
            if context.purpose:
                base_sig += f"|purpose:{context.purpose.value}"
            if context.stage:
                # Handle both enum and string stages
                stage_val = (
                    context.stage.value if hasattr(context.stage, "value") else str(context.stage)
                )
                base_sig += f"|stage:{stage_val}"

        # Hash for compact signature (but keep first part readable for debugging)
        readable_part = base_sig[:80] if len(base_sig) > 80 else base_sig
        # MD5 used for loop detection signature, not security
        hash_part = hashlib.md5(base_sig.encode(), usedforsecurity=False).hexdigest()[:8]

        return f"{readable_part}|{hash_part}"

    @staticmethod
    def _get_stable_args(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get stable arguments excluding volatile fields.

        Args:
            tool_name: Name of the tool
            arguments: All arguments

        Returns:
            Filtered arguments dict
        """
        # Normalize tool name to canonical form for lookup
        canonical_name = get_canonical_name(tool_name)

        # Get volatile fields for this tool
        volatile = LoopSignature.VOLATILE_BY_TOOL.get(canonical_name, set())

        # Also exclude universal volatile fields
        universal_volatile = {"timeout", "retry", "verbose"}
        volatile = volatile | universal_volatile

        return {k: v for k, v in arguments.items() if k not in volatile}

    @staticmethod
    def infer_purpose(tool_name: str, arguments: Dict[str, Any], stage: Any) -> OperationPurpose:
        """Infer the purpose of a tool call from context.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            stage: Current conversation stage

        Returns:
            Inferred operation purpose
        """
        # Get canonical tool name
        canonical_name = tool_name.lower()

        # Modify tools
        if canonical_name in ("edit", "write", "create", "delete"):
            return OperationPurpose.MODIFY

        # Verify tools (checking results)
        if canonical_name in ("read",) and stage:
            stage_val = stage.value if hasattr(stage, "value") else str(stage)
            if "verif" in stage_val.lower() or "test" in stage_val.lower():
                return OperationPurpose.VERIFY

        # Explore tools
        if canonical_name in ("ls", "glob", "search", "grep", "code_search"):
            return OperationPurpose.EXPLORE

        # Default to analyze
        return OperationPurpose.ANALYZE


# Factory function for DI registration
def create_loop_signature() -> LoopSignature:
    """Create a LoopSignature instance for DI registration.

    Returns:
        New LoopSignature instance
    """
    return LoopSignature()


__all__ = [
    "LoopSignature",
    "LoopContext",
    "OperationPurpose",
    "create_loop_signature",
]
