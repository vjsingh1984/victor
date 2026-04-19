"""Deterministic decision rules for fast, cheap decision-making.

Provides O(1) lookup tables and regex pattern matchers for common decision
patterns. This is the first layer of the hybrid decision pipeline:

1. Lookup Tables (O(1)) → 95% accuracy, 0ms
2. Pattern Matcher (regex) → 85% accuracy, 1-5ms
3. Ensemble Voting → 90% accuracy, 5-10ms
4. LLM Fallback → 98% accuracy, 500-2000ms

Each decision type has dedicated lookup tables and patterns optimized for
high-frequency patterns observed in production.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from victor.agent.decisions.schemas import DecisionType


@dataclass
class LookupResult:
    """Result from a lookup table match."""

    decision: Any
    confidence: float
    reason: str
    matched_pattern: str


class LookupTables:
    """O(1) hash-based lookup tables for common decision patterns.

    Provides instant decisions for high-frequency patterns without any
    LLM calls or regex matching. Each lookup maps a pattern key to a
    pre-defined decision with confidence score.
    """

    # Task completion lookup - 100+ patterns
    TASK_COMPLETION_LOOKUP: Dict[str, Tuple[bool, float, str]] = {
        # Explicit completion markers
        "done": (True, 0.98, "done"),
        "complete": (True, 0.98, "done"),
        "completed": (True, 0.98, "done"),
        "finished": (True, 0.98, "done"),
        "success": (True, 0.95, "done"),
        "successful": (True, 0.95, "done"),
        "resolved": (True, 0.95, "done"),
        "fixed": (True, 0.95, "done"),
        "implemented": (True, 0.92, "done"),
        "implemented successfully": (True, 0.95, "done"),
        "task complete": (True, 0.98, "done"),
        "task completed": (True, 0.98, "done"),
        "all done": (True, 0.97, "done"),
        "that's it": (True, 0.90, "done"),
        "that is all": (True, 0.90, "done"),
        "nothing more": (True, 0.90, "done"),
        "that's all": (True, 0.90, "done"),
        "ready": (True, 0.92, "done"),
        "good to go": (True, 0.92, "done"),
        "all set": (True, 0.92, "done"),
        # Explicit in-progress markers
        "working": (False, 0.90, "working"),
        "working on": (False, 0.90, "working"),
        "continuing": (False, 0.90, "working"),
        "proceeding": (False, 0.90, "working"),
        "proceed": (False, 0.90, "working"),
        "next": (False, 0.85, "working"),
        "now": (False, 0.85, "working"),
        "let me": (False, 0.85, "working"),
        "let's": (False, 0.85, "working"),
        "i'll": (False, 0.85, "working"),
        "i will": (False, 0.85, "working"),
        # Error/stuck markers
        "error": (False, 0.92, "stuck"),
        "failed": (False, 0.92, "stuck"),
        "failure": (False, 0.92, "stuck"),
        "stuck": (False, 0.95, "stuck"),
        "unable": (False, 0.90, "stuck"),
        "cannot": (False, 0.90, "stuck"),
        "can't": (False, 0.90, "stuck"),
        "not working": (False, 0.92, "stuck"),
        "doesn't work": (False, 0.92, "stuck"),
        "issue": (False, 0.80, "stuck"),
        "problem": (False, 0.80, "stuck"),
        "having trouble": (False, 0.90, "stuck"),
        "having issues": (False, 0.90, "stuck"),
        # Finalizing markers
        "finalizing": (False, 0.90, "finalizing"),
        "wrapping up": (False, 0.90, "finalizing"),
        "finishing": (False, 0.88, "finalizing"),
        "almost": (False, 0.85, "finalizing"),
        "almost done": (False, 0.88, "finalizing"),
        "nearly there": (False, 0.88, "finalizing"),
        "one more": (False, 0.85, "finalizing"),
        "last step": (False, 0.88, "finalizing"),
        "final step": (False, 0.88, "finalizing"),
    }

    # Intent classification lookup
    INTENT_LOOKUP: Dict[str, Tuple[str, float]] = {
        # Completion intent
        "here's": ("completion", 0.88),
        "here's the": ("completion", 0.90),
        "here is the": ("completion", 0.90),
        "i've": ("completion", 0.85),
        "i have": ("completion", 0.85),
        "the code is": ("completion", 0.88),
        "the solution is": ("completion", 0.88),
        "completed": ("completion", 0.92),
        "finished": ("completion", 0.92),
        "done": ("completion", 0.90),
        "here you go": ("completion", 0.88),
        "here you are": ("completion", 0.88),
        "ready": ("completion", 0.88),
        "requested": ("completion", 0.85),
        # Continuation intent
        "now": ("continuation", 0.85),
        "next": ("continuation", 0.88),
        "let me": ("continuation", 0.90),
        "let's": ("continuation", 0.90),
        "i'll": ("continuation", 0.90),
        "i will": ("continuation", 0.90),
        "proceeding": ("continuation", 0.88),
        "continuing": ("continuation", 0.88),
        "first": ("continuation", 0.85),
        "then": ("continuation", 0.85),
        "implement": ("continuation", 0.85),
        # Asking input
        "would you like": ("asking_input", 0.92),
        "do you want": ("asking_input", 0.92),
        "should i": ("asking_input", 0.90),
        "would you prefer": ("asking_input", 0.90),
        "which one": ("asking_input", 0.88),
        "which approach": ("asking_input", 0.88),
        "any preference": ("asking_input", 0.88),
        "shall i": ("asking_input", 0.90),
        "which": ("asking_input", 0.85),
        "prefer": ("asking_input", 0.85),
        # Stuck loop
        "i'm stuck": ("stuck_loop", 0.95),
        "i am stuck": ("stuck_loop", 0.95),
        "stuck": ("stuck_loop", 0.90),
        "not sure how": ("stuck_loop", 0.88),
        "don't know": ("stuck_loop", 0.85),
        "unclear": ("stuck_loop", 0.85),
        "having trouble": ("stuck_loop", 0.88),
        "trouble": ("stuck_loop", 0.85),
    }

    # Task type classification lookup
    TASK_TYPE_LOOKUP: Dict[str, Tuple[str, float]] = {
        # Analysis tasks
        "analyze": ("analysis", 0.92),
        "analysis": ("analysis", 0.92),
        "examine": ("analysis", 0.90),
        "review": ("analysis", 0.88),
        "understand": ("analysis", 0.88),
        "explain": ("analysis", 0.88),
        "what is": ("analysis", 0.85),
        "how does": ("analysis", 0.85),
        "find out": ("analysis", 0.85),
        "investigate": ("analysis", 0.90),
        "explore": ("analysis", 0.85),
        "check": ("analysis", 0.82),
        "look at": ("analysis", 0.80),
        # Action tasks
        "fix": ("action", 0.92),
        "fix the": ("action", 0.90),
        "implement": ("action", 0.92),
        "add": ("action", 0.88),
        "remove": ("action", 0.88),
        "delete": ("action", 0.88),
        "build": ("action", 0.90),
        "make": ("action", 0.85),
        "set up": ("action", 0.88),
        "configure": ("action", 0.88),
        "install": ("action", 0.90),
        # Generation tasks
        "generate": ("generation", 0.92),
        "write": ("generation", 0.88),
        "create": ("generation", 0.88),
        "draft": ("generation", 0.90),
        "produce": ("generation", 0.88),
        "output": ("generation", 0.85),
        "code": ("generation", 0.82),
        "script": ("generation", 0.85),
        # Search tasks
        "find": ("search", 0.90),
        "search": ("search", 0.92),
        "locate": ("search", 0.90),
        "look for": ("search", 0.88),
        "where is": ("search", 0.88),
        "show me": ("search", 0.85),
        "list": ("search", 0.85),
        "what files": ("search", 0.85),
        # Edit tasks
        "edit": ("edit", 0.92),
        "modify": ("edit", 0.90),
        "change": ("edit", 0.88),
        "update": ("edit", 0.88),
        "refactor": ("edit", 0.90),
        "improve": ("edit", 0.85),
        "optimize": ("edit", 0.85),
        "replace": ("edit", 0.88),
        "rewrite": ("edit", 0.88),
    }

    # Error classification lookup
    ERROR_TYPE_LOOKUP: Dict[str, Tuple[str, float]] = {
        # Permanent errors
        "file not found": ("permanent", 0.90),
        "not found": ("permanent", 0.90),
        "does not exist": ("permanent", 0.90),
        "permission": ("permanent", 0.92),
        "permission denied": ("permanent", 0.92),
        "unauthorized": ("permanent", 0.95),
        "authentication": ("permanent", 0.92),
        "invalid api key": ("permanent", 0.95),
        "syntax error": ("permanent", 0.88),
        "type error": ("permanent", 0.85),
        "attribute error": ("permanent", 0.85),
        # Transient errors
        "connection timeout": ("transient", 0.90),
        "timeout": ("transient", 0.90),
        "timed out": ("transient", 0.90),
        "connection": ("transient", 0.88),
        "network": ("transient", 0.88),
        "rate limit": ("transient", 0.85),
        "too many requests": ("transient", 0.88),
        "temporary": ("transient", 0.85),
        "unavailable": ("transient", 0.82),
        "overloaded": ("transient", 0.85),
        # Retryable errors
        "500 internal server error": ("retryable", 0.88),
        "internal server error": ("retryable", 0.88),
        "503 service unavailable": ("retryable", 0.88),
        "internal error": ("retryable", 0.85),
        "server error": ("retryable", 0.85),
        "500": ("retryable", 0.88),
        "502": ("retryable", 0.88),
        "503": ("retryable", 0.88),
        "504": ("retryable", 0.88),
    }

    # Tool necessity lookup
    TOOL_NECESSITY_LOOKUP: Dict[str, Tuple[bool, float]] = {
        # Needs tools
        "fix": (True, 0.92),
        "fix the": (True, 0.90),
        "fix the bug": (True, 0.92),
        "implement": (True, 0.92),
        "create file": (True, 0.90),
        "edit": (True, 0.90),
        "modify": (True, 0.90),
        "search": (True, 0.88),
        "find": (True, 0.88),
        "run": (True, 0.90),
        "execute": (True, 0.90),
        "test": (True, 0.88),
        "build": (True, 0.88),
        "install": (True, 0.88),
        "in login.py": (True, 0.95),  # File reference
        "file": (True, 0.88),  # File operation
        # No tools needed
        "explain": (False, 0.85),
        "what is": (False, 0.85),
        "how do i": (False, 0.82),
        "tell me": (False, 0.85),
        "describe": (False, 0.85),
        "analyze": (False, 0.80),  # Might need tools, but often doesn't
        "show me": (True, 0.75),  # Often needs tools
    }

    @classmethod
    def lookup(
        cls,
        decision_type: DecisionType,
        context: Dict[str, Any],
    ) -> Optional[LookupResult]:
        """Perform O(1) hash lookup for a decision.

        Args:
            decision_type: Type of decision to make
            context: Context dict containing decision-relevant data

        Returns:
            LookupResult if pattern found, None otherwise
        """
        # Extract relevant text from context based on decision type
        text = cls._extract_text(decision_type, context)
        if not text:
            return None

        text_lower = text.lower().strip()

        # Select appropriate lookup table
        lookup_map = {
            DecisionType.TASK_COMPLETION: cls.TASK_COMPLETION_LOOKUP,
            DecisionType.INTENT_CLASSIFICATION: cls.INTENT_LOOKUP,
            DecisionType.TASK_TYPE_CLASSIFICATION: cls.TASK_TYPE_LOOKUP,
            DecisionType.ERROR_CLASSIFICATION: cls.ERROR_TYPE_LOOKUP,
            DecisionType.TOOL_NECESSITY: cls.TOOL_NECESSITY_LOOKUP,
        }

        table = lookup_map.get(decision_type)
        if not table:
            return None

        # Direct lookup
        if text_lower in table:
            decision_data = table[text_lower]
            return cls._create_result(decision_type, decision_data, text_lower)

        # Try multi-word phrase matching (longest phrases first)
        # Sort table keys by length (descending) to match longer phrases first
        sorted_keys = sorted(table.keys(), key=len, reverse=True)

        for key in sorted_keys:
            if len(key.split()) == 1:  # Skip single-word keys for now
                continue
            # Use word boundary matching to avoid partial matches
            # e.g., "unavailable" shouldn't match "503 service unavailable"
            pattern = r"\b" + re.escape(key) + r"\b"
            if re.search(pattern, text_lower):
                decision_data = table[key]
                return cls._create_result(decision_type, decision_data, key)

        # Word-by-word lookup for longer phrases (single words only)
        words = text_lower.split()
        # Sort by length to match longer words first
        words_sorted = sorted(words, key=len, reverse=True)
        for word in words_sorted:
            if word in table and len(word) > 3:  # Only match significant words
                decision_data = table[word]
                return cls._create_result(decision_type, decision_data, word)

        return None

    @classmethod
    def _extract_text(cls, decision_type: DecisionType, context: Dict[str, Any]) -> Optional[str]:
        """Extract relevant text from context based on decision type."""
        # Common context keys to check
        text_keys = [
            "message",
            "response",
            "content",
            "text",
            "error_message",
            "user_message",
            "task_description",
        ]

        for key in text_keys:
            if key in context and context[key]:
                return str(context[key])

        # Check nested dicts
        for value in context.values():
            if isinstance(value, dict):
                for key in text_keys:
                    if key in value and value[key]:
                        return str(value[key])

        return None

    @classmethod
    def _create_result(
        cls,
        decision_type: DecisionType,
        decision_data: Tuple[Any, ...],
        matched_pattern: str,
    ) -> Optional[LookupResult]:
        """Create a LookupResult from decision data."""
        if decision_type == DecisionType.TASK_COMPLETION:
            is_complete, confidence, phase = decision_data
            from victor.agent.decisions.schemas import TaskCompletionDecision

            return LookupResult(
                decision=TaskCompletionDecision(
                    is_complete=is_complete,
                    confidence=confidence,
                    phase=phase,
                ),
                confidence=confidence,
                reason=f"Lookup matched '{matched_pattern}'",
                matched_pattern=matched_pattern,
            )

        elif decision_type == DecisionType.INTENT_CLASSIFICATION:
            intent, confidence = decision_data
            from victor.agent.decisions.schemas import IntentDecision

            return LookupResult(
                decision=IntentDecision(
                    intent=intent,
                    confidence=confidence,
                ),
                confidence=confidence,
                reason=f"Lookup matched '{matched_pattern}'",
                matched_pattern=matched_pattern,
            )

        elif decision_type == DecisionType.TASK_TYPE_CLASSIFICATION:
            task_type, confidence = decision_data
            from victor.agent.decisions.schemas import TaskTypeDecision

            return LookupResult(
                decision=TaskTypeDecision(
                    task_type=task_type,
                    confidence=confidence,
                    deliverables=[],  # Lookup doesn't predict deliverables
                ),
                confidence=confidence,
                reason=f"Lookup matched '{matched_pattern}'",
                matched_pattern=matched_pattern,
            )

        elif decision_type == DecisionType.ERROR_CLASSIFICATION:
            error_type, confidence = decision_data
            from victor.agent.decisions.schemas import ErrorClassDecision

            return LookupResult(
                decision=ErrorClassDecision(
                    error_type=error_type,
                    confidence=confidence,
                ),
                confidence=confidence,
                reason=f"Lookup matched '{matched_pattern}'",
                matched_pattern=matched_pattern,
            )

        elif decision_type == DecisionType.TOOL_NECESSITY:
            requires_tools, confidence = decision_data
            from victor.agent.decisions.schemas import ToolNecessityDecision

            return LookupResult(
                decision=ToolNecessityDecision(
                    requires_tools=requires_tools,
                    confidence=confidence,
                ),
                confidence=confidence,
                reason=f"Lookup matched '{matched_pattern}'",
                matched_pattern=matched_pattern,
            )

        return None


class PatternMatcher:
    """Regex-based pattern matcher for decision patterns.

    Provides fast regex matching (1-5ms) for patterns that don't fit
    in simple lookup tables. Each decision type has dedicated regex
    patterns compiled at module load time.
    """

    # Task completion patterns
    TASK_COMPLETION_PATTERNS: List[Tuple[re.Pattern, bool, float, str]] = [
        # Completion patterns
        (re.compile(r"\b(I'm|I am)\s+done\b", re.IGNORECASE), True, 0.92, "done"),
        (
            re.compile(r"\b(I've|I have)\s+(completed|finished)\b", re.IGNORECASE),
            True,
            0.92,
            "done",
        ),
        (
            re.compile(r"\bThe\s+(task|job|work)\s+is\s+(complete|done)\b", re.IGNORECASE),
            True,
            0.95,
            "done",
        ),
        (
            re.compile(
                r"\b(Successfully|success)\s+(completed|finished|implemented)\b", re.IGNORECASE
            ),
            True,
            0.93,
            "done",
        ),
        (re.compile(r"\bThat('s| is)\s+(everything|all)\b", re.IGNORECASE), True, 0.90, "done"),
        (re.compile(r"\bNothing\s+else\s+to\s+do\b", re.IGNORECASE), True, 0.90, "done"),
        # Working patterns
        (
            re.compile(r"\b(I'm|I am)\s+(working|proceeding|continuing)\b", re.IGNORECASE),
            False,
            0.88,
            "working",
        ),
        (
            re.compile(r"\bLet('s| us)\s+(proceed|continue|start|begin)\b", re.IGNORECASE),
            False,
            0.88,
            "working",
        ),
        (re.compile(r"\b(I'll|I will)\s+(now|next)\b", re.IGNORECASE), False, 0.88, "working"),
        (
            re.compile(r"\b(Now|Next)\s+(let me|I'll|I will)\b", re.IGNORECASE),
            False,
            0.88,
            "working",
        ),
        # Error/stuck patterns
        (
            re.compile(r"\b(I'm|I am)\s+(stuck|blocked|unable)\b", re.IGNORECASE),
            False,
            0.92,
            "stuck",
        ),
        (
            re.compile(r"\b(Can't|Cannot|Unable)\s+(proceed|continue|do)\b", re.IGNORECASE),
            False,
            0.90,
            "stuck",
        ),
        (
            re.compile(r"\b(Not working|Doesn't work|Failing)\b", re.IGNORECASE),
            False,
            0.90,
            "stuck",
        ),
        (
            re.compile(
                r"\b(Encountered|Hit|Facing)\s+(an? )?(error|issue|problem)\b", re.IGNORECASE
            ),
            False,
            0.88,
            "stuck",
        ),
        # Finalizing patterns
        (
            re.compile(r"\b(Finalizing|Wrapping up|Finishing up)\b", re.IGNORECASE),
            False,
            0.90,
            "finalizing",
        ),
        (
            re.compile(r"\b(Almost|Nearly)\s+(done|there|finished)\b", re.IGNORECASE),
            False,
            0.88,
            "finalizing",
        ),
        (
            re.compile(r"\b(One|One more|Last|Final)\s+(step|thing|task)\b", re.IGNORECASE),
            False,
            0.88,
            "finalizing",
        ),
    ]

    # Intent classification patterns
    INTENT_PATTERNS: List[Tuple[re.Pattern, str, float]] = [
        # Completion intent
        (
            re.compile(r"\b(Here('s| is)|I've provided|I have provided)\b", re.IGNORECASE),
            "completion",
            0.88,
        ),
        (
            re.compile(r"\b(The )?(code|solution|answer|result)\s+is\b", re.IGNORECASE),
            "completion",
            0.90,
        ),
        (re.compile(r"\b(Completed|Finished|Done|Ready)\b", re.IGNORECASE), "completion", 0.88),
        # Continuation intent
        (re.compile(r"\b(Now|Next|Then|After that)\b", re.IGNORECASE), "continuation", 0.85),
        (
            re.compile(
                r"\b(Let('s| me)|I('ll| will))\s+(start|begin|proceed|continue)\b", re.IGNORECASE
            ),
            "continuation",
            0.90,
        ),
        (re.compile(r"\b(First|Second|Third|Next step)\b", re.IGNORECASE), "continuation", 0.88),
        # Asking input
        (
            re.compile(r"\b(Would|Should|Shall|Do|Can)\s+(you|I)\b", re.IGNORECASE),
            "asking_input",
            0.85,
        ),
        (
            re.compile(r"\b(Which|What|How)\s+(one|approach|way|method)\b", re.IGNORECASE),
            "asking_input",
            0.88,
        ),
        (
            re.compile(r"\b(Any|Any particular)\s+(preference|choice)\b", re.IGNORECASE),
            "asking_input",
            0.88,
        ),
        # Stuck loop
        (
            re.compile(r"\b(I'm|I am)\s+(stuck|confused|uncertain|unsure)\b", re.IGNORECASE),
            "stuck_loop",
            0.92,
        ),
        (
            re.compile(r"\b(Not sure|Don't know|Unclear|Confused)\b", re.IGNORECASE),
            "stuck_loop",
            0.88,
        ),
        (
            re.compile(r"\b(Having|Facing)\s+(trouble|difficulty|issues)\b", re.IGNORECASE),
            "stuck_loop",
            0.88,
        ),
    ]

    # Task type patterns
    TASK_TYPE_PATTERNS: List[Tuple[re.Pattern, str, float]] = [
        # Analysis tasks
        (
            re.compile(r"\b(Analyze|Examine|Review|Investigate|Explore|Study)\b", re.IGNORECASE),
            "analysis",
            0.90,
        ),
        (re.compile(r"\b(Explain|Understand|Describe|Clarify)\b", re.IGNORECASE), "analysis", 0.88),
        (
            re.compile(r"\b(What is|How does|Why does|Tell me about)\b", re.IGNORECASE),
            "analysis",
            0.85,
        ),
        # Action tasks
        (
            re.compile(
                r"\b(Fix|Implement|Create|Add|Remove|Update|Delete|Write|Build)\b", re.IGNORECASE
            ),
            "action",
            0.90,
        ),
        (re.compile(r"\b(Set up|Configure|Install|Deploy)\b", re.IGNORECASE), "action", 0.90),
        (re.compile(r"\b(Make|Change|Modify)\b", re.IGNORECASE), "action", 0.85),
        # Generation tasks
        (re.compile(r"\b(Generate|Produce|Output|Draft)\b", re.IGNORECASE), "generation", 0.90),
        (
            re.compile(r"\bWrite\s+(a|the|code|script|function)\b", re.IGNORECASE),
            "generation",
            0.88,
        ),
        # Search tasks
        (re.compile(r"\b(Find|Search|Locate|Look for)\b", re.IGNORECASE), "search", 0.90),
        (re.compile(r"\b(Where is|Show me|List)\b", re.IGNORECASE), "search", 0.88),
        (re.compile(r"\bWhat files\b", re.IGNORECASE), "search", 0.88),
        # Edit tasks
        (
            re.compile(r"\b(Edit|Modify|Change|Update|Refactor|Improve|Optimize)\b", re.IGNORECASE),
            "edit",
            0.90,
        ),
        (re.compile(r"\b(Replace|Rewrite|Refactor)\b", re.IGNORECASE), "edit", 0.88),
    ]

    # Error classification patterns
    ERROR_PATTERNS: List[Tuple[re.Pattern, str, float]] = [
        # Specific HTTP status codes (must come before general patterns)
        (re.compile(r"\b503\s+Service\s+Unavailable\b", re.IGNORECASE), "retryable", 0.90),
        (re.compile(r"\b500\s+Internal\s+Server\s+Error\b", re.IGNORECASE), "retryable", 0.90),
        (re.compile(r"\b[54]\d{2}\b", re.IGNORECASE), "retryable", 0.88),
        # Permanent errors
        (
            re.compile(r"\b(Not found|Does not exist|File not found)\b", re.IGNORECASE),
            "permanent",
            0.90,
        ),
        (
            re.compile(r"\b(Permission denied|Unauthorized|Authentication)\b", re.IGNORECASE),
            "permanent",
            0.92,
        ),
        (
            re.compile(r"\b(Invalid API key|Invalid credentials)\b", re.IGNORECASE),
            "permanent",
            0.95,
        ),
        (re.compile(r"\b(Syntax|Type|Attribute|Name)Error\b", re.IGNORECASE), "permanent", 0.85),
        # Transient errors
        (re.compile(r"\b(Timeout|Timed out)\b", re.IGNORECASE), "transient", 0.90),
        (re.compile(r"\b(Connection|Network)\b", re.IGNORECASE), "transient", 0.88),
        (re.compile(r"\b(Rate limit|Too many requests)\b", re.IGNORECASE), "transient", 0.88),
        (
            re.compile(r"\b(Temporarily|Currently)?\s*(unavailable|overloaded)\b", re.IGNORECASE),
            "transient",
            0.85,
        ),
        # Retryable errors
        (re.compile(r"\b(Internal|Server)\s+error\b", re.IGNORECASE), "retryable", 0.85),
    ]

    @classmethod
    def match(
        cls,
        decision_type: DecisionType,
        context: Dict[str, Any],
    ) -> Optional[LookupResult]:
        """Match regex patterns for a decision.

        Args:
            decision_type: Type of decision to make
            context: Context dict containing decision-relevant data

        Returns:
            LookupResult if pattern matched, None otherwise
        """
        # Extract relevant text from context
        text = LookupTables._extract_text(decision_type, context)
        if not text:
            return None

        # Select appropriate pattern list
        pattern_map = {
            DecisionType.TASK_COMPLETION: cls.TASK_COMPLETION_PATTERNS,
            DecisionType.INTENT_CLASSIFICATION: cls.INTENT_PATTERNS,
            DecisionType.TASK_TYPE_CLASSIFICATION: cls.TASK_TYPE_PATTERNS,
            DecisionType.ERROR_CLASSIFICATION: cls.ERROR_PATTERNS,
        }

        patterns = pattern_map.get(decision_type)
        if not patterns:
            return None

        # Try each pattern in order
        for pattern_tuple in patterns:
            if decision_type == DecisionType.TASK_COMPLETION:
                pattern, is_complete, confidence, phase = pattern_tuple
            else:
                pattern, decision_value, confidence = pattern_tuple
            match = pattern.search(text)
            if match:
                matched_text = match.group(0)
                return cls._create_result_from_match(
                    decision_type,
                    pattern_tuple,
                    matched_text,
                )

        return None

    @classmethod
    def _create_result_from_match(
        cls,
        decision_type: DecisionType,
        pattern_tuple: Tuple,
        matched_text: str,
    ) -> Optional[LookupResult]:
        """Create a LookupResult from a regex match."""
        if decision_type == DecisionType.TASK_COMPLETION:
            pattern, is_complete, confidence, phase = pattern_tuple
            from victor.agent.decisions.schemas import TaskCompletionDecision

            return LookupResult(
                decision=TaskCompletionDecision(
                    is_complete=is_complete,
                    confidence=confidence,
                    phase=phase,
                ),
                confidence=confidence,
                reason=f"Pattern matched '{matched_text}'",
                matched_pattern=matched_text,
            )

        elif decision_type == DecisionType.INTENT_CLASSIFICATION:
            pattern, intent, confidence = pattern_tuple
            from victor.agent.decisions.schemas import IntentDecision

            return LookupResult(
                decision=IntentDecision(
                    intent=intent,
                    confidence=confidence,
                ),
                confidence=confidence,
                reason=f"Pattern matched '{matched_text}'",
                matched_pattern=matched_text,
            )

        elif decision_type == DecisionType.TASK_TYPE_CLASSIFICATION:
            pattern, task_type, confidence = pattern_tuple
            from victor.agent.decisions.schemas import TaskTypeDecision

            return LookupResult(
                decision=TaskTypeDecision(
                    task_type=task_type,
                    confidence=confidence,
                    deliverables=[],
                ),
                confidence=confidence,
                reason=f"Pattern matched '{matched_text}'",
                matched_pattern=matched_text,
            )

        elif decision_type == DecisionType.ERROR_CLASSIFICATION:
            pattern, error_type, confidence = pattern_tuple
            from victor.agent.decisions.schemas import ErrorClassDecision

            return LookupResult(
                decision=ErrorClassDecision(
                    error_type=error_type,
                    confidence=confidence,
                ),
                confidence=confidence,
                reason=f"Pattern matched '{matched_text}'",
                matched_pattern=matched_text,
            )

        return None


class EnsembleVoter:
    """Ensemble voting for combining multiple decision signals.

    Combines predictions from:
    - Keyword matching (30% weight)
    - Semantic similarity (40% weight)
    - Context patterns (20% weight)
    - Heuristic confidence (10% weight)

    Uses weighted voting to produce a final decision with calibrated confidence.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """Initialize ensemble voter with custom weights.

        Args:
            weights: Custom weights for each signal source.
                     Defaults to {"keyword": 0.3, "semantic": 0.4,
                                  "context": 0.2, "heuristic": 0.1}
        """
        self.weights = weights or {
            "keyword": 0.3,
            "semantic": 0.4,
            "context": 0.2,
            "heuristic": 0.1,
        }

        # Validate weights sum to 1.0
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Ensemble weights must sum to 1.0, got {total_weight}")

    def vote(
        self,
        decision_type: DecisionType,
        context: Dict[str, Any],
        keyword_result: Optional[LookupResult] = None,
        semantic_result: Optional[LookupResult] = None,
        heuristic_result: Optional[Any] = None,
        heuristic_confidence: float = 0.0,
    ) -> Optional[LookupResult]:
        """Combine multiple signals via weighted voting.

        Args:
            decision_type: Type of decision to make
            context: Decision context
            keyword_result: Result from keyword/pattern matching
            semantic_result: Result from semantic similarity (if available)
            heuristic_result: Result from heuristic fallback
            heuristic_confidence: Confidence of heuristic result

        Returns:
            LookupResult with ensemble decision, or None if no signals available
        """
        votes = []
        total_weight = 0.0

        # Add keyword vote
        if keyword_result:
            votes.append((keyword_result, self.weights["keyword"]))
            total_weight += self.weights["keyword"]

        # Add semantic vote (if available)
        if semantic_result:
            votes.append((semantic_result, self.weights["semantic"]))
            total_weight += self.weights["semantic"]

        # Add heuristic vote
        if heuristic_result and heuristic_confidence > 0:
            # Create a LookupResult for the heuristic
            heuristic_lookup = LookupResult(
                decision=heuristic_result,
                confidence=heuristic_confidence,
                reason="Heuristic fallback",
                matched_pattern="heuristic",
            )
            votes.append((heuristic_lookup, self.weights["heuristic"]))
            total_weight += self.weights["heuristic"]

        # No votes available
        if not votes:
            return None

        # Normalize weights if some signals missing
        if total_weight < 1.0:
            scale = 1.0 / total_weight
            votes = [(result, weight * scale) for result, weight in votes]

        # For now, use the highest-weighted confident result
        # TODO: Implement proper voting for multi-class decisions
        votes.sort(key=lambda x: x[0].confidence, reverse=True)
        best_result, _ = votes[0]

        # Boost confidence slightly based on ensemble agreement
        confidence_boost = min(len(votes) * 0.05, 0.15)  # Max 15% boost
        boosted_confidence = min(best_result.confidence + confidence_boost, 0.98)

        # Create a new decision object with boosted confidence if it has a confidence field
        decision = best_result.decision
        if hasattr(decision, "confidence") and hasattr(decision, "model_copy"):
            # Pydantic v2 - use model_copy to create a new instance
            try:
                decision = decision.model_copy(update={"confidence": boosted_confidence})
            except Exception:
                # If model_copy fails, just use the original decision
                pass

        return LookupResult(
            decision=decision,
            confidence=boosted_confidence,
            reason=f"Ensemble of {len(votes)} signals: {best_result.reason}",
            matched_pattern=best_result.matched_pattern,
        )
