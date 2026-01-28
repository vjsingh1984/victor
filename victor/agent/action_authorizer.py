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

"""Action Authorization via Intent Detection.

This module determines whether user requests authorize file modifications:
- File writes when user only asked to "show" or "create" code
- Destructive operations without explicit consent

Purpose: Action Authorization (can the agent write files?)
Note: For model continuation detection (is the model done?),
      see victor.embeddings.intent_classifier (IntentClassifier).

Design Principles:
- Default to safe (display-only) when intent is ambiguous
- Explicit write signals required for file modifications
- Extensible pattern system for custom intents
- Clear audit trail for authorization decisions

Aliases: ActionAuthorizer = IntentDetector (for semantic clarity)
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Pattern, Set, Tuple

logger = logging.getLogger(__name__)


class ActionIntent(Enum):
    """User intent regarding actions.

    Values:
        DISPLAY_ONLY: User wants to see output but not modify files
        WRITE_ALLOWED: User explicitly requested file modifications
        AMBIGUOUS: Intent unclear, requires clarification or safe default
        READ_ONLY: User is only asking for information, no generation
    """

    DISPLAY_ONLY = "display_only"
    WRITE_ALLOWED = "write_allowed"
    AMBIGUOUS = "ambiguous"
    READ_ONLY = "read_only"


# =============================================================================
# Metadata-Based Authorization
# =============================================================================
# Hard-coded tool lists removed in v0.5.1
# All tools now use ToolAuthMetadataRegistry for authorization
# See victor/tools/auth_metadata.py for metadata definitions


def get_metadata_blocked_tools(intent: ActionIntent) -> frozenset[str]:
    """Get blocked tools using metadata-based authorization.

    Phase 5: Metadata-based authorization replaces hard-coded tool lists.
    This function uses ToolAuthMetadataRegistry to determine which tools
    should be blocked for a given intent.

    Args:
        intent: User intent (ActionIntent enum)

    Returns:
        Frozenset of tool names that are blocked for the intent

    Raises:
        ImportError: If MetadataActionAuthorizer is not available

    Example:
        blocked_tools = get_metadata_blocked_tools(ActionIntent.DISPLAY_ONLY)
        if "write_file" in blocked_tools:
            print("write_file is blocked for DISPLAY_ONLY intent")
    """
    from victor.agent.metadata_authorizer import MetadataActionAuthorizer

    authorizer = MetadataActionAuthorizer()
    blocked_tools = authorizer.get_blocked_tools(intent)
    return frozenset(blocked_tools)


def get_blocked_tools_for_intent(intent: ActionIntent) -> frozenset[str]:
    """Get blocked tools for an intent using metadata-based authorization.

    This is the canonical function for getting blocked tools.
    Hard-coded tool lists have been removed in v0.5.1.

    Args:
        intent: User intent (ActionIntent enum)

    Returns:
        Frozenset of tool names that are blocked for the intent

    Example:
        blocked = get_blocked_tools_for_intent(ActionIntent.DISPLAY_ONLY)
    """
    return get_metadata_blocked_tools(intent)


@dataclass
class IntentClassification:
    """Result of intent detection.

    Attributes:
        intent: The detected user intent
        confidence: Confidence score (0.0-1.0)
        matched_signals: List of signal names that matched
        safe_actions: Set of action categories that are safe to perform
        prompt_guard: Text to inject into system prompt if needed
    """

    intent: ActionIntent
    confidence: float
    matched_signals: List[str]
    safe_actions: Set[str]
    prompt_guard: str


# Signals that indicate "display only" intent
DISPLAY_SIGNALS: List[Tuple[str, float, str]] = [
    (r"\bshow\s+me\b", 1.0, "show_me"),
    (r"\bgive\s+me\s+(an?\s+)?(example|code)\b", 0.9, "give_example"),
    (r"\bcreate\s+(a|an)\s+\w*\s*(function|class|method|script)\b", 0.8, "create_function"),
    (r"\bwrite\s+(a|an)\s+\w*\s*(function|class|method)\b", 0.7, "write_function"),
    (r"\bhow\s+(do|would)\s+(i|you)\s+(write|create|implement)\b", 0.9, "how_to_write"),
    (r"\bwhat\s+(does|would)\s+\w+\s+look\s+like\b", 0.8, "what_looks_like"),
    (r"\bjust\s+(show|display|print)\b", 1.0, "just_show"),
    (r"\bdon'?t\s+(save|write|create)\s+(any\s+)?(files?|to\s+disk)\b", 1.0, "dont_write"),
    (r"\bdisplay\s+(only|it)\b", 1.0, "display_only"),
    (r"\bwithout\s+(saving|writing)\b", 0.9, "without_saving"),
]

# Signals that explicitly authorize file writing
WRITE_SIGNALS: List[Tuple[str, float, str]] = [
    (r"\b(save|write)\s+(\w+\s+)?(to|as)\s+\w+", 1.0, "save_to_file"),
    (r"\bsave\s+(this|it)\s+(to|as)\b", 1.0, "save_this_to"),
    (r"\bcreate\s+(a\s+)?(new\s+)?file\b", 1.0, "create_file"),
    (
        r"\b(add|put|place)\s+(\w+\s+)?(in|into)\s+\w+\.(py|js|ts|java|cpp|go|rs)",
        0.9,
        "add_to_file",
    ),
    (r"\badd\s+(this|it)\s+(function\s+)?(into|to)\s+\w+\.(py|js|ts)", 0.9, "add_into_file"),
    (r"\bupdate\s+(the\s+)?file\b", 0.9, "update_file"),
    (r"\bmodify\s+(the\s+)?\w+\.(py|js|ts|java|cpp)", 0.9, "modify_file"),
    (r"\b(edit|change)\s+(the\s+)?file\b", 0.8, "edit_file"),
    (r"\bsave\s+changes\b", 0.9, "save_changes"),
    (r"\bwrite\s+(this|it)\s+to\s+disk\b", 1.0, "write_to_disk"),
    (r"\badd\s+(this\s+)?to\s+\w+\.(py|js|ts)", 0.8, "add_to_ext"),
    (r"\bimplement\s+(this\s+)?(in|into)\s+\w+\.(py|js|ts)", 0.8, "implement_into_file"),
]

# Signals that indicate read-only intent (no generation)
READ_ONLY_SIGNALS: List[Tuple[str, float, str]] = [
    (r"\b(list|show)\s+(all\s+)?(files?|directories?)\b", 0.9, "list_files"),
    (r"\bwhat\s+(are|is)\s+(in|inside)\b", 0.8, "what_is_in"),
    (r"\bexplain\s+(the\s+)?(file|code|function)\b", 0.9, "explain"),
    (r"\bsummarize\b", 0.9, "summarize"),
    (r"\bdescribe\b", 0.8, "describe"),
    (r"\bread\s+(and\s+)?\w+\b", 0.7, "read"),
    (r"\bgit\s+(status|log|branch|diff)\b", 1.0, "git_read"),
]

# Compound signals: "analyze AND fix" patterns that authorize writes after analysis
# These take precedence over READ_ONLY signals when both analysis and action words present
COMPOUND_WRITE_SIGNALS: List[Tuple[str, float, str]] = [
    # Analyze/review + fix/update/modify
    (
        r"\b(analyze|review|check|find)\b.*\b(and|then)\s+(fix|update|modify|correct|improve)\b",
        1.0,
        "analyze_then_fix",
    ),
    (r"\b(identify|detect|find)\b.*\b(and\s+)?(fix|correct|resolve)\b", 1.0, "find_and_fix"),
    (
        r"\b(fix|correct|resolve)\s+(any|all|the)?\s*(bugs?|issues?|errors?|problems?)\b",
        0.9,
        "fix_bugs",
    ),
    # "Fix the bug in X" patterns
    (r"\bfix\s+(the\s+)?(bug|issue|error|problem)\s+(in|with)\b", 0.9, "fix_in"),
    # "Apply the fix/changes" patterns
    (r"\b(apply|implement)\s+(the\s+)?(fix|changes?|improvements?)\b", 0.9, "apply_fix"),
    # "Make the changes" / "make it work"
    (r"\bmake\s+(the\s+)?(changes?|it\s+work|corrections?)\b", 0.8, "make_changes"),
    # "Refactor and improve"
    (r"\b(refactor|clean\s*up)\s+(and\s+)?(improve|optimize)?\b", 0.8, "refactor_improve"),
    # =============================================================================
    # Implementation task patterns - tasks that require creating/modifying artifacts
    # =============================================================================
    # "Create/Write a Dockerfile/docker-compose/manifest/config" - infrastructure files
    (
        r"\b(create|write|generate|build)\s+(a\s+)?(dockerfile|docker-compose|makefile|"
        r"jenkinsfile|gitlab-ci|github\s*actions?|kubernetes|k8s|helm|terraform|ansible)",
        0.9,
        "create_infra_file",
    ),
    # "Generate/Create a CI/CD/pipeline/workflow" - pipeline configuration
    (
        r"\b(create|write|generate|set\s*up)\s+(a\s+)?(ci/?cd|pipeline|workflow|"
        r"deployment|build)\s*(configuration|config|file|yaml|yml)?",
        0.85,
        "create_pipeline",
    ),
    # "Create a {report/analysis/document}" - document creation tasks
    (
        r"\b(create|write|generate)\s+(a\s+)?([\w\s]+)?(report|analysis|document|documentation)",
        0.8,
        "create_document",
    ),
    # "Create/Write/Generate a {artifact} for {project/this}" - project-specific creation
    (
        r"\b(create|write|generate)\s+(a\s+)?[\w\s]+\s+(for|in)\s+(this|the|my)\s+(project|repo|codebase)",
        0.85,
        "create_for_project",
    ),
    # "Implement {feature}" - explicit implementation request
    (
        r"\bimplement\s+(\w+\s+)?(feature|functionality|module|component|system|service)",
        0.85,
        "implement_feature",
    ),
    # "Implement {something}" - any implementation task (user authentication, login, etc.)
    (
        r"\bimplement\s+(the\s+)?[\w\s]+(authentication|login|logout|auth|api|endpoint|handler|service)",
        0.85,
        "implement_auth",
    ),
    # Generic "implement" followed by a noun phrase
    (r"\bimplement\s+[\w\s]{3,30}$", 0.75, "implement_generic"),
    # "Add {feature/functionality} to" - adding to existing codebase
    (
        r"\badd\s+(\w+\s+)?(feature|functionality|support|capability|module)\s+(to|for)\b",
        0.85,
        "add_feature",
    ),
    # "Set up/Configure {tool/system}" - setup tasks
    (r"\b(set\s*up|configure|initialize|bootstrap)\s+(a\s+)?\w+", 0.8, "setup_configure"),
    # "Build/Deploy {service/app}" - deployment tasks
    (
        r"\b(build|deploy|release)\s+(the\s+)?\w+\s*(service|app|application|container)?",
        0.8,
        "build_deploy",
    ),
    # "Create a {type} that {does something}" - functional creation for project artifacts
    # EXCLUDES function/class/method/script (those are DISPLAY_ONLY patterns)
    (
        r"\bcreate\s+(a\s+)?(?!.*\b(function|class|method|script)\b)[\w\s]+\s+that\s+",
        0.75,
        "create_functional",
    ),
    # "Write/Create tests for" - test creation
    (
        r"\b(write|create|add|implement)\s+(unit\s+|integration\s+)?tests?\s+(for|to)\b",
        0.85,
        "write_tests",
    ),
]

# Safe actions by intent type
SAFE_ACTIONS: dict[ActionIntent, Set[str]] = {
    ActionIntent.READ_ONLY: {"read_file", "list_directory", "code_search", "git_status"},
    ActionIntent.DISPLAY_ONLY: {
        "read_file",
        "list_directory",
        "code_search",
        "git_status",
        "generate_response",  # Can generate code in response, just not write
    },
    ActionIntent.WRITE_ALLOWED: {
        "read_file",
        "list_directory",
        "code_search",
        "git_status",
        "generate_response",
        "write_file",
        "edit_file",
        "create_file",
        "execute_bash",
    },
    ActionIntent.AMBIGUOUS: {
        "read_file",
        "list_directory",
        "code_search",
        "git_status",
        "generate_response",
    },
}

# Prompt guards for different intents
PROMPT_GUARDS: dict[ActionIntent, str] = {
    ActionIntent.DISPLAY_ONLY: """
IMPORTANT: The user wants to SEE code, not save it to a file.
- Generate and DISPLAY the requested code directly in your response
- Do NOT use write_file or create_file tools
- Do NOT modify any existing files
- Present code in markdown code blocks
- Be concise and complete the task in one response
- Do NOT ask follow-up questions like "Would you like me to elaborate?" or "Should I explain more?"
- Simply provide the answer and stop
""",
    ActionIntent.READ_ONLY: """
IMPORTANT: This is a read-only query.
- Only use read operations (read_file, list_directory, code_search)
- Be concise and answer directly
- Do NOT ask follow-up questions - just provide the information requested
- Do NOT write, create, or modify any files
- Do NOT generate new code unless specifically asked
""",
    ActionIntent.WRITE_ALLOWED: "",  # No guard needed
    ActionIntent.AMBIGUOUS: """
NOTE: User intent is unclear. Default to safe behavior:
- Present code in markdown blocks rather than writing to files
- Ask for confirmation before creating or modifying files
""",
}


class IntentDetector:
    """Detects user intent for action authorization.

    This detector analyzes user messages to determine whether file
    modifications are authorized or if output should be display-only.

    Example:
        detector = IntentDetector()
        result = detector.detect("Show me a function that calculates factorial")
        print(result.intent)  # ActionIntent.DISPLAY_ONLY
        print(result.safe_actions)  # {"read_file", "generate_response", ...}
    """

    def __init__(
        self,
        custom_display_signals: Optional[List[Tuple[str, float, str]]] = None,
        custom_write_signals: Optional[List[Tuple[str, float, str]]] = None,
        custom_detectors: Optional[List[Callable[[str], Optional[IntentClassification]]]] = None,
        default_intent: ActionIntent = ActionIntent.DISPLAY_ONLY,
    ):
        """Initialize the intent detector.

        Args:
            custom_display_signals: Additional patterns for display-only intent
            custom_write_signals: Additional patterns for write-allowed intent
            custom_detectors: Custom detector functions to try first
            default_intent: Default when no patterns match (safe by default)
        """
        self.default_intent = default_intent
        self.custom_detectors = custom_detectors or []

        # Compile patterns
        self._display_patterns: List[Tuple[Pattern[str], float, str]] = []
        self._write_patterns: List[Tuple[Pattern[str], float, str]] = []
        self._read_only_patterns: List[Tuple[Pattern[str], float, str]] = []
        self._compound_write_patterns: List[Tuple[Pattern[str], float, str]] = []

        self._compile_patterns(DISPLAY_SIGNALS, self._display_patterns)
        self._compile_patterns(WRITE_SIGNALS, self._write_patterns)
        self._compile_patterns(READ_ONLY_SIGNALS, self._read_only_patterns)
        self._compile_patterns(COMPOUND_WRITE_SIGNALS, self._compound_write_patterns)

        if custom_display_signals:
            self._compile_patterns(custom_display_signals, self._display_patterns)
        if custom_write_signals:
            self._compile_patterns(custom_write_signals, self._write_patterns)

    def _compile_patterns(
        self,
        signals: List[Tuple[str, float, str]],
        target: List[Tuple[Pattern[str], float, str]],
    ) -> None:
        """Compile regex patterns."""
        for pattern_str, weight, name in signals:
            try:
                compiled = re.compile(pattern_str, re.IGNORECASE)
                target.append((compiled, weight, name))
            except re.error as e:
                logger.warning(f"Invalid pattern '{pattern_str}': {e}")

    def detect(self, message: str) -> IntentClassification:
        """Detect user intent from message.

        Args:
            message: User's input message

        Returns:
            IntentClassification with intent, confidence, and safe actions
        """
        # Try custom detectors first
        for detector in self.custom_detectors:
            result = detector(message)
            if result is not None:
                return result

        # Score each intent type
        display_score, display_matched = self._score_patterns(message, self._display_patterns)
        write_score, write_matched = self._score_patterns(message, self._write_patterns)
        read_only_score, read_only_matched = self._score_patterns(message, self._read_only_patterns)
        compound_score, compound_matched = self._score_patterns(
            message, self._compound_write_patterns
        )

        # Determine intent based on scores
        # Compound write signals (e.g., "analyze and fix") take highest precedence
        # They indicate user wants both analysis AND file modifications
        if compound_score > 0:
            intent = ActionIntent.WRITE_ALLOWED
            confidence = min(1.0, compound_score)
            matched = compound_matched
            logger.debug(f"Compound write intent detected: {compound_matched}")
        # Write signals are explicit and take precedence when strong
        elif write_score > 0.5 and write_score > display_score:
            intent = ActionIntent.WRITE_ALLOWED
            confidence = min(1.0, write_score)
            matched = write_matched
        elif read_only_score > 0 and read_only_score >= display_score:
            intent = ActionIntent.READ_ONLY
            confidence = min(1.0, read_only_score)
            matched = read_only_matched
        elif display_score > 0:
            intent = ActionIntent.DISPLAY_ONLY
            confidence = min(1.0, display_score)
            matched = display_matched
        elif write_score > 0:
            # Weak write signal, but no display signal
            intent = ActionIntent.AMBIGUOUS
            confidence = 0.3
            matched = write_matched
        else:
            # No signals - use safe default
            intent = self.default_intent
            confidence = 0.2
            matched = []

        return IntentClassification(
            intent=intent,
            confidence=confidence,
            matched_signals=matched,
            safe_actions=SAFE_ACTIONS[intent].copy(),
            prompt_guard=PROMPT_GUARDS[intent],
        )

    def _score_patterns(
        self,
        message: str,
        patterns: List[Tuple[Pattern[str], float, str]],
    ) -> Tuple[float, List[str]]:
        """Score patterns against message.

        Args:
            message: User message
            patterns: List of (pattern, weight, name) tuples

        Returns:
            Tuple of (total_score, matched_names)
        """
        total_score = 0.0
        matched = []
        for pattern, weight, name in patterns:
            if pattern.search(message):
                total_score += weight
                matched.append(name)
        return total_score, matched

    def is_write_authorized(self, message: str) -> bool:
        """Quick check if file writing is authorized.

        Args:
            message: User message

        Returns:
            True if writing is explicitly authorized
        """
        result = self.detect(message)
        return result.intent == ActionIntent.WRITE_ALLOWED


def detect_intent(message: str) -> IntentClassification:
    """Convenience function to detect intent.

    Args:
        message: User message

    Returns:
        IntentClassification
    """
    detector = IntentDetector()
    return detector.detect(message)


def is_write_authorized(message: str) -> bool:
    """Quick check if file writing is authorized.

    Args:
        message: User message

    Returns:
        True if writing is explicitly authorized
    """
    detector = IntentDetector()
    return detector.is_write_authorized(message)


def get_prompt_guard(message: str) -> str:
    """Get prompt guard text for a message.

    Args:
        message: User message

    Returns:
        Prompt guard text to inject into system prompt
    """
    result = detect_intent(message)
    return result.prompt_guard


def get_safe_tools(message: str, all_tools: Set[str]) -> Set[str]:
    """Get tools that are safe to use based on intent.

    Args:
        message: User message
        all_tools: Set of all available tool names

    Returns:
        Set of tool names that are safe to use
    """
    result = detect_intent(message)

    # Map safe action categories to actual tool names
    # This is a simple mapping - can be extended
    action_to_tools = {
        "read_file": {"read_file"},
        "list_directory": {"list_directory"},
        "code_search": {"code_search", "semantic_code_search"},
        "git_status": {"execute_bash"},  # Only for git read commands
        "generate_response": set(),  # Not a tool, just allow text response
        "write_file": {"write_file", "create_file"},
        "edit_file": {"edit_file", "patch_file"},
        "create_file": {"write_file", "create_file"},
        "execute_bash": {"execute_bash"},
    }

    safe = set()
    for action in result.safe_actions:
        tools = action_to_tools.get(action, set())
        safe.update(tools)

    return safe.intersection(all_tools)


# Semantic alias for clarity
ActionAuthorizer = IntentDetector
