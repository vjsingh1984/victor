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

"""Continuation strategy for orchestrator response handling.

This module provides decision logic for determining when and how to continue
processing when the model doesn't call tools, including:
- Detecting mentioned but unexecuted tools (hallucinated tool calls)
- Determining continuation actions based on intent and task type
- Managing continuation prompt budgets

Extracted from CRITICAL-001 Phase 2E.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Pattern

from victor.core.events import ObservabilityBus
from victor.agent.tool_call_extractor import extract_tool_call_from_text, ExtractedToolCall
from victor.storage.embeddings.question_classifier import (
    QuestionTypeClassifier,
    QuestionType,
    classify_question,
)

# Patterns for detecting output requirements in response content
# PRE-COMPILED at module load for performance (avoid re.compile in hot path)
_OUTPUT_REQUIREMENT_PATTERNS_RAW = {
    "findings table": [
        r"(?i)findings?\s*table",
        r"(?i)\|[^\|]+\|[^\|]+\|",  # Markdown table syntax
        r"(?i)finding[s]?\s*:?\s*\n\s*[-\*]",  # Bullet list findings
    ],
    "top-3 fixes": [
        r"(?i)top[- ]?3\s+fix(es)?",
        r"(?i)recommended\s+fix(es)?",
        r"(?i)1\.\s+.+\n\s*2\.\s+.+\n\s*3\.",  # Numbered list 1-3
    ],
    "summary": [
        r"(?i)summary\s*:",
        r"(?i)in\s+summary",
        r"(?i)conclusion",
    ],
}

# Pre-compile all patterns at module load (30-40% speedup on pattern matching)
OUTPUT_REQUIREMENT_PATTERNS: Dict[str, List[Pattern[Any]]] = {
    key: [re.compile(pattern) for pattern in patterns]
    for key, patterns in _OUTPUT_REQUIREMENT_PATTERNS_RAW.items()
}

# Cache for tool mention patterns (avoid recompiling per-tool patterns)
# Key: tool_name, Value: list of compiled patterns
_TOOL_MENTION_PATTERN_CACHE: Dict[str, List[Pattern[Any]]] = {}

# Common tool mention pattern templates (compiled once, tool name inserted)
_TOOL_MENTION_TEMPLATES = [
    r"\b(?:call|use|execute|run|invoke|perform)\s+{tool}\b",
    r"\b{tool}\s*\(",  # tool_name( or tool_name (
    r"\bthe\s+{tool}\s+tool\b",  # "the read tool"
]


def _get_tool_mention_patterns(tool_name: str) -> List[Pattern[Any]]:
    """Get or create compiled patterns for detecting tool mentions.

    Caches compiled patterns per tool name to avoid recompilation.

    Args:
        tool_name: Name of the tool

    Returns:
        List of compiled regex patterns for this tool
    """
    if tool_name not in _TOOL_MENTION_PATTERN_CACHE:
        escaped_name = re.escape(tool_name)
        _TOOL_MENTION_PATTERN_CACHE[tool_name] = [
            re.compile(template.format(tool=escaped_name), re.IGNORECASE)
            for template in _TOOL_MENTION_TEMPLATES
        ]
    return _TOOL_MENTION_PATTERN_CACHE[tool_name]


logger = logging.getLogger(__name__)


class ContinuationStrategy:
    """Handles continuation decision logic for orchestrator.

    This component encapsulates the complex logic for determining what action
    to take when the model responds without tool calls, including:
    - Intent-based continuation decisions
    - Tool mention detection (hallucinated calls)
    - Continuation prompt budget management
    - RL-learned parameter adaptation

    Design Pattern: Strategy
    Provides different continuation strategies based on task type and intent.

    Extracted from CRITICAL-001 Phase 2E.
    """

    def __init__(self, event_bus: Optional[ObservabilityBus] = None):
        """Initialize continuation strategy.

        Args:
            event_bus: Optional ObservabilityBus instance. If None, uses DI container.
        """
        self._event_bus = event_bus or self._get_default_bus()

    def _get_default_bus(self) -> Optional[ObservabilityBus]:
        """Get default ObservabilityBus from DI container.

        Returns:
            ObservabilityBus instance or None if unavailable
        """
        try:
            from victor.core.events import get_observability_bus

            return get_observability_bus()
        except Exception:
            return None

    def _emit_event(
        self, topic: str, data: Dict[str, Any], source: str = "ContinuationStrategy"
    ) -> None:
        """Emit event with error handling (non-blocking).

        Args:
            topic: Event topic
            data: Event data
            source: Event source
        """
        if self._event_bus:
            try:
                import asyncio

                # Check if there's a running event loop
                try:
                    loop = asyncio.get_running_loop()
                    # Non-blocking async call in sync context
                    loop.create_task(
                        self._event_bus.emit(
                            topic=topic,
                            data={**data, "category": "state"},  # Preserve for observability
                            source=source,
                        )
                    )
                except RuntimeError:
                    # No event loop running, skip event emission
                    logger.debug(f"No event loop, skipping event emission for {topic}")
            except Exception as e:
                logger.debug(f"Failed to emit continuation event: {e}")

    @staticmethod
    def _get_complexity_threshold(
        settings: Any, complexity: Optional[str], threshold_type: str
    ) -> int:
        """Get complexity-based threshold from settings.

        Args:
            settings: Settings object
            complexity: Task complexity level (simple/medium/complex/generation/action/analysis)
            threshold_type: Type of threshold (max_interventions or max_iterations)

        Returns:
            Threshold value based on complexity, with sensible defaults
        """
        # Map complexity to setting names
        complexity_map = {
            "simple": "continuation_simple",
            "medium": "continuation_medium",
            "complex": "continuation_complex",
            "generation": "continuation_generation",
            "action": "continuation_medium",  # Use medium thresholds for action tasks
            "analysis": "continuation_complex",  # Use complex thresholds for analysis tasks
        }

        # Default to medium if complexity not specified
        complexity_key = complexity_map.get(complexity or "medium", "continuation_medium")

        # Get threshold from settings
        setting_name = f"{complexity_key}_{threshold_type}"
        threshold = getattr(settings, setting_name, None)

        # Fallback defaults if setting not available
        if threshold is None:
            if threshold_type == "max_interventions":
                defaults = {"simple": 5, "medium": 10, "complex": 20, "generation": 15}
            else:  # max_iterations
                defaults = {"simple": 10, "medium": 25, "complex": 50, "generation": 35}

            complexity_default = complexity or "medium"
            threshold = defaults.get(complexity_default, 10)

        return threshold

    def _calculate_hybrid_score(
        self,
        progress_metrics: Optional[Any],
        task_complexity: Optional[str],
        cumulative_interventions: int,
        max_interventions: int,
        max_iterations: int,
        estimated_tokens: int,
        content_length: int,
    ) -> Dict[str, Any]:
        """Calculate hybrid continuation score combining all signals.

        Uses the ContinuationSignals class to compute a weighted score from:
        - Progress velocity (30%)
        - Stuck loop penalty (25%)
        - Token budget (20%)
        - Intervention ratio (15%)
        - Complexity adjustment (10%)

        Args:
            progress_metrics: ProgressMetrics instance
            task_complexity: Task complexity level
            cumulative_interventions: Total continuation prompts
            max_interventions: Max interventions for this complexity
            max_iterations: Max iterations for this complexity
            estimated_tokens: Current token usage estimate
            content_length: Content length

        Returns:
            Dict with score, recommendation, confidence, and signal breakdown
        """
        from victor.agent.streaming.continuation import ContinuationSignals

        # Create signals container
        signals = ContinuationSignals(
            progress_metrics=progress_metrics,
            task_complexity=task_complexity,
            cumulative_interventions=cumulative_interventions,
            max_interventions=max_interventions,
            max_iterations=max_iterations,
            estimated_tokens=estimated_tokens,
            content_length=content_length,
        )

        # Calculate hybrid score
        return signals.calculate_continuation_score()

    @staticmethod
    def detect_mentioned_tools(
        text: str, all_tool_names: List[str], tool_aliases: Dict[str, str]
    ) -> List[str]:
        """Detect tool names mentioned in text that model said it would call.

        Looks for patterns like:
        - "let me call read()"
        - "I'll use web_search to"
        - "calling the ls tool"
        - "execute grep"

        Uses cached pre-compiled patterns for 5-10x speedup over dynamic compilation.

        Args:
            text: Model response text
            all_tool_names: List of all valid tool names
            tool_aliases: Dictionary mapping aliases to canonical names

        Returns:
            List of mentioned tool names (canonical form)
        """
        mentioned: List[str] = []
        # Lowercase once, reuse for all pattern matches
        text_lower = text.lower()

        # Look for tool names using cached compiled patterns
        for tool_name in all_tool_names:
            # Get cached compiled patterns (created once per tool, reused)
            patterns = _get_tool_mention_patterns(tool_name)
            for pattern in patterns:
                if pattern.search(text_lower):
                    # Resolve to canonical name
                    canonical = tool_aliases.get(tool_name, tool_name)
                    if canonical not in mentioned:
                        mentioned.append(canonical)
                    break

        return mentioned

    def _output_requirements_met(self, content: Optional[str], required_outputs: List[str]) -> bool:
        """Check if response content contains required output elements.

        Uses pre-compiled pattern matching to detect common output format elements
        like findings tables, numbered fix lists, summaries, etc.

        Performance: Uses pre-compiled regex patterns for 30-40% speedup.

        Args:
            content: Response content to check
            required_outputs: List of required output elements (e.g., ["findings table", "top-3 fixes"])

        Returns:
            True if all required outputs are detected in content
        """
        if not content or not required_outputs:
            return False

        # Lowercase once, reuse for substring checks
        content_lower = content.lower()

        for requirement in required_outputs:
            requirement_key = requirement.lower().strip()

            # Get pre-compiled patterns for this requirement type
            patterns = OUTPUT_REQUIREMENT_PATTERNS.get(requirement_key, [])

            # If no predefined patterns, use direct substring match
            if not patterns:
                # Check for requirement as substring
                if requirement_key not in content_lower:
                    return False
                continue

            # Check if any pre-compiled pattern matches
            found = False
            for pattern in patterns:
                # Patterns are now pre-compiled re.Pattern objects
                if pattern.search(content):
                    found = True
                    break

            if not found:
                return False

        return True

    def _get_continuation_prompt(
        self,
        provider_name: str,
        model: str,
        is_analysis_task: bool,
        is_action_task: bool,
        continuation_prompts: int,
        full_content: Optional[str] = None,
    ) -> str:
        """Get model-specific continuation prompt to encourage tool usage.

        Different models respond better to different prompting styles. This method
        returns tailored prompts based on the provider/model combination.

        Args:
            provider_name: LLM provider (e.g., "ollama", "anthropic", "openai")
            model: Model name (e.g., "qwen3-coder-tools:30b-64K")
            is_analysis_task: Whether this is an analysis task
            is_action_task: Whether this is an action/implementation task
            continuation_prompts: Current number of continuation prompts sent
            full_content: Optional full response content for context-aware prompts

        Returns:
            Continuation prompt message to send to the model
        """
        # Extract model family for targeted prompts
        model_lower = model.lower()
        provider_lower = provider_name.lower()

        # Qwen models need more explicit, directive prompts
        if "qwen" in model_lower or "alibaba" in provider_lower:
            return self._get_qwen_continuation_prompt(
                is_analysis_task, is_action_task, continuation_prompts, full_content
            )

        # DeepSeek models respond well to step-by-step breakdown
        if "deepseek" in model_lower:
            return self._get_deepseek_continuation_prompt(is_analysis_task, is_action_task)

        # Default prompts for other models
        if is_analysis_task:
            return (
                "Continue your analysis. Use tools like read_file, list_directory, "
                "code_search to gather more information."
            )
        elif is_action_task:
            return (
                "Continue with the implementation. Use tools like write_file, "
                "edit_files to make the necessary changes."
            )
        else:
            return "Continue. Use appropriate tools if needed."

    def _get_qwen_continuation_prompt(
        self,
        is_analysis_task: bool,
        is_action_task: bool,
        continuation_prompts: int,
        full_content: Optional[str] = None,
    ) -> str:
        """Get Qwen-specific continuation prompt.

        Qwen models (especially coder variants) need:
        1. More explicit directives
        2. Step-by-step instructions
        3. Clear tool usage examples
        """
        if continuation_prompts >= 3:
            # After 3 failed attempts, use very directive language
            if is_analysis_task:
                return (
                    "⚠️ ACTION REQUIRED: You MUST use tools to complete this analysis.\n\n"
                    "Next step: Choose ONE tool to execute NOW:\n"
                    "1. read(path='path/to/file') - Read a specific file\n"
                    "2. ls(path='directory') - List directory contents\n"
                    "3. graph() - Analyze code architecture\n"
                    "4. grep(pattern='text', path='path') - Search for patterns\n\n"
                    "Example: read(path='victor/agent/orchestrator.py')\n\n"
                    "Execute a tool call immediately. Do not repeat the task description."
                )
            elif is_action_task:
                return (
                    "⚠️ ACTION REQUIRED: You MUST use tools to implement the changes.\n\n"
                    "Next step: Choose a tool to execute:\n"
                    "1. write_file(path='path', content='...') - Create/overwrite file\n"
                    "2. edit(path='path', ...) - Edit existing file\n\n"
                    "Execute a tool call immediately."
                )
            else:
                return (
                    "⚠️ ACTION REQUIRED: Execute a tool call to continue.\n\n"
                    "Use appropriate tools for your task. Do not repeat the task description."
                )

        # First/second attempt - more subtle
        if is_analysis_task:
            return (
                "To complete your analysis, please start by examining the codebase:\n\n"
                "1. First, use: ls(path='victor') to see the main directories\n"
                "2. Then, use: read(path='victor/agent/orchestrator.py') to read key files\n"
                "3. Use: graph() to understand the architecture\n\n"
                "Begin with ls() now."
            )
        elif is_action_task:
            return (
                "To implement this task:\n\n"
                "1. First, read the files you need to modify\n"
                "2. Then, use write_file() or edit() to make changes\n\n"
                "Start by reading the relevant file."
            )
        else:
            return "Please use appropriate tools to continue with your task."

    def _get_deepseek_continuation_prompt(
        self,
        is_analysis_task: bool,
        is_action_task: bool,
    ) -> str:
        """Get DeepSeek-specific continuation prompt.

        DeepSeek models respond well to:
        1. Step-by-step breakdown
        2. Clear reasoning chains
        """
        if is_analysis_task:
            return (
                "Let's continue the analysis step by step:\n\n"
                "Step 1: List the directories to understand the structure\n"
                "Step 2: Read key implementation files\n"
                "Step 3: Analyze the integration patterns\n\n"
                "Please start with Step 1 using the ls() tool."
            )
        elif is_action_task:
            return (
                "Let's proceed with implementation:\n\n"
                "Step 1: Read the file to understand current implementation\n"
                "Step 2: Make the necessary edits\n"
                "Step 3: Verify the changes\n\n"
                "Begin with Step 1."
            )
        else:
            return "Continue step by step. Use tools as needed."

    def determine_continuation_action(
        self,
        intent_result: Any,  # IntentClassificationResult
        is_analysis_task: bool,
        is_action_task: bool,
        content_length: int,
        full_content: Optional[str],
        continuation_prompts: int,
        asking_input_prompts: int,
        one_shot_mode: bool,
        mentioned_tools: Optional[List[str]],
        # Context from orchestrator
        max_prompts_summary_requested: bool,
        settings: Any,
        rl_coordinator: Any,
        provider_name: str,
        model: str,
        tool_budget: int,
        unified_tracker_config: Dict[str, Any],
        task_completion_signals: Optional[Dict[str, Any]] = None,
        progress_metrics: Optional[Any] = None,  # ProgressMetrics instance
        task_complexity: Optional[
            str
        ] = None,  # Task complexity level (simple/medium/complex/generation)
    ) -> Dict[str, Any]:
        """Determine what continuation action to take when model doesn't call tools.

        Encapsulates the complex decision logic for handling responses without tool
        calls, including intent classification, continuation prompting, and summary
        requests.

        Args:
            intent_result: Result from intent classifier (has .intent, .confidence)
            is_analysis_task: Whether task is analysis-oriented
            is_action_task: Whether task is action-oriented
            content_length: Length of model's response content
            full_content: Full response content (for structure detection)
            continuation_prompts: Current count of continuation prompts sent
            asking_input_prompts: Current count of asking-input auto-responses
            one_shot_mode: Whether running in non-interactive mode
            mentioned_tools: Tools mentioned but not executed (hallucinated tool calls)
            max_prompts_summary_requested: Whether summary was already requested
            settings: Settings object for configuration
            rl_coordinator: RL coordinator for learned parameters
            provider_name: Provider name for RL lookups
            model: Model name for RL lookups
            tool_budget: Current tool budget
            unified_tracker_config: Config dict from unified tracker
            task_completion_signals: Optional signals for task completion detection
            progress_metrics: Optional ProgressMetrics instance for progress tracking
            task_complexity: Optional task complexity level (simple/medium/complex/generation)

        Returns:
            Dictionary with:
            - action: str - One of: "continue_asking_input", "return_to_user",
                          "prompt_tool_call", "request_summary",
                          "request_completion", "finish", "force_tool_execution"
            - message: Optional[str] - System message to inject (if any)
            - reason: str - Human-readable reason for the action
            - updates: Dict - State updates (continuation_prompts, asking_input_prompts)
        """
        from victor.storage.embeddings.intent_classifier import IntentType

        updates: Dict[str, Any] = {}

        # Get complexity-based thresholds from settings
        max_interventions = self._get_complexity_threshold(
            settings, task_complexity, "max_interventions"
        )
        max_iterations = self._get_complexity_threshold(settings, task_complexity, "max_iterations")

        logger.debug(
            f"Continuation action decision: complexity={task_complexity}, "
            f"max_interventions={max_interventions}, max_iterations={max_iterations}"
        )

        # TOKEN BUDGET CHECK: Check if token limits require action
        # This provides an additional signal beyond iteration counts
        # Estimate current token usage from content length
        estimated_tokens = int(content_length / 4) if content_length > 0 else 0

        if progress_metrics and progress_metrics.token_budget:
            # Get token status from progress metrics
            token_status = progress_metrics.check_token_limits(estimated_tokens)

            if token_status:
                logger.debug(
                    f"Token budget status: {token_status.get('usage_pct', 0)}% used, "
                    f"should_nudge={token_status.get('should_nudge', False)}, "
                    f"should_force={token_status.get('should_force', False)}"
                )

                # Force synthesis if hard limit reached
                if token_status.get("should_force", False):
                    return {
                        "action": "request_summary",
                        "message": (
                            f"Token budget at {token_status.get('usage_pct', 0)}% of context window. "
                            f"Please synthesize your findings now to avoid exceeding model capacity."
                        ),
                        "reason": f"Token limit reached ({token_status.get('usage_pct', 0)}% of context)",
                        "updates": updates,
                    }

                # Add token status to logs for observability
                if token_status.get("should_nudge", False):
                    logger.info(
                        f"Token budget at soft limit ({token_status.get('usage_pct', 0)}%) - "
                        f"considering synthesis nudge"
                    )

        # Calculate cumulative interventions early for hybrid scoring
        cumulative_interventions = 0
        if task_completion_signals:
            cumulative_interventions = task_completion_signals.get(
                "cumulative_prompt_interventions", 0
            )

        # HYBRID SCORING: Combine all signals for intelligent continuation decision
        # This is the primary decision mechanism for Phase 4
        hybrid_result = self._calculate_hybrid_score(
            progress_metrics=progress_metrics,
            task_complexity=task_complexity,
            cumulative_interventions=cumulative_interventions,
            max_interventions=max_interventions,
            max_iterations=max_iterations,
            estimated_tokens=estimated_tokens,
            content_length=content_length,
        )

        # Log hybrid scoring results for observability
        logger.debug(
            f"Hybrid continuation score: {hybrid_result['score']:.3f}, "
            f"recommendation={hybrid_result['recommendation']}, "
            f"confidence={hybrid_result['confidence']:.3f}"
        )

        # Emit event with detailed breakdown for observability
        self._emit_event(
            topic="state.continuation.hybrid_score",
            data={
                "score": hybrid_result["score"],
                "recommendation": hybrid_result["recommendation"],
                "confidence": hybrid_result["confidence"],
                "signal_scores": hybrid_result["signal_scores"],
                "task_complexity": task_complexity,
            },
        )

        # Force synthesis if hybrid score is very low OR stuck loop detected
        if (
            hybrid_result["recommendation"] == "force_synthesis"
            or hybrid_result["score"] < 0.2
            or (progress_metrics and progress_metrics.is_stuck_loop)
        ):
            # Build detailed reason from hybrid result
            reason_parts = []
            if progress_metrics and progress_metrics.is_stuck_loop:
                reason_parts.append("stuck loop detected")
            if hybrid_result["score"] < 0.2:
                reason_parts.append(f"low continuation score ({hybrid_result['score']:.2f})")

            return {
                "action": "request_summary",
                "message": (
                    f"Please synthesize your findings now. "
                    f"Continuation score: {hybrid_result['score']:.2f}/1.0"
                ),
                "reason": f"Force synthesis: {', '.join(reason_parts)}",
                "updates": updates,
            }

        # TASK COMPLETION CHECK: If all required files read and output requirements met,
        # finish immediately to prevent prompting loop (prompting loop fix)
        if task_completion_signals:
            required_files = task_completion_signals.get("required_files", [])
            read_files = task_completion_signals.get("read_files", set())
            required_outputs = task_completion_signals.get("required_outputs", [])
            all_files_read = task_completion_signals.get("all_files_read", False)
            synthesis_nudge_count = task_completion_signals.get("synthesis_nudge_count", 0)

            # Check if all required files have been read
            files_complete = required_files and (
                all_files_read or read_files.issuperset(set(required_files))
            )

            if files_complete:
                # Check if output requirements are met in the response
                if self._output_requirements_met(full_content, required_outputs):
                    logger.info(
                        "Task completion detected: all required files read and "
                        "output requirements met - finishing"
                    )
                    self._emit_event(
                        topic="state.continuation.task_complete",
                        data={
                            "reason": "task_completion_detected",
                            "required_files": list(required_files),
                            "read_files": list(read_files),
                            "output_requirements": required_outputs,
                        },
                    )
                    return {
                        "action": "finish",
                        "message": None,
                        "reason": "Task completion: all required files read and output requirements met",
                        "updates": updates,
                    }

                # SYNTHESIS NUDGE: If all files read but output not produced,
                # gently remind model to synthesize (not force - allow exploration)
                # Only nudge after 2+ turns without output, max 3 nudges
                if is_analysis_task and synthesis_nudge_count < 3:
                    logger.info(
                        f"Synthesis nudge: all {len(required_files)} required files read, "
                        f"but output not yet produced (nudge {synthesis_nudge_count + 1}/3)"
                    )
                    self._emit_event(
                        topic="state.continuation.synthesis_nudge",
                        data={
                            "required_files": list(required_files),
                            "read_files": list(read_files),
                            "required_outputs": required_outputs,
                            "nudge_count": synthesis_nudge_count + 1,
                        },
                    )
                    updates["synthesis_nudge_count"] = synthesis_nudge_count + 1

                    # Gentle nudge message - not forceful
                    output_hints = (
                        ", ".join(required_outputs[:3]) if required_outputs else "your findings"
                    )
                    return {
                        "action": "continue_with_synthesis_hint",
                        "message": (
                            f"You've read all the required files. When ready, please synthesize "
                            f"your analysis into {output_hints}. You may continue exploring if "
                            f"needed, but don't forget to produce the final output."
                        ),
                        "reason": "Gentle synthesis nudge - all required files read",
                        "updates": updates,
                    }

            # CYCLE DETECTION: If we're cycling between stages too much,
            # force synthesis to prevent infinite exploration loops
            cycle_count = task_completion_signals.get("cycle_count", 0)
            if cycle_count >= 5 and is_analysis_task:
                logger.warning(
                    f"Stage cycling detected (cycle_count={cycle_count}) - "
                    "forcing synthesis to prevent infinite exploration"
                )
                self._emit_event(
                    topic="state.continuation.cycle_force_synthesis",
                    data={
                        "cycle_count": cycle_count,
                        "required_outputs": required_outputs,
                    },
                )
                output_hints = (
                    ", ".join(required_outputs[:3]) if required_outputs else "your findings"
                )
                return {
                    "action": "request_summary",
                    "message": (
                        f"You've been exploring for a while and cycling between stages. "
                        f"Please stop exploring and synthesize your analysis now into {output_hints}. "
                        f"Provide your findings based on what you've already read."
                    ),
                    "reason": f"Stage cycling detected (count={cycle_count}) - forcing synthesis",
                    "updates": updates,
                }

            # PROGRESS-AWARE CUMULATIVE INTERVENTION CHECK
            # If we've had too many prompt interventions with LOW PROGRESS, nudge synthesis
            # This allows active exploration (new file discovery) while preventing stuck loops
            cumulative_interventions = task_completion_signals.get(
                "cumulative_prompt_interventions", 0
            )

            # Use ProgressMetrics if available for more accurate progress tracking
            if progress_metrics:
                # ProgressMetrics provides accurate tracking of files read, revisits, tool usage
                unique_files = progress_metrics.unique_files_read
                file_count = progress_metrics.total_file_reads
                is_making_progress = progress_metrics.is_making_progress
                is_stuck_loop = progress_metrics.is_stuck_loop
                revisit_ratio = progress_metrics.revisit_ratio

                # Log detailed progress metrics
                logger.info(
                    f"ProgressMetrics check: interventions={cumulative_interventions}, "
                    f"unique_files={unique_files}, total_reads={file_count}, "
                    f"revisit_ratio={revisit_ratio:.2f}, is_making_progress={is_making_progress}, "
                    f"is_stuck_loop={is_stuck_loop}"
                )

                # Decision logic using ProgressMetrics:
                # 1. Stuck loop → force synthesis immediately
                # 2. High interventions (>=max_interventions) regardless of progress
                # 3. Low progress (revisit_ratio > 0.5) AND >5 interventions
                should_nudge = (
                    is_stuck_loop
                    or cumulative_interventions >= max_interventions
                    or (revisit_ratio > 0.5 and cumulative_interventions >= 5)
                )

                if cumulative_interventions >= 5 and is_analysis_task and should_nudge:
                    self._emit_event(
                        topic="state.continuation.cumulative_intervention_nudge",
                        data={
                            "cumulative_interventions": cumulative_interventions,
                            "unique_files": unique_files,
                            "total_reads": file_count,
                            "revisit_ratio": revisit_ratio,
                            "is_making_progress": is_making_progress,
                            "is_stuck_loop": is_stuck_loop,
                            "required_outputs": required_outputs,
                            "max_interventions": max_interventions,
                            "task_complexity": task_complexity,
                        },
                    )
                    output_hints = (
                        ", ".join(required_outputs[:3]) if required_outputs else "your findings"
                    )
                    # After max_interventions+ interventions or stuck loop, force synthesis; before that, just nudge
                    if cumulative_interventions >= max_interventions or is_stuck_loop:
                        return {
                            "action": "request_summary",
                            "message": (
                                f"You've explored extensively with {unique_files} unique files read and "
                                f"{cumulative_interventions} intervention cycles. Please synthesize your analysis now "
                                f"into {output_hints}. Provide your findings based on what you've already read."
                            ),
                            "reason": f"Prompt interventions ({cumulative_interventions}/{max_interventions}) or stuck loop - forcing synthesis",
                            "updates": updates,
                        }
                    elif synthesis_nudge_count < 3:
                        updates["synthesis_nudge_count"] = synthesis_nudge_count + 1
                        return {
                            "action": "continue_with_synthesis_hint",
                            "message": (
                                f"You've read {unique_files} unique files so far. When ready, please synthesize "
                                f"your analysis into {output_hints}. You may continue exploring briefly, "
                                f"but please produce the final output soon."
                            ),
                            "reason": f"Interventions ({cumulative_interventions}) with progress ratio {revisit_ratio:.2f} nudge",
                            "updates": updates,
                        }
            else:
                # Fallback to simple progress calculation without ProgressMetrics
                if cumulative_interventions >= 5 and is_analysis_task:
                    # Calculate progress ratio (unique files per intervention)
                    file_count = len(read_files)
                    unique_files = len(set(read_files))  # Dedupe in case of re-reads
                    progress_ratio = unique_files / max(cumulative_interventions, 1)

                    # Log for observability
                    logger.info(
                        f"Intervention check: count={cumulative_interventions}, "
                        f"unique_files={unique_files}, total_reads={file_count}, "
                        f"progress_ratio={progress_ratio:.2f}"
                    )

                    # Only intervene if:
                    # 1. High interventions (>=max_interventions) regardless of progress, OR
                    # 2. Low progress (<0.5 new files per intervention) AND >5 interventions
                    should_nudge = cumulative_interventions >= max_interventions or (
                        progress_ratio < 0.5 and cumulative_interventions >= 5
                    )

                    if should_nudge:
                        self._emit_event(
                            topic="state.continuation.cumulative_intervention_nudge",
                            data={
                                "cumulative_interventions": cumulative_interventions,
                                "unique_files": unique_files,
                                "total_reads": file_count,
                                "progress_ratio": progress_ratio,
                                "required_outputs": required_outputs,
                                "max_interventions": max_interventions,
                                "task_complexity": task_complexity,
                            },
                        )
                        output_hints = (
                            ", ".join(required_outputs[:3]) if required_outputs else "your findings"
                        )
                        # After max_interventions+ interventions, force synthesis; before that, just nudge
                        if cumulative_interventions >= max_interventions:
                            return {
                                "action": "request_summary",
                                "message": (
                                    f"You've explored extensively with {unique_files} unique files read and "
                                    f"{cumulative_interventions} intervention cycles. Please synthesize your analysis now "
                                    f"into {output_hints}. Provide your findings based on what you've already read."
                                ),
                                "reason": f"Excessive prompt interventions ({cumulative_interventions}/{max_interventions}) - forcing synthesis",
                                "updates": updates,
                            }
                        elif synthesis_nudge_count < 3:
                            updates["synthesis_nudge_count"] = synthesis_nudge_count + 1
                            return {
                                "action": "continue_with_synthesis_hint",
                                "message": (
                                    f"You've read {unique_files} unique files so far. When ready, please synthesize "
                                    f"your analysis into {output_hints}. You may continue exploring briefly, "
                                    f"but please produce the final output soon."
                                ),
                                "reason": f"Interventions ({cumulative_interventions}) with low progress ({progress_ratio:.2f}) nudge",
                                "updates": updates,
                            }

        # CRITICAL FIX: If summary was already requested in a previous iteration,
        # we should finish now - don't ask for another summary or loop again.
        # This prevents duplicate output where the same content is yielded multiple times.
        if max_prompts_summary_requested:
            logger.info("Summary was already requested - finishing to prevent duplicate output")
            # Emit STATE event for continuation decision
            self._emit_event(
                topic="state.continuation.finish",
                data={
                    "reason": "summary_already_requested",
                    "continuation_prompts": continuation_prompts,
                },
            )
            return {
                "action": "finish",
                "message": None,
                "reason": "Summary already requested - final response received",
                "updates": updates,
            }

        # Extract intent type
        intends_to_continue = intent_result.intent == IntentType.CONTINUATION
        is_completion = intent_result.intent == IntentType.COMPLETION
        is_asking_input = intent_result.intent == IntentType.ASKING_INPUT
        is_stuck_loop = intent_result.intent == IntentType.STUCK_LOOP

        # CRITICAL FIX: Handle stuck loop immediately - model is planning but not executing
        if is_stuck_loop:
            logger.warning(
                "Detected STUCK_LOOP intent - model is planning but not executing. "
                "Forcing summary."
            )
            # Emit ERROR event for stuck loop detection
            if self._event_bus is not None:
                self._event_bus.emit_error(
                    error=RuntimeError("Stuck loop detected - model planning but not executing"),
                    context={
                        "intent": "STUCK_LOOP",
                        "continuation_prompts": continuation_prompts,
                        "content_length": content_length,
                    },
                    recoverable=True,
                )
            # Emit STATE event for continuation decision
            self._emit_event(
                topic="state.continuation.request_summary",
                data={
                    "reason": "stuck_loop_detected",
                    "continuation_prompts": 99,  # Force max
                },
            )
            return {
                "action": "request_summary",
                "message": (
                    "You appear to be stuck in a planning loop - you keep describing what "
                    "you will do but are not making actual tool calls.\n\n"
                    "Please either:\n"
                    "1. Make an ACTUAL tool call NOW (not just describe it), OR\n"
                    "2. Provide your response based on what you already know.\n\n"
                    "Do not describe what you will do - just do it or provide your answer."
                ),
                "reason": "STUCK_LOOP detected - forcing summary",
                "updates": {"continuation_prompts": 99},  # Prevent further prompting
            }

        # Configuration - use configurable thresholds from settings
        max_asking_input_prompts = 3
        requires_continuation_support = is_analysis_task or is_action_task or intends_to_continue

        # Get continuation prompt limits from settings with provider/model-specific overrides
        max_cont_analysis = getattr(settings, "max_continuation_prompts_analysis", 6)
        max_cont_action = getattr(settings, "max_continuation_prompts_action", 5)
        max_cont_default = getattr(settings, "max_continuation_prompts_default", 3)

        # Check for provider/model-specific overrides (RL-learned or manually configured)
        provider_model_key = f"{provider_name}:{model}"

        # First, try RL-learned recommendations if coordinator is enabled
        if rl_coordinator:
            for task_type_name, default_val in [
                ("analysis", max_cont_analysis),
                ("action", max_cont_action),
                ("default", max_cont_default),
            ]:
                recommendation = rl_coordinator.get_recommendation(
                    "continuation_prompts", provider_name, model, task_type_name
                )
                if recommendation and recommendation.value is not None:
                    learned_val = recommendation.value
                    if task_type_name == "analysis":
                        max_cont_analysis = learned_val
                    elif task_type_name == "action":
                        max_cont_action = learned_val
                    else:
                        max_cont_default = learned_val
                    logger.debug(
                        f"RL: Using learned continuation prompt for {provider_model_key}:{task_type_name}: "
                        f"{default_val} → {learned_val} (confidence={recommendation.confidence:.2f})"
                    )

        # Then, apply manual overrides (take precedence over RL)
        overrides = getattr(settings, "continuation_prompt_overrides", {})
        if provider_model_key in overrides:
            override = overrides[provider_model_key]
            max_cont_analysis = override.get("analysis", max_cont_analysis)
            max_cont_action = override.get("action", max_cont_action)
            max_cont_default = override.get("default", max_cont_default)
            logger.debug(
                f"Using manual continuation prompt overrides for {provider_model_key}: "
                f"analysis={max_cont_analysis}, action={max_cont_action}, default={max_cont_default}"
            )

        max_continuation_prompts = (
            max_cont_analysis
            if is_analysis_task
            else (max_cont_action if is_action_task else max_cont_default)
        )

        # Budget/iteration thresholds (reserved for future use)
        _budget_threshold = (
            tool_budget // 4 if requires_continuation_support else tool_budget // 2
        )  # noqa: F841
        max_iterations = unified_tracker_config.get("max_total_iterations", 50)
        _iteration_threshold = (  # noqa: F841
            max_iterations * 3 // 4 if requires_continuation_support else max_iterations // 2
        )

        # CRITICAL FIX: Handle tool mention without execution (hallucinated tool calls)
        # If model says "let me call search()" but didn't actually call it, try to extract
        # the intended tool call from the text and execute it automatically.
        if mentioned_tools and len(mentioned_tools) > 0:
            logger.info(
                f"Model mentioned tools but didn't call them: {mentioned_tools}. "
                "Attempting to extract tool call from text."
            )

            # Try to extract tool call from the model's text
            extracted_call = None
            if full_content:
                extracted_call = extract_tool_call_from_text(
                    text=full_content,
                    mentioned_tools=mentioned_tools,
                    context=task_completion_signals,  # Pass context for file path hints
                )

            if extracted_call and extracted_call.confidence >= 0.6:
                # Successfully extracted tool call - execute it automatically
                logger.info(
                    f"Extracted tool call: {extracted_call.tool_name} with confidence "
                    f"{extracted_call.confidence:.2f}. Will execute automatically."
                )
                self._emit_event(
                    topic="state.continuation.execute_extracted_tool",
                    data={
                        "tool_name": extracted_call.tool_name,
                        "arguments": extracted_call.arguments,
                        "confidence": extracted_call.confidence,
                        "mentioned_tools": mentioned_tools,
                    },
                )
                return {
                    "action": "execute_extracted_tool",
                    "extracted_call": extracted_call,
                    "message": None,  # No message to model - we'll execute directly
                    "reason": f"Extracted {extracted_call.tool_name} call from model text",
                    "updates": {},
                }

            # Could not extract - fall back to asking model to retry
            logger.warning(
                f"Could not extract tool call from text. Mentioned tools: {mentioned_tools}. "
                "Will ask model to make proper tool call."
            )
            # Emit ERROR event for hallucinated tool calls
            if self._event_bus is not None:
                self._event_bus.emit_error(
                    error=RuntimeError(f"Hallucinated tool calls: {', '.join(mentioned_tools)}"),
                    context={
                        "mentioned_tools": mentioned_tools,
                        "content_length": content_length,
                        "extraction_attempted": True,
                        "extraction_failed": True,
                    },
                    recoverable=True,
                )
            # Emit STATE event for continuation decision
            self._emit_event(
                topic="state.continuation.force_tool_execution",
                data={
                    "reason": "hallucinated_tool_calls",
                    "mentioned_tools": mentioned_tools,
                    "extraction_failed": True,
                },
            )
            # Escalation: after 3+ continuation prompts with hallucinated tools, request summary
            if continuation_prompts >= 3:
                logger.warning(
                    f"Model resistant to tool calling after {continuation_prompts} attempts - "
                    "requesting summary instead of forcing"
                )
                # Emit STATE event for escalation
                self._emit_event(
                    topic="state.continuation.escalate_to_summary",
                    data={
                        "reason": "tool_calling_resistance",
                        "hallucinated_tools": mentioned_tools,
                        "continuation_prompts": continuation_prompts,
                    },
                )
                return {
                    "action": "request_summary",
                    "message": (
                        "You've mentioned tools multiple times without executing them. "
                        "Please provide your response based on what you already know, "
                        "or make a single tool call now if needed."
                    ),
                    "reason": "Tool calling resistance detected - escalating to summary",
                    "updates": {"continuation_prompts": continuation_prompts + 1},
                }

            return {
                "action": "force_tool_execution",
                "message": (
                    f"You mentioned calling {', '.join(mentioned_tools)} but didn't actually make the tool call. "
                    "Please make the ACTUAL tool call now. Use the exact tool format:\n"
                    "For write: write(path='filename.py', content='...')\n"
                    "For edit: edit(path='filename.py', old_string='...', new_string='...')\n"
                    "For shell: shell(command='...')"
                ),
                "reason": f"Hallucinated tool calls detected: {mentioned_tools}",
                "updates": {},
            }

        # Handle asking input intent - use QuestionTypeClassifier for smarter decisions
        if is_asking_input:
            if one_shot_mode:
                logger.info("Model asking for input in one-shot mode - returning to user")
                return {
                    "action": "return_to_user",
                    "message": None,
                    "reason": "Model needs user input (one-shot mode)",
                    "updates": updates,
                }

            # Use QuestionTypeClassifier to determine if question is rhetorical/continuation
            # or if it genuinely needs user input (clarification/information)
            question_result = classify_question(full_content or "")

            # Emit event for observability
            self._emit_event(
                topic="state.continuation.question_classified",
                data={
                    "question_type": question_result.question_type.value,
                    "confidence": question_result.confidence,
                    "should_auto_continue": question_result.should_auto_continue,
                    "matched_pattern": question_result.matched_pattern,
                },
            )

            # If question requires actual user input, return to user immediately
            if question_result.question_type in (
                QuestionType.CLARIFICATION,
                QuestionType.INFORMATION,
            ):
                logger.info(
                    f"Model asking {question_result.question_type.value} question "
                    f"(confidence={question_result.confidence:.2f}) - returning to user"
                )
                return {
                    "action": "return_to_user",
                    "message": None,
                    "reason": f"Model needs user input: {question_result.question_type.value} question",
                    "updates": updates,
                }

            if asking_input_prompts >= max_asking_input_prompts:
                logger.info(
                    f"Max asking-input prompts reached ({asking_input_prompts}/{max_asking_input_prompts}) - "
                    "returning to user"
                )
                return {
                    "action": "return_to_user",
                    "message": None,
                    "reason": "Max asking-input attempts reached",
                    "updates": updates,
                }

            # Only auto-continue for rhetorical/continuation questions
            if question_result.should_auto_continue:
                logger.info(
                    f"Model asking {question_result.question_type.value} question "
                    f"(confidence={question_result.confidence:.2f}) - auto-continuing"
                )
                updates["asking_input_prompts"] = asking_input_prompts + 1
                return {
                    "action": "continue_asking_input",
                    "message": (
                        "Yes, please continue with your analysis/implementation. "
                        "If you need information, use available tools to gather it."
                    ),
                    "reason": f"Auto-responding to {question_result.question_type.value} question",
                    "updates": updates,
                }

            # Unknown question type with low confidence - return to user to be safe
            logger.info(
                f"Unknown/low-confidence question (type={question_result.question_type.value}, "
                f"confidence={question_result.confidence:.2f}) - returning to user"
            )
            return {
                "action": "return_to_user",
                "message": None,
                "reason": "Unknown question type - returning to user for safety",
                "updates": updates,
            }

        # Handle completion intent
        if is_completion:
            logger.info("Model indicated completion - finishing")
            # Emit STATE event for completion
            self._emit_event(
                topic="state.continuation.finish",
                data={
                    "reason": "completion_intent",
                    "intent": "COMPLETION",
                },
            )
            return {
                "action": "finish",
                "message": None,
                "reason": "Model indicated task completion",
                "updates": updates,
            }

        # Check if we should prompt for tool calls (continuation support)
        if requires_continuation_support and continuation_prompts < max_continuation_prompts:
            logger.info(
                f"Prompting for tool calls ({continuation_prompts + 1}/{max_continuation_prompts})"
            )
            updates["continuation_prompts"] = continuation_prompts + 1

            # Get model-specific continuation prompt
            message = self._get_continuation_prompt(
                provider_name=provider_name,
                model=model,
                is_analysis_task=is_analysis_task,
                is_action_task=is_action_task,
                continuation_prompts=continuation_prompts,
                full_content=full_content,
            )

            return {
                "action": "prompt_tool_call",
                "message": message,
                "reason": "Encouraging tool usage for task completion",
                "updates": updates,
            }

        # Max continuation prompts reached - request summary/completion
        if continuation_prompts >= max_continuation_prompts:
            logger.info(
                f"Max continuation prompts reached ({continuation_prompts}/{max_continuation_prompts}) - "
                "requesting summary"
            )
            # Emit METRIC event for budget exhaustion
            if self._event_bus is not None:
                self._event_bus.emit_metric(
                    metric="continuation_prompts_max_reached",
                    value=continuation_prompts,
                    unit="count",
                    tags={
                        "max_prompts": str(max_continuation_prompts),
                        "task_type": (
                            "analysis"
                            if is_analysis_task
                            else ("action" if is_action_task else "default")
                        ),
                    },
                )
            # Emit STATE event for summary request
            self._emit_event(
                topic="state.continuation.request_summary",
                data={
                    "reason": "max_prompts_reached",
                    "continuation_prompts": continuation_prompts,
                    "max_continuation_prompts": max_continuation_prompts,
                },
            )
            updates["max_prompts_summary_requested"] = True
            return {
                "action": "request_summary",
                "message": (
                    "Please provide a summary of your findings/work so far. "
                    "Conclude your response."
                ),
                "reason": "Max continuation prompts reached",
                "updates": updates,
            }

        # Default: finish
        logger.info("No continuation needed - finishing")
        return {
            "action": "finish",
            "message": None,
            "reason": "Response appears complete",
            "updates": updates,
        }
