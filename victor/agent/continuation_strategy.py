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
from typing import Any, Dict, List, Optional

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
OUTPUT_REQUIREMENT_PATTERNS: Dict[str, List[re.Pattern]] = {
    key: [re.compile(pattern) for pattern in patterns]
    for key, patterns in _OUTPUT_REQUIREMENT_PATTERNS_RAW.items()
}

# Cache for tool mention patterns (avoid recompiling per-tool patterns)
# Key: tool_name, Value: list of compiled patterns
_TOOL_MENTION_PATTERN_CACHE: Dict[str, List[re.Pattern]] = {}

# Common tool mention pattern templates (compiled once, tool name inserted)
_TOOL_MENTION_TEMPLATES = [
    r"\b(?:call|use|execute|run|invoke|perform)\s+{tool}\b",
    r"\b{tool}\s*\(",  # tool_name( or tool_name (
    r"\bthe\s+{tool}\s+tool\b",  # "the read tool"
]


def _get_tool_mention_patterns(tool_name: str) -> List[re.Pattern]:
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
        """Emit event with error handling.

        Args:
            topic: Event topic
            data: Event data
            source: Event source
        """
        if self._event_bus:
            try:
                import asyncio

                asyncio.run(
                    self._event_bus.emit(
                        topic=topic,
                        data={**data, "category": "state"},  # Preserve for observability
                        source=source,
                    )
                )
            except Exception as e:
                logger.debug(f"Failed to emit continuation event: {e}")

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

            # CUMULATIVE INTERVENTION CHECK: If we've had too many prompt interventions
            # across the session, nudge or force synthesis (prevents sessions that never finish)
            cumulative_interventions = task_completion_signals.get(
                "cumulative_prompt_interventions", 0
            )
            if cumulative_interventions >= 5 and is_analysis_task:
                # Log for observability
                logger.info(
                    f"Cumulative prompt interventions ({cumulative_interventions}) reached threshold - "
                    "nudging synthesis"
                )
                self._emit_event(
                    topic="state.continuation.cumulative_intervention_nudge",
                    data={
                        "cumulative_interventions": cumulative_interventions,
                        "required_outputs": required_outputs,
                        "read_files_count": len(read_files),
                    },
                )
                output_hints = (
                    ", ".join(required_outputs[:3]) if required_outputs else "your findings"
                )
                # After 8+ interventions, force synthesis; before that, just nudge
                if cumulative_interventions >= 8:
                    return {
                        "action": "request_summary",
                        "message": (
                            f"You've explored extensively with {len(read_files)} files read and "
                            f"multiple continuation prompts. Please synthesize your analysis now "
                            f"into {output_hints}. Provide your findings based on what you've already read."
                        ),
                        "reason": f"Excessive prompt interventions ({cumulative_interventions}) - forcing synthesis",
                        "updates": updates,
                    }
                elif synthesis_nudge_count < 3:
                    updates["synthesis_nudge_count"] = synthesis_nudge_count + 1
                    return {
                        "action": "continue_with_synthesis_hint",
                        "message": (
                            f"You've read {len(read_files)} files so far. When ready, please synthesize "
                            f"your analysis into {output_hints}. You may continue exploring briefly, "
                            f"but please produce the final output soon."
                        ),
                        "reason": f"Cumulative interventions ({cumulative_interventions}) nudge",
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

            # Determine message based on task type
            if is_analysis_task:
                message = (
                    "Continue your analysis. Use tools like read_file, list_directory, "
                    "code_search to gather more information."
                )
            elif is_action_task:
                message = (
                    "Continue with the implementation. Use tools like write_file, "
                    "edit_files to make the necessary changes."
                )
            else:
                message = "Continue. Use appropriate tools if needed."

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
            self._event_bus.emit_metric(
                metric_name="continuation_prompts_max_reached",
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
