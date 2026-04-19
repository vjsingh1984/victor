"""GEPA v2 reflection/mutation service.

Wraps a provider with GEPA-specific prompts for the three core operations:
- reflect(): Analyze ASI execution traces, diagnose failure patterns
- mutate(): Generate improved prompt section from reflection
- merge(): Combine strengths of two Pareto-optimal candidates

Follows the DecisionService pattern: provider-agnostic, configurable
via settings, sync interface using run_sync_in_thread.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

REFLECT_SYSTEM = """You are a prompt engineering expert analyzing execution traces \
for an AI coding agent. Your goal is to diagnose WHY the agent's current prompt \
guidance causes failures, and propose specific fixes.

You will receive:
1. Aggregated execution trace data (tool calls, reasoning, results, errors)
2. The current prompt section being evaluated

Produce exactly 3-5 specific, actionable bullet points. Each must:
- Reference a concrete failure pattern from the traces
- Propose a precise wording change to the prompt
- Explain the expected impact

Do NOT produce vague advice like "improve clarity". Be specific."""

MUTATE_SYSTEM = """You are rewriting a prompt section for an AI coding agent. \
You will receive the current prompt text and a reflection analyzing its failures.

Requirements:
- Address EVERY failure pattern from the reflection
- Keep the output under {max_chars} characters (HARD LIMIT)
- Be specific and actionable — no vague platitudes
- Preserve the section's core purpose
- Output ONLY the improved prompt text — no explanation, no preamble

Current length: {current_len} characters. Target: under {max_chars} characters."""

MERGE_SYSTEM = """You are combining two prompt section variants for an AI coding \
agent. Each variant excels at different task types.

Candidate A:
{candidate_a}

Candidate B:
{candidate_b}

Create a merged version that preserves the strengths of both. Requirements:
- Keep under {max_chars} characters
- Identify what each does best and unify
- Output ONLY the merged text — no explanation"""


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class GEPAServiceProtocol(Protocol):
    """Protocol for GEPA reflection and mutation LLM calls."""

    def reflect(self, traces_summary: str, section_name: str, current_text: str) -> str: ...

    def mutate(
        self,
        current_text: str,
        reflection: str,
        section_name: str,
        max_chars: int,
    ) -> str: ...

    def merge(
        self,
        candidate_a: str,
        candidate_b: str,
        section_name: str,
        max_chars: int,
    ) -> str: ...

    def get_tier(self) -> str: ...


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


class GEPAService:
    """Wraps a provider with GEPA-specific prompts for reflect/mutate/merge.

    Sync interface — uses run_sync_in_thread internally since providers
    are async. Matches the existing GEPAStrategy calling convention.
    """

    def __init__(
        self,
        provider: Any,
        model: str,
        tier: str = "balanced",
        max_prompt_chars: int = 1500,
        timeout_s: float = 30.0,
        max_tokens: int = 1000,
    ):
        self._provider = provider
        self._model = model
        self._tier = tier
        self._max_prompt_chars = max_prompt_chars
        self._timeout_s = timeout_s
        self._max_tokens = max_tokens

    def get_tier(self) -> str:
        return self._tier

    def reflect(self, traces_summary: str, section_name: str, current_text: str) -> str:
        """Analyze ASI traces and produce actionable reflection."""
        user_prompt = (
            f"Execution traces for section '{section_name}':\n\n"
            f"{traces_summary}\n\n"
            f"Current prompt section:\n{current_text[:1000]}\n\n"
            f"Diagnose the failure patterns and propose specific fixes."
        )
        result = self._call_llm(REFLECT_SYSTEM, user_prompt)
        if result:
            return result
        return f"[Reflection unavailable — {self._tier} tier LLM call failed]"

    def mutate(
        self,
        current_text: str,
        reflection: str,
        section_name: str,
        max_chars: int = 0,
    ) -> str:
        """Generate improved prompt section from reflection."""
        limit = max_chars or self._max_prompt_chars
        system = MUTATE_SYSTEM.format(max_chars=limit, current_len=len(current_text))
        user_prompt = (
            f"Section: {section_name}\n\n"
            f"Current text:\n{current_text}\n\n"
            f"Reflection on failures:\n{reflection}\n\n"
            f"Generate the improved version (under {limit} characters):"
        )
        result = self._call_llm(system, user_prompt, max_tokens=self._max_tokens)
        if result:
            # Hard-truncate if LLM exceeds limit
            if len(result) > limit:
                result = result[:limit]
                logger.info(
                    "GEPA mutate output truncated from %d to %d chars",
                    len(result),
                    limit,
                )
            return result
        return current_text  # Fallback: no change

    def merge(
        self,
        candidate_a: str,
        candidate_b: str,
        section_name: str,
        max_chars: int = 0,
    ) -> str:
        """Combine strengths of two Pareto-optimal candidates."""
        limit = max_chars or self._max_prompt_chars
        system = MERGE_SYSTEM.format(
            candidate_a=candidate_a[:800],
            candidate_b=candidate_b[:800],
            max_chars=limit,
        )
        user_prompt = (
            f"Merge these two variants of '{section_name}' into one "
            f"that combines the best of both. Under {limit} characters."
        )
        result = self._call_llm(system, user_prompt, max_tokens=self._max_tokens)
        if result and len(result) > limit:
            result = result[:limit]
        return result or candidate_a  # Fallback to first candidate

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        """Call the provider synchronously."""
        try:
            from victor.core.async_utils import run_sync_in_thread
            from victor.providers.base import Message

            # Suppress thinking for Qwen models
            effective_user = user_prompt
            if "qwen" in self._model.lower():
                effective_user = f"/no_think\n{user_prompt}"

            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=effective_user),
            ]
            response = run_sync_in_thread(
                self._provider.chat(
                    messages=messages,
                    model=self._model,
                    max_tokens=max_tokens or self._max_tokens,
                    temperature=0.7,
                ),
                timeout=self._timeout_s,
            )
            content = response.content if response else ""
            content = self._strip_thinking(content)
            if content and len(content) > 20:
                return content.strip()
        except Exception as e:
            logger.debug("GEPA %s tier LLM call failed: %s", self._tier, e)
        return None

    @staticmethod
    def _strip_thinking(content: str) -> str:
        """Strip <think> blocks and 'Thinking Process:' preamble."""
        if "<think>" in content:
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if "Thinking Process:" in content:
            parts = re.split(r"\n(?=[A-Z]{3,}[:\s])", content)
            for part in reversed(parts):
                if "Thinking Process" not in part and len(part.strip()) > 50:
                    return part.strip()
        return content
