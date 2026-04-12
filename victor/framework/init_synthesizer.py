"""Init.md synthesis via Agent framework.

Generates project context files using the full AgentOrchestrator
infrastructure (logging, events, GEPA traces, tool execution).

Two modes:
- synthesize(base_content, agent=ctx.agent) — reuses existing orchestrator (slash command)
- synthesize(base_content, provider="ollama") — creates fresh Agent (CLI)
- synthesize_with_tools(agent=...) — fallback when victor-coding unavailable
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from victor.agent.orchestrator import AgentOrchestrator

logger = logging.getLogger(__name__)

# Frame: fixed structure that wraps the evolvable RULES section.
# GEPA evolves SYNTHESIS_RULES only; the frame is never mutated.
_SYNTHESIS_FRAME_BEFORE = """You are writing an init.md file — a compact system-prompt context
file that an AI coding assistant reads at the start of every conversation about this codebase.

Below is raw auto-generated data about the project. Your job is to SYNTHESIZE it into a
compact, high-signal init.md (target: 80–120 lines, under 2000 tokens).

{rules}

RAW DATA:

```markdown
{base_content}
```

Return ONLY the final init.md markdown. No preamble, no explanation."""

# The evolvable rules section — this is what GEPA v2 optimizes.
SYNTHESIS_RULES = """RULES:
- Write project-specific content only. No generic advice.
- Use these sections in order: Project Overview, Package Layout, Key Entry Points,
  Development Commands, Dependencies, Configuration, Architecture Notes, Codebase Scale.
- Project Overview: 2–3 sentences covering what the project is, its primary language,
  and its key capabilities.
- Package Layout: table with Path, Description. Only include directories that exist.
- Key Entry Points: table with Component, Path, Description. Pick the 8–12 most
  architecturally important classes/functions — entry points, facades, registries,
  core abstractions. NOT alphabetically sorted Manager classes.
- Development Commands: the essential build/test/run commands in a code block.
- Dependencies: one line listing core deps count and top 5–8 packages.
- Configuration: 1–2 lines on how config works.
- Architecture Notes: 3–5 bullet points on the system's key architectural patterns,
  data flow, or design decisions. INCLUDE graph insights if present: inheritance backbone
  (most-subclassed base classes), hub classes (high connectivity), coupling hotspots,
  and key modules by role.
- Codebase Scale: one line with total symbols, files, and graph edges if available.
- OMIT these sections entirely (they waste tokens): Analyzer Coverage, Performance Hints,
  Embeddings & Chunking, raw PageRank numbers, graph node IDs.
- Do NOT include sections about the indexer or analysis tooling.
- End with a one-line note: "Run `/init --update` to refresh after code changes.\""""


def _build_synthesis_prompt(base_content: str, rules: str = SYNTHESIS_RULES) -> str:
    """Assemble the full synthesis prompt from frame + rules + data."""
    return _SYNTHESIS_FRAME_BEFORE.format(rules=rules, base_content=base_content)


# Public constant for backward compatibility (assembled with static rules)
SYNTHESIS_PROMPT = _build_synthesis_prompt("{base_content}")
# Note: SYNTHESIS_PROMPT.format(base_content=...) still works because
# _build_synthesis_prompt leaves {base_content} as a literal when called
# with "{base_content}" as the data argument... except it would double-format.
# So we define it directly for backward compat:
SYNTHESIS_PROMPT = _SYNTHESIS_FRAME_BEFORE.format(
    rules=SYNTHESIS_RULES, base_content="{base_content}"
)

TOOLS_FALLBACK_PROMPT = """Analyze this codebase and generate a comprehensive init.md file.

Use the available tools (overview, ls, read) to understand the project structure,
key components, and architecture. Then write a concise init.md with sections:
Project Overview, Package Layout, Key Entry Points, Development Commands,
Dependencies, Configuration, Architecture Notes.

Target: 80-120 lines, project-specific content only, no generic advice.
Return ONLY the init.md markdown content."""


class InitSynthesizer:
    """Generates init.md via Agent framework for full observability.

    Provides two synthesis modes:
    - synthesize(): Takes pre-built base_content (from CodebaseAnalyzer + graph)
      and synthesizes via LLM. Used when victor-coding is available.
    - synthesize_with_tools(): Uses Agent tools (overview, ls, read) to gather
      context and synthesize in one pass. Fallback when victor-coding unavailable.

    Both modes support agent reuse (slash command) or fresh Agent creation (CLI).
    """

    async def synthesize(
        self,
        base_content: str,
        *,
        agent: Optional["AgentOrchestrator"] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """Synthesize init.md from pre-built base content via Agent framework.

        Args:
            base_content: Raw analysis data (from CodebaseAnalyzer + graph insights).
            agent: Existing orchestrator to reuse (from slash command context).
            provider: Provider name for fresh Agent (CLI path). Ignored if agent given.
            model: Model name for fresh Agent. Ignored if agent given.

        Returns:
            Synthesized init.md markdown content.
        """
        # Check for GEPA-evolved RULES section (not the full prompt)
        evolved_rules = self._get_evolved_rules(provider)
        if evolved_rules:
            prompt = _build_synthesis_prompt(base_content, rules=evolved_rules)
            logger.info(
                "[init] Using GEPA-evolved rules (%d chars) in fixed frame",
                len(evolved_rules),
            )
        else:
            prompt = _build_synthesis_prompt(base_content)

        if agent:
            return await self._run_with_orchestrator(agent, prompt)
        else:
            return await self._run_with_fresh_agent(prompt, provider, model)

    async def synthesize_with_tools(
        self,
        *,
        agent: Optional["AgentOrchestrator"] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """Fallback: use Agent tools to gather context and synthesize init.md.

        Used when victor-coding is not installed — no CodebaseAnalyzer or graph
        analysis available. The Agent uses tools (overview, ls, read) to explore
        the codebase and generate init.md in one pass.

        Unlike synthesize(), this DOES use the agentic loop (needs tool calling).
        """
        if agent:
            return await self._run_with_orchestrator(agent, TOOLS_FALLBACK_PROMPT)
        else:
            return await self._run_agent_with_tools(
                TOOLS_FALLBACK_PROMPT, provider, model, vertical="coding"
            )

    async def _run_with_orchestrator(
        self, agent: "AgentOrchestrator", prompt: str
    ) -> str:
        """Run synthesis using an existing orchestrator (slash command path)."""
        try:
            response = await agent.chat(prompt)
            content = response.content if response else ""
            return self._clean(content)
        except Exception as e:
            logger.warning("Init synthesis via orchestrator failed: %s", e)
            return ""

    async def _run_with_fresh_agent(
        self,
        prompt: str,
        provider: Optional[str],
        model: Optional[str],
        vertical: Optional[str] = None,
    ) -> str:
        """Run synthesis using a direct provider call with framework logging.

        Uses ProviderRegistry (not Agent.run()) to avoid the agentic loop:
        - No tool calling, no continuation nudges, no multi-turn
        - Single LLM call: prompt in → markdown out
        - Still gets provider-level logging (API_CALL_START/SUCCESS)
        """
        try:
            from victor.providers.base import Message
            from victor.providers.registry import ProviderRegistry

            if not provider:
                from victor.config.settings import load_settings

                settings = load_settings()
                provider = getattr(settings, "default_provider", "ollama")
                model = model or getattr(settings, "default_model", None)

            provider_instance = ProviderRegistry.create(provider)
            if not provider_instance:
                logger.warning("Could not create provider %s", provider)
                return ""

            import time as _time

            messages = [Message(role="user", content=prompt)]

            logger.info(
                "[init→LLM] provider=%s model=%s prompt_chars=%d prompt_lines=%d",
                provider, model, len(prompt), prompt.count("\n"),
            )
            _start = _time.monotonic()

            response = await provider_instance.chat(
                messages=messages,
                model=model,
                temperature=0.7,
                max_tokens=4096,
            )
            _elapsed_ms = (_time.monotonic() - _start) * 1000
            content = response.content if response else ""
            result = self._clean(content)

            logger.info(
                "[init←LLM] provider=%s model=%s duration=%.1fs "
                "response_chars=%d response_lines=%d usage=%s",
                provider, model, _elapsed_ms / 1000,
                len(result), result.count("\n"),
                getattr(response, "usage", None),
            )

            # Log to usage.jsonl for GEPA/MIPROv2/CoT learning
            try:
                from victor.observability.analytics.logger import UsageLogger
                from pathlib import Path

                logs_dir = Path.home() / ".victor" / "logs"
                logs_dir.mkdir(parents=True, exist_ok=True)
                usage = UsageLogger(log_file=logs_dir / "usage.jsonl", enabled=True)

                # Emit task_classification so GEPA trace collection
                # categorizes this session as init_synthesis (not 'default')
                usage.log_event("task_classification", {
                    "task_type": "init_synthesis",
                    "provider": provider,
                    "model": model,
                })

                usage.log_event("tool_call", {
                    "tool_name": "init_synthesis",
                    "tool_args": {
                        "provider": provider,
                        "model": model,
                        "prompt_chars": len(prompt),
                    },
                })
                usage.log_event("tool_result", {
                    "tool_name": "init_synthesis",
                    "success": bool(result),
                    "duration_ms": round(_elapsed_ms, 1),
                    "result_lines": result.count("\n"),
                    "result_chars": len(result),
                })

                # Log init output quality signal for GEPA evolution
                self._log_init_quality(usage, result)
            except Exception:
                pass  # Usage logging is best-effort

            await provider_instance.close()
            return result
        except Exception as e:
            logger.warning("Init synthesis via provider failed: %s", e)
            return ""

    async def _run_agent_with_tools(
        self,
        prompt: str,
        provider: Optional[str],
        model: Optional[str],
        vertical: Optional[str] = None,
    ) -> str:
        """Run synthesis with Agent tools (fallback path — needs agentic loop)."""
        try:
            from victor.framework.agent import Agent

            kwargs: dict[str, Any] = {"enable_observability": True}
            if provider:
                kwargs["provider"] = provider
            if model:
                kwargs["model"] = model
            if vertical:
                kwargs["vertical"] = vertical

            agent = await Agent.create(**kwargs)
            result = await agent.run(prompt)
            return self._clean(result.content) if result.success else ""
        except Exception as e:
            logger.warning("Init synthesis with tools failed: %s", e)
            return ""

    @staticmethod
    def _clean(content: str) -> str:
        """Clean LLM output — strip code fences, validate markdown."""
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            lines = lines[1:] if lines[0].startswith("```") else lines
            lines = lines[:-1] if lines and lines[-1].strip() == "```" else lines
            content = "\n".join(lines)
        return content

    @staticmethod
    def _log_init_quality(usage: Any, result: str) -> None:
        """Score init output quality and log for GEPA learning.

        Quality signals:
        - Section completeness: how many expected sections are present
        - Line count: within 80-120 target range
        - Conciseness: no bloated or trivially short output
        """
        expected_sections = [
            "Project Overview", "Package Layout", "Key Entry Points",
            "Development Commands", "Dependencies", "Configuration",
            "Architecture Notes", "Codebase Scale",
        ]
        sections_found = sum(1 for s in expected_sections if s in result)
        section_score = sections_found / len(expected_sections)

        line_count = result.count("\n") + 1
        # Penalize being outside 60-140 range (relaxed from 80-120)
        if 60 <= line_count <= 140:
            length_score = 1.0
        elif 40 <= line_count <= 200:
            length_score = 0.7
        else:
            length_score = 0.3

        # Combined quality score
        quality_score = section_score * 0.6 + length_score * 0.2 + (0.2 if result else 0.0)

        usage.log_event("init_quality", {
            "sections_found": sections_found,
            "sections_total": len(expected_sections),
            "section_score": round(section_score, 2),
            "line_count": line_count,
            "length_score": round(length_score, 2),
            "quality_score": round(quality_score, 2),
            "char_count": len(result),
        })

    @staticmethod
    def _get_evolved_rules(provider: Optional[str] = None) -> Optional[str]:
        """Check if GEPA has evolved RULES for the init synthesis prompt.

        Returns the evolved RULES text if a candidate exists with sufficient
        confidence, otherwise None (caller uses static SYNTHESIS_RULES).
        The rules are injected into the fixed frame — {base_content} is
        always safe because the frame handles it.

        Gated by prompt_optimization.enabled setting.
        """
        try:
            from victor.config.settings import get_settings

            po = getattr(get_settings(), "prompt_optimization", None)
            if po is None or not po.enabled:
                return None
            strategies = po.get_strategies_for_section("INIT_SYNTHESIS_RULES")
            if not strategies:
                return None
        except Exception:
            return None

        try:
            from victor.framework.rl.coordinator import get_rl_coordinator

            coordinator = get_rl_coordinator()
            learner = coordinator.get_learner("prompt_optimizer")
            if learner is None:
                return None
            rec = learner.get_recommendation(
                provider or "default",
                "",  # model not relevant for rules template
                "init_synthesis",
                section_name="INIT_SYNTHESIS_RULES",
            )
            if rec and rec.confidence > 0.6 and not rec.is_baseline:
                logger.info(
                    "Using GEPA-evolved init rules "
                    "(gen=%s, confidence=%.2f, %d chars)",
                    rec.reason,
                    rec.confidence,
                    len(rec.value),
                )
                return rec.value
        except Exception:
            pass
        return None
