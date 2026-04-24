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
Think of it like a CLAUDE.md or Cursor rules file: the goal is to give an AI assistant
enough context to navigate the codebase confidently without reading every file.

Below is raw auto-generated data about the project (symbol graph + project docs).
Your job is to SYNTHESIZE it into a compact, high-signal init.md (100–150 lines, <2500 tokens).

First: detect the primary language(s) and project type from the raw data. Then apply
language-appropriate patterns. An AI reading this file should immediately understand
the architecture, key design patterns, where to find important code, and how pieces fit
together. Do NOT re-list the raw data — INTERPRET and SYNTHESIZE it.

{rules}

RAW DATA:

```markdown
{base_content}
```

Return ONLY the final init.md markdown. No preamble, no explanation."""

# The evolvable rules section — this is what GEPA v2 optimizes.
SYNTHESIS_RULES = """RULES:
- Write project-specific content only. No generic advice.
- Sections in order: Project Overview, System Flow, Package Layout, Key Entry Points,
  Architecture Patterns, Development Commands, Dependencies, Configuration, Codebase Scale.

- Project Overview: 3–4 sentences: what the project does, primary language/framework,
  key capabilities, intended users. Include version/release info if present.

- System Flow: one arrow-chain line showing end-to-end data/control flow. Examples:
  - Web service:    "Request → Router → Handler → Service → Repository → DB"
  - CLI tool:       "User → CLI Parser → Command → Core Logic → Output"
  - Library/SDK:    "User Code → Public API → Core Engine → Backends/Adapters"
  - Agent/AI:       "User → Orchestrator → Provider → Tools → Storage"
  - Compiler/tool:  "Source → Lexer → Parser → AST → Codegen → Output"

- Package Layout: table (Path | Type | Description). Language conventions:
  - Go:        cmd/ (binaries), pkg/ (libraries), internal/ (private), api/ (protos)
  - Rust:      src/ (lib root), each workspace crate, bin/ for binaries
  - Node/TS:   src/, packages/ (monorepo), lib/; omit dist/ and node_modules/
  - Python:    top-level package dir, scripts/, config files
  - Java/Kotlin: src/main/java, modules/ (multi-module), resources/
  Only include directories that actually exist. Omit build artifacts and test subfolders.

- Key Entry Points: table (Component | Type | Path:line | Description).
  Pick 10–15 most architecturally important: entry points, core abstractions, base
  classes/interfaces/traits, registries, facades, public API surfaces.
  Use real line numbers from raw data. Exclude alphabetically sorted utility classes.

- Architecture Patterns: MOST IMPORTANT SECTION. 5–8 bullet points. Cite concrete names.
  Detect and report patterns appropriate to the language:

  Python patterns to look for:
  * Inheritance backbone — most-subclassed base classes with subclass count
  * Hub classes — highest-connectivity types (many imports/references)
  * Facade/Orchestrator — large coordinator classes delegating to sub-components
  * Plugin/Extension system — how plugins register (entry_points, subclassing, decorators)
  * Async-first design — asyncio coroutine boundaries, sync/async adapter layers
  * Feature flag / settings gating — config-driven behavior variants

  Go patterns to look for:
  * Interface segregation — key interfaces and their main implementors
  * Constructor injection — how dependencies are wired (wire/dig/fx or manual)
  * Package cohesion — cmd/ vs internal/ vs pkg/ boundary discipline
  * Error handling — sentinel errors, wrapped errors (%w), custom error types
  * Concurrency — goroutine/channel patterns, sync.Mutex, errgroup usage

  TypeScript/Node patterns to look for:
  * Module system — ESM vs CJS, barrel exports (index.ts), path aliases
  * DI framework — NestJS decorators, inversify, tsyringe, or manual factories
  * Type hierarchy — discriminated unions, branded types, generic constraints
  * Async patterns — Promise, async/await, RxJS observables, event emitters
  * Monorepo structure — workspaces (nx/turborepo/pnpm), shared packages

  Rust patterns to look for:
  * Trait system — key traits and their main implementors (analogous to interfaces)
  * Ownership strategy — Arc<Mutex<>>, RwLock, channels for shared state
  * Error handling — thiserror/anyhow, custom Error enums, Result propagation
  * Crate architecture — workspace layout, public API surface via lib.rs pub use
  * Async runtime — tokio/async-std task spawning, Pin/Future patterns

  Java/Kotlin patterns to look for:
  * DI framework — Spring beans, Guice modules, Dagger components
  * Layered architecture — Controller → Service → Repository boundary
  * Domain model — key entities, value objects, aggregates (DDD markers)
  * Exception hierarchy — checked vs unchecked, custom exception classes

  Universal patterns (apply to any language):
  * Coupling hotspots — files/modules with highest fan-in (note as "many callers")
  * Extensibility mechanism — the primary way the system is extended by users/plugins
  * Configuration pattern — how config is loaded and propagated through the system

- Development Commands: essential commands in a code block (build, test, run, lint).
  Language defaults if not found in raw data:
  - Python:  pip install -e ".[dev]", pytest, ruff/black/mypy
  - Go:      go build ./..., go test ./..., go vet ./..., golangci-lint run
  - Node/TS: npm install, npm run build, npm test, npm run lint
  - Rust:    cargo build, cargo test, cargo clippy, cargo fmt
  - Java:    ./mvnw package, ./mvnw test  OR  ./gradlew build, ./gradlew test
  Include environment setup (venv activate, nvm use, etc.) if present in raw data.

- Dependencies: one line — package manager, total count, top 6–8 package names.
  Source from whichever manifest is present: pyproject.toml, package.json, go.mod,
  Cargo.toml, pom.xml, build.gradle.

- Configuration: 2–3 lines on how config works: sources, override order, key classes/files.

- Codebase Scale: one line — LOC, file count, symbol/node count and graph edges if
  available, test coverage if known.

- OMIT: Analyzer Coverage, Performance Hints, Embeddings & Chunking, raw PageRank
  numbers, graph node IDs, Learned from Conversations, Common Topics, FAQ, indexer
  internals, sections about the analysis tooling itself.
- End with: "Run `/init --update` to refresh after code changes.\""""


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


def _parse_makefile_targets(path: Any) -> str:
    """Extract phony/documented make targets (up to 1000 chars)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    targets = [
        line
        for line in lines
        if line
        and not line.startswith("\t")
        and ":" in line
        and not line.startswith("#")
        and len(line) < 100
    ][:25]
    return "\n".join(targets)[:1000]


def _read_task_file(path: Any) -> str:
    """Read justfile/Taskfile, trimmed to 1000 chars."""
    return path.read_text(encoding="utf-8")[:1000]


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
        # Enrich base_content with CLAUDE.md / existing project docs before synthesis.
        # This gives the LLM the architectural ground truth (patterns, service layer,
        # feature flags, dev commands) that static graph analysis alone can't surface.
        base_content = self._enrich_with_project_docs(base_content)

        # Pre-synthesis discovery: use graph tool for structural and semantic insights
        # This fills the "0 patterns" gap by finding loosely coupled relationships
        # and hotspots (coupling, centrality) before the LLM sees the data.
        discovery_data = await self._pre_synthesis_discovery(agent)
        if discovery_data:
            base_content = f"## Architectural Discovery (Heuristic + Semantic)\n\n{discovery_data}\n\n---\n\n{base_content}"

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
            # synthesize_with_tools intentionally uses the agentic loop — the agent
            # needs to call tools (overview, ls, read) to explore the codebase.
            # Use agent.chat() directly, NOT _run_with_orchestrator (which bypasses the loop).
            try:
                response = await agent.chat(TOOLS_FALLBACK_PROMPT)
                content = response.content if response else ""
                return self._clean(content)
            except Exception as e:
                logger.warning("Init synthesis with tools failed: %s", e)
                return ""
        else:
            return await self._run_agent_with_tools(
                TOOLS_FALLBACK_PROMPT, provider, model, vertical="coding"
            )

    async def _run_with_orchestrator(self, agent: "AgentOrchestrator", prompt: str) -> str:
        """Run synthesis using an existing orchestrator or framework Agent.

        Handles two agent types:
        - AgentOrchestrator: async chat() → CompletionResponse (slash command path)
        - victor.framework.agent.Agent: async run() → TaskResult (CLI path via _create_init_agent)
        """
        import inspect

        try:
            # Always prefer a direct provider call for init synthesis — bypass AgenticLoop.
            # AgentOrchestrator.chat() routes through USE_AGENTIC_LOOP → edge model (Ollama)
            # calls for task classification + tool selection even when we just need one LLM
            # call: prompt → markdown. Use the already-initialized provider directly instead.

            # 1. AgentOrchestrator (from _create_init_agent profile path or slash command)
            provider_instance = getattr(agent, "provider", None)
            model = getattr(agent, "model", None)

            # 2. Framework Agent (wraps orchestrator via _orchestrator)
            if provider_instance is None:
                inner = getattr(agent, "_orchestrator", None)
                if inner is not None:
                    provider_instance = getattr(inner, "provider", None)
                    model = getattr(inner, "model", None)

            if provider_instance is not None:
                logger.debug(
                    "[init] Using initialized provider %s (bypassing AgenticLoop)",
                    type(provider_instance).__name__,
                )
                return await self._call_initialized_provider(prompt, provider_instance, model)

            # 3. Last resort: fall back to agent.chat() for slash command contexts where
            #    the orchestrator's provider attribute may not be directly accessible.
            if inspect.iscoroutinefunction(getattr(agent, "chat", None)):
                logger.debug("[init] No direct provider found, falling back to agent.chat()")
                response = await agent.chat(prompt)
                content = response.content if response else ""
                return self._clean(content)

            # 4. Bare framework Agent with no accessible orchestrator
            provider_name = getattr(agent, "_provider", None)
            model = getattr(agent, "_model", None)
            logger.debug("[init] Falling back to fresh provider: %s", provider_name)
            return await self._run_with_fresh_agent(prompt, provider_name, model)
        except Exception as e:
            logger.warning("Init synthesis via orchestrator failed: %s", e)
            return ""

    async def _call_initialized_provider(
        self,
        prompt: str,
        provider_instance: Any,
        model: Optional[str],
    ) -> str:
        """Single direct call using an already-initialized provider instance.

        Used when a framework Agent is passed: reuses the provider that was
        initialized by AgentFactory (with keyring access) rather than creating a
        new instance via ProviderRegistry.create() which skips keyring in
        non-interactive mode.  This is the same mechanism `victor chat` uses.
        """
        import time as _time

        from victor.providers.base import Message

        provider_name = getattr(provider_instance, "name", type(provider_instance).__name__)
        messages = [Message(role="user", content=prompt)]

        logger.info(
            "[init→LLM] provider=%s model=%s prompt_chars=%d prompt_lines=%d (reused)",
            provider_name,
            model,
            len(prompt),
            prompt.count("\n"),
        )
        _start = _time.monotonic()

        chat_kwargs: dict = {"temperature": 0.7, "max_tokens": 4096}
        chat_kwargs["model"] = model
        response = await provider_instance.chat(messages=messages, **chat_kwargs)

        _elapsed_ms = (_time.monotonic() - _start) * 1000
        content = response.content if response else ""
        result = self._clean(content)

        logger.info(
            "[init←LLM] provider=%s model=%s duration=%.1fs "
            "response_chars=%d response_lines=%d usage=%s",
            provider_name,
            model,
            _elapsed_ms / 1000,
            len(result),
            result.count("\n"),
            getattr(response, "usage", None),
        )
        return result

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
                provider,
                model,
                len(prompt),
                prompt.count("\n"),
            )
            _start = _time.monotonic()

            # Only pass model when explicitly set — lets provider use its own default
            # when model=None (passing None overrides the method's default parameter)
            chat_kwargs: dict = {"temperature": 0.7, "max_tokens": 4096}
            chat_kwargs["model"] = model
            response = await provider_instance.chat(messages=messages, **chat_kwargs)
            _elapsed_ms = (_time.monotonic() - _start) * 1000
            content = response.content if response else ""
            result = self._clean(content)

            logger.info(
                "[init←LLM] provider=%s model=%s duration=%.1fs "
                "response_chars=%d response_lines=%d usage=%s",
                provider,
                model,
                _elapsed_ms / 1000,
                len(result),
                result.count("\n"),
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
                usage.log_event(
                    "task_classification",
                    {
                        "task_type": "init_synthesis",
                        "provider": provider,
                        "model": model,
                    },
                )

                usage.log_event(
                    "tool_call",
                    {
                        "tool_name": "init_synthesis",
                        "tool_args": {
                            "provider": provider,
                            "model": model,
                            "prompt_chars": len(prompt),
                        },
                    },
                )
                usage.log_event(
                    "tool_result",
                    {
                        "tool_name": "init_synthesis",
                        "success": bool(result),
                        "duration_ms": round(_elapsed_ms, 1),
                        "result_lines": result.count("\n"),
                        "result_chars": len(result),
                    },
                )

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

    async def _pre_synthesis_discovery(
        self, agent: Optional["AgentOrchestrator"] = None
    ) -> Optional[str]:
        """Perform automated discovery of architectural patterns and hotspots.

        Uses the graph tool (victor.tools.graph_tool) to:
        1. Find coupling hotspots via SQL-powered aggregate analysis.
        2. Find loosely coupled patterns (registries, decorators) via semantic edges.
        """
        try:
            from victor.tools.graph_tool import graph, GraphMode

            discovery_lines = []

            # 1. Structural Hotspots (SQL-Powered)
            # Find the most-imported modules — reliable indicator of "Hub" classes.
            coupling_sql = (
                "SELECT count(*) as count, dst "
                "FROM graph_edge "
                "WHERE type='IMPORTS' "
                "GROUP BY dst "
                "ORDER BY count DESC "
                "LIMIT 5"
            )
            try:
                coupling_result = await graph(mode=GraphMode.QUERY, query=coupling_sql)
                if coupling_result["success"] and coupling_result["result"]["results"]:
                    discovery_lines.append("### Highly Coupled Modules (Fan-in)")
                    for row in coupling_result["result"]["results"]:
                        discovery_lines.append(f"- {row['dst']} ({row['count']} importers)")
                    discovery_lines.append("")
            except Exception:
                pass

            # 2. Key Components Identification
            # Find modules with the most symbols — indicates core implementation depth.
            size_sql = (
                "SELECT file, count(*) as count "
                "FROM graph_node "
                "WHERE type IN ('class', 'function') "
                "GROUP BY file "
                "ORDER BY count DESC "
                "LIMIT 5"
            )
            try:
                size_result = await graph(mode=GraphMode.QUERY, query=size_sql)
                if size_result["success"] and size_result["result"]["results"]:
                    discovery_lines.append("### Module Implementation Depth")
                    for row in size_result["result"]["results"]:
                        discovery_lines.append(f"- {row['file']} ({row['count']} symbols)")
                    discovery_lines.append("")
            except Exception:
                pass

            # 3. Semantic Pattern Discovery
            # Attempt to find relationships for the top hub (from step 1).
            # This identifies registries, providers, etc.
            try:
                if coupling_result["success"] and coupling_result["result"]["results"]:
                    top_hub = coupling_result["result"]["results"][0]["dst"]
                    semantic_result = await graph(
                        mode=GraphMode.SEMANTIC, node=top_hub, threshold=0.6, top_k=5
                    )
                    if (
                        semantic_result["success"]
                        and semantic_result["result"]["potential_relationships"]
                    ):
                        discovery_lines.append(f"### Semantic Relationships for '{top_hub}'")
                        for rel in semantic_result["result"]["potential_relationships"]:
                            discovery_lines.append(
                                f"- {rel['name']} ({rel['file']}) - score: {rel['similarity']}"
                            )
                        discovery_lines.append("")
            except Exception:
                pass

            return "\n".join(discovery_lines) if discovery_lines else None

        except Exception as e:
            logger.debug(f"Pre-synthesis discovery failed: {e}")
            return None

    @staticmethod
    def _log_init_quality(usage: Any, result: str) -> None:
        """Score init output quality and log for GEPA learning.

        Quality signals:
        - Section completeness: how many expected sections are present
        - Line count: within 80-120 target range
        - Conciseness: no bloated or trivially short output
        """
        expected_sections = [
            "Project Overview",
            "System Flow",
            "Package Layout",
            "Key Entry Points",
            "Architecture Patterns",
            "Development Commands",
            "Dependencies",
            "Configuration",
            "Codebase Scale",
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

        usage.log_event(
            "init_quality",
            {
                "sections_found": sections_found,
                "sections_total": len(expected_sections),
                "section_score": round(section_score, 2),
                "line_count": line_count,
                "length_score": round(length_score, 2),
                "quality_score": round(quality_score, 2),
                "char_count": len(result),
            },
        )

    @staticmethod
    def _enrich_with_project_docs(base_content: str) -> str:
        """Prepend high-signal project docs to base_content before synthesis.

        Static graph analysis surfaces connectivity metrics but misses: project
        narrative, declared dependencies, dev commands, and architectural decisions.
        This enrichment step reads 4 targeted file categories to fill those gaps —
        deterministic and fast (~milliseconds), works for any language/framework.

        Reading budget (to avoid crowding out graph data):
          CLAUDE.md / AI rules: 5000 chars
          README:               1500 chars
          Build manifest:       1500 chars
          Task runner:          1000 chars
        """
        from pathlib import Path

        cwd = Path.cwd()
        enrichments: list[str] = []

        # 1. AI assistant rules files — highest signal: architecture, patterns, dev commands
        for name in (
            "CLAUDE.md",
            ".claude/CLAUDE.md",
            ".cursor/rules",
            ".github/copilot-instructions.md",
        ):
            path = cwd / name
            if path.exists():
                try:
                    text = path.read_text(encoding="utf-8")
                    trimmed = text[:5000]
                    if len(text) > 5000:
                        trimmed += "\n... (truncated)"
                    label = Path(name).name
                    enrichments.append(f"## Project AI Rules ({label})\n\n{trimmed}")
                    logger.info("[init] Enriched with %s (%d chars)", name, len(trimmed))
                    break
                except Exception:
                    pass

        # 2. README — project narrative, purpose, quick-start (language-agnostic)
        for name in ("README.md", "README.rst", "README.txt", "README"):
            path = cwd / name
            if path.exists():
                try:
                    text = path.read_text(encoding="utf-8")
                    trimmed = text[:1500]
                    if len(text) > 1500:
                        trimmed += "\n... (truncated)"
                    enrichments.append(f"## README (first 1500 chars)\n\n{trimmed}")
                    logger.info("[init] Enriched with %s (%d chars)", name, len(trimmed))
                    break
                except Exception:
                    pass

        # 3. Build manifest — declared dependencies, project metadata, scripts
        #    Priority: language-specific manifest wins over generic
        build_manifests = [
            # Python
            ("pyproject.toml", "pyproject.toml"),
            ("setup.cfg", "setup.cfg"),
            # Node/TS/Deno
            ("package.json", "package.json"),
            ("deno.json", "deno.json"),
            # Go
            ("go.mod", "go.mod"),
            # Rust
            ("Cargo.toml", "Cargo.toml"),
            # JVM
            ("pom.xml", "pom.xml"),
            ("build.gradle.kts", "build.gradle.kts"),
            ("build.gradle", "build.gradle"),
            # .NET
            ("*.csproj", None),  # glob — handled below
            # Ruby
            ("Gemfile", "Gemfile"),
            # PHP
            ("composer.json", "composer.json"),
            # Swift
            ("Package.swift", "Package.swift"),
        ]
        for name, _ in build_manifests:
            if name == "*.csproj":
                import glob as _glob

                matches = _glob.glob(str(cwd / "**" / "*.csproj"), recursive=True)
                if matches:
                    path = Path(matches[0])
                    name = path.name
                else:
                    continue
            else:
                path = cwd / name
            if path.exists():
                try:
                    text = path.read_text(encoding="utf-8")
                    trimmed = text[:1500]
                    if len(text) > 1500:
                        trimmed += "\n... (truncated)"
                    enrichments.append(f"## Build Manifest ({name})\n\n```\n{trimmed}\n```")
                    logger.info("[init] Enriched with %s (%d chars)", name, len(trimmed))
                    break
                except Exception:
                    pass

        # 4. Task runner — dev commands, build targets (always read, complements manifest)
        task_runners = [
            ("Makefile", _parse_makefile_targets),
            ("justfile", _read_task_file),
            ("Taskfile.yml", _read_task_file),
            ("Taskfile.yaml", _read_task_file),
        ]
        for name, reader in task_runners:
            path = cwd / name
            if path.exists():
                try:
                    snippet = reader(path)
                    if snippet:
                        enrichments.append(f"## Task Runner ({name})\n\n```\n{snippet}\n```")
                        logger.info("[init] Enriched with %s", name)
                    break
                except Exception:
                    pass

        if not enrichments:
            return base_content

        prefix = "\n\n".join(enrichments)
        return f"{prefix}\n\n---\n\n{base_content}"

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
                    "Using GEPA-evolved init rules " "(gen=%s, confidence=%.2f, %d chars)",
                    rec.reason,
                    rec.confidence,
                    len(rec.value),
                )
                return rec.value
        except Exception:
            pass
        return None
