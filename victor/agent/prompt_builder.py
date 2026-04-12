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

"""System prompt builder for Victor.

Builds provider-specific system prompts based on:
- Provider type (cloud vs local)
- Model capabilities (native tool calling vs fallback)
- Tool calling adapter hints
"""

import logging
from typing import TYPE_CHECKING, Optional, Set, List, Dict, Any

from victor.agent.tool_calling import BaseToolCallingAdapter, ToolCallingCapabilities
from victor.agent.provider_tool_guidance import (
    get_tool_guidance_strategy,
    ToolGuidanceStrategy,
)
from victor.agent.prompt_normalizer import get_prompt_normalizer

if TYPE_CHECKING:
    from victor.agent.query_classifier import QueryClassification
    from victor.framework.enrichment import (
        PromptEnrichmentService,
        EnrichmentContext,
        EnrichedPrompt,
    )

logger = logging.getLogger(__name__)


# Provider classifications
CLOUD_PROVIDERS: Set[str] = {
    "anthropic",
    "openai",
    "google",
    "xai",
    "moonshot",
    "kimi",
    "deepseek",
}
LOCAL_PROVIDERS: Set[str] = {"ollama", "lmstudio", "vllm"}

# Critical grounding rules to prevent hallucination
# Concise version for cloud providers - they handle context well
GROUNDING_RULES = """
GROUNDING: Base ALL responses on tool output only. Never invent file paths or content.
Quote code exactly from tool output. If more info needed, call another tool.
""".strip()

# Parallel read optimization guidance
PARALLEL_READ_GUIDANCE = """
PARALLEL READS: For exploration tasks, batch multiple read calls together.
- Call read on 5-10 files simultaneously when analyzing a codebase
- Each file read is limited to ~8K chars (~230 lines) to fit context
- List files first (ls), then batch-read relevant ones in parallel
- Example: To understand a module, read all .py files in that directory at once
""".strip()

# Concise mode guidance to reduce verbosity
CONCISE_MODE_GUIDANCE = """
OUTPUT STYLE: CONCISE
- Be direct and brief. No unnecessary preamble or summary.
- Skip "I'll" and "Let me" phrases - just do the action.
- No explanations unless explicitly requested.
- For code: Show the code, minimal commentary.
- For actions: Report result, not the process.
- For questions: Answer directly, then stop.
- Maximum 3 sentences for simple queries.
""".strip()

# Active completion signaling - deterministic detection
# This guidance is always included to ensure predictable task completion detection
# Uses bold-wrapped markers with colon suffix for markdown-safe rendering:
#   **DONE**: description  → rendered as bold "DONE:" → stripped to "DONE:"
#   __DONE__: description  → rendered as italic "DONE:" → stripped to "DONE:"
# The detector strips markdown formatting (**, __, *, _) and matches "KEYWORD:"
COMPLETION_GUIDANCE = """
TASK COMPLETION (MANDATORY):
When you complete a task, you MUST signal completion using these EXACT markers.
Use the bold format with a colon suffix exactly as shown:

1. For FILE OPERATIONS (create/edit/write):
   **DONE**: Created/Modified <filename>

2. For BUG FIXES / ISSUE RESOLUTION:
   **TASK_DONE**: <what was fixed>

3. For ANALYSIS / QUESTIONS / RESEARCH:
   **SUMMARY**: <key findings>

4. For FAILED / BLOCKED TASKS:
   **BLOCKED**: <reason>

IMPORTANT:
- These markers are REQUIRED for the system to detect task completion
- Always use the bold format with colon: **DONE**: or **SUMMARY**: or **TASK_DONE**:
- After signaling completion, STOP - do NOT ask follow-up questions
- Do NOT say "would you like me to continue?" after completing the task
- Do NOT re-read files you have already read
- Signal completion ONCE - do not repeat the marker multiple times
""".strip()

# Extended grounding rules for local models that need more explicit guidance
GROUNDING_RULES_EXTENDED = """
CRITICAL - TOOL OUTPUT GROUNDING:
When you receive tool output in <TOOL_OUTPUT> tags:
1. The content between ═══ markers is ACTUAL file/command output - NEVER ignore it
2. You MUST base your analysis ONLY on this actual content
3. NEVER fabricate, invent, or imagine file contents that differ from tool output
4. If you need more information, call another tool - do NOT guess
5. When citing code, quote EXACTLY from the tool output
6. If tool output is empty or truncated, acknowledge this limitation

VIOLATION OF THESE RULES WILL RESULT IN INCORRECT ANALYSIS.
""".strip()

# Prescriptive guidance for handling large/truncated files
LARGE_FILE_PAGINATION_GUIDANCE = """
LARGE FILE HANDLING (MANDATORY):
When you see "LARGE FILE" or "TRUNCATED" in tool output, you received PARTIAL content only.
The file contains more data at different offsets. To find what you need:

1. **File shows structure summary only**: Use search parameter to find specific content:
   read(path='file.py', search='function_name')
   read(path='file.py', search='class ClassName')

2. **File was truncated mid-content**: Use offset to continue reading:
   read(path='file.py', offset=X, limit=300)  # Where X = last line shown

3. **Searching for a specific line number**: If function is at line 3000:
   read(path='file.py', offset=2980, limit=100)  # Read lines 2981-3080

DO NOT re-read the full file without parameters - you will get the same truncated view.
DO NOT assume content is missing - use offset/search to access additional sections.
""".strip()

# ASI-derived guidance: Lessons learned from execution trace analysis (GEPA-inspired).
# These rules were extracted from 64K+ tool execution events across 11 days:
# - 165 read(dir) errors → directory vs file guidance
# - 60% literal code_search with 0 results → search mode guidance
# - 70:9 read:code_search ratio → search-first discovery guidance
# - 33% edit failure rate → edit precision guidance
ASI_TOOL_EFFECTIVENESS_GUIDANCE = """
TOOL EFFECTIVENESS (from execution data):
- Use code_search(query='...') FIRST to discover relevant files before reading them.
  Do NOT browse with read→read→read — search finds the right file in one call.
- code_search works best with mode='semantic' for concepts and patterns.
  Use mode='literal' only for exact identifiers you know exist.
- For edits: include 3+ surrounding lines of context in old_str to ensure a unique match.
  Ambiguous matches (old_str appears 2+ times) will fail — add more context.
- Use ls() for directories, read() for files. read('directory_name') will auto-convert
  but wastes a tool call.
- Only access files within the current project. Never guess paths from other projects.
  If read('victor') or read('../') fails, you are in the WRONG directory.
  Use ls('.') to orient yourself in the workspace.
- Do NOT use shell('rg ...') or shell('grep ...') to search code.
  Use code_search(query='...') instead — it uses the semantic index.
- After a failed edit (old_str not found), RE-READ the file at the exact location
  and copy the text character-by-character. Do NOT guess from memory.
""".strip()

# Task-type hints are now in vertical prompt contributors (E5 M3).
# Use get_task_type_hint(task_type, prompt_contributors=[...]) instead.


def get_task_type_hint(task_type: str, prompt_contributors: Optional[list] = None) -> str:
    """Get prompt hint for a specific task type.

    Hints come from vertical prompt contributors (the canonical source).
    Pass prompt_contributors from the vertical for full task hint support.

    Args:
        task_type: The detected task type (e.g., "create_simple", "edit")
        prompt_contributors: Optional list of PromptContributorProtocol implementations

    Returns:
        Task-specific prompt hint or empty string if not found
    """
    if prompt_contributors:
        for contributor in sorted(prompt_contributors, key=lambda c: c.get_priority()):
            hints = contributor.get_task_type_hints()
            if task_type.lower() in hints:
                task_hint = hints[task_type.lower()]
                # Handle TaskTypeHint objects or plain strings
                if hasattr(task_hint, "hint"):
                    hint_text = task_hint.hint
                else:
                    hint_text = str(task_hint)
                contributor_name = type(contributor).__name__
                logger.info(
                    "Applied vertical task hint: task_type=%s, contributor=%s",
                    task_type,
                    contributor_name,
                )
                return hint_text

    return ""


# Models with known good native tool calling support
NATIVE_TOOL_MODELS = [
    "qwen2.5",
    "qwen-2.5",
    "qwen3",
    "qwen-3",
    "llama-3.1",
    "llama3.1",
    "llama-3.2",
    "llama3.2",
    "llama-3.3",
    "llama3.3",
    "ministral",
    "mistral",
    "mixtral",
    "command-r",
    "firefunction",
    "hermes",
    "functionary",
]


class SystemPromptBuilder:
    """Builds system prompts tailored to provider and model capabilities.

    Different providers have different tool calling capabilities:
    - Cloud providers (Anthropic, OpenAI, Google, xAI): Robust native tool calling
    - vLLM: Production-grade OpenAI-compatible with tool parsers
    - LMStudio: OpenAI-compatible with Native vs Default mode
    - Ollama: Native tool_calls for Llama3.1+, Qwen2.5+, Mistral; fallback otherwise

    Vertical Integration:
    - Accepts prompt contributors from verticals via DI container
    - Merges vertical-specific task hints and system prompt sections
    - Falls back to deprecated hints for backward compatibility
    """

    def __init__(
        self,
        provider_name: str,
        model: str,
        tool_adapter: Optional[BaseToolCallingAdapter] = None,
        capabilities: Optional[ToolCallingCapabilities] = None,
        prompt_contributors: Optional[list] = None,
        tool_guidance_strategy: Optional[ToolGuidanceStrategy] = None,
        task_type: str = "medium",
        available_tools: Optional[List[str]] = None,
        enrichment_service: Optional["PromptEnrichmentService"] = None,
        vertical: Optional[str] = None,
        concise_mode: bool = False,
        query_classification: Optional["QueryClassification"] = None,
    ):
        """Initialize the prompt builder.

        Args:
            provider_name: Name of the provider (e.g., "ollama", "anthropic")
            model: Model name/identifier
            tool_adapter: Optional tool calling adapter for getting hints
            capabilities: Optional pre-computed capabilities
            prompt_contributors: Optional list of PromptContributorProtocol implementations
            tool_guidance_strategy: Optional provider-specific tool guidance strategy (GAP-5)
            task_type: Task complexity level (simple, medium, complex) for guidance
            available_tools: List of available tool names for guidance context
            enrichment_service: Optional prompt enrichment service for context injection
            vertical: Current vertical (coding, research, devops, data_analysis) for enrichment
            concise_mode: If True, adds guidance to produce brief, direct responses
        """
        # Handle both string and ProviderSettings object for provider_name
        # (backward compatibility with settings refactor)
        if provider_name and hasattr(provider_name, "default_provider"):
            # provider_name is a ProviderSettings object
            actual_provider_name = provider_name.default_provider
        else:
            # provider_name is a string or None
            actual_provider_name = provider_name

        self.provider_name = (actual_provider_name or "").lower()
        self.model = model or ""
        self.model_lower = self.model.lower()
        self.tool_adapter = tool_adapter
        self.capabilities = capabilities
        self.prompt_contributors = prompt_contributors or []
        self.task_type = task_type
        self.available_tools = available_tools or []
        self.enrichment_service = enrichment_service
        self.vertical = vertical or "coding"
        self.concise_mode = concise_mode
        self.query_classification = query_classification

        # Initialize tool guidance strategy (GAP-5: Provider-specific tool guidance)
        # Use provided strategy or auto-detect based on provider name
        if tool_guidance_strategy:
            self._tool_guidance = tool_guidance_strategy
        else:
            self._tool_guidance = get_tool_guidance_strategy(self.provider_name)

        # Cache merged task hints from vertical contributors
        self._merged_task_hints = None

    def is_cloud_provider(self) -> bool:
        """Check if the provider is a cloud-based API with robust tool calling."""
        return self.provider_name in CLOUD_PROVIDERS

    def is_local_provider(self) -> bool:
        """Check if the provider is a local model (Ollama, LMStudio, vLLM)."""
        return self.provider_name in LOCAL_PROVIDERS

    def has_native_tool_support(self) -> bool:
        """Check if the model has known native tool calling support."""
        return any(pattern in self.model_lower for pattern in NATIVE_TOOL_MODELS)

    def get_merged_task_hints(self) -> dict:
        """Get merged task hints from vertical contributors.

        Returns:
            Dict of task type -> hint string, merged from all contributors
        """
        if self._merged_task_hints is not None:
            return self._merged_task_hints

        merged: dict = {}  # Populated by vertical contributors below

        # Override with vertical contributors (sorted by priority)
        for contributor in sorted(self.prompt_contributors, key=lambda c: c.get_priority()):
            hints = contributor.get_task_type_hints()
            for task_type, task_hint in hints.items():
                # Extract hint string from TaskTypeHint objects
                if hasattr(task_hint, "hint"):
                    merged[task_type] = task_hint.hint
                else:
                    merged[task_type] = str(task_hint)

        self._merged_task_hints = merged
        return merged

    def get_vertical_grounding_rules(self) -> str:
        """Get grounding rules from vertical contributors with deduplication.

        Uses PromptNormalizer.deduplicate_sections() to remove duplicate
        grounding rules that may come from multiple contributors.

        Returns:
            Merged and deduplicated grounding rules from all contributors
        """
        if not self.prompt_contributors:
            return ""

        # Collect grounding rules from all contributors
        rules = []
        for contributor in sorted(self.prompt_contributors, key=lambda c: c.get_priority()):
            grounding = contributor.get_grounding_rules()
            if grounding:
                rules.append(grounding)

        # Deduplicate sections using PromptNormalizer
        if rules:
            normalizer = get_prompt_normalizer()
            rules = normalizer.deduplicate_sections(rules)

        return "\n\n".join(rules) if rules else ""

    def get_vertical_system_prompt_sections(self) -> str:
        """Get system prompt sections from vertical contributors with deduplication.

        Uses PromptNormalizer.deduplicate_sections() to remove duplicate
        system prompt sections that may come from multiple contributors.

        Returns:
            Merged and deduplicated system prompt sections from all contributors
        """
        if not self.prompt_contributors:
            return ""

        # Collect sections from all contributors
        sections = []
        for contributor in sorted(self.prompt_contributors, key=lambda c: c.get_priority()):
            section = contributor.get_system_prompt_section()
            if section:
                sections.append(section)

        # Deduplicate sections using PromptNormalizer
        if sections:
            normalizer = get_prompt_normalizer()
            sections = normalizer.deduplicate_sections(sections)

        return "\n\n".join(sections) if sections else ""

    def get_provider_tool_guidance(self) -> str:
        """Get provider-specific tool usage guidance.

        This implements GAP-5: Provider-specific tool guidance (Strategy pattern).
        Different providers have different tool calling behaviors that benefit from
        tailored guidance. For example:
        - DeepSeek tends to over-explore with redundant tool calls
        - Grok handles tools efficiently with minimal redundancy
        - Ollama needs stricter guidance due to weaker tool calling

        Returns:
            Provider-specific tool guidance prompt or empty string if no special guidance needed
        """
        if not self._tool_guidance:
            return ""

        guidance = self._tool_guidance.get_guidance_prompt(
            task_type=self.task_type, available_tools=self.available_tools
        )

        if guidance:
            logger.debug(
                f"Applied provider tool guidance for {self.provider_name}, "
                f"task_type={self.task_type}"
            )

        return guidance

    def get_max_exploration_depth(self) -> int:
        """Get the maximum exploration depth for the current provider and task complexity.

        Returns:
            Maximum number of tool calls before synthesis checkpoint
        """
        if not self._tool_guidance:
            return 10  # Default

        return self._tool_guidance.get_max_exploration_depth(self.task_type)

    def should_consolidate_calls(self, tool_history: List[Dict[str, Any]]) -> bool:
        """Check if tool calls should be consolidated based on history.

        Args:
            tool_history: List of recent tool calls with 'tool' and 'args' keys

        Returns:
            True if consolidation/synthesis is recommended
        """
        if not self._tool_guidance:
            return False

        return self._tool_guidance.should_consolidate_calls(tool_history)

    def get_synthesis_checkpoint_prompt(self, tool_count: int) -> str:
        """Get synthesis checkpoint prompt if we've reached the threshold.

        Args:
            tool_count: Number of tool calls made so far

        Returns:
            Synthesis prompt or empty string if not at checkpoint
        """
        if not self._tool_guidance:
            return ""

        return self._tool_guidance.get_synthesis_checkpoint_prompt(tool_count)

    async def enrich_prompt(
        self,
        prompt: str,
        context: Optional["EnrichmentContext"] = None,
    ) -> "EnrichedPrompt":
        """Enrich a user prompt with contextual information.

        Uses the enrichment service to inject relevant context based on the
        vertical (coding, research, devops, data_analysis). This can include:
        - Knowledge graph symbols for coding tasks
        - Web search results for research tasks
        - Infrastructure context for devops tasks
        - Schema context for data analysis tasks

        Args:
            prompt: The user prompt to enrich
            context: Optional enrichment context with session/task metadata.
                     If not provided, a basic context will be created.

        Returns:
            EnrichedPrompt with the enriched prompt text and metadata.
            If enrichment is disabled or unavailable, returns the original prompt.
        """
        # Import here to avoid circular imports
        from victor.framework.enrichment import EnrichmentContext, EnrichedPrompt

        # If no enrichment service, return original prompt
        if not self.enrichment_service:
            logger.debug("No enrichment service available, returning original prompt")
            return EnrichedPrompt(
                original_prompt=prompt,
                enriched_prompt=prompt,
            )

        # Create default context if not provided
        if context is None:
            context = EnrichmentContext(
                task_type=self.task_type,
            )

        try:
            result = await self.enrichment_service.enrich(
                prompt=prompt,
                vertical=self.vertical,
                context=context,
            )
            logger.info(
                "Prompt enriched: vertical=%s, enrichments=%d, tokens_added=%d",
                self.vertical,
                result.enrichment_count,
                result.total_tokens_added,
            )
            return result
        except Exception as e:
            logger.warning("Prompt enrichment failed: %s", e)
            return EnrichedPrompt(
                original_prompt=prompt,
                enriched_prompt=prompt,
            )

    def _get_task_guidance_section(self) -> str:
        """Get task-specific guidance based on query classification."""
        if not self.query_classification:
            return ""

        from victor.agent.query_classifier import QueryType

        guidance_map = {
            QueryType.EXPLORATION: (
                "TASK GUIDANCE: Explore systematically. List directories before reading files. "
                "Map structure first. Build a mental model of the codebase before diving deep."
            ),
            QueryType.IMPLEMENTATION: (
                "TASK GUIDANCE: Plan before coding. Break into discrete steps. "
                "Test incrementally. Verify each step before proceeding."
            ),
            QueryType.DEBUGGING: (
                "TASK GUIDANCE: Focus on error messages and stack traces. "
                "Check recent changes first. Reproduce the issue before fixing."
            ),
            QueryType.REVIEW: (
                "TASK GUIDANCE: Examine for correctness, style, and security. "
                "Reference specific line numbers. Be thorough but constructive."
            ),
            QueryType.QUICK_QUESTION: (
                "TASK GUIDANCE: Answer directly and concisely. "
                "Only use tools if the answer requires checking the codebase."
            ),
        }
        return guidance_map.get(self.query_classification.query_type, "")

    def _get_tool_constraint_section(self) -> str:
        """Get tool constraint section listing available tools."""
        if not self.available_tools:
            return ""
        tool_list = ", ".join(sorted(self.available_tools))
        return (
            f"IMPORTANT: Only use tools from this list: {tool_list}. "
            "Do not attempt to call unlisted tools."
        )

    def build(self) -> str:
        """Build the system prompt.

        Uses adapter hints if available, otherwise falls back to
        provider-specific prompt construction. Includes provider-specific
        tool guidance (GAP-5) when available.

        The prompt is built in this order:
        1. Concise mode guidance (if enabled)
        2. Base prompt (provider-specific)
        3. Task-specific guidance (based on query classification)
        4. Tool constraint section (available tools list)
        5. Completion guidance (always included for deterministic task completion)
        6. Provider-specific tool guidance (GAP-5)

        Returns:
            System prompt string tailored to the provider/model
        """
        # Try adapter-based prompt first
        if self.tool_adapter:
            base_prompt = self._build_with_adapter()
        else:
            # Fall back to provider-specific prompt
            base_prompt = self._build_for_provider()

        # Determine which prompt sections to include.
        # Edge model can select only task-relevant sections to save tokens.
        sections_to_include = self._get_active_sections()

        if "concise_mode" in sections_to_include and self.concise_mode:
            base_prompt = f"{CONCISE_MODE_GUIDANCE}\n\n{base_prompt}"
            logger.debug("Concise mode enabled - added brevity guidance to prompt")

        if "task_guidance" in sections_to_include:
            task_guidance = self._get_task_guidance_section()
            if task_guidance:
                base_prompt = f"{base_prompt}\n\n{task_guidance}"

        if "tool_constraint" in sections_to_include:
            tool_constraint = self._get_tool_constraint_section()
            if tool_constraint:
                base_prompt = f"{base_prompt}\n\n{tool_constraint}"

        if "completion" in sections_to_include:
            optimized_completion = self._get_optimized_section("COMPLETION_GUIDANCE")
            completion = optimized_completion or COMPLETION_GUIDANCE
            base_prompt = f"{base_prompt}\n\n{completion}"

        if "tool_guidance" in sections_to_include:
            tool_guidance = self.get_provider_tool_guidance()
            if tool_guidance:
                base_prompt = f"{base_prompt}\n\n{tool_guidance}"

        # GEPA: Replace static GROUNDING_RULES with evolved version if available
        optimized_grounding = self._get_optimized_section("GROUNDING_RULES")
        if optimized_grounding:
            base_prompt = base_prompt.replace(GROUNDING_RULES, optimized_grounding)

        # ASI-derived tool effectiveness guidance (GEPA-inspired)
        # Check if prompt optimizer has an evolved version, else use static default
        if "tool_guidance" in sections_to_include:
            optimized = self._get_optimized_section("ASI_TOOL_EFFECTIVENESS_GUIDANCE")
            guidance = optimized or ASI_TOOL_EFFECTIVENESS_GUIDANCE
            base_prompt = f"{base_prompt}\n\n{guidance}"

        # Few-shot examples (MIPROv2-mined from successful traces)
        if "few_shot_examples" in sections_to_include:
            examples = self._get_optimized_section("FEW_SHOT_EXAMPLES")
            if examples:
                base_prompt = f"{base_prompt}\n\n{examples}"

        # Log system prompt composition sent to LLM
        logger.debug(
            "[SystemPrompt→LLM] provider=%s sections=%s tool_constraint=%s "
            "tool_guidance_len=%d prompt_total_len=%d",
            self.provider_name,
            sorted(sections_to_include),
            self._get_tool_constraint_section()[:300] if self.available_tools else "(none)",
            len(self.get_provider_tool_guidance()),
            len(base_prompt),
        )

        return base_prompt

    def _get_optimized_section(self, section_name: str) -> Optional[str]:
        """Check if the prompt optimizer has an evolved version of a section.

        Returns the evolved text if a candidate exists with sufficient
        confidence, otherwise None (caller uses the static default).
        Gated by prompt_optimization.enabled setting.
        Checks section_strategies to skip sections with empty strategy list.

        IMPORTANT: Results are cached per session in _optimized_section_cache
        to ensure Thompson Sampling picks ONE candidate per section per session.
        Without this cache, each build() call would re-sample and potentially
        pick a different candidate, mutating the system prompt mid-session
        and breaking provider prefix caching (90% discount).
        """
        # Session cache: sample once, reuse for the rest of the session
        cache = getattr(self, "_optimized_section_cache", None)
        if cache is None:
            self._optimized_section_cache: Dict[str, Optional[str]] = {}
            cache = self._optimized_section_cache
        if section_name in cache:
            return cache[section_name]

        result = self._sample_optimized_section(section_name)
        cache[section_name] = result
        return result

    def _sample_optimized_section(self, section_name: str) -> Optional[str]:
        """Sample an evolved section from GEPA/MIPROv2/CoT (called once per session).

        This does the actual Thompson Sampling via get_recommendation().
        Results are cached by _get_optimized_section() to prevent mid-session mutation.
        """
        # Gate on settings.prompt_optimization.enabled (authoritative source)
        try:
            from victor.config.settings import get_settings

            po = getattr(get_settings(), "prompt_optimization", None)
            if po is None or not po.enabled:
                return None
            strategies = po.get_strategies_for_section(section_name)
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
                getattr(self, "provider_name", "") or "",
                getattr(self, "model", "") or "",
                getattr(self, "task_type", "default"),
                section_name=section_name,
            )
            if rec and rec.confidence > 0.6 and not rec.is_baseline:
                logger.info(
                    "Using GEPA-evolved prompt section '%s' (gen=%s, confidence=%.2f)",
                    section_name,
                    rec.reason,
                    rec.confidence,
                )
                return rec.value
        except Exception:
            pass
        return None

    def _get_active_sections(self) -> set:
        """Determine which prompt sections to include.

        Uses edge model to select relevant sections when available,
        reducing system prompt token count for focused tasks.
        Falls back to all sections if edge model unavailable.
        """
        all_sections = {
            "concise_mode",
            "task_guidance",
            "tool_constraint",
            "completion",
            "tool_guidance",
            "few_shot_examples",
        }

        try:
            from victor.core import get_container

            container = get_container()

            from victor.agent.services.protocols.decision_service import (
                LLMDecisionServiceProtocol,
            )

            service = container.get(LLMDecisionServiceProtocol)
            if service is None:
                return all_sections

            from victor.agent.edge_model import select_prompt_sections_with_edge_model

            # Use cached task type from classification if available
            task_type = getattr(self, "_task_type", "action")
            user_msg = getattr(self, "_user_message", "")

            selected = select_prompt_sections_with_edge_model(
                service=service,
                user_message=user_msg[:200] if user_msg else "",
                task_type=task_type,
                available_sections=list(all_sections),
            )

            if selected:
                # Always include completion guidance (required for detection)
                result = set(selected) | {"completion"}
                logger.debug(f"Edge prompt focus: {len(result)}/{len(all_sections)} sections")
                return result

        except Exception:
            pass

        return all_sections

    def _build_with_adapter(self) -> str:
        """Build system prompt using the tool calling adapter.

        Returns:
            System prompt string tailored to the provider/model
        """
        base_prompt = (
            "You are an expert coding assistant. You can analyze, explain, and generate code.\n"
            "When asked to write or complete code, provide working implementations directly.\n"
            "When asked to explore or analyze code, use the available tools."
        )

        # Get adapter-specific hints
        hints = self.tool_adapter.get_system_prompt_hints() if self.tool_adapter else None

        if hints:
            return f"{base_prompt}\n\n{hints}\n\n{GROUNDING_RULES}"

        # For providers with robust native tool calling, use minimal prompt
        caps = self.capabilities or (
            self.tool_adapter.get_capabilities() if self.tool_adapter else None
        )
        if caps and caps.native_tool_calls and not caps.requires_strict_prompting:
            return (
                f"{base_prompt}\n\n"
                "Tool usage guidelines:\n"
                "1. For code generation tasks: write the code directly in your response.\n"
                "2. For exploration tasks: use list_directory and read_file to examine code.\n"
                "3. For modification tasks: use write_file or edit_files "
                "after understanding context.\n"
                "4. Provide clear, working solutions.\n\n"
                f"{GROUNDING_RULES}"
            )

        return f"{base_prompt}\n\n{GROUNDING_RULES}"

    def _build_for_provider(self) -> str:
        """Build an appropriate system prompt based on the provider type.

        Returns:
            Appropriate system prompt string for the provider
        """
        # Cloud providers with robust native tool calling
        if self.is_cloud_provider():
            return self._build_cloud_prompt()

        # vLLM - Production-grade OpenAI-compatible
        if self.provider_name == "vllm":
            return self._build_vllm_prompt()

        # LMStudio - OpenAI-compatible with Native vs Default mode
        if self.provider_name == "lmstudio":
            return self._build_lmstudio_prompt()

        # Ollama - Native tool_calls for supported models
        if self.provider_name == "ollama":
            return self._build_ollama_prompt()

        # Default/unknown provider
        return self._build_default_prompt()

    def _build_cloud_prompt(self) -> str:
        """Build prompt for cloud providers (Anthropic, OpenAI, xAI, DeepSeek).

        Delegates to provider-specific prompts for optimal behavior.
        """
        if self.provider_name == "google":
            return self._build_google_prompt()

        if self.provider_name == "deepseek":
            return self._build_deepseek_prompt()

        if self.provider_name == "xai":
            return self._build_xai_prompt()

        return (
            "You are an expert code analyst with access to tools for exploring "
            "and modifying code. Use them effectively:\n\n"
            "1. Use list_directory and read_file to examine code before conclusions.\n"
            "2. If asked to modify code, use write_file or edit_files "
            "after understanding context.\n"
            "3. For call-graph questions, prefer graph(mode='callers'|'callees'|'trace').\n"
            "4. Provide clear, actionable responses based on actual file contents.\n"
            "5. Always cite specific file paths and line numbers when referencing code.\n"
            "6. You may call multiple tools in parallel when they are independent.\n\n"
            f"{PARALLEL_READ_GUIDANCE}\n\n"
            f"{GROUNDING_RULES}"
        )

    def _build_google_prompt(self) -> str:
        """Build optimized prompt for Google/Gemini models.

        Gemini Flash/Pro have excellent native tool calling and work best with:
        - Concise, action-oriented instructions
        - Minimal redundancy (don't repeat what's in tool definitions)
        - Focus on task completion over verbose guidance
        """
        return (
            "Expert coding assistant with tool access.\n\n"
            "CORE RULES:\n"
            "• Read files before making claims about their contents\n"
            "• Cite file:line when referencing code\n"
            "• Parallel tool calls allowed for independent operations\n"
            "• Ground all responses in actual tool output\n\n"
            "TASK EXECUTION:\n"
            "• Generation: Write code directly, minimal tool use\n"
            "• Exploration: Use tools systematically, then summarize\n"
            "• Modification: Read → understand → edit\n"
            "• For call-graph questions, use graph(mode='callers'|'callees'|'trace')\n"
            "• Actions: Execute fully, report results\n\n"
            f"{PARALLEL_READ_GUIDANCE}\n\n"
            f"{GROUNDING_RULES}"
        )

    def _build_deepseek_prompt(self) -> str:
        """Build optimized prompt for DeepSeek models.

        DeepSeek models have good tool calling but tend to:
        - Read the same file multiple times (repetitive tool calls)
        - Generate code snippets without verification
        - Need explicit grounding reminders

        This prompt emphasizes:
        - Anti-repetition rules
        - Strict grounding requirements
        - Efficient tool usage patterns
        - Pagination for large files
        """
        return (
            "Expert coding assistant with tool access.\n\n"
            "CRITICAL RULES (MUST FOLLOW):\n"
            "• NEVER read the same file twice with identical arguments\n"
            "• NEVER call the same tool with identical arguments\n"
            "• If you've read a file, use that content for all future references\n"
            "• Only call tools when you need NEW information\n\n"
            f"{LARGE_FILE_PAGINATION_GUIDANCE}\n\n"
            "GROUNDING (MANDATORY):\n"
            "• ALL code snippets must be directly from tool output\n"
            "• Do NOT generate or imagine file contents\n"
            "• Quote code EXACTLY as it appears in tool output\n"
            "• If unsure about content, read the file ONCE\n"
            "• Never cite line numbers you haven't verified\n\n"
            "TOOL EFFICIENCY:\n"
            "• list_directory first to understand structure\n"
            "• read_file ONCE per file, remember contents\n"
            "• For large files, use search/offset parameters (see above)\n"
            "• Use semantic_code_search for specific symbols\n"
            "• Use graph(mode='callers'|'callees'|'trace') for call-graph questions\n"
            "• Stop tool calls when you have enough info (usually 3-5 calls)\n\n"
            "TASK EXECUTION:\n"
            "• Generation: Write code directly without excessive exploration\n"
            "• Analysis: Read files ONCE, provide grounded analysis\n"
            "• Modification: Read target file ONCE, then edit\n\n"
            f"{GROUNDING_RULES_EXTENDED}"
        )

    def _build_xai_prompt(self) -> str:
        """Build optimized prompt for xAI/Grok models.

        Grok models are generally good at tool calling but benefit from:
        - Clear task structure
        - Explicit completion criteria
        - Balanced exploration guidance
        """
        return (
            "You are an expert code analyst with access to tools.\n\n"
            "EFFECTIVE TOOL USAGE:\n"
            "• Use list_directory to understand project structure first\n"
            "• Use read_file to examine specific files (one read per file)\n"
            "• Use semantic_code_search for finding specific code patterns\n"
            "• Use graph(mode='callers'|'callees'|'trace') for call-graph questions\n"
            "• Parallel tool calls are allowed for independent operations\n\n"
            "TASK APPROACH:\n"
            "• For analysis tasks: Read relevant files, then provide structured findings\n"
            "• For generation tasks: Write code directly with minimal exploration\n"
            "• For modification tasks: Read → understand → modify\n"
            "• Stop exploring when you have sufficient information\n\n"
            "RESPONSE QUALITY:\n"
            "• Cite specific file paths and line numbers\n"
            "• Base all claims on actual tool output\n"
            "• Provide actionable, concrete suggestions\n"
            "• Structure responses clearly with headers when appropriate\n\n"
            f"{PARALLEL_READ_GUIDANCE}\n\n"
            f"{GROUNDING_RULES}"
        )

    def _build_vllm_prompt(self) -> str:
        """Build prompt for vLLM provider."""
        return (
            "You are a code analyst. You have access to tools via OpenAI-compatible API.\n\n"
            "TOOL CALLING:\n"
            "- Tools are called using the standard OpenAI function calling format.\n"
            "- You may call tools when needed to gather information.\n"
            "- After gathering sufficient information (2-4 tool calls), provide your answer.\n"
            "- Do NOT repeat the same tool call with identical arguments.\n\n"
            "RESPONSE FORMAT:\n"
            "- When you have enough information, respond with a clear answer.\n"
            "- Your final response should be in plain text, not JSON or tool syntax.\n"
            "- Be concise and focus on answering the user's question.\n\n"
            "IMPORTANT:\n"
            "- Do not output raw JSON tool calls in your text response.\n"
            "- Do not output XML tags like </function> or </parameter>.\n"
            "- When you're done using tools, provide a human-readable answer.\n\n"
            f"{GROUNDING_RULES}"
        )

    def _build_lmstudio_prompt(self) -> str:
        """Build prompt for LMStudio provider."""
        if self.has_native_tool_support():
            return (
                "You are an expert coding assistant. "
                "You can analyze, explain, and generate code.\n\n"
                "CAPABILITIES:\n"
                "- Code generation: Write working implementations directly in your response.\n"
                "- Code analysis: Use tools to explore and understand existing code.\n"
                "- Code modification: Use tools to read files before making changes.\n\n"
                "TOOL USAGE:\n"
                "- For code generation tasks: write code directly, no tools needed.\n"
                "- For exploration tasks: use tools via OpenAI function calling format.\n"
                "- Call tools one at a time and wait for results.\n"
                "- After 2-3 tool calls, provide your answer.\n"
                "- Do NOT repeat identical tool calls.\n\n"
                "RESPONSE FORMAT:\n"
                "- Provide answers in plain, readable text.\n"
                "- For code generation, include the complete implementation.\n"
                "- Do NOT include JSON, XML, or tool syntax in your response.\n\n"
                f"{GROUNDING_RULES}"
            )
        else:
            # Default/non-native mode - stricter guidance needed
            return (
                "You are a code analyst. Follow these rules EXACTLY:\n\n"
                "CRITICAL RULES:\n"
                "1. Call tools ONE AT A TIME. Wait for each result.\n"
                "2. After reading 2-3 files, STOP and provide your answer.\n"
                "3. Do NOT repeat the same tool call.\n\n"
                "OUTPUT FORMAT:\n"
                "1. Your answer must be in plain English text.\n"
                "2. Do NOT output JSON objects in your response.\n"
                "3. Do NOT output XML tags like </function> or <parameter>.\n"
                "4. Do NOT output [TOOL_REQUEST] or similar markers.\n\n"
                "WHEN TO STOP:\n"
                "1. When you have read the relevant files.\n"
                "2. When you can answer the user's question.\n"
                "3. After calling any tool 3 times.\n\n"
                f"{GROUNDING_RULES}"
            )

    def _build_ollama_prompt(self) -> str:
        """Build prompt for Ollama provider."""
        if self.has_native_tool_support():
            base_prompt = (
                "You are a code analyst with tool calling capability.\n\n"
                "TOOL USAGE:\n"
                "- Use list_directory and read_file to inspect code.\n"
                "- You can call multiple read tools in parallel for efficiency.\n"
                "- After reading relevant files, provide your answer.\n"
                "- Do NOT make identical repeated tool calls.\n\n"
                f"{PARALLEL_READ_GUIDANCE}\n\n"
                "RESPONSE FORMAT:\n"
                "- Write your answer in plain, readable text.\n"
                "- Do NOT output raw JSON in your response.\n"
                "- Do NOT output XML tags or function call syntax.\n"
                "- Be concise and answer the question directly.\n\n"
                "COMPLETION:\n"
                "- Stop calling tools when you have enough information.\n"
                "- If you've called a tool 3+ times, stop and summarize.\n"
                "- Always end with a human-readable answer.\n\n"
                f"{GROUNDING_RULES}"
            )
            # Add Qwen3-specific thinking mode guidance
            if "qwen3" in self.model_lower or "qwen-3" in self.model_lower:
                base_prompt += (
                    "\n\nQWEN3 MODE:\n"
                    "- Use /no_think for simple questions.\n"
                    "- Provide direct answers without excessive reasoning."
                )
            return base_prompt
        else:
            # Models without reliable tool calling - strictest guidance
            return (
                "You are a code analyst. Follow these rules EXACTLY:\n\n"
                "CRITICAL TOOL RULES:\n"
                "1. Call tools ONE AT A TIME. Never batch calls.\n"
                "2. After reading 2-3 files, STOP and answer.\n"
                "3. Do NOT repeat the same tool call.\n\n"
                "CRITICAL OUTPUT RULES:\n"
                "1. Write your answer in plain English.\n"
                '2. Do NOT output JSON objects like {"name": ...}.\n'
                "3. Do NOT output XML tags like </function> or </parameter>.\n"
                "4. Do NOT output function call syntax.\n"
                "5. Keep your answer focused and concise.\n\n"
                "STOP IMMEDIATELY WHEN:\n"
                "1. You have read the relevant files.\n"
                "2. You can answer the user's question.\n"
                "3. You have called any tool 3+ times.\n\n"
                f"{GROUNDING_RULES}"
            )

    def _build_default_prompt(self) -> str:
        """Build default prompt for unknown providers."""
        return (
            "You are a code analyst. Follow these rules strictly:\n\n"
            "TOOL USAGE:\n"
            "- Use list_directory or read_file to inspect files before answering.\n"
            "- Call tools ONE AT A TIME. Wait for results before calling the next tool.\n"
            "- After reading 2-3 relevant files, STOP and provide your answer.\n"
            "- Do NOT repeatedly call the same tool with similar arguments.\n\n"
            "RESPONSE FORMAT:\n"
            "- After gathering information, provide a CLEAR ANSWER in plain text.\n"
            "- Do NOT output raw JSON, XML tags, or tool call syntax in your response.\n"
            "- Keep responses concise and focused on the user's question.\n\n"
            "WHEN TO STOP:\n"
            "- Stop calling tools when you have enough information to answer.\n"
            "- If you've called the same tool 3+ times, stop and summarize.\n"
            "- Always end with a human-readable answer, not more tool calls.\n\n"
            f"{GROUNDING_RULES}"
        )


def build_system_prompt(
    provider_name: str,
    model: str,
    tool_adapter: Optional[BaseToolCallingAdapter] = None,
    capabilities: Optional[ToolCallingCapabilities] = None,
    prompt_contributors: Optional[list] = None,
    concise_mode: bool = False,
) -> str:
    """Build a system prompt (convenience function).

    Args:
        provider_name: Provider name
        model: Model name
        tool_adapter: Optional tool calling adapter
        capabilities: Optional pre-computed capabilities
        prompt_contributors: Optional list of PromptContributorProtocol implementations
        concise_mode: If True, adds guidance for brief, direct responses

    Returns:
        System prompt string
    """
    builder = SystemPromptBuilder(
        provider_name=provider_name,
        model=model,
        tool_adapter=tool_adapter,
        capabilities=capabilities,
        prompt_contributors=prompt_contributors,
        concise_mode=concise_mode,
    )
    return builder.build()
