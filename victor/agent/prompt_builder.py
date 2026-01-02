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

if TYPE_CHECKING:
    from victor.framework.enrichment import (
        PromptEnrichmentService,
        EnrichmentContext,
        EnrichedPrompt,
    )

logger = logging.getLogger(__name__)


# Provider classifications
CLOUD_PROVIDERS: Set[str] = {"anthropic", "openai", "google", "xai", "moonshot", "kimi", "deepseek"}
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

# Task-type-specific prompt hints
# These are appended to system prompts when task type is detected
# Concise format for cloud providers - local models use extended hints from complexity_classifier
TASK_TYPE_HINTS = {
    "code_generation": """[GENERATE] Write code directly. No exploration needed. Complete implementation.""",
    "create_simple": """[CREATE] Write file immediately. Skip codebase exploration. One tool call max.""",
    "create": """[CREATE+CONTEXT] Read 1-2 relevant files, then create. Follow existing patterns.""",
    "edit": """[EDIT] Read target file first, then modify. Focused changes only.""",
    "search": """[SEARCH] Use code_search/list_directory. Summarize after 2-4 calls.""",
    "action": """[ACTION] Execute git/test/build operations. Multiple tool calls allowed. Continue until complete.""",
    "analysis_deep": """[ANALYSIS] Thorough codebase exploration. Read all relevant modules. Comprehensive output.""",
    "analyze": """[ANALYZE] Examine code carefully. Read related files. Structured findings.""",
    "design": """[ARCHITECTURE] For architecture/component questions:
USE STRUCTURED GRAPH FIRST:
- Call architecture_summary to get module pagerank/centrality with edge_counts + 2–3 callsites (runtime-only). Avoid ad-hoc graph/find hops unless data is missing.
- Keep modules vs symbols separate; cite CALLS/INHERITS/IMPORTS counts and callsites (file:line) per hotspot.
- Prefer runtime code; ignore tests/venv/build outputs unless explicitly requested.
DOC-FIRST STRATEGY (mandatory order):
1. FIRST: Read architecture docs if they exist:
   - read_file CLAUDE.md, .victor/init.md, README.md, ARCHITECTURE.md
   - These contain component lists, named implementations, and key relationships
2. SECOND: Explore implementation directories systematically:
   - list_directory on src/, lib/, engines/, impls/, modules/, core/, services/
   - Directory names under impls/ or engines/ are often named implementations
   - Look for ALL-CAPS directory/file names - these are typically named engines/components
3. THIRD: Read key implementation files for each component found
4. FOURTH: Look for benchmark/test files (benches/, *_bench*, *_test*) for performance insights

DISCOVERY PATTERNS - Look for:
- Named implementations: Directories with ALL-CAPS names (engines, stores, protocols)
- Factories/registries: Files named *_factory.*, *_registry.*, mod.rs, index.ts
- Core abstractions: base.py, interface.*, trait definitions
- Configuration: *.yaml, *.toml in config/ directories

Output requirements:
- Use discovered component names (not generic descriptions like "storage module")
- Include file:line references (e.g., "src/engines/impl.rs:42")
- Verify improvements reference ACTUAL code patterns (grep first)
Use 15-20 tool calls minimum. Prioritize by architectural importance.""",
    "general": """[GENERAL] Moderate exploration. 3-6 tool calls. Answer concisely.""",
    # DevOps vertical task types (aligned with TaskType enum)
    "infrastructure": """[INFRASTRUCTURE] Deploy infrastructure (Kubernetes, Terraform, Docker, cloud):
1. Use Infrastructure as Code (Terraform, CloudFormation, Pulumi)
2. Implement multi-stage Docker builds for smaller images
3. Define resource limits and requests for Kubernetes
4. Use ConfigMaps/Secrets for configuration management
5. Tag all resources for cost tracking and organization""",
    "ci_cd": """[CI/CD] Configure continuous integration/deployment:
1. Define clear stages: lint, test, build, deploy
2. Cache dependencies for faster builds
3. Use matrix builds for cross-platform testing
4. Implement proper secret management (GitHub Secrets, Vault)
5. Add manual approval for production deployments""",
    # Data Analysis vertical task types (aligned with TaskType enum)
    "data_analysis": """[DATA ANALYSIS] Comprehensive data exploration and analysis:
1. Load data and check shape/types with df.info(), df.describe()
2. Calculate summary statistics (mean, median, std, quartiles)
3. Identify missing values and their patterns (df.isnull().sum())
4. Check for duplicates and data quality issues
5. Analyze correlations and distributions before modeling""",
    "visualization": """[VISUALIZATION] Create informative charts and dashboards:
1. Choose appropriate chart type for the data (bar, line, scatter, heatmap)
2. Use clear labels, titles, and legends
3. Add context (units, time periods, annotations)
4. Consider colorblind-friendly palettes (viridis, cividis)
5. Save as high-resolution images (plt.savefig('fig.png', dpi=300))""",
    # Research vertical task types (aligned with TaskType enum)
    "fact_check": """[FACT-CHECK] Verify claims with multiple independent sources:
1. Search for original sources and official documentation
2. Cross-reference with authoritative databases
3. Check recency and relevance of sources
4. Note any conflicting information found""",
    "literature_review": """[LITERATURE] Systematic review of existing knowledge:
1. Define scope and search criteria
2. Search academic and authoritative sources
3. Extract key findings and methodologies
4. Synthesize patterns and gaps
5. Provide structured bibliography""",
    "competitive_analysis": """[ANALYSIS] Compare products, services, or approaches:
1. Identify key comparison criteria
2. Gather data from official sources
3. Create objective comparison matrix
4. Note strengths, weaknesses, limitations
5. Avoid promotional language""",
    "trend_research": """[TRENDS] Identify patterns and emerging developments:
1. Search recent news and publications
2. Look for quantitative data and statistics
3. Identify key players and innovations
4. Note methodology limitations
5. Distinguish facts from speculation""",
    "technical_research": """[TECHNICAL] Deep dive into technical topics:
1. Start with official documentation
2. Search code repositories and examples
3. Look for benchmarks and comparisons
4. Note version-specific information
5. Verify with multiple technical sources""",
    # Coding vertical granular task types (aligned with TaskType enum)
    "refactor": """[REFACTOR] Restructure existing code without changing behavior:
1. Analyze current code structure and identify issues
2. Plan incremental changes to minimize risk
3. Apply refactoring patterns (extract method, rename, move)
4. Verify behavior unchanged with existing tests
5. Document architectural decisions""",
    "debug": """[DEBUG] Find and fix bugs systematically:
1. Reproduce the issue consistently
2. Read error messages and stack traces carefully
3. Trace execution flow to find root cause
4. Isolate the problem with minimal test case
5. Fix root cause, not just symptoms""",
    "test": """[TEST] Write comprehensive tests for code:
1. Identify critical paths and edge cases
2. Write unit tests for individual functions
3. Add integration tests for component interactions
4. Mock external dependencies appropriately
5. Aim for meaningful coverage, not just metrics""",
    # DevOps vertical granular task types (aligned with TaskType enum)
    "dockerfile": """[DOCKERFILE] Create optimized Docker images:
1. Use official base images with specific version tags
2. Implement multi-stage builds for smaller images
3. Order layers for optimal cache utilization
4. Add health checks and proper signal handling
5. Run as non-root user for security""",
    "docker_compose": """[COMPOSE] Configure multi-container applications:
1. Define all services with explicit dependencies
2. Use named volumes for persistent data
3. Configure proper network isolation
4. Add health checks for service readiness
5. Use environment files for secrets""",
    "kubernetes": """[K8S] Create Kubernetes configurations:
1. Use Deployments for stateless, StatefulSets for stateful apps
2. Define resource requests and limits appropriately
3. Add liveness and readiness probes
4. Use ConfigMaps for config, Secrets for sensitive data
5. Implement NetworkPolicies for security""",
    "terraform": """[TERRAFORM] Write Infrastructure as Code:
1. Organize code into reusable modules
2. Use remote state with locking (S3+DynamoDB, etc.)
3. Implement proper variable typing and validation
4. Tag all resources for cost tracking
5. Use data sources instead of hardcoded IDs""",
    "monitoring": """[MONITORING] Set up observability infrastructure:
1. Define key metrics and SLIs/SLOs
2. Configure alerting with appropriate thresholds
3. Set up distributed tracing for microservices
4. Implement structured logging with context
5. Create dashboards for visibility""",
    # Data Analysis vertical granular task types (aligned with TaskType enum)
    "data_profiling": """[PROFILE] Comprehensive data profiling:
1. Load data and check shape/types (df.info(), df.dtypes)
2. Calculate summary statistics (mean, median, std, quartiles)
3. Identify missing values and their patterns
4. Check for duplicates and uniqueness constraints
5. Analyze value distributions and outliers""",
    "statistical_analysis": """[STATISTICS] Perform rigorous statistical analysis:
1. State null and alternative hypotheses clearly
2. Check assumptions (normality, variance homogeneity)
3. Choose appropriate test (t-test, ANOVA, chi-square)
4. Calculate test statistic and p-value
5. Interpret results with effect size and confidence intervals""",
    "correlation_analysis": """[CORRELATION] Analyze variable relationships:
1. Calculate correlation matrix for numeric variables
2. Use appropriate method (Pearson for linear, Spearman for monotonic)
3. Visualize with heatmap or scatter matrix
4. Identify strong correlations (|r| > 0.7)
5. Note potential confounders and causation vs correlation""",
    "regression": """[REGRESSION] Build predictive regression models:
1. Define target and feature variables clearly
2. Split data into train/test sets (or use cross-validation)
3. Check for multicollinearity (VIF analysis)
4. Fit model and assess coefficients significance
5. Evaluate with R², RMSE, residual plots""",
    "clustering": """[CLUSTERING] Segment data into meaningful groups:
1. Scale features appropriately (StandardScaler, MinMaxScaler)
2. Determine optimal cluster count (elbow method, silhouette score)
3. Apply appropriate algorithm (K-means, hierarchical, DBSCAN)
4. Visualize clusters (PCA/t-SNE for high dimensions)
5. Profile cluster characteristics and interpret business meaning""",
    "time_series": """[TIMESERIES] Analyze temporal data patterns:
1. Check datetime format and ensure proper frequency
2. Plot time series and identify patterns (trend, seasonality, cycles)
3. Decompose into trend, seasonal, and residual components
4. Check stationarity (ADF test) and apply differencing if needed
5. Apply appropriate forecasting method (ARIMA, Prophet, etc.)""",
    # Research vertical granular task types (aligned with TaskType enum)
    "general_query": """[QUERY] Answer general research questions:
1. Clarify the scope and specific aspects of the question
2. Search for authoritative sources and documentation
3. Synthesize information from multiple perspectives
4. Provide clear, structured explanation
5. Note limitations and areas of uncertainty""",
}


def get_task_type_hint(task_type: str, prompt_contributors: Optional[list] = None) -> str:
    """Get prompt hint for a specific task type.

    This function now supports vertical prompt contributors. It merges hints from:
    1. Vertical prompt contributors (if provided)
    2. Hardcoded TASK_TYPE_HINTS (fallback for backward compatibility)

    Args:
        task_type: The detected task type (e.g., "create_simple", "edit")
        prompt_contributors: Optional list of PromptContributorProtocol implementations

    Returns:
        Task-specific prompt hint or empty string if not found
    """
    # Try vertical contributors first
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
                # Log when vertical hint is applied
                contributor_name = type(contributor).__name__
                logger.info(
                    "Applied vertical task hint: task_type=%s, contributor=%s",
                    task_type,
                    contributor_name,
                )
                return hint_text

    # Fallback to hardcoded hints
    hint = TASK_TYPE_HINTS.get(task_type.lower(), "")
    if hint:
        logger.debug("Applied default task hint for task_type=%s", task_type)
    return hint


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
    - Falls back to hardcoded TASK_TYPE_HINTS for backward compatibility
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
        self.provider_name = (provider_name or "").lower()
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

        merged = TASK_TYPE_HINTS.copy()  # Start with hardcoded hints

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
        """Get grounding rules from vertical contributors.

        Returns:
            Merged grounding rules from all contributors
        """
        if not self.prompt_contributors:
            return ""

        # Collect grounding rules from all contributors
        rules = []
        for contributor in sorted(self.prompt_contributors, key=lambda c: c.get_priority()):
            grounding = contributor.get_grounding_rules()
            if grounding:
                rules.append(grounding)

        return "\n\n".join(rules) if rules else ""

    def get_vertical_system_prompt_sections(self) -> str:
        """Get system prompt sections from vertical contributors.

        Returns:
            Merged system prompt sections from all contributors
        """
        if not self.prompt_contributors:
            return ""

        # Collect sections from all contributors
        sections = []
        for contributor in sorted(self.prompt_contributors, key=lambda c: c.get_priority()):
            section = contributor.get_system_prompt_section()
            if section:
                sections.append(section)

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

    def build(self) -> str:
        """Build the system prompt.

        Uses adapter hints if available, otherwise falls back to
        provider-specific prompt construction. Includes provider-specific
        tool guidance (GAP-5) when available.

        Returns:
            System prompt string tailored to the provider/model
        """
        # Try adapter-based prompt first
        if self.tool_adapter:
            base_prompt = self._build_with_adapter()
        else:
            # Fall back to provider-specific prompt
            base_prompt = self._build_for_provider()

        # Prepend concise mode guidance if enabled
        if self.concise_mode:
            base_prompt = f"{CONCISE_MODE_GUIDANCE}\n\n{base_prompt}"
            logger.debug("Concise mode enabled - added brevity guidance to prompt")

        # Append provider-specific tool guidance if available (GAP-5)
        tool_guidance = self.get_provider_tool_guidance()
        if tool_guidance:
            return f"{base_prompt}\n\n{tool_guidance}"

        return base_prompt

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
                "3. For modification tasks: use write_file or edit_files after understanding context.\n"
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
            "2. If asked to modify code, use write_file or edit_files after understanding context.\n"
            "3. Provide clear, actionable responses based on actual file contents.\n"
            "4. Always cite specific file paths and line numbers when referencing code.\n"
            "5. You may call multiple tools in parallel when they are independent.\n\n"
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
                "You are an expert coding assistant. You can analyze, explain, and generate code.\n\n"
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
