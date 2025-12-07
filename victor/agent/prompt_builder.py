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
from typing import Optional, Set

from victor.agent.tool_calling import BaseToolCallingAdapter, ToolCallingCapabilities

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
1. FIRST: Read README.md, CLAUDE.md, or project docs to understand high-level structure
2. THEN: List main src/ directories to identify actual modules
3. Filter out test/, demo/, examples/, clients/ when listing "key" components
4. Rank by: file size, import frequency, centrality - NOT alphabetically
5. Improvements must reference ACTUAL code patterns (e.g., "line 150 uses X, could be Y")
Do NOT list components alphabetically. Prioritize by architectural importance.""",
    "general": """[GENERAL] Moderate exploration. 3-6 tool calls. Answer concisely.""",
}


def get_task_type_hint(task_type: str) -> str:
    """Get prompt hint for a specific task type.

    Args:
        task_type: The detected task type (e.g., "create_simple", "edit")

    Returns:
        Task-specific prompt hint or empty string if not found
    """
    return TASK_TYPE_HINTS.get(task_type.lower(), "")


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
    """

    def __init__(
        self,
        provider_name: str,
        model: str,
        tool_adapter: Optional[BaseToolCallingAdapter] = None,
        capabilities: Optional[ToolCallingCapabilities] = None,
    ):
        """Initialize the prompt builder.

        Args:
            provider_name: Name of the provider (e.g., "ollama", "anthropic")
            model: Model name/identifier
            tool_adapter: Optional tool calling adapter for getting hints
            capabilities: Optional pre-computed capabilities
        """
        self.provider_name = (provider_name or "").lower()
        self.model = model or ""
        self.model_lower = self.model.lower()
        self.tool_adapter = tool_adapter
        self.capabilities = capabilities

    def is_cloud_provider(self) -> bool:
        """Check if the provider is a cloud-based API with robust tool calling."""
        return self.provider_name in CLOUD_PROVIDERS

    def is_local_provider(self) -> bool:
        """Check if the provider is a local model (Ollama, LMStudio, vLLM)."""
        return self.provider_name in LOCAL_PROVIDERS

    def has_native_tool_support(self) -> bool:
        """Check if the model has known native tool calling support."""
        return any(pattern in self.model_lower for pattern in NATIVE_TOOL_MODELS)

    def build(self) -> str:
        """Build the system prompt.

        Uses adapter hints if available, otherwise falls back to
        provider-specific prompt construction.

        Returns:
            System prompt string tailored to the provider/model
        """
        # Try adapter-based prompt first
        if self.tool_adapter:
            return self._build_with_adapter()

        # Fall back to provider-specific prompt
        return self._build_for_provider()

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
        """Build prompt for cloud providers (Anthropic, OpenAI, xAI).

        Delegates to Google-specific prompt for Gemini models.
        """
        if self.provider_name == "google":
            return self._build_google_prompt()

        return (
            "You are an expert code analyst with access to tools for exploring "
            "and modifying code. Use them effectively:\n\n"
            "1. Use list_directory and read_file to examine code before conclusions.\n"
            "2. If asked to modify code, use write_file or edit_files after understanding context.\n"
            "3. Provide clear, actionable responses based on actual file contents.\n"
            "4. Always cite specific file paths and line numbers when referencing code.\n"
            "5. You may call multiple tools in parallel when they are independent.\n\n"
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
                "- Call tools one at a time, waiting for results.\n"
                "- After 2-3 successful tool calls, provide your answer.\n"
                "- Do NOT make identical repeated tool calls.\n\n"
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
) -> str:
    """Build a system prompt (convenience function).

    Args:
        provider_name: Provider name
        model: Model name
        tool_adapter: Optional tool calling adapter
        capabilities: Optional pre-computed capabilities

    Returns:
        System prompt string
    """
    builder = SystemPromptBuilder(
        provider_name=provider_name,
        model=model,
        tool_adapter=tool_adapter,
        capabilities=capabilities,
    )
    return builder.build()
