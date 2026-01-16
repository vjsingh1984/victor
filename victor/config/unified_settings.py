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

"""Unified configuration settings with explicit precedence.

This module provides a single source of truth for all Victor configuration,
consolidating 8+ configuration sources into a clear precedence hierarchy:

Precedence (highest to lowest):
1. CLI arguments (passed via override)
2. Environment variables (VICTOR_*)
3. .env file
4. ~/.victor/settings.yaml (loaded manually)
5. ~/.victor/profiles.yaml (loaded manually)
6. Default values

Design:
- Uses Pydantic Settings for type-safe configuration
- Provides from_sources() classmethod for precedence-aware loading
- All settings use direct typed access (no getattr needed)
- Field validators ensure configuration consistency

Usage:
    # Load with all sources
    settings = VictorSettings.from_sources(
        cli_args={"provider": "ollama", "model": "qwen3-coder:30b"},
        profile_name="default"
    )

    # Access settings with type safety
    provider = settings.provider
    max_tools = settings.max_tools
    airgapped = settings.airgapped_mode
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from victor.config.model_capabilities import _load_tool_capable_patterns_from_yaml
from victor.config.orchestrator_constants import BUDGET_LIMITS


class VictorSettings(BaseSettings):
    """Single source of truth for all Victor configuration.

    Consolidates all configuration sources with explicit precedence.
    Use from_sources() classmethod to load with proper precedence.

    Precedence (highest to lowest):
    1. CLI arguments (passed via override)
    2. Environment variables (VICTOR_*)
    3. .env file
    4. ~/.victor/settings.yaml
    5. ~/.victor/profiles.yaml
    6. Default values
    """

    model_config = SettingsConfigDict(
        env_prefix="VICTOR_",
        env_file=".env" if not os.getenv("VICTOR_SKIP_ENV_FILE") else None,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
        protected_namespaces=(),
    )

    # ==========================================================================
    # Provider Settings
    # ==========================================================================

    default_provider: str = Field(
        default="ollama",
        description="Default LLM provider (ollama, anthropic, openai, google, groq, lmstudio, vllm)",
    )
    default_model: str = Field(default="qwen3-coder:30b", description="Default model identifier")
    model_name: Optional[str] = Field(
        default=None, description="Runtime model override (alias for default_model)"
    )
    default_temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Default temperature for generation"
    )
    default_max_tokens: int = Field(
        default=4096, gt=0, description="Default maximum tokens for generation"
    )

    # ==========================================================================
    # API Keys
    # ==========================================================================

    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    google_api_key: Optional[str] = Field(default=None, description="Google Gemini API key")
    moonshot_api_key: Optional[str] = Field(
        default=None, description="Moonshot AI API key (Kimi K2 models)"
    )
    deepseek_api_key: Optional[str] = Field(
        default=None, description="DeepSeek API key (DeepSeek-V3 models)"
    )

    # ==========================================================================
    # Local Server URLs
    # ==========================================================================

    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama server base URL"
    )
    lmstudio_base_urls: List[str] = Field(
        default=["http://127.0.0.1:1234"], description="LMStudio tiered endpoints (try in order)"
    )
    vllm_base_url: str = Field(default="http://localhost:8000", description="vLLM server base URL")

    # ==========================================================================
    # Logging
    # ==========================================================================

    log_level: str = Field(
        default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    log_file: Optional[str] = Field(
        default=None, description="Log file path (None for console only)"
    )

    # ==========================================================================
    # Privacy and Security
    # ==========================================================================

    airgapped_mode: bool = Field(
        default=False, description="Enable airgapped mode (no external network calls)"
    )

    # ==========================================================================
    # Vertical Configuration
    # ==========================================================================

    default_vertical: str = Field(
        default="coding", description="Default vertical when --vertical not specified"
    )
    auto_detect_vertical: bool = Field(
        default=False, description="Auto-detect vertical from project context (experimental)"
    )

    # ==========================================================================
    # Server Security (FastAPI/WebSocket)
    # ==========================================================================

    server_api_key: Optional[str] = Field(
        default=None,
        description="API key for server authentication (Authorization: Bearer <token>)",
    )
    server_session_secret: Optional[str] = Field(
        default=None, description="HMAC secret for session tokens (defaults to random per-process)"
    )
    server_max_sessions: int = Field(
        default=100, ge=1, description="Hard cap on simultaneous sessions"
    )
    server_max_message_bytes: int = Field(
        default=32768,
        ge=1024,
        description="Maximum inbound message payload size (bytes) for WebSocket",
    )
    server_session_ttl_seconds: int = Field(
        default=86400, ge=60, description="Session token time-to-live in seconds"
    )
    render_max_payload_bytes: int = Field(
        default=20000, ge=1024, description="Diagram rendering max payload size"
    )
    render_timeout_seconds: int = Field(default=10, ge=1, description="Diagram rendering timeout")
    render_max_concurrency: int = Field(
        default=2, ge=1, description="Diagram rendering max concurrency"
    )

    # ==========================================================================
    # Code Execution Sandbox
    # ==========================================================================

    code_executor_network_disabled: bool = Field(
        default=True, description="Disable network in code execution sandbox"
    )
    code_executor_memory_limit: Optional[str] = Field(
        default="512m", description="Memory limit for code execution sandbox"
    )
    code_executor_cpu_shares: Optional[int] = Field(
        default=256, ge=1, description="CPU shares for code execution sandbox"
    )

    # ==========================================================================
    # Write Approval Mode
    # ==========================================================================

    write_approval_mode: str = Field(
        default="risky_only", description="Write approval mode: off, risky_only, all_writes"
    )

    # ==========================================================================
    # Headless Mode Settings
    # ==========================================================================

    headless_mode: bool = Field(
        default=False, description="Run without prompts, auto-approve safe actions"
    )
    dry_run_mode: bool = Field(default=False, description="Preview changes without applying them")
    auto_approve_safe: bool = Field(
        default=False, description="Auto-approve read-only and LOW risk operations"
    )
    max_file_changes: Optional[int] = Field(
        default=None, ge=1, description="Limit file modifications per session"
    )
    one_shot_mode: bool = Field(default=False, description="Exit after completing a single request")

    # ==========================================================================
    # Embedding Models
    # ==========================================================================

    unified_embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Unified embedding model for tool selection and codebase search",
    )

    # ==========================================================================
    # Tool Selection Strategy
    # ==========================================================================

    use_semantic_tool_selection: bool = Field(
        default=True, description="Use embeddings instead of keywords for tool selection"
    )
    embedding_provider: str = Field(
        default="sentence-transformers",
        description="Embedding provider: sentence-transformers, ollama, vllm, lmstudio",
    )
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5", description="Embedding model name"
    )

    # ==========================================================================
    # Codebase Semantic Search
    # ==========================================================================

    codebase_vector_store: str = Field(
        default="lancedb", description="Vector store backend: lancedb, chromadb"
    )
    codebase_embedding_provider: str = Field(
        default="sentence-transformers", description="Codebase embedding provider"
    )
    codebase_embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5", description="Codebase embedding model"
    )
    codebase_persist_directory: Optional[str] = Field(
        default=None, description="Codebase embeddings persist directory"
    )
    codebase_dimension: int = Field(default=384, ge=1, description="Embedding dimension")
    codebase_batch_size: int = Field(
        default=32, ge=1, description="Batch size for embedding generation"
    )
    codebase_graph_store: str = Field(default="sqlite", description="Graph backend: sqlite, duckdb")
    codebase_graph_path: Optional[str] = Field(
        default=None, description="Optional explicit graph db path"
    )
    core_readonly_tools: Optional[List[str]] = Field(
        default=None, description="Override/extend curated read-only tool set"
    )

    # ==========================================================================
    # Semantic Search Quality
    # ==========================================================================

    semantic_similarity_threshold: float = Field(
        default=0.5, ge=0.1, le=0.9, description="Minimum semantic similarity score"
    )
    semantic_query_expansion_enabled: bool = Field(
        default=True, description="Expand queries with synonyms/related terms"
    )
    semantic_max_query_expansions: int = Field(
        default=5, ge=1, description="Max query variations to try (including original)"
    )

    # ==========================================================================
    # Hybrid Search
    # ==========================================================================

    enable_hybrid_search: bool = Field(
        default=False, description="Enable hybrid search combining semantic + keyword"
    )
    hybrid_search_semantic_weight: float = Field(
        default=0.6, ge=0.0, le=1.0, description="Weight for semantic search"
    )
    hybrid_search_keyword_weight: float = Field(
        default=0.4, ge=0.0, le=1.0, description="Weight for keyword search"
    )

    # ==========================================================================
    # RL-based Threshold Learning
    # ==========================================================================

    enable_semantic_threshold_rl_learning: bool = Field(
        default=False, description="Enable automatic threshold learning"
    )
    semantic_threshold_overrides: Dict[str, float] = Field(
        default_factory=dict, description="Model:task:tool threshold overrides"
    )

    # ==========================================================================
    # Tool Deduplication
    # ==========================================================================

    enable_tool_deduplication: bool = Field(
        default=True, description="Enable deduplication tracker to prevent redundant calls"
    )
    tool_deduplication_window_size: int = Field(
        default=20, ge=1, description="Number of recent calls to track"
    )

    # ==========================================================================
    # UI Settings
    # ==========================================================================

    theme: str = Field(default="monokai", description="UI theme name")
    show_token_count: bool = Field(default=True, description="Show token counts in UI")
    stream_responses: bool = Field(
        default=True, description="Stream responses instead of waiting for completion"
    )
    use_emojis: bool = Field(default=True, description="Enable emoji indicators in output")

    # ==========================================================================
    # MCP Settings
    # ==========================================================================

    use_mcp_tools: bool = Field(default=False, description="Enable MCP tool integration")
    mcp_command: Optional[str] = Field(
        default=None, description="MCP server command (e.g., 'python mcp_server.py')"
    )
    mcp_prefix: str = Field(default="mcp", description="Prefix for MCP tool names")

    # ==========================================================================
    # Tool Execution Settings
    # ==========================================================================

    tool_call_budget: int = Field(
        default=BUDGET_LIMITS.max_session_budget, ge=1, description="Maximum tool calls per session"
    )
    tool_call_budget_warning_threshold: int = Field(
        default=int(BUDGET_LIMITS.max_session_budget * BUDGET_LIMITS.warning_threshold_pct),
        ge=1,
        description="Warn when approaching budget limit",
    )
    tool_calling_models: Dict[str, List[str]] = Field(
        default_factory=_load_tool_capable_patterns_from_yaml,
        description="Models known to support structured tool calls per provider",
    )

    # ==========================================================================
    # Tool Retry Settings
    # ==========================================================================

    tool_retry_enabled: bool = Field(
        default=True, description="Enable automatic retry for failed tool executions"
    )
    tool_retry_max_attempts: int = Field(
        default=3, ge=1, description="Maximum retry attempts per tool call"
    )
    tool_retry_base_delay: float = Field(
        default=1.0, ge=0.1, description="Base delay in seconds for exponential backoff"
    )
    tool_retry_max_delay: float = Field(
        default=10.0, ge=1.0, description="Maximum delay in seconds between retries"
    )

    # ==========================================================================
    # Tool Selection Fallback
    # ==========================================================================

    fallback_max_tools: int = Field(
        default=8, ge=1, description="Cap tool list when stage pruning removes everything"
    )

    # ==========================================================================
    # Tool Result Caching
    # ==========================================================================

    tool_cache_enabled: bool = Field(default=True, description="Enable tool result caching")
    tool_cache_ttl: int = Field(default=600, ge=0, description="Tool cache TTL in seconds")
    tool_cache_allowlist: List[str] = Field(
        default=[
            "code_search",
            "semantic_code_search",
            "list_directory",
            "plan_files",
        ],
        description="Tools allowed to be cached",
    )

    # ==========================================================================
    # Tool Argument Validation
    # ==========================================================================

    tool_validation_mode: str = Field(
        default="lenient", description="Validation mode: strict, lenient, off"
    )

    # ==========================================================================
    # Context Compaction
    # ==========================================================================

    context_compaction_strategy: str = Field(
        default="tiered", description="Compaction strategy: simple, tiered, semantic, hybrid"
    )
    context_min_messages_to_keep: int = Field(
        default=6, ge=1, description="Minimum messages to retain after compaction"
    )
    context_tool_retention_weight: float = Field(
        default=1.5, ge=0.0, description="Boost for tool result retention"
    )
    context_recency_weight: float = Field(
        default=2.0, ge=0.0, description="Boost for recent messages"
    )
    context_semantic_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Min similarity for semantic retention"
    )

    # ==========================================================================
    # Plugin System
    # ==========================================================================

    plugin_enabled: bool = Field(default=True, description="Enable plugin system")
    plugin_packages: List[str] = Field(
        default_factory=list, description="Python packages to load as plugins"
    )
    plugin_disabled: List[str] = Field(
        default_factory=list, description="List of plugin names to disable"
    )
    plugin_config: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Plugin-specific configuration"
    )

    # ==========================================================================
    # Security Scan Extensions
    # ==========================================================================

    security_dependency_scan: bool = Field(
        default=False, description="Enable dependency vulnerability scanning"
    )
    security_iac_scan: bool = Field(
        default=False, description="Enable infrastructure-as-code scanning"
    )

    # ==========================================================================
    # LMStudio Resource Guard
    # ==========================================================================

    lmstudio_max_vram_gb: Optional[float] = Field(
        default=48.0, ge=1.0, description="Cap model selection to this VRAM budget (GB)"
    )

    # ==========================================================================
    # Exploration Loop Settings
    # ==========================================================================

    max_exploration_iterations: int = Field(
        default=8, ge=1, description="Max consecutive read-only tool calls with minimal output"
    )
    chat_max_iterations: int = Field(
        default=50, ge=1, description="Maximum chat iterations per session"
    )
    max_consecutive_tool_calls: int = Field(
        default=20, ge=1, description="Maximum consecutive tool calls before forcing synthesis"
    )
    max_exploration_iterations_action: int = Field(
        default=12, ge=1, description="More lenient for action tasks"
    )
    max_exploration_iterations_analysis: int = Field(
        default=50, ge=1, description="Very lenient for analysis tasks"
    )
    min_content_threshold: int = Field(
        default=150, ge=1, description="Minimum chars to consider 'substantial' output"
    )
    max_research_iterations: int = Field(
        default=6, ge=1, description="Force synthesis after N consecutive web searches"
    )

    # ==========================================================================
    # Recovery & Loop Detection Thresholds
    # ==========================================================================

    recovery_empty_response_threshold: int = Field(
        default=5, ge=1, description="Force after N consecutive empty responses from model"
    )
    recovery_blocked_consecutive_threshold: int = Field(
        default=6, ge=1, description="Force after N consecutive blocked attempts"
    )
    recovery_blocked_total_threshold: int = Field(
        default=9, ge=1, description="Force after N total blocked attempts"
    )

    # ==========================================================================
    # Continuation Prompts
    # ==========================================================================

    max_continuation_prompts_analysis: int = Field(
        default=6, ge=1, description="For analysis tasks"
    )
    max_continuation_prompts_action: int = Field(default=5, ge=1, description="For action tasks")
    max_continuation_prompts_default: int = Field(default=3, ge=1, description="For other tasks")
    continuation_prompt_overrides: Dict[str, Dict[str, int]] = Field(
        default_factory=dict, description="Provider/model-specific continuation prompt overrides"
    )
    enable_continuation_rl_learning: bool = Field(
        default=False, description="Enable RL-based learning of optimal continuation prompts"
    )

    # ==========================================================================
    # Session Timeout
    # ==========================================================================

    session_idle_timeout: int = Field(
        default=180, ge=10, description="Maximum seconds of inactivity before forcing completion"
    )

    # ==========================================================================
    # Conversation Memory
    # ==========================================================================

    conversation_memory_enabled: bool = Field(
        default=True, description="Enable SQLite-backed conversation persistence"
    )
    conversation_embeddings_enabled: bool = Field(
        default=True, description="Enable LanceDB embeddings for semantic retrieval"
    )
    max_context_tokens: int = Field(
        default=100000, ge=1000, description="Maximum tokens in context window"
    )
    max_context_chars: Optional[int] = Field(
        default=None,
        ge=1000,
        description="Maximum characters in context (alternative to token-based limit)",
    )
    response_token_reserve: int = Field(
        default=4096, ge=100, description="Tokens reserved for model response"
    )

    # ==========================================================================
    # Provider Resilience
    # ==========================================================================

    resilience_enabled: bool = Field(
        default=True, description="Enable circuit breaker and retry logic"
    )
    circuit_breaker_failure_threshold: int = Field(
        default=5, ge=1, description="Failures before circuit opens"
    )
    circuit_breaker_success_threshold: int = Field(
        default=2, ge=1, description="Successes before circuit closes"
    )
    circuit_breaker_timeout: float = Field(
        default=60.0, ge=1.0, description="Seconds before half-open state"
    )
    circuit_breaker_half_open_max: int = Field(
        default=3, ge=1, description="Max requests in half-open state"
    )

    # ==========================================================================
    # Retry Settings
    # ==========================================================================

    retry_max_attempts: int = Field(default=3, ge=1, description="Maximum retry attempts")
    retry_base_delay: float = Field(default=1.0, ge=0.1, description="Base delay in seconds")
    retry_max_delay: float = Field(
        default=60.0, ge=1.0, description="Maximum delay between retries"
    )
    retry_exponential_base: float = Field(
        default=2.0, ge=1.0, description="Exponential backoff multiplier"
    )

    # ==========================================================================
    # Rate Limiting
    # ==========================================================================

    rate_limiting_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests_per_minute: int = Field(
        default=50, ge=1, description="Requests per minute limit"
    )
    rate_limit_tokens_per_minute: int = Field(
        default=50000, ge=1, description="Tokens per minute limit"
    )
    rate_limit_max_concurrent: int = Field(
        default=5, ge=1, description="Maximum concurrent requests"
    )
    rate_limit_queue_size: int = Field(
        default=100, ge=1, description="Maximum pending requests in queue"
    )
    rate_limit_num_workers: int = Field(default=3, ge=1, description="Number of queue worker tasks")

    # ==========================================================================
    # Streaming Metrics
    # ==========================================================================

    streaming_metrics_enabled: bool = Field(
        default=True, description="Enable streaming performance metrics"
    )
    streaming_metrics_history_size: int = Field(
        default=1000, ge=10, description="Number of metrics samples to retain"
    )

    # ==========================================================================
    # Serialization
    # ==========================================================================

    serialization_enabled: bool = Field(
        default=True, description="Enable token-optimized serialization"
    )
    serialization_default_format: Optional[str] = Field(
        default=None,
        description="Default format: json, toon, csv, markdown_table, reference_encoded",
    )
    serialization_min_savings_threshold: float = Field(
        default=0.15, ge=0.0, le=1.0, description="Min savings to use alternative format"
    )
    serialization_include_format_hint: bool = Field(
        default=True, description="Include format description in output"
    )
    serialization_min_rows_for_tabular: int = Field(
        default=3, ge=1, description="Min rows to consider tabular formats"
    )
    serialization_debug_mode: bool = Field(
        default=False, description="Include data characteristics in output"
    )

    # ==========================================================================
    # Intelligent Agent Pipeline
    # ==========================================================================

    intelligent_pipeline_enabled: bool = Field(
        default=True, description="Master switch for intelligent features"
    )
    intelligent_quality_scoring: bool = Field(
        default=True, description="Enable multi-dimensional quality scoring"
    )
    intelligent_mode_learning: bool = Field(
        default=True, description="Enable Q-learning for mode transitions"
    )
    intelligent_prompt_optimization: bool = Field(
        default=True, description="Enable embedding-based prompt selection"
    )
    intelligent_grounding_verification: bool = Field(
        default=True, description="Enable hallucination detection"
    )
    intelligent_min_quality_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum quality to accept response"
    )
    intelligent_grounding_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for grounding (alias for grounding_confidence_threshold)",
    )
    grounding_confidence_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Confidence threshold for grounding verification"
    )
    intelligent_exploration_rate: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Initial exploration vs exploitation"
    )
    intelligent_learning_rate: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Q-learning alpha parameter"
    )
    intelligent_discount_factor: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Q-learning gamma parameter"
    )

    # ==========================================================================
    # Analytics
    # ==========================================================================

    analytics_enabled: bool = Field(default=True, description="Enable usage analytics")

    # ==========================================================================
    # Code Correction
    # ==========================================================================

    code_correction_enabled: bool = Field(
        default=True, description="Enable automatic code correction suggestions"
    )
    code_correction_auto_fix: bool = Field(
        default=True, description="Automatically apply code corrections"
    )
    code_correction_max_iterations: int = Field(
        default=3, ge=1, description="Maximum correction iterations per file"
    )

    # ==========================================================================
    # Auto Commit
    # ==========================================================================

    auto_commit_enabled: bool = Field(
        default=False, description="Enable automatic git commits for file changes"
    )

    # ==========================================================================
    # Tool Output Limits
    # ==========================================================================

    max_tool_output_chars: int = Field(
        default=15000, ge=100, description="Maximum characters in tool output"
    )
    max_tool_output_lines: int = Field(
        default=200, ge=10, description="Maximum lines in tool output"
    )
    tool_result_truncation: bool = Field(
        default=True, description="Enable truncation of large tool results"
    )

    # ==========================================================================
    # Parallel Tool Execution
    # ==========================================================================

    parallel_tool_execution: bool = Field(
        default=True, description="Enable parallel execution of independent tools"
    )
    max_concurrent_tools: int = Field(
        default=5, ge=1, description="Maximum tools to execute concurrently"
    )

    # ==========================================================================
    # Response Completion
    # ==========================================================================

    response_completion_retries: int = Field(
        default=3, ge=1, description="Max retries for incomplete responses"
    )
    force_response_on_error: bool = Field(
        default=True, description="Force synthesis even if errors occurred"
    )

    # ==========================================================================
    # Context Compaction Advanced
    # ==========================================================================

    context_proactive_compaction: bool = Field(
        default=True, description="Enable proactive compaction before hitting limit"
    )
    context_proactive_threshold: float = Field(
        default=0.90,
        ge=0.5,
        le=0.99,
        description="Trigger compaction at this fraction of max tokens",
    )
    context_min_messages_after_compact: int = Field(
        default=8, ge=1, description="Minimum messages after compaction"
    )
    context_truncation_strategy: str = Field(
        default="smart", description="Truncation strategy: simple, smart, preserve_code"
    )
    file_structure_threshold: int = Field(
        default=50000, ge=1000, description="File size threshold for structure-based truncation"
    )

    # ==========================================================================
    # Conversation History
    # ==========================================================================

    max_conversation_history: int = Field(
        default=100, ge=10, description="Maximum messages to retain in conversation history"
    )

    # ==========================================================================
    # Tool Cache Storage
    # ==========================================================================

    tool_cache_dir: Optional[str] = Field(
        default=None, description="Custom directory for tool result cache"
    )

    # ==========================================================================
    # Plugin Directories
    # ==========================================================================

    plugin_dirs: List[str] = Field(
        default_factory=list, description="Additional directories to scan for plugins"
    )
    disabled_plugins: List[str] = Field(
        default_factory=list,
        description="List of plugin names to disable (renamed from plugin_disabled)",
    )

    # ==========================================================================
    # Provider Resilience Advanced
    # ==========================================================================

    provider_health_checks: bool = Field(
        default=True, description="Enable periodic provider health checks"
    )
    provider_auto_fallback: bool = Field(
        default=True, description="Automatically fallback to secondary providers on failure"
    )
    fallback_providers: List[str] = Field(
        default_factory=list, description="Ordered list of fallback providers"
    )

    # ==========================================================================
    # Subagent Orchestration (Experimental)
    # ==========================================================================

    subagent_orchestration_enabled: bool = Field(
        default=True, description="Enable hierarchical subagent task decomposition"
    )

    # ==========================================================================
    # Observability
    # ==========================================================================

    enable_observability: bool = Field(
        default=True, description="Enable observability integration (Langfuse, etc.)"
    )

    # ==========================================================================
    # Debug Settings
    # ==========================================================================

    debug_logging: bool = Field(default=False, description="Enable verbose debug logging")

    # ==========================================================================
    # Validators
    # ==========================================================================

    @field_validator("default_provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate provider name."""
        valid_providers = [
            "ollama",
            "anthropic",
            "openai",
            "google",
            "groq",
            "lmstudio",
            "vllm",
            "deepseek",
            "moonshot",
            "xai",
        ]
        if v.lower() not in valid_providers:
            raise ValueError(f"Invalid provider: {v}. Must be one of {valid_providers}")
        return v.lower()

    @field_validator("write_approval_mode")
    @classmethod
    def validate_write_approval_mode(cls, v: str) -> str:
        """Validate write approval mode."""
        valid_modes = ["off", "risky_only", "all_writes"]
        if v not in valid_modes:
            raise ValueError(f"Invalid write_approval_mode: {v}. Must be one of {valid_modes}")
        return v

    @field_validator("tool_validation_mode")
    @classmethod
    def validate_tool_validation_mode(cls, v: str) -> str:
        """Validate tool validation mode."""
        valid_modes = ["strict", "lenient", "off"]
        if v not in valid_modes:
            raise ValueError(f"Invalid tool_validation_mode: {v}. Must be one of {valid_modes}")
        return v

    @field_validator("context_compaction_strategy")
    @classmethod
    def validate_context_compaction_strategy(cls, v: str) -> str:
        """Validate context compaction strategy."""
        valid_strategies = ["simple", "tiered", "semantic", "hybrid"]
        if v not in valid_strategies:
            raise ValueError(
                f"Invalid context_compaction_strategy: {v}. Must be one of {valid_strategies}"
            )
        return v

    @model_validator(mode="after")
    def validate_hybrid_search_weights(self) -> "VictorSettings":
        """Validate that hybrid search weights sum to 1.0."""
        if self.enable_hybrid_search:
            total_weight = self.hybrid_search_semantic_weight + self.hybrid_search_keyword_weight
            if abs(total_weight - 1.0) > 0.01:
                raise ValueError(f"Hybrid search weights must sum to 1.0, got {total_weight}")
        return self

    # ==========================================================================
    # Loading with Precedence
    # ==========================================================================

    @classmethod
    def from_sources(
        cls,
        cli_args: Optional[Dict[str, Any]] = None,
        profile_name: Optional[str] = None,
        config_dir: Optional[Path] = None,
    ) -> "VictorSettings":
        """Load settings with proper precedence.

        Precedence (highest to lowest):
        1. CLI arguments (passed via cli_args)
        2. Environment variables (VICTOR_*)
        3. .env file
        4. ~/.victor/settings.yaml
        5. ~/.victor/profiles.yaml (specific profile)
        6. Default values

        Args:
            cli_args: CLI argument overrides (highest priority)
            profile_name: Active profile name to load from profiles.yaml
            config_dir: Custom config directory (defaults to ~/.victor)

        Returns:
            VictorSettings instance with all sources merged

        Example:
            >>> settings = VictorSettings.from_sources(
            ...     cli_args={"provider": "ollama", "model": "qwen3-coder:30b"},
            ...     profile_name="default"
            ... )
            >>> settings.provider
            'ollama'
        """
        # Determine config directory
        if config_dir is None:
            config_dir = Path.home() / ".victor"

        # Start with defaults + env vars + .env file (Pydantic handles these)
        settings_dict: Dict[str, Any] = {}

        # Layer 5: Load profiles.yaml if exists
        profiles_path = config_dir / "profiles.yaml"
        if profiles_path.exists():
            try:
                with open(profiles_path) as f:
                    profiles_data = yaml.safe_load(f) or {}

                # Extract profile if specified
                if profile_name and profile_name in profiles_data.get("profiles", {}):
                    profile_config = profiles_data["profiles"][profile_name]
                    # Only take settings that exist in VictorSettings
                    for key, value in profile_config.items():
                        if key in cls.model_fields:
                            settings_dict[key] = value
            except Exception as e:
                print(f"Warning: Failed to load profiles.yaml: {e}")

        # Layer 4: Load settings.yaml if exists
        settings_path = config_dir / "settings.yaml"
        if settings_path.exists():
            try:
                with open(settings_path) as f:
                    user_settings = yaml.safe_load(f) or {}
                    # Override with user settings
                    for key, value in user_settings.items():
                        if key in cls.model_fields:
                            settings_dict[key] = value
            except Exception as e:
                print(f"Warning: Failed to load settings.yaml: {e}")

        # Layer 3: .env file + Layer 2: Environment variables
        # Pydantic Settings handles these automatically in __init__

        # Create base settings with layers 3-6 (defaults, .env, env vars)
        settings = cls(**settings_dict)

        # Layer 1: Apply CLI overrides (highest priority)
        if cli_args:
            # Filter out None values and non-field keys
            filtered_args = {
                k: v for k, v in cli_args.items() if v is not None and k in cls.model_fields
            }
            if filtered_args:
                settings = settings.model_copy(update=filtered_args)

        return settings


__all__ = ["VictorSettings"]
