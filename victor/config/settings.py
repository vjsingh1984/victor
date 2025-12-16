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

"""Configuration management for CodingAgent."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union, List

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from victor.config.model_capabilities import _load_tool_capable_patterns_from_yaml
from victor.config.orchestrator_constants import BUDGET_LIMITS, TOOL_SELECTION_PRESETS


# =============================================================================
# CENTRALIZED PATH CONFIGURATION
# =============================================================================
# All Victor paths are centralized here for consistency and easy configuration.
# Project-local paths are stored in {project_root}/.victor/
# Global paths are stored in ~/.victor/
#
# SECURITY: Uses secure_paths module to protect against:
# - HOME environment variable manipulation (SEC-001)
# - Path traversal attacks via VICTOR_DIR_NAME (SEC-002)
# - Symlink attacks (SEC-003)
# =============================================================================

# Context file name (configurable)
VICTOR_CONTEXT_FILE = os.getenv("VICTOR_CONTEXT_FILE", "init.md")


def _get_secure_victor_dir_name() -> str:
    """Get validated VICTOR_DIR_NAME with path traversal protection.

    Security: Blocks path traversal attempts like '../../../etc' or '/tmp/evil'
    """
    try:
        from victor.config.secure_paths import validate_victor_dir_name

        raw_name = os.getenv("VICTOR_DIR_NAME", ".victor")
        validated_name, is_valid = validate_victor_dir_name(raw_name)
        return validated_name
    except ImportError:
        # Fallback if secure_paths not available yet (during initial import)
        return os.getenv("VICTOR_DIR_NAME", ".victor")


def _get_secure_global_victor_dir() -> Path:
    """Get global Victor directory with secure home resolution.

    Security: Validates HOME against passwd database to detect manipulation.
    """
    try:
        from victor.config.secure_paths import get_victor_dir

        return get_victor_dir()
    except ImportError:
        # Fallback if secure_paths not available yet
        return Path.home() / _get_secure_victor_dir_name()


# Directory name with validation (lazy property to avoid circular imports)
VICTOR_DIR_NAME = _get_secure_victor_dir_name()

# Global config directory with secure home resolution
GLOBAL_VICTOR_DIR = _get_secure_global_victor_dir()


class ProjectPaths:
    """Centralized path management for Victor.

    Provides consistent paths for both project-local and global storage.
    Project-local paths are preferred for isolation between projects.

    Directory structure:
        {project_root}/.victor/
        ├── init.md              # Project context (was .victor.md)
        ├── conversation.db      # Conversation history
        ├── embeddings/          # Vector embeddings for semantic search
        ├── index_metadata.json  # Codebase index metadata
        ├── backups/             # File edit backups
        ├── changes/             # Undo/redo history
        ├── sessions/            # Session snapshots
        └── mcp.yaml             # MCP server configuration

        ~/.victor/
        ├── profiles.yaml        # Global profiles configuration
        ├── plugins/             # Plugins directory
        ├── cache/               # Global cache
        ├── logs/                # Log files
        └── embeddings/          # Global embedding cache (task classifier, etc.)
    """

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize paths for a project.

        Args:
            project_root: Project root directory. Defaults to current working directory.
        """
        self._project_root = Path(project_root) if project_root else Path.cwd()

    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return self._project_root

    # -------------------------------------------------------------------------
    # Project-local paths (stored in {project}/.victor/)
    # -------------------------------------------------------------------------

    @property
    def project_victor_dir(self) -> Path:
        """Get project-local .victor directory."""
        return self._project_root / VICTOR_DIR_NAME

    @property
    def project_context_file(self) -> Path:
        """Get project context file path (.victor/init.md)."""
        return self.project_victor_dir / VICTOR_CONTEXT_FILE

    @property
    def conversation_db(self) -> Path:
        """Get project-local conversation database path."""
        return self.project_victor_dir / "conversation.db"

    @property
    def embeddings_dir(self) -> Path:
        """Get project-local embeddings directory."""
        return self.project_victor_dir / "embeddings"

    @property
    def graph_dir(self) -> Path:
        """Get project-local graph directory."""
        return self.project_victor_dir / "graph"

    @property
    def index_metadata(self) -> Path:
        """Get codebase index metadata file path."""
        return self.project_victor_dir / "index_metadata.json"

    @property
    def backups_dir(self) -> Path:
        """Get project-local backups directory."""
        return self.project_victor_dir / "backups"

    @property
    def changes_dir(self) -> Path:
        """Get project-local changes (undo/redo) directory."""
        return self.project_victor_dir / "changes"

    @property
    def sessions_dir(self) -> Path:
        """Get project-local sessions directory."""
        return self.project_victor_dir / "sessions"

    @property
    def conversations_export_dir(self) -> Path:
        """Get project-local conversations export directory."""
        return self.project_victor_dir / "conversations"

    @property
    def mcp_config(self) -> Path:
        """Get project-local MCP configuration file."""
        return self.project_victor_dir / "mcp.yaml"

    # -------------------------------------------------------------------------
    # Global paths (stored in ~/.victor/)
    # -------------------------------------------------------------------------

    @property
    def global_victor_dir(self) -> Path:
        """Get global .victor directory."""
        return GLOBAL_VICTOR_DIR

    @property
    def global_profiles(self) -> Path:
        """Get global profiles.yaml path."""
        return GLOBAL_VICTOR_DIR / "profiles.yaml"

    @property
    def global_plugins_dir(self) -> Path:
        """Get global plugins directory."""
        return GLOBAL_VICTOR_DIR / "plugins"

    @property
    def global_cache_dir(self) -> Path:
        """Get global cache directory."""
        return GLOBAL_VICTOR_DIR / "cache"

    @property
    def global_logs_dir(self) -> Path:
        """Get global logs directory."""
        return GLOBAL_VICTOR_DIR / "logs"

    @property
    def global_embeddings_dir(self) -> Path:
        """Get global embeddings cache directory (for task classifier, etc.)."""
        return GLOBAL_VICTOR_DIR / "embeddings"

    @property
    def global_mcp_config(self) -> Path:
        """Get global MCP configuration file."""
        return GLOBAL_VICTOR_DIR / "mcp.yaml"

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def ensure_project_dirs(self) -> None:
        """Create project-local directories if they don't exist."""
        self.project_victor_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        self.backups_dir.mkdir(parents=True, exist_ok=True)
        self.changes_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def ensure_global_dirs(self) -> None:
        """Create global directories if they don't exist."""
        self.global_victor_dir.mkdir(parents=True, exist_ok=True)
        self.global_plugins_dir.mkdir(parents=True, exist_ok=True)
        self.global_cache_dir.mkdir(parents=True, exist_ok=True)
        self.global_logs_dir.mkdir(parents=True, exist_ok=True)
        self.global_embeddings_dir.mkdir(parents=True, exist_ok=True)

    def find_context_file(self) -> Optional[Path]:
        """Find project context file at .victor/init.md.

        Returns:
            Path to context file if found, None otherwise.

        Location: .victor/init.md (configurable via VICTOR_DIR_NAME, VICTOR_CONTEXT_FILE)
        """
        if self.project_context_file.exists():
            return self.project_context_file

        return None


# Global singleton for current project
_current_project_paths: Optional[ProjectPaths] = None


def get_project_paths(project_root: Optional[Path] = None) -> ProjectPaths:
    """Get ProjectPaths instance for a project.

    Args:
        project_root: Project root directory. If None, uses cached instance or cwd.

    Returns:
        ProjectPaths instance for the project.
    """
    global _current_project_paths

    if project_root is not None:
        return ProjectPaths(project_root)

    if _current_project_paths is None:
        _current_project_paths = ProjectPaths()

    return _current_project_paths


def set_project_root(project_root: Path) -> ProjectPaths:
    """Set the current project root and return paths.

    Args:
        project_root: Project root directory.

    Returns:
        ProjectPaths instance for the project.
    """
    global _current_project_paths
    _current_project_paths = ProjectPaths(project_root)
    return _current_project_paths


def reset_project_paths() -> None:
    """Reset the cached project paths singleton.

    This is primarily useful for testing when you need to switch project roots.
    """
    global _current_project_paths
    _current_project_paths = None


class ProviderConfig(BaseSettings):
    """Configuration for a specific provider."""

    api_key: Optional[str] = None
    base_url: Optional[Union[str, List[str]]] = None
    timeout: int = 300  # 5 minutes - increased for CPU-only inference
    max_retries: int = 3
    organization: Optional[str] = None  # For OpenAI


class ProfileConfig(BaseSettings):
    """Configuration for a model profile."""

    model_config = SettingsConfigDict(extra="allow")

    provider: str = Field(..., description="Provider name (ollama, anthropic, openai, google)")
    model: str = Field(..., description="Model identifier")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(4096, gt=0)
    description: Optional[str] = Field(None, description="Optional profile description")
    tool_selection: Optional[Dict[str, Any]] = Field(
        None, description="Tool selection configuration for adaptive thresholds"
    )

    @field_validator("tool_selection")
    @classmethod
    def validate_tool_selection(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate tool_selection configuration.

        Args:
            v: Tool selection configuration dictionary

        Returns:
            Validated configuration with expanded tier shortcuts

        Raises:
            ValueError: If configuration is invalid
        """
        if v is None:
            return None

        # Predefined model size tiers for convenience (from orchestrator_constants)
        TIER_PRESETS = {
            "tiny": TOOL_SELECTION_PRESETS.tiny,  # 0.5B-3B
            "small": TOOL_SELECTION_PRESETS.small,  # 7B-8B
            "medium": TOOL_SELECTION_PRESETS.medium,  # 13B-15B
            "large": TOOL_SELECTION_PRESETS.large,  # 30B+
            "cloud": TOOL_SELECTION_PRESETS.cloud,  # Claude/GPT
        }

        # Expand tier shortcuts
        if "model_size_tier" in v:
            tier = v["model_size_tier"]
            if tier in TIER_PRESETS:
                # Apply preset values, but allow manual overrides
                preset = TIER_PRESETS[tier].copy()
                preset.update(v)  # Manual values override preset
                v = preset

        # Validate base_threshold
        if "base_threshold" in v:
            threshold = v["base_threshold"]
            if not isinstance(threshold, (int, float)):
                raise ValueError(f"base_threshold must be a number, got {type(threshold)}")
            if not (0.0 <= threshold <= 1.0):
                raise ValueError(f"base_threshold must be between 0.0 and 1.0, got {threshold}")

        # Validate base_max_tools
        if "base_max_tools" in v:
            max_tools = v["base_max_tools"]
            if not isinstance(max_tools, int):
                raise ValueError(f"base_max_tools must be an integer, got {type(max_tools)}")
            if max_tools < 1:
                raise ValueError(f"base_max_tools must be positive, got {max_tools}")

        return v


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env" if not os.getenv("VICTOR_SKIP_ENV_FILE") else None,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
    )

    # Default provider settings (LMStudio by default for local observability)
    default_provider: str = "ollama"
    default_model: str = "qwen3-coder:30b"
    default_temperature: float = 0.7
    default_max_tokens: int = 4096

    # API Keys
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    moonshot_api_key: Optional[str] = None  # Moonshot AI for Kimi K2 models
    deepseek_api_key: Optional[str] = None  # DeepSeek for DeepSeek-V3 models

    # Local server URLs
    # Can be overridden via environment variables:
    #   OLLAMA_BASE_URL, LMSTUDIO_BASE_URLS (comma-separated), VLLM_BASE_URL
    # For LAN servers, set: LMSTUDIO_BASE_URLS="http://<your-server>:1234,http://localhost:1234"
    ollama_base_url: str = "http://localhost:11434"
    # LMStudio tiered endpoints (try in order) - defaults to localhost only
    # Set LMSTUDIO_BASE_URLS env var to add LAN servers
    lmstudio_base_urls: List[str] = [
        "http://127.0.0.1:1234",
    ]
    vllm_base_url: str = "http://localhost:8000"

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # Privacy and Security
    airgapped_mode: bool = False

    # ==========================================================================
    # Vertical Configuration
    # ==========================================================================
    # Verticals are domain-specific configurations that customize Victor's behavior.
    # Available verticals: coding, research, devops (extensible via plugins)
    default_vertical: str = "coding"  # Default vertical when --vertical not specified
    auto_detect_vertical: bool = False  # Auto-detect vertical from project context (experimental)

    # Server Security (FastAPI/WebSocket layer)
    # When set, API key is required for HTTP + WebSocket requests (Authorization: Bearer <token>)
    server_api_key: Optional[str] = None
    # HMAC secret for issuing/verifying session tokens (defaults to random per-process secret)
    server_session_secret: Optional[str] = None
    # Hard cap on simultaneous sessions to avoid resource exhaustion
    server_max_sessions: int = 100
    # Maximum inbound message payload size (bytes) for WebSocket messages
    server_max_message_bytes: int = 32768
    # Session token time-to-live in seconds
    server_session_ttl_seconds: int = 86400
    # Diagram rendering limits
    render_max_payload_bytes: int = 20000
    render_timeout_seconds: int = 10
    render_max_concurrency: int = 2

    # Code execution sandbox defaults (used by code_executor_tool)
    code_executor_network_disabled: bool = True
    code_executor_memory_limit: Optional[str] = "512m"
    code_executor_cpu_shares: Optional[int] = 256

    # Write Approval Mode (safety for autonomous/task mode)
    # Controls when user confirmation is required for file modifications:
    #   - "off": Never require approval (dangerous, testing only)
    #   - "risky_only": Only for HIGH/CRITICAL risk operations (default)
    #   - "all_writes": Require for ALL write operations (recommended for task mode)
    write_approval_mode: str = "risky_only"

    # Headless Mode Settings (for CI/CD and automation)
    # These can be set via CLI flags or environment variables:
    #   - VICTOR_HEADLESS_MODE=true
    #   - VICTOR_DRY_RUN_MODE=true
    #   - VICTOR_MAX_FILE_CHANGES=10
    headless_mode: bool = False  # Run without prompts, auto-approve safe actions
    dry_run_mode: bool = False  # Preview changes without applying them
    auto_approve_safe: bool = False  # Auto-approve read-only and LOW risk operations
    max_file_changes: Optional[int] = None  # Limit file modifications per session
    one_shot_mode: bool = False  # Exit after completing a single request

    # Unified Embedding Model (Optimized for Memory + Cache Efficiency)
    # Using same model for tool selection AND codebase search provides:
    # - 40% memory reduction (130MB vs 200MB)
    # - Better OS page cache utilization (1 model file instead of 2)
    # - Improved CPU L2/L3 cache hit rates
    # - Simpler management (1 model to download/update)
    #
    # Model: BAAI/bge-small-en-v1.5 (130MB, 384-dim, ~6ms)
    # - MTEB score: 62.2 (vs 58.8 for all-MiniLM-L6-v2)
    # - Excellent for code search (trained on code-related tasks)
    # - CPU-optimized, works great on consumer-grade hardware
    # - Native sentence-transformers support (no API needed)
    unified_embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Tool Selection Strategy
    use_semantic_tool_selection: bool = True  # Use embeddings instead of keywords (DEFAULT)
    embedding_provider: str = (
        "sentence-transformers"  # sentence-transformers (local), ollama, vllm, lmstudio
    )
    embedding_model: str = unified_embedding_model  # Shared with codebase search

    # Codebase Semantic Search (Air-gapped by Default)
    codebase_vector_store: str = "lancedb"  # lancedb (recommended), chromadb
    codebase_embedding_provider: str = "sentence-transformers"  # Local, offline, fast
    codebase_embedding_model: str = unified_embedding_model  # Shared with tool selection
    codebase_persist_directory: Optional[str] = None  # Default: ~/.victor/embeddings/codebase
    codebase_dimension: int = 384  # Embedding dimension
    codebase_batch_size: int = 32  # Batch size for embedding generation
    codebase_graph_store: str = "sqlite"  # Graph backend (sqlite default)
    codebase_graph_path: Optional[str] = None  # Optional explicit graph db path
    core_readonly_tools: Optional[List[str]] = None  # Override/extend curated read-only tool set

    # UI
    theme: str = "monokai"
    show_token_count: bool = True
    stream_responses: bool = True

    # Interaction Mode
    # When True (one-shot mode), auto-continue when model asks for user input
    # When False (interactive mode), return to user for choice
    one_shot_mode: bool = False

    # MCP
    use_mcp_tools: bool = False
    mcp_command: Optional[str] = None  # e.g., "python mcp_server.py" or "node mcp-server.js"
    mcp_prefix: str = "mcp"

    # Tool Execution Settings
    tool_call_budget: int = BUDGET_LIMITS.max_session_budget  # Maximum tool calls per session
    tool_call_budget_warning_threshold: int = int(
        BUDGET_LIMITS.max_session_budget * BUDGET_LIMITS.warning_threshold_pct
    )  # Warn when approaching budget limit

    # Models known to support structured tool calls per provider
    # Loaded from model_capabilities.yaml, can be extended in profiles.yaml
    tool_calling_models: Dict[str, list[str]] = Field(
        default_factory=_load_tool_capable_patterns_from_yaml
    )

    # Tool Retry Settings
    tool_retry_enabled: bool = True  # Enable automatic retry for failed tool executions
    tool_retry_max_attempts: int = 3  # Maximum retry attempts per tool call
    tool_retry_base_delay: float = 1.0  # Base delay in seconds for exponential backoff
    tool_retry_max_delay: float = 10.0  # Maximum delay in seconds between retries

    # Tool selection fallback
    fallback_max_tools: int = 8  # Cap tool list when stage pruning removes everything

    # Tool result caching (opt-in per tool)
    tool_cache_enabled: bool = True
    tool_cache_ttl: int = 600  # seconds
    # Note: tool_cache_dir now uses get_project_paths().global_cache_dir
    tool_cache_allowlist: List[str] = [
        "code_search",
        "semantic_code_search",
        "list_directory",
        "plan_files",
    ]

    # Tool Argument Validation
    # Controls pre-execution JSON Schema validation of tool arguments
    # Options: "strict" (block on errors), "lenient" (warn only), "off" (disable)
    tool_validation_mode: str = "lenient"

    # Context Compaction Settings
    # Controls how conversation history is managed when context grows too large
    # Options: "simple" (keep N recent), "tiered" (prioritize tool results),
    #          "semantic" (use embeddings), "hybrid" (combine tiered + semantic)
    context_compaction_strategy: str = "tiered"
    context_min_messages_to_keep: int = 6  # Minimum messages to retain after compaction
    context_tool_retention_weight: float = 1.5  # Boost for tool result retention
    context_recency_weight: float = 2.0  # Boost for recent messages
    context_semantic_threshold: float = 0.3  # Min similarity for semantic retention

    # Plugin System
    plugin_enabled: bool = True  # Enable plugin system
    # Note: plugin_dirs now uses get_project_paths().global_plugins_dir
    plugin_packages: List[str] = []  # Python packages to load as plugins
    plugin_disabled: List[str] = []  # List of plugin names to disable
    plugin_config: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Plugin-specific configuration (plugin_name -> config dict)",
    )

    # Security scan extensions
    security_dependency_scan: bool = False
    security_iac_scan: bool = False

    # LMStudio resource guard
    lmstudio_max_vram_gb: Optional[float] = (
        48.0  # Cap model selection to this budget (GB); override via env/config
    )

    # Exploration Loop Settings (prevents endless exploration without output)
    # Higher values = more thorough exploration, slower responses
    max_exploration_iterations: int = 8  # Max consecutive read-only tool calls with minimal output
    max_exploration_iterations_action: int = (
        12  # More lenient for action tasks (create, write, etc.)
    )
    max_exploration_iterations_analysis: int = (
        50  # Very lenient for analysis tasks (uses loop detection instead)
    )
    min_content_threshold: int = 150  # Minimum chars to consider "substantial" output
    max_research_iterations: int = 6  # Force synthesis after N consecutive web searches

    # ==========================================================================
    # Recovery & Loop Detection Thresholds
    # ==========================================================================
    # These control when Victor forces completion after detecting stuck behavior.
    # Lower values = faster recovery but may cut off legitimate long operations.
    # Higher values = more patience but may waste tokens on stuck loops.

    # Empty response recovery: Force after N consecutive empty responses from model
    recovery_empty_response_threshold: int = (
        5  # Default: force after 5 empty responses (3 * 1.5 = 4.5 → 5)
    )

    # Loop detection patience: How many consecutive blocked attempts before forcing completion
    # This is separate from the per-task loop_repeat_threshold (which controls when to warn/block)
    recovery_blocked_consecutive_threshold: int = (
        6  # Default: force after 6 consecutive blocks (4 * 1.5 = 6)
    )
    recovery_blocked_total_threshold: int = (
        9  # Default: force after 9 total blocked attempts (6 * 1.5 = 9)
    )

    # Continuation prompts: How many times to prompt model to continue before forcing
    max_continuation_prompts_analysis: int = 6  # For analysis tasks (4 * 1.5 = 6)
    max_continuation_prompts_action: int = 5  # For action tasks (3 * 1.5 = 4.5 → 5)
    max_continuation_prompts_default: int = 3  # For other tasks (2 * 1.5 = 3)

    # Session time limit: Maximum seconds before forcing completion
    # Set below provider timeout (300s default) to provide graceful completion
    session_time_limit: int = 240  # 4 minutes default, leaves 60s buffer for summary

    # ==========================================================================
    # Conversation Memory (Multi-turn Context Retention)
    # ==========================================================================
    conversation_memory_enabled: bool = True  # Enable SQLite-backed conversation persistence
    conversation_embeddings_enabled: bool = True  # Enable LanceDB embeddings for semantic retrieval
    # Note: conversation_db now uses get_project_paths().conversation_db (project-local)
    # Embeddings stored at get_project_paths().embeddings_dir / "conversations"
    max_context_tokens: int = 100000  # Maximum tokens in context window
    response_token_reserve: int = 4096  # Tokens reserved for model response

    # ==========================================================================
    # Provider Resilience (Circuit Breaker, Retry, Fallback)
    # ==========================================================================
    resilience_enabled: bool = True  # Enable circuit breaker and retry logic

    # Circuit Breaker Settings
    circuit_breaker_failure_threshold: int = 5  # Failures before circuit opens
    circuit_breaker_success_threshold: int = 2  # Successes before circuit closes
    circuit_breaker_timeout: float = 60.0  # Seconds before half-open state
    circuit_breaker_half_open_max: int = 3  # Max requests in half-open state

    # Retry Settings
    retry_max_attempts: int = 3  # Maximum retry attempts
    retry_base_delay: float = 1.0  # Base delay in seconds
    retry_max_delay: float = 60.0  # Maximum delay between retries
    retry_exponential_base: float = 2.0  # Exponential backoff multiplier

    # ==========================================================================
    # Rate Limiting (Request Throttling)
    # ==========================================================================
    rate_limiting_enabled: bool = True  # Enable rate limiting
    rate_limit_requests_per_minute: int = 50  # Requests per minute limit
    rate_limit_tokens_per_minute: int = 50000  # Tokens per minute limit
    rate_limit_max_concurrent: int = 5  # Maximum concurrent requests
    rate_limit_queue_size: int = 100  # Maximum pending requests in queue
    rate_limit_num_workers: int = 3  # Number of queue worker tasks

    # ==========================================================================
    # Streaming Metrics (Performance Monitoring)
    # ==========================================================================
    streaming_metrics_enabled: bool = True  # Enable streaming performance metrics
    streaming_metrics_history_size: int = 1000  # Number of metrics samples to retain

    # ==========================================================================
    # Serialization (Token Optimization for Tool Output)
    # ==========================================================================
    # Controls how tool outputs are serialized for token efficiency.
    # Formats: json, json_minified, toon, csv, markdown_table, reference_encoded
    # TOON (Token-Oriented Object Notation) provides 30-60% savings for tabular data.
    serialization_enabled: bool = True  # Enable token-optimized serialization
    serialization_default_format: Optional[str] = None  # None = auto-select best format
    serialization_min_savings_threshold: float = 0.15  # Min savings to use alternative format

    # ==========================================================================
    # Intelligent Agent Pipeline (RL-based Learning, Quality Scoring)
    # ==========================================================================
    # Controls the intelligent agent features including:
    # - Q-learning based mode transitions (explore -> plan -> build -> review)
    # - Response quality scoring (coherence, completeness, relevance, grounding)
    # - Provider resilience integration (circuit breaker, retries)
    # - Embedding-based prompt optimization
    intelligent_pipeline_enabled: bool = True  # Master switch for intelligent features
    intelligent_quality_scoring: bool = True  # Enable multi-dimensional quality scoring
    intelligent_mode_learning: bool = True  # Enable Q-learning for mode transitions
    intelligent_prompt_optimization: bool = True  # Enable embedding-based prompt selection
    intelligent_grounding_verification: bool = True  # Enable hallucination detection

    # Quality thresholds
    intelligent_min_quality_threshold: float = 0.5  # Minimum quality to accept response
    intelligent_grounding_threshold: float = 0.7  # Confidence threshold for grounding

    # Learning rate for Q-learning (default exploration rate = 0.3, decay = 0.995)
    intelligent_exploration_rate: float = 0.3  # Initial exploration vs exploitation
    intelligent_learning_rate: float = 0.1  # Q-learning alpha parameter
    intelligent_discount_factor: float = 0.9  # Q-learning gamma parameter
    serialization_include_format_hint: bool = True  # Include format description in output
    serialization_min_rows_for_tabular: int = 3  # Min rows to consider tabular formats
    serialization_debug_mode: bool = False  # Include data characteristics in output

    # Analytics
    analytics_enabled: bool = True
    # Note: analytics_log_file now uses get_project_paths().global_logs_dir / "usage.jsonl"

    @staticmethod
    def _estimate_model_vram_gb(model_id: str) -> Optional[float]:
        """Rough VRAM requirement (GB) by model name heuristic."""
        requirements = [
            ("70b", 64.0),
            ("65b", 60.0),
            ("33b", 40.0),
            ("32b", 36.0),
            ("30b", 34.0),
            ("34b", 34.0),
            ("14b", 18.0),
            ("13b", 16.0),
            ("12b", 14.0),
            ("8x7b", 40.0),  # Mixture
            ("8b", 10.0),
            ("7b", 8.0),
            ("6.7b", 8.0),
            ("3b", 6.0),
            ("1.5b", 4.0),
        ]
        m = model_id.lower()
        for key, vram in requirements:
            if key in m:
                return vram
        return None

    @staticmethod
    def _detect_vram_gb() -> Optional[float]:
        """Detect available GPU VRAM in GB (best-effort)."""
        try:
            import subprocess

            # Try NVIDIA GPUs
            cmd = [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ]
            output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
            values = [float(x.strip()) for x in output.splitlines() if x.strip()]
            if values:
                return max(values) / 1024.0
        except Exception:
            pass

        # macOS: try system_profiler for VRAM
        try:
            import subprocess

            output = subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            for line in output.splitlines():
                if "VRAM" in line and "Total" in line:
                    # e.g., "      Total VRAM (Dynamic, Max): 4096 MB"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if (
                            part.replace(",", "").isdigit()
                            and i + 1 < len(parts)
                            and parts[i + 1].upper().startswith("MB")
                        ):
                            return float(part.replace(",", "")) / 1024.0
        except Exception:
            pass

        return None

    @classmethod
    def _choose_default_lmstudio_model(
        cls, urls: list[str], max_vram_gb: Optional[float] = None
    ) -> str:
        """Pick a sane default model from reachable LMStudio servers.

        Preference order favors small coder models for latency, then falls
        back to the first advertised model from the first reachable server.
        """
        preferred_models = [
            "qwen2.5-coder:7b",
            "qwen2.5-coder:14b",
            "qwen2.5-coder:32b",
            "qwen2.5:7b",
            "llama-3.1-8b-instruct",
        ]

        detected_vram = cls._detect_vram_gb()
        max_vram = (
            max_vram_gb if max_vram_gb is not None else getattr(cls, "lmstudio_max_vram_gb", None)
        )
        available_vram = None
        if detected_vram and max_vram:
            available_vram = min(detected_vram, max_vram)
        else:
            available_vram = detected_vram or max_vram

        try:
            import httpx  # Local network call only; safe in airgapped mode
        except Exception:
            return preferred_models[0]

        for url in urls:
            try:
                resp = httpx.get(f"{url}/v1/models", timeout=1.0)
                if resp.status_code != 200:
                    continue
                data = resp.json() or {}
                models = [
                    m.get("id") or m.get("model")
                    for m in data.get("data", [])
                    if isinstance(m, dict)
                ]
                if not models:
                    continue

                # If VRAM is known, choose the most capable coder/instruct model that fits
                if available_vram:
                    candidates = []
                    for m_id in models:
                        if not m_id:
                            continue
                        requirement = cls._estimate_model_vram_gb(m_id or "")
                        if requirement and requirement <= available_vram:
                            # Prefer coder/ instruct models
                            is_coder = "coder" in m_id.lower()
                            candidates.append((requirement, is_coder, m_id))
                    if candidates:
                        # Choose largest VRAM within budget; coder preferred
                        candidates.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
                        return str(candidates[0][2])

                for pref in preferred_models:
                    if pref in models:
                        return pref
                return str(models[0])
            except Exception:
                continue

        return str(preferred_models[0])

    @classmethod
    def get_config_dir(cls) -> Path:
        """Get configuration directory path.

        Returns:
            Path to global config directory (~/.victor)
        """
        config_dir = GLOBAL_VICTOR_DIR
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir

    @classmethod
    def load_profiles(cls) -> Dict[str, ProfileConfig]:
        """Load profiles from YAML file.

        Returns:
            Dictionary of profile configurations
        """
        profiles_file = cls.get_config_dir() / "profiles.yaml"

        if not profiles_file.exists():
            urls = getattr(cls, "lmstudio_base_urls", []) or [
                "http://localhost:1234",
            ]
            default_model = cls._choose_default_lmstudio_model(
                urls, max_vram_gb=cls().lmstudio_max_vram_gb
            )
            # Return default profiles
            return {
                "default": ProfileConfig(
                    provider="lmstudio",
                    model=default_model,
                    temperature=0.7,
                    max_tokens=4096,
                    description=None,
                    tool_selection=None,
                )
            }

        try:
            with open(profiles_file, "r") as f:
                data = yaml.safe_load(f)

            profiles = {}
            for name, config in data.get("profiles", {}).items():
                profiles[name] = ProfileConfig(**config)

            return profiles

        except Exception as e:
            print(f"Warning: Failed to load profiles: {e}")
            return {}

    @classmethod
    def load_provider_config(cls, provider: str) -> Optional[ProviderConfig]:
        """Load provider-specific configuration.

        Args:
            provider: Provider name

        Returns:
            ProviderConfig or None
        """
        profiles_file = cls.get_config_dir() / "profiles.yaml"

        if not profiles_file.exists():
            return None

        try:
            with open(profiles_file, "r") as f:
                data = yaml.safe_load(f)

            provider_data = data.get("providers", {}).get(provider, {})
            if provider_data:
                # Expand environment variables
                for key, value in provider_data.items():
                    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                        env_var = value[2:-1]
                        provider_data[key] = os.getenv(env_var)

                return ProviderConfig(**provider_data)

        except Exception as e:
            print(f"Warning: Failed to load provider config for {provider}: {e}")

        return None

    @classmethod
    def load_tool_config(cls) -> Dict[str, Any]:
        """Load tool-specific configuration from profiles.yaml (top-level 'tools' key)."""
        profiles_file = cls.get_config_dir() / "profiles.yaml"
        if not profiles_file.exists():
            return {}
        try:
            with open(profiles_file, "r") as f:
                data = yaml.safe_load(f) or {}
            return data.get("tools", {}) or {}
        except Exception as e:
            print(f"Warning: Failed to load tool config: {e}")
            return {}

    def get_provider_settings(self, provider: str) -> Dict[str, Any]:
        """Get settings for a specific provider.

        Args:
            provider: Provider name

        Returns:
            Dictionary of provider settings
        """
        settings = {}

        # Load from profiles.yaml
        provider_config = self.load_provider_config(provider)
        if provider_config:
            settings.update(provider_config.model_dump(exclude_none=True))

        # Use secure API key manager for provider isolation
        # Only loads the specific provider's key, not all keys
        from victor.config.api_keys import get_api_key

        # Set provider-specific defaults and load API key
        # Check Settings instance attributes first, then fallback to API key manager
        if provider == "anthropic":
            api_key = self.anthropic_api_key or get_api_key("anthropic")
            if api_key:
                settings["api_key"] = api_key
            settings.setdefault("base_url", "https://api.anthropic.com")

        elif provider == "openai":
            api_key = self.openai_api_key or get_api_key("openai")
            if api_key:
                settings["api_key"] = api_key
            settings.setdefault("base_url", "https://api.openai.com/v1")

        elif provider == "google":
            api_key = self.google_api_key or get_api_key("google")
            if api_key:
                settings["api_key"] = api_key

        elif provider == "xai":
            # xai_api_key might not exist in Settings, use getattr with default
            api_key = getattr(self, "xai_api_key", None) or get_api_key("xai")
            if api_key:
                settings["api_key"] = api_key
            settings.setdefault("base_url", "https://api.x.ai/v1")

        elif provider == "ollama":
            # Local provider - no API key needed
            settings.setdefault("base_url", self.ollama_base_url)

        elif provider == "lmstudio":
            # Local provider - no API key needed
            urls = getattr(self, "lmstudio_base_urls", []) or []
            # If provider config supplied a list, merge/override
            if "base_url" in settings:
                cfg_url = settings["base_url"]
                if isinstance(cfg_url, list):
                    urls = cfg_url
                elif isinstance(cfg_url, str):
                    urls = [cfg_url]
            chosen = None
            try:
                import httpx

                for url in urls:
                    try:
                        resp = httpx.get(f"{url}/v1/models", timeout=1.5)
                        if resp.status_code == 200:
                            chosen = url
                            break
                    except Exception:
                        continue
            except Exception:
                pass
            settings["base_url"] = f"{(chosen or urls[0]).rstrip('/')}/v1"

        elif provider == "vllm":
            # Local provider - no API key needed
            settings.setdefault("base_url", self.vllm_base_url)

        elif provider in ("moonshot", "kimi"):
            api_key = get_api_key("moonshot")
            if api_key:
                settings["api_key"] = api_key
            settings.setdefault("base_url", "https://api.moonshot.cn/v1")

        elif provider == "deepseek":
            api_key = get_api_key("deepseek")
            if api_key:
                settings["api_key"] = api_key
            settings.setdefault("base_url", "https://api.deepseek.com/v1")

        elif provider == "groqcloud":
            api_key = get_api_key("groqcloud")
            if api_key:
                settings["api_key"] = api_key
            settings.setdefault("base_url", "https://api.groq.com/openai/v1")

        return settings


def load_settings() -> Settings:
    """Load application settings.

    Returns:
        Settings instance
    """
    return Settings()
