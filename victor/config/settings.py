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
from victor.config.model_capabilities import _default_tool_calling_models


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

        # Predefined model size tiers for convenience
        TIER_PRESETS = {
            "tiny": {"base_threshold": 0.35, "base_max_tools": 5},  # 0.5B-3B
            "small": {"base_threshold": 0.25, "base_max_tools": 7},  # 7B-8B
            "medium": {"base_threshold": 0.20, "base_max_tools": 10},  # 13B-15B
            "large": {"base_threshold": 0.15, "base_max_tools": 12},  # 30B+
            "cloud": {"base_threshold": 0.18, "base_max_tools": 10},  # Claude/GPT
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
        env_file=".env",
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

    # Unified Embedding Model (Optimized for Memory + Cache Efficiency)
    # Using same model for tool selection AND codebase search provides:
    # - 40% memory reduction (120MB vs 200MB)
    # - Better OS page cache utilization (1 model file instead of 2)
    # - Improved CPU L2/L3 cache hit rates
    # - Simpler management (1 model to download/update)
    unified_embedding_model: str = "all-MiniLM-L12-v2"  # 120MB, 384-dim, ~8ms

    # Tool Selection Strategy
    use_semantic_tool_selection: bool = True  # Use embeddings instead of keywords (DEFAULT)
    embedding_provider: str = (
        "sentence-transformers"  # sentence-transformers (local), ollama, vllm, lmstudio
    )
    embedding_model: str = unified_embedding_model  # Shared with codebase search

    # Codebase Semantic Search (Air-gapped by Default)
    codebase_vector_store: str = "lancedb"  # lancedb (recommended), chromadb, proximadb
    codebase_embedding_provider: str = "sentence-transformers"  # Local, offline, fast
    codebase_embedding_model: str = unified_embedding_model  # Shared with tool selection
    codebase_persist_directory: Optional[str] = None  # Default: ~/.victor/embeddings/codebase
    codebase_dimension: int = 384  # Embedding dimension
    codebase_batch_size: int = 32  # Batch size for embedding generation

    # UI
    theme: str = "monokai"
    show_token_count: bool = True
    stream_responses: bool = True

    # MCP
    use_mcp_tools: bool = False
    mcp_command: Optional[str] = None  # e.g., "python mcp_server.py" or "node mcp-server.js"
    mcp_prefix: str = "mcp"

    # Tool Execution Settings
    tool_call_budget: int = (
        300  # Maximum tool calls per session (increased from 20 for long operations)
    )
    tool_call_budget_warning_threshold: int = 250  # Warn when approaching budget limit

    # Models known to support structured tool calls per provider
    tool_calling_models: Dict[str, list[str]] = Field(default_factory=_default_tool_calling_models)

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
    tool_cache_dir: str = "~/.victor/cache"
    tool_cache_allowlist: List[str] = [
        "code_search",
        "semantic_code_search",
        "list_directory",
        "plan_files",
    ]

    # Plugin System
    plugin_enabled: bool = True  # Enable plugin system
    plugin_dirs: List[str] = ["~/.victor/plugins"]  # Directories to search for plugins
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

    # Analytics
    analytics_enabled: bool = True
    analytics_log_file: str = "~/.victor/logs/usage.jsonl"

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
            Path to config directory
        """
        config_dir = Path.home() / ".victor"
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

        # Override with environment variables and default settings
        if provider == "anthropic":
            settings["api_key"] = self.anthropic_api_key or settings.get("api_key")
            settings.setdefault("base_url", "https://api.anthropic.com")

        elif provider == "openai":
            settings["api_key"] = self.openai_api_key or settings.get("api_key")
            settings.setdefault("base_url", "https://api.openai.com/v1")

        elif provider == "google":
            settings["api_key"] = self.google_api_key or settings.get("api_key")

        elif provider == "ollama":
            settings.setdefault("base_url", self.ollama_base_url)

        elif provider == "lmstudio":
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
            settings.setdefault("base_url", self.vllm_base_url)

        return settings


def load_settings() -> Settings:
    """Load application settings.

    Returns:
        Settings instance
    """
    return Settings()
