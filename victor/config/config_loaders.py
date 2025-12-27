# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Configuration loaders for YAML-based settings.

This module provides loaders for externalized configuration:
- Provider context limits (provider_context_limits.yaml)
- Stage keywords (stage_keywords.yaml)

Design Principles:
- Hot-reload support via file watching
- Caching with TTL for performance
- Fallback to defaults if files missing
- Type-safe dataclasses for configuration
"""

from __future__ import annotations

import fnmatch
import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Configuration file paths
CONFIG_DIR = Path(__file__).parent
PROVIDER_LIMITS_FILE = CONFIG_DIR / "provider_context_limits.yaml"
STAGE_KEYWORDS_FILE = CONFIG_DIR / "stage_keywords.yaml"

# Cache settings
_cache_ttl = 300  # 5 minutes
_cache_timestamps: Dict[str, float] = {}
_cache_data: Dict[str, Any] = {}


@dataclass
class ProviderLimits:
    """Context window and rate limits for a provider."""

    context_window: int = 128000
    response_reserve: int = 4096
    supports_extended_context: bool = False
    rate_limit_rpm: int = 60
    rate_limit_tpm: int = 100000

    @property
    def effective_context(self) -> int:
        """Get effective context window (minus response reserve)."""
        return self.context_window - self.response_reserve


@dataclass
class StageConfig:
    """Configuration for a conversation stage."""

    keywords: List[str] = field(default_factory=list)
    weight: float = 1.0
    min_score: int = 2
    tool_preferences: List[str] = field(default_factory=list)


def _load_yaml_cached(file_path: Path, cache_key: str) -> Optional[Dict[str, Any]]:
    """Load YAML file with caching.

    Args:
        file_path: Path to YAML file
        cache_key: Key for caching

    Returns:
        Parsed YAML data or None if file doesn't exist
    """
    now = time.time()

    # Check cache freshness
    if cache_key in _cache_data:
        cached_time = _cache_timestamps.get(cache_key, 0)
        if now - cached_time < _cache_ttl:
            return _cache_data[cache_key]

    if not file_path.exists():
        logger.debug(f"Config file not found: {file_path}")
        return None

    try:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        _cache_data[cache_key] = data
        _cache_timestamps[cache_key] = now
        return data

    except Exception as e:
        logger.warning(f"Failed to load {file_path}: {e}")
        return None


def invalidate_config_cache() -> None:
    """Invalidate all configuration caches (for hot-reload)."""
    _cache_data.clear()
    _cache_timestamps.clear()


def get_provider_limits(provider: str, model: Optional[str] = None) -> ProviderLimits:
    """Get context limits for a provider/model.

    Args:
        provider: Provider name (e.g., 'anthropic', 'ollama')
        model: Optional model name for model-specific overrides

    Returns:
        ProviderLimits with context window and rate limits
    """
    data = _load_yaml_cached(PROVIDER_LIMITS_FILE, "provider_limits")

    if not data:
        # Return defaults if no config
        return ProviderLimits()

    providers = data.get("providers", {})
    models = data.get("models", {})

    # Start with provider defaults
    provider_data = providers.get(provider, {})
    limits = ProviderLimits(
        context_window=provider_data.get("context_window", 128000),
        response_reserve=provider_data.get("response_reserve", 4096),
        supports_extended_context=provider_data.get("supports_extended_context", False),
        rate_limit_rpm=provider_data.get("rate_limit_rpm", 60),
        rate_limit_tpm=provider_data.get("rate_limit_tpm", 100000),
    )

    # Apply model-specific overrides
    if model:
        for pattern, overrides in models.items():
            if fnmatch.fnmatch(model, pattern):
                if "context_window" in overrides:
                    limits.context_window = overrides["context_window"]
                if "response_reserve" in overrides:
                    limits.response_reserve = overrides["response_reserve"]
                break

    return limits


def get_all_provider_limits() -> Dict[str, ProviderLimits]:
    """Get limits for all configured providers.

    Returns:
        Dictionary mapping provider names to their limits
    """
    data = _load_yaml_cached(PROVIDER_LIMITS_FILE, "provider_limits")

    if not data:
        return {}

    providers = data.get("providers", {})
    result = {}

    for name, config in providers.items():
        result[name] = ProviderLimits(
            context_window=config.get("context_window", 128000),
            response_reserve=config.get("response_reserve", 4096),
            supports_extended_context=config.get("supports_extended_context", False),
            rate_limit_rpm=config.get("rate_limit_rpm", 60),
            rate_limit_tpm=config.get("rate_limit_tpm", 100000),
        )

    return result


def get_stage_keywords() -> Dict[str, StageConfig]:
    """Get stage keywords configuration.

    Returns:
        Dictionary mapping stage names to their configuration
    """
    data = _load_yaml_cached(STAGE_KEYWORDS_FILE, "stage_keywords")

    if not data:
        # Return defaults if no config
        return _get_default_stage_keywords()

    stages = data.get("stages", {})
    tool_prefs = data.get("tool_preferences", {})

    result = {}
    for name, config in stages.items():
        result[name] = StageConfig(
            keywords=config.get("keywords", []),
            weight=config.get("weight", 1.0),
            min_score=config.get("min_score", 2),
            tool_preferences=tool_prefs.get(name, []),
        )

    return result


def get_stage_tool_preferences(stage: str) -> List[str]:
    """Get preferred tools for a conversation stage.

    Args:
        stage: Stage name (e.g., 'EXPLORING', 'IMPLEMENTING')

    Returns:
        List of preferred tool names
    """
    data = _load_yaml_cached(STAGE_KEYWORDS_FILE, "stage_keywords")

    if not data:
        return []

    tool_prefs = data.get("tool_preferences", {})
    return tool_prefs.get(stage, [])


def _get_default_stage_keywords() -> Dict[str, StageConfig]:
    """Get default stage keywords (fallback if YAML missing)."""
    return {
        "INITIAL": StageConfig(
            keywords=["help me", "i need", "can you", "please"],
            weight=1.0,
            min_score=2,
        ),
        "EXPLORING": StageConfig(
            keywords=["where", "find", "search", "look for", "locate"],
            weight=1.2,
            min_score=2,
        ),
        "ANALYZING": StageConfig(
            keywords=["why", "explain", "understand", "analyze", "debug"],
            weight=1.3,
            min_score=2,
        ),
        "IMPLEMENTING": StageConfig(
            keywords=["create", "write", "implement", "add", "fix", "update"],
            weight=1.5,
            min_score=2,
        ),
        "REVIEWING": StageConfig(
            keywords=["review", "test", "verify", "commit", "done"],
            weight=1.2,
            min_score=2,
        ),
    }
