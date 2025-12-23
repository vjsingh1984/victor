"""Model capability helpers for tool-calling support across providers.

SINGLE SOURCE OF TRUTH: ~/.victor/profiles.yaml

This module derives tool-capable model patterns from the user's profiles.yaml.
Models with `native_tool_calls: true` in the `model_capabilities` section are
considered tool-capable.

Users should copy examples/profiles.yaml.example to ~/.victor/profiles.yaml
to get the full list of pre-configured model capabilities.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Module-level cache to avoid re-loading the same file multiple times
_tool_capable_patterns_cache: Optional[Dict[str, List[str]]] = None


def _load_tool_capable_patterns_from_yaml(
    user_profiles_path: Optional[Path] = None,
) -> Dict[str, List[str]]:
    """Load tool-capable model patterns from model_capabilities.yaml.

    PRIMARY SOURCE: victor/config/model_capabilities.yaml
    OVERRIDE SOURCE: ~/.victor/profiles.yaml (model_capabilities section)

    Returns model patterns where native_tool_calls is true, organized by provider.
    Results are cached at module level to avoid repeated file reads.
    """
    global _tool_capable_patterns_cache

    # Return cached result if available (for default path only)
    if user_profiles_path is None and _tool_capable_patterns_cache is not None:
        return _tool_capable_patterns_cache

    result: Dict[str, List[str]] = {}

    # First, load from the main model_capabilities.yaml (bundled with victor)
    config_dir = Path(__file__).parent
    model_caps_path = config_dir / "model_capabilities.yaml"
    if model_caps_path.exists():
        try:
            model_caps_data = yaml.safe_load(model_caps_path.read_text()) or {}
            _extract_tool_capable_patterns(model_caps_data, result)
            logger.debug(f"Loaded model capabilities from {model_caps_path}")
        except Exception as e:
            logger.warning(f"Failed to load model_capabilities.yaml: {e}")

    # Then, load overrides from user's profiles.yaml
    if user_profiles_path is None:
        from victor.config.settings import get_project_paths

        user_path = get_project_paths().global_victor_dir / "profiles.yaml"
    else:
        user_path = user_profiles_path
    if user_path.exists():
        try:
            user_data = yaml.safe_load(user_path.read_text()) or {}
            model_caps = user_data.get("model_capabilities", {})
            if model_caps:
                _extract_tool_capable_patterns(model_caps, result)
                logger.debug(f"Loaded model capability overrides from {user_path}")
        except Exception as e:
            logger.warning(f"Failed to load profiles.yaml: {e}")

    # If nothing loaded, use minimal built-in defaults
    if not result:
        logger.debug("No model_capabilities found, using minimal defaults")
        result = _minimal_builtin_defaults()

    # Cache for future calls (only for default path)
    if user_profiles_path is None:
        _tool_capable_patterns_cache = result

    return result


def _minimal_builtin_defaults() -> Dict[str, List[str]]:
    """Minimal built-in defaults for when profiles.yaml is missing.

    These are just enough to get started. Users should copy
    examples/profiles.yaml.example to get the full list.
    """
    # Cloud providers support all models
    defaults: Dict[str, List[str]] = {
        "anthropic": ["*"],
        "openai": ["*"],
        "google": ["*"],
        "xai": ["*"],
        "vllm": ["*"],
    }

    # Common local model patterns
    local_patterns = [
        "llama3.1*",
        "llama3.2*",
        "llama3.3*",
        "qwen2.5*",
        "qwen3*",
        "mistral*",
        "mixtral*",
        "deepseek*",
        "hermes*",
        "command-r*",
    ]
    for provider in ["ollama", "lmstudio"]:
        defaults[provider] = local_patterns.copy()

    return defaults


def _extract_tool_capable_patterns(data: Dict, result: Dict[str, List[str]]) -> None:
    """Extract tool-capable patterns from capability data into result dict."""
    # Providers with native_tool_calls: true at provider_defaults level
    provider_defaults = data.get("provider_defaults", {})
    for provider, caps in provider_defaults.items():
        if isinstance(caps, dict) and caps.get("native_tool_calls", False):
            # This provider supports tools by default (e.g., anthropic, openai)
            result.setdefault(provider.lower(), []).append("*")

    # Model-specific patterns - check providers.<provider>.native_tool_calls
    models = data.get("models", {})
    for pattern, model_config in models.items():
        if not isinstance(model_config, dict):
            continue

        # Check for native_tool_calls in the providers section of each model
        model_providers = model_config.get("providers", {})
        for provider, provider_caps in model_providers.items():
            if isinstance(provider_caps, dict) and provider_caps.get("native_tool_calls", False):
                provider_l = provider.lower()
                if pattern not in result.get(provider_l, []):
                    result.setdefault(provider_l, []).append(pattern)


def _flatten_yaml_manifest(data: Dict[str, Iterable]) -> Dict[str, List[str]]:
    """Flatten the tiered YAML manifest into provider -> model list."""
    result: Dict[str, List[str]] = {}

    def _add(provider: str, name: Optional[str]) -> None:
        if not name:
            return
        result.setdefault(provider, []).append(name)

    for provider, tiers in data.items():
        if isinstance(tiers, dict):
            for tier_models in tiers.values():
                if isinstance(tier_models, list):
                    for entry in tier_models:
                        if isinstance(entry, dict):
                            _add(provider, entry.get("name"))
                        else:
                            _add(provider, str(entry))
        elif isinstance(tiers, list):
            for entry in tiers:
                if isinstance(entry, dict):
                    _add(provider, entry.get("name"))
                else:
                    _add(provider, str(entry))
    # Deduplicate while preserving order
    for provider, models in result.items():
        seen = set()
        deduped: List[str] = []
        for model in models:
            if model not in seen:
                seen.add(model)
                deduped.append(model)
        result[provider] = deduped
    return result


class ToolCallingMatrix:
    """Capability matrix to decide whether a model supports structured tool calls.

    SINGLE SOURCE OF TRUTH: model_capabilities.yaml

    Tool-capable models are derived from patterns with `native_tool_calls: true`
    in model_capabilities.yaml. No hardcoded model lists.
    """

    def __init__(
        self,
        manifest: Optional[Dict[str, List[str]]] = None,
        manifest_path: Optional[Path] = None,  # noqa: ARG002 - kept for API compat
        always_allow_providers: Optional[List[str]] = None,
    ) -> None:
        # Load tool-capable patterns from model_capabilities.yaml
        base = _load_tool_capable_patterns_from_yaml()

        # Merge any additional manifest patterns (for extensibility)
        if manifest:
            for provider, models in manifest.items():
                base.setdefault(provider.lower(), [])
                for model in models:
                    if model not in base[provider.lower()]:
                        base[provider.lower()].append(model)

        # Normalize to lowercase for matching
        self.manifest: Dict[str, List[str]] = {
            provider.lower(): [m.lower() for m in models] for provider, models in base.items()
        }
        self.always_allow = {p.lower() for p in (always_allow_providers or [])}

    @staticmethod
    def _matches(model: str, pattern: str) -> bool:
        """Check if a model matches a pattern with simple wildcards."""
        model_l = model.lower()
        pattern_l = pattern.lower()
        if pattern_l == model_l:
            return True
        if pattern_l.endswith("*"):
            return model_l.startswith(pattern_l[:-1])
        if pattern_l.startswith("*") and pattern_l.endswith("*"):
            return pattern_l.strip("*") in model_l
        if pattern_l.startswith("*"):
            return model_l.endswith(pattern_l[1:])
        return pattern_l in model_l

    def is_tool_call_supported(self, provider: str, model: str) -> bool:
        """Return True if the provider/model pair is marked as tool-capable."""
        provider_l = (provider or "").lower()
        model_l = (model or "").lower()

        if provider_l in self.always_allow:
            return True

        patterns = self.manifest.get(provider_l)
        if not patterns:
            return False

        return any(self._matches(model_l, pattern) for pattern in patterns)

    def get_supported_models(self, provider: str) -> List[str]:
        """Return the list of known tool-capable models for a provider."""
        return self.manifest.get(provider.lower(), [])

    def to_json(self) -> str:
        """Serialize manifest for debugging or logging."""
        return json.dumps(self.manifest, indent=2, sort_keys=True)


__all__ = ["ToolCallingMatrix", "_load_tool_capable_patterns_from_yaml"]
