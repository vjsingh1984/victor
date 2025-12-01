"""Model capability helpers for tool-calling support across providers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml


def _default_tool_calling_models() -> Dict[str, List[str]]:
    """Built-in manifest for providers that support tool calling."""
    return {
        # Local/open-weight: ordered by coding/tool proficiency and size bands for ~64GB hosts (Q4 focus).
        # Source: https://ollama.com/search?c=tools
        "ollama": [
            # 50B-70B tier (need 48GB+ VRAM)
            "llama3.1:70b",  # ~40GB Q4, native tools
            "llama3.3:70b",  # ~40GB Q4, native tools (newer)
            "llama3-groq-tool-use:70b",  # ~40GB Q4, specialized for tools
            "hermes3:70b",  # ~40GB Q4, tool calling
            "nemotron:70b",  # ~40GB Q4, tool calling
            "athene-v2:72b",  # ~44GB Q4, tool calling
            # 25B-50B tier (best quality per GB)
            "mixtral:8x7b",  # MoE; ~26GB Q4, good tools
            "mixtral-8x7b",  # alias
            "mixtral-8x22b",  # MoE; heavier, use only with headroom
            "qwen3:32b",  # ~20GB Q4, native tools
            "qwen3-coder:30b",  # ~18GB Q4, strong coder + tools
            "qwen3:30b",  # ~18GB Q4, MoE variant
            "qwen2.5-coder:32b",  # ~20GB Q4, strong coder
            "qwen2.5:32b",  # ~20GB Q4, general purpose
            "deepseek-coder:33b",  # ~20GB Q4, strong code/tool
            "deepseek-coder-v2:34b",  # ~20GB Q4, newer
            "command-r:35b",  # ~22GB Q4, enterprise tools
            "seed-36b",  # est ~22GB Q4
            "gemma3:27b",  # ~17GB Q4, tool calling
            "aya-expanse:32b",  # ~20GB Q4, multilingual + tools
            # 10B-25B tier
            "qwen2.5-coder:14b",  # ~9GB Q4, fast coder
            "qwen2.5-coder:7b",  # ~5GB Q4, lightweight coder
            "deepseek-coder-v2:16b",  # ~10GB Q4
            "seed-14b",  # ~9GB Q4
            "gemma3:12b",  # ~8GB Q4
            # <10B for light use
            "llama3.1:8b",  # ~5GB Q4
            "llama3.2:3b",  # ~2GB Q4
            "llama3-groq-tool-use:8b",  # ~5GB Q4, specialized for tools
            "mistral:7b-instruct",  # ~4GB Q4
            "mistral:7b",  # ~4GB Q4
            "phi-4",  # ~8GB Q4
            "phi-4-mini",  # ~4GB Q4
            "hermes3:8b",  # ~5GB Q4
            "firefunction-v2:70b",  # specialized function calling
        ],
        "lmstudio": [
            # 50B-70B tier
            "llama3.1:70b",
            "llama3.3:70b",
            # 25B-50B tier
            "qwen3:32b",
            "qwen3-coder:30b",
            "qwen3:30b",
            "qwen2.5-coder:32b",
            "qwen2.5:32b",
            "deepseek-coder:33b",
            "deepseek-coder-v2:34b",
            "seed-36b",
            "gemma3:27b",
            "mixtral:8x7b",
            # 10B-25B tier
            "qwen2.5-coder:14b",
            "qwen2.5-coder:7b",
            "deepseek-coder-v2:16b",
            "seed-14b",
            "gemma3:12b",
            # <10B
            "llama3.1:8b",
            "mistral:7b-instruct",
            "mistral:7b",
            "phi-4",
            "phi-4-mini",
        ],
        "vllm": [
            # 50B-70B tier
            "llama3.1:70b",
            "llama3.3:70b",
            # 25B-50B tier
            "qwen3:32b",
            "qwen3-coder:30b",
            "qwen3:30b",
            "qwen2.5-coder:32b",
            "qwen2.5:32b",
            "deepseek-coder:33b",
            "deepseek-coder-v2:34b",
            "seed-36b",
            "gemma3:27b",
            "mixtral:8x7b",
            "mixtral-8x7b",
            "mixtral-8x22b",
            # 10B-25B tier
            "qwen2.5-coder:14b",
            "qwen2.5-coder:7b",
            "deepseek-coder-v2:16b",
            "seed-14b",
            "gemma3:12b",
            # <10B
            "llama3.1:8b",
            "mistral:7b-instruct",
            "mistral:7b",
            "phi-4",
            "phi-4-mini",
        ],
        "anthropic": [
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
            "claude-3-5-haiku-20241022",
            "claude-sonnet-4-5",
            "claude-opus-4-5",
            "claude-haiku-4-5",
        ],
        "openai": [
            # Order by expected coding/tool strength
            "gpt-5.1",
            "gpt-5",
            "gpt-4.5-preview",
            "gpt-4.1",
            "gpt-4o",
            "gpt-4.1-mini",
            "gpt-4o-mini",
        ],
        "google": [
            # Order by capability; 3.x then 2.5
            "gemini-3-pro",
            "gemini-3-pro-preview",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
            "gemini-2.0-flash-001",
        ],
        "xai": [
            "grok-2",
        ],
    }


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
    """Capability matrix to decide whether a model supports structured tool calls."""

    def __init__(
        self,
        manifest: Optional[Dict[str, List[str]]] = None,
        manifest_path: Optional[Path] = None,
        always_allow_providers: Optional[List[str]] = None,
    ) -> None:
        base = _default_tool_calling_models()
        if manifest:
            for provider, models in manifest.items():
                base.setdefault(provider.lower(), [])
                for model in models:
                    if model not in base[provider.lower()]:
                        base[provider.lower()].append(model)

        # Merge YAML manifest if available
        path = manifest_path or Path(__file__).with_name("tool_calling_models.yaml")
        if path.exists():
            try:
                yaml_data = yaml.safe_load(path.read_text()) or {}
                yaml_manifest = _flatten_yaml_manifest(yaml_data.get("tool_calling_models", {}))
                for provider, models in yaml_manifest.items():
                    base.setdefault(provider.lower(), [])
                    for model in models:
                        if model not in base[provider.lower()]:
                            base[provider.lower()].append(model)
            except Exception:
                # If the manifest is malformed, fall back to the base set
                pass

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


__all__ = ["ToolCallingMatrix", "_default_tool_calling_models"]
