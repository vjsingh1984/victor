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
        "ollama": [
            # 25B-50B first (best quality per GB)
            "qwen3-coder:30b",  # ~34GB Q4, strong coder + tools
            "deepseek-coder:33b",  # ~36GB Q4, strong code/tool
            "deepseek-coder-v2:34b",  # ~36GB Q4, newer
            "seed-36b",  # est ~38-42GB Q4
            "gemma3:27b",  # est ~30-34GB Q4
            "qwen2.5-coder:32b",  # ~36GB Q4
            # 10B-25B next
            "deepseek-coder-v2:16b",  # ~18GB Q4
            "qwen2.5-coder:14b",  # ~18GB Q4
            "seed-14b",  # ~18-20GB Q4
            "gemma3:12b",  # ~14-16GB Q4
            "mixtral:8x7b",  # MoE; ~24GB Q4, good tools
            "mixtral-8x7b",  # alias
            "mixtral-8x22b",  # MoE; heavier, use only with headroom
            # 50B-70B tier (borderline on 64GB shared GPU)
            "llama3.1:70b",  # ~64GB Q4, borderline
            # <10B for light use
            "mistral:7b-instruct",  # ~10GB Q4
            "phi-4",  # ~12-14GB Q4
            "phi-4-mini",  # ~8-10GB Q4
            "llama3.1:8b",  # ~10GB Q4
        ],
        "lmstudio": [
            # 25B-50B
            "qwen3-coder:30b",  # ~34GB Q4
            "deepseek-coder:33b",  # ~36GB Q4
            "seed-36b",  # ~38-42GB Q4
            "gemma3:27b",  # ~30-34GB Q4
            # 10B-25B
            "deepseek-coder-v2:16b",  # ~18GB Q4
            "qwen2.5-coder:14b",  # ~18GB Q4
            "seed-14b",  # ~18-20GB Q4
            "gemma3:12b",  # ~14-16GB Q4
            "mixtral:8x7b",  # ~24GB Q4 MoE
            # <10B
            "mistral:7b-instruct",  # ~10GB Q4
            "phi-4",  # ~12-14GB Q4
            "phi-4-mini",  # ~8-10GB Q4
            "llama3.1:8b",  # ~10GB Q4
        ],
        "vllm": [
            # 25B-50B
            "qwen3-coder:30b",  # ~34GB Q4
            "deepseek-coder:33b",  # ~36GB Q4
            "deepseek-coder-v2:34b",  # ~36GB Q4
            "seed-36b",  # ~38-42GB Q4
            "gemma3:27b",  # ~30-34GB Q4
            "qwen2.5-coder:32b",  # ~36GB Q4
            # 10B-25B
            "deepseek-coder-v2:16b",  # ~18GB Q4
            "qwen2.5-coder:14b",  # ~18GB Q4
            "seed-14b",  # ~18-20GB Q4
            "gemma3:12b",  # ~14-16GB Q4
            "mixtral:8x7b",  # ~24GB Q4 MoE
            "mixtral-8x7b",  # alias
            "mixtral-8x22b",  # heavier MoE
            # 50B-70B (borderline)
            "llama3.1:70b",  # ~64GB Q4, borderline
            # <10B
            "mistral:7b-instruct",  # ~10GB Q4
            "phi-4",  # ~12-14GB Q4
            "phi-4-mini",  # ~8-10GB Q4
            "llama3.1:8b",  # ~10GB Q4
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
