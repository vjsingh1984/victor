"""Provider connection and model defaults."""

from __future__ import annotations

import warnings
from typing import List, Optional

from pydantic import BaseModel, Field, SecretStr


class ProviderSettings(BaseModel):
    """Provider connection and model defaults."""

    default_provider: str = "ollama"
    default_model: str = "qwen3-coder:30b"
    default_temperature: float = 0.7
    default_max_tokens: int = 4096
    anthropic_api_key: Optional[SecretStr] = None
    openai_api_key: Optional[SecretStr] = None
    google_api_key: Optional[SecretStr] = None
    moonshot_api_key: Optional[SecretStr] = None
    deepseek_api_key: Optional[SecretStr] = None
    ollama_base_url: str = "http://localhost:11434"
    lmstudio_base_urls: List[str] = Field(default_factory=lambda: ["http://127.0.0.1:1234"])
    vllm_base_url: str = "http://localhost:8000"
    lmstudio_max_vram_gb: Optional[float] = 48.0

    def __str__(self) -> str:
        """Return provider name for string operations.

        Enables backward compatibility with code that treats provider as a string.
        Allows operations like str(provider), f"{provider}", etc.
        """
        return self.default_provider

    # String-like methods for backward compatibility (deprecated)
    def lower(self) -> str:
        """Return lowercase provider name."""
        warnings.warn(
            "ProviderSettings.lower() is deprecated. " "Use .default_provider.lower() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.default_provider.lower()

    def upper(self) -> str:
        """Return uppercase provider name."""
        warnings.warn(
            "ProviderSettings.upper() is deprecated. " "Use .default_provider.upper() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.default_provider.upper()

    def title(self) -> str:
        """Return title-case provider name."""
        warnings.warn(
            "ProviderSettings.title() is deprecated. " "Use .default_provider.title() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.default_provider.title()

    def startswith(self, prefix: str) -> bool:
        """Check if provider name starts with prefix."""
        warnings.warn(
            "ProviderSettings.startswith() is deprecated. "
            "Use .default_provider.startswith() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.default_provider.startswith(prefix)

    def endswith(self, suffix: str) -> bool:
        """Check if provider name ends with suffix."""
        warnings.warn(
            "ProviderSettings.endswith() is deprecated. "
            "Use .default_provider.endswith() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.default_provider.endswith(suffix)

    def replace(self, old: str, new: str) -> str:
        """Replace substrings in provider name."""
        warnings.warn(
            "ProviderSettings.replace() is deprecated. " "Use .default_provider.replace() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.default_provider.replace(old, new)

    def split(self, sep: str = None, maxsplit: int = -1):
        """Split provider name."""
        warnings.warn(
            "ProviderSettings.split() is deprecated. " "Use .default_provider.split() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.default_provider.split(sep, maxsplit)

    def strip(self) -> str:
        """Strip whitespace from provider name."""
        warnings.warn(
            "ProviderSettings.strip() is deprecated. " "Use .default_provider.strip() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.default_provider.strip()

    def __eq__(self, other: object) -> bool:
        """Compare provider name with other (string or ProviderSettings)."""
        if isinstance(other, str):
            return self.default_provider == other
        if isinstance(other, ProviderSettings):
            return self.default_provider == other.default_provider
        return NotImplemented

    def __hash__(self) -> int:
        """Hash provider name for dict/set operations."""
        return hash(self.default_provider)
