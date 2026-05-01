"""Provider connection and model defaults."""

from __future__ import annotations

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
        """Return provider name for string contexts (f-strings, str(), logging)."""
        return self.default_provider

    def __eq__(self, other: object) -> bool:
        """Compare provider name with string or ProviderSettings."""
        if isinstance(other, str):
            return self.default_provider == other
        if isinstance(other, ProviderSettings):
            return self.default_provider == other.default_provider
        return NotImplemented

    def __hash__(self) -> int:
        """Hash by provider name for dict/set operations."""
        return hash(self.default_provider)
