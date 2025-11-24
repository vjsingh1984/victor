"""Configuration management for CodingAgent."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderConfig(BaseSettings):
    """Configuration for a specific provider."""

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3
    organization: Optional[str] = None  # For OpenAI


class ProfileConfig(BaseSettings):
    """Configuration for a model profile."""

    provider: str = Field(..., description="Provider name (ollama, anthropic, openai, google)")
    model: str = Field(..., description="Model identifier")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(4096, gt=0)


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
    )

    # Default provider settings
    default_provider: str = "ollama"
    default_model: str = "qwen2.5-coder:7b"
    default_temperature: float = 0.7
    default_max_tokens: int = 4096

    # API Keys
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None

    # Local server URLs
    ollama_base_url: str = "http://localhost:11434"
    lmstudio_base_url: str = "http://localhost:1234"
    vllm_base_url: str = "http://localhost:8000"

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # UI
    theme: str = "monokai"
    show_token_count: bool = True
    stream_responses: bool = True

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
            # Return default profiles
            return {
                "default": ProfileConfig(
                    provider="ollama",
                    model="qwen2.5-coder:7b",
                    temperature=0.7,
                    max_tokens=4096,
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
            settings.setdefault("base_url", self.lmstudio_base_url)

        elif provider == "vllm":
            settings.setdefault("base_url", self.vllm_base_url)

        return settings


def load_settings() -> Settings:
    """Load application settings.

    Returns:
        Settings instance
    """
    return Settings()
