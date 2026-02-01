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

"""Provider validation utility for integration tests.

This module provides robust API key validation that checks:
- API key existence (environment variable or keyring)
- API key validity (makes a test call)
- Billing/credit status (detects common error patterns)
"""

import os
import re
import asyncio
from typing import Optional
from dataclasses import dataclass


@dataclass
class ProviderValidationResult:
    """Result of provider validation."""

    provider: str
    available: bool
    reason: str
    error_type: Optional[str] = (
        None  # 'missing_key', 'auth_error', 'billing_error', 'rate_limit', 'network_error'
    )

    def __str__(self) -> str:
        status = "✓" if self.available else "✗"
        return f"{status} {self.provider}: {self.reason}"


class ProviderValidator:
    """Validates provider availability by testing API keys.

    Checks for:
    - Missing API keys
    - Invalid/expired keys
    - Billing/credit issues
    - Rate limits
    - Network errors
    """

    # Common error patterns for different failure types
    AUTH_PATTERNS = [
        r"invalid api key",
        r"authentication.*failed",
        r"unauthorized",
        r"forbidden.*40",
        r"credentials.*invalid",
        r"token.*invalid",
        r"access.*denied",
        r"auth failed",
    ]

    BILLING_PATTERNS = [
        r"billing",
        r"payment",
        r"credit.*limit",
        r"quota.*exceed",
        r"insufficient.*fund",
        r"insufficient.*credit",
        r"balance.*insufficient",
        r"account.*suspended",
        r"usage.*limit",
    ]

    RATE_LIMIT_PATTERNS = [
        r"rate.*limit",
        r"too.*many.*request",
        r"429",
        r"quota.*exceeded",
        r"throttl",
        r"retry.*after",
    ]

    def __init__(self):
        """Initialize provider validator."""
        self._cache: dict[str, ProviderValidationResult] = {}

    def classify_error(self, error_message: str) -> str:
        """Classify error type from error message.

        Args:
            error_message: Error message from provider

        Returns:
            Error type: 'auth_error', 'billing_error', 'rate_limit', 'network_error', or 'unknown'
        """
        error_lower = error_message.lower()

        # Check for authentication errors
        for pattern in self.AUTH_PATTERNS:
            if re.search(pattern, error_lower):
                return "auth_error"

        # Check for billing errors
        for pattern in self.BILLING_PATTERNS:
            if re.search(pattern, error_lower):
                return "billing_error"

        # Check for rate limits
        for pattern in self.RATE_LIMIT_PATTERNS:
            if re.search(pattern, error_lower):
                return "rate_limit"

        # Check for network errors
        network_patterns = [
            r"connection.*refused",
            r"timeout",
            r"network.*error",
            r"dns.*fail",
            r"host.*unreachable",
        ]
        for pattern in network_patterns:
            if re.search(pattern, error_lower):
                return "network_error"

        return "unknown"

    async def validate_provider(
        self,
        provider_name: str,
        provider_class,
        model: str,
        api_key: Optional[str],
    ) -> ProviderValidationResult:
        """Validate a provider by making a test API call.

        Args:
            provider_name: Provider name
            provider_class: Provider class
            model: Model to test with
            api_key: API key (if applicable)

        Returns:
            ProviderValidationResult with availability status
        """
        # Check cache first
        if provider_name in self._cache:
            return self._cache[provider_name]

        # For local providers, check if service is running
        if provider_name in ["ollama", "lmstudio", "vllm", "llamacpp", "llama.cpp"]:
            return await self._validate_local_provider(provider_name)

        # For cloud providers, need API key
        if not api_key:
            result = ProviderValidationResult(
                provider=provider_name,
                available=False,
                reason=f"{self._get_env_var(provider_name)} not set",
                error_type="missing_key",
            )
            self._cache[provider_name] = result
            return result

        # Test API key with a simple call
        try:
            provider = provider_class(api_key=api_key, timeout=10)

            # Import Message from base
            from victor.providers.base import Message

            # Make minimal test call
            response = await provider.chat(
                messages=[Message(role="user", content="Hi")], model=model, max_tokens=5
            )

            if response and response.content:
                result = ProviderValidationResult(
                    provider=provider_name, available=True, reason=f"API key valid (model: {model})"
                )
            else:
                result = ProviderValidationResult(
                    provider=provider_name,
                    available=False,
                    reason="No response content",
                    error_type="auth_error",
                )

            # Cleanup
            if hasattr(provider, "close"):
                await provider.close()
            elif hasattr(provider, "client"):
                await provider.client.aclose()

        except Exception as e:
            error_msg = str(e).lower()
            error_type = self.classify_error(str(e))

            # User-friendly reason
            if error_type == "auth_error":
                reason = f"Invalid API key: {str(e)[:100]}"
            elif error_type == "billing_error":
                reason = f"Billing/credit issue: {str(e)[:100]}"
            elif error_type == "rate_limit":
                reason = f"Rate limit exceeded: {str(e)[:100]}"
            elif error_type == "network_error":
                reason = f"Network error: {str(e)[:100]}"
            else:
                reason = f"Error: {str(e)[:100]}"

            result = ProviderValidationResult(
                provider=provider_name, available=False, reason=reason, error_type=error_type
            )

        self._cache[provider_name] = result
        return result

    async def _validate_local_provider(self, provider_name: str) -> ProviderValidationResult:
        """Validate local provider (Ollama, LMStudio, etc.).

        Args:
            provider_name: Provider name

        Returns:
            ProviderValidationResult
        """
        import socket

        if provider_name == "ollama":
            host, port = "localhost", 11434
        elif provider_name in ["lmstudio", "vllm"]:
            host, port = "localhost", 1234
        elif provider_name in ["llamacpp", "llama.cpp"]:
            # llama.cpp runs on dynamic ports, just check if binary exists
            return ProviderValidationResult(
                provider=provider_name, available=True, reason="Local provider (check if running)"
            )
        else:
            return ProviderValidationResult(
                provider=provider_name, available=False, reason="Unknown local provider"
            )

        # Try to connect
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((host, port))
            sock.close()

            if result == 0:
                return ProviderValidationResult(
                    provider=provider_name, available=True, reason=f"Running at {host}:{port}"
                )
            else:
                return ProviderValidationResult(
                    provider=provider_name,
                    available=False,
                    reason=f"Not running at {host}:{port}",
                    error_type="network_error",
                )
        except Exception as e:
            return ProviderValidationResult(
                provider=provider_name,
                available=False,
                reason=f"Connection failed: {str(e)[:50]}",
                error_type="network_error",
            )

    def _get_env_var(self, provider_name: str) -> str:
        """Get environment variable name for provider's API key.

        Args:
            provider_name: Provider name

        Returns:
            Environment variable name
        """
        env_vars = {
            # Cloud providers
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
            "xai": "XAI_API_KEY",
            "zai": "ZAI_API_KEY",
            "moonshot": "MOONSHOT_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "groqcloud": "GROQCLOUD_API_KEY",
            "mistral": "MISTRAL_API_KEY",
            "together": "TOGETHER_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "fireworks": "FIREWORKS_API_KEY",
            "cerebras": "CEREBRAS_API_KEY",
            "huggingface": "HF_TOKEN",
            "replicate": "REPLICATE_API_TOKEN",
            # Enterprise
            "vertexai": "GOOGLE_APPLICATION_CREDENTIALS",
            "azure-openai": "AZURE_OPENAI_API_KEY",
            "bedrock": "AWS_ACCESS_KEY_ID",
            # Local providers (no API key)
            "ollama": None,
            "lmstudio": None,
            "vllm": None,
            "llamacpp": None,
            "llama.cpp": None,
        }

        return env_vars.get(provider_name.lower(), "")

    async def validate_all_providers(
        self, providers_config: dict[str, dict]
    ) -> dict[str, ProviderValidationResult]:
        """Validate all providers concurrently.

        Args:
            providers_config: Dict mapping provider name to config dict with:
                - 'class': Provider class
                - 'model': Model to test
                - 'api_key_getter': Function to get API key (optional)

        Returns:
            Dict mapping provider name to validation result
        """
        results = {}

        # Create validation tasks
        tasks = []
        provider_names = []

        for provider_name, config in providers_config.items():
            provider_class = config["class"]
            model = config["model"]

            # Get API key
            if "api_key_getter" in config:
                api_key = config["api_key_getter"]()
            else:
                env_var = self._get_env_var(provider_name)
                api_key = os.getenv(env_var)

            task = self.validate_provider(provider_name, provider_class, model, api_key)
            tasks.append(task)
            provider_names.append(provider_name)

        # Run validations concurrently
        validation_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for provider_name, result in zip(provider_names, validation_results):
            if isinstance(result, Exception):
                results[provider_name] = ProviderValidationResult(
                    provider=provider_name,
                    available=False,
                    reason=f"Validation error: {str(result)[:100]}",
                    error_type="unknown",
                )
            else:
                results[provider_name] = result

        return results


# Global validator instance
_provider_validator = ProviderValidator()


def get_provider_validator() -> ProviderValidator:
    """Get the global provider validator instance."""
    return _provider_validator
