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

"""
Unified API Key Resolution for Victor Providers.

This module provides centralized API key resolution with support for:
- Non-interactive environments (CI/CD, containers, daemons)
- Multiple key sources with full attribution
- Actionable error messages
- Debug logging

Resolution Order:
1. Explicit api_key parameter (highest priority)
2. Environment variable (VICTOR_NONINTERACTIVE=true uses this as default)
3. System keyring (only when VICTOR_NONINTERACTIVE=false/undefined)
4. Config file (~/.victor/api_keys.yaml)

Usage:
    from victor.providers.resolution import UnifiedApiKeyResolver

    resolver = UnifiedApiKeyResolver()
    result = resolver.get_api_key("deepseek", explicit_key=None)

    if result.key is None:
        raise APIKeyNotFoundError(
            provider="deepseek",
            sources_attempted=result.sources_attempted,
            non_interactive=result.non_interactive,
        )
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from victor.config.api_keys import (
    PROVIDER_ENV_VARS,
    _get_key_from_keyring,
    is_keyring_available,
)


logger = logging.getLogger(__name__)


@dataclass
class KeySource:
    """Represents a single key source that was attempted."""

    source: str  # "explicit", "environment", "keyring", "file"
    description: str  # Human-readable description
    found: bool  # Whether key was found
    value_preview: Optional[str] = None  # First few chars of key (for logging)
    interactive_required: bool = False  # Whether this source requires user interaction

    def __str__(self) -> str:
        status = "✓" if self.found else "✗"
        return f"{status} {self.description}"


@dataclass
class APIKeyResult:
    """Result of API key resolution attempt."""

    key: Optional[str]
    source: str  # "explicit", "environment", "keyring", "file", "none"
    source_detail: str  # Specific source (e.g., "DEEPSEEK_API_KEY env var")
    sources_attempted: List[KeySource] = field(default_factory=list)
    non_interactive: bool = False
    confidence: str = "low"  # "high", "medium", "low"


class APIKeyNotFoundError(Exception):
    """
    API key not found with actionable suggestions.

    This exception provides a user-friendly error message with:
    - List of all sources attempted
    - Status of each source (found/not found)
    - Actionable suggestions for fixing the issue
    - Context about the environment (interactive/non-interactive)
    """

    def __init__(
        self,
        provider: str,
        sources_attempted: List[KeySource],
        non_interactive: bool,
        model: Optional[str] = None,
    ):
        self.provider = provider
        self.sources_attempted = sources_attempted
        self.non_interactive = non_interactive
        self.model = model
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Generate user-friendly error with solutions."""
        lines = [
            f"{self.provider.upper()} API key not found. "
            f"Tried {len(self.sources_attempted)} source(s):",
        ]

        for i, source in enumerate(self.sources_attempted, 1):
            lines.append(f"  {i}. {source}")

        lines.append("\nSolutions:")

        if self.non_interactive:
            env_var = _get_provider_env_var(self.provider)
            if env_var:
                lines.append(
                    f"  • Set {env_var} environment variable "
                    "(recommended for servers/containers/CI)"
                )
            lines.append(
                "  • Pass api_key parameter to provider constructor"
            )
        else:
            lines.append(
                f"  • Run: victor keys set {self.provider} --keyring "
                "(for interactive CLI use)"
            )
            env_var = _get_provider_env_var(self.provider)
            if env_var:
                lines.append(f"  • Set {env_var} environment variable")

        if self.model:
            lines.append(f"\n  Context: Provider={self.provider}, Model={self.model}")
        if self.non_interactive:
            lines.append("  Environment: Non-interactive mode (daemon/container/CI)")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "error_type": "APIKeyNotFound",
            "provider": self.provider,
            "model": self.model,
            "non_interactive": self.non_interactive,
            "sources_attempted": [
                {
                    "source": s.source,
                    "description": s.description,
                    "found": s.found,
                }
                for s in self.sources_attempted
            ],
            "message": str(self),
        }


def _get_provider_env_var(provider: str) -> Optional[str]:
    """Get environment variable name for a provider."""
    return PROVIDER_ENV_VARS.get(provider.lower())


class UnifiedApiKeyResolver:
    """
    Centralized API key resolution with non-interactive support.

    Features:
    - Automatic detection of non-interactive environments
    - Multiple key sources with full attribution
    - Structured results for error handling
    - Debug logging for troubleshooting

    Detection of Non-Interactive Environments:
    - VICTOR_NONINTERACTIVE=true (explicit)
    - CI environment variable (GitHub Actions, GitLab CI, etc.)
    - KUBERNETES_SERVICE_HOST (Kubernetes)
    - container environment variable (Docker)
    - No TTY attached to stdin

    Usage:
        # Auto-detect environment
        resolver = UnifiedApiKeyResolver()
        result = resolver.get_api_key("deepseek")

        # Explicit mode
        resolver = UnifiedApiKeyResolver(non_interactive=True)
        result = resolver.get_api_key("deepseek")
    """

    def __init__(self, non_interactive: Optional[bool] = None):
        """
        Initialize resolver.

        Args:
            non_interactive: Force non-interactive mode (None = auto-detect)
        """
        self.non_interactive = (
            non_interactive if non_interactive is not None
            else self._detect_non_interactive()
        )
        self._cache: Dict[str, APIKeyResult] = {}

        if self.non_interactive:
            logger.debug("UnifiedApiKeyResolver: Non-interactive mode detected")

    def _detect_non_interactive(self) -> bool:
        """
        Detect if running in non-interactive environment.

        Returns:
            True if non-interactive environment detected
        """
        # Explicit env var
        if os.environ.get("VICTOR_NONINTERACTIVE", "").lower() == "true":
            logger.debug("VICTOR_NONINTERACTIVE=true detected")
            return True

        # CI/CD detection
        if os.environ.get("CI"):
            logger.debug("CI environment detected")
            return True

        # Kubernetes detection
        if os.environ.get("KUBERNETES_SERVICE_HOST"):
            logger.debug("Kubernetes environment detected")
            return True

        # Docker detection (common env vars)
        if os.environ.get("container"):
            logger.debug("Container environment detected")
            return True

        # TTY detection (Unix)
        try:
            if not sys.stdin.isatty():
                logger.debug("No TTY detected on stdin")
                return True
        except Exception:
            pass

        return False

    def get_api_key(
        self,
        provider: str,
        explicit_key: Optional[str] = None,
        check_keyring: Optional[bool] = None,
    ) -> APIKeyResult:
        """
        Resolve API key with full attribution.

        Args:
            provider: Provider name (e.g., "deepseek", "anthropic")
            explicit_key: Explicit key passed by caller (highest priority)
            check_keyring: Whether to check keyring (None = auto-detect)

        Returns:
            APIKeyResult with key, source, and full attribution
        """
        # Check cache
        if provider in self._cache:
            cached = self._cache[provider]
            # If explicit key provided, don't use cached result
            if explicit_key is None:
                return cached

        provider = provider.lower()
        sources: List[KeySource] = []

        # Priority 1: Explicit parameter
        if explicit_key is not None:
            sources.append(KeySource(
                source="explicit",
                description="Explicit api_key parameter",
                found=True,
                value_preview=self._preview_key(explicit_key),
                interactive_required=False,
            ))
            return self._cache_result(provider, APIKeyResult(
                key=explicit_key,
                source="explicit",
                source_detail="Explicit api_key parameter",
                sources_attempted=sources,
                non_interactive=self.non_interactive,
                confidence="high",
            ))

        sources.append(KeySource(
            source="explicit",
            description="Explicit api_key parameter",
            found=False,
            interactive_required=False,
        ))

        # Priority 2: Environment variable
        env_var = _get_provider_env_var(provider)
        env_key = None
        if env_var:
            env_key = os.environ.get(env_var)
            if env_key:
                sources.append(KeySource(
                    source="environment",
                    description=f"{env_var} environment variable",
                    found=True,
                    value_preview=self._preview_key(env_key),
                    interactive_required=False,
                ))
                result = APIKeyResult(
                    key=env_key,
                    source="environment",
                    source_detail=f"{env_var} environment variable",
                    sources_attempted=sources,
                    non_interactive=self.non_interactive,
                    confidence="high",
                )
                return self._cache_result(provider, result)
            else:
                sources.append(KeySource(
                    source="environment",
                    description=f"{env_var} environment variable",
                    found=False,
                    interactive_required=False,
                ))

        # Priority 3: Keyring (skip in non-interactive mode)
        check_keyring = (
            check_keyring if check_keyring is not None
            else not self.non_interactive
        )

        if check_keyring and is_keyring_available():
            from victor.config.api_keys import KEYRING_SERVICE

            keyring_key = _get_key_from_keyring(provider)
            if keyring_key:
                sources.append(KeySource(
                    source="keyring",
                    description=f"System keyring ({KEYRING_SERVICE})",
                    found=True,
                    value_preview=self._preview_key(keyring_key),
                    interactive_required=False,
                ))
                result = APIKeyResult(
                    key=keyring_key,
                    source="keyring",
                    source_detail=f"System keyring ({KEYRING_SERVICE})",
                    sources_attempted=sources,
                    non_interactive=self.non_interactive,
                    confidence="medium",
                )
                return self._cache_result(provider, result)
            else:
                sources.append(KeySource(
                    source="keyring",
                    description=f"System keyring ({KEYRING_SERVICE})",
                    found=False,
                    interactive_required=True,  # May need user to unlock
                ))
        elif self.non_interactive:
            sources.append(KeySource(
                source="keyring",
                description="System keyring (skipped in non-interactive mode)",
                found=False,
                interactive_required=False,
            ))

        # Priority 4: Config file
        try:
            from victor.config.api_keys import _get_secure_keys_file, APIKeyManager

            keys_file = _get_secure_keys_file()
            if keys_file.exists():
                manager = APIKeyManager(keys_file=keys_file)
                file_key = manager._load_key_from_file(provider)
                if file_key:
                    sources.append(KeySource(
                        source="file",
                        description=f"Config file ({keys_file})",
                        found=True,
                        value_preview=self._preview_key(file_key),
                        interactive_required=False,
                    ))
                    result = APIKeyResult(
                        key=file_key,
                        source="file",
                        source_detail=f"Config file ({keys_file})",
                        sources_attempted=sources,
                        non_interactive=self.non_interactive,
                        confidence="medium",
                    )
                    return self._cache_result(provider, result)
                else:
                    sources.append(KeySource(
                        source="file",
                        description=f"Config file ({keys_file})",
                        found=False,
                        interactive_required=False,
                    ))
        except Exception as e:
            logger.debug(f"Config file check failed: {e}")

        # No key found
        result = APIKeyResult(
            key=None,
            source="none",
            source_detail="No key found in any source",
            sources_attempted=sources,
            non_interactive=self.non_interactive,
            confidence="low",
        )
        return self._cache_result(provider, result)

    def _cache_result(self, provider: str, result: APIKeyResult) -> APIKeyResult:
        """Cache result for future lookups."""
        self._cache[provider] = result
        return result

    def _preview_key(self, key: str) -> str:
        """Get safe preview of key (first few chars)."""
        if not key:
            return "(empty)"
        if len(key) <= 8:
            return f"{key[:4]}..."
        return f"{key[:8]}..."

    def clear_cache(self) -> None:
        """Clear cached results."""
        self._cache.clear()


def get_api_key_with_resolution(
    provider: str,
    api_key: Optional[str] = None,
    non_interactive: Optional[bool] = None,
    raise_on_not_found: bool = True,
) -> Optional[str]:
    """
    Convenience function to get API key with full resolution.

    Args:
        provider: Provider name
        api_key: Explicit key (highest priority)
        non_interactive: Force non-interactive mode
        raise_on_not_found: Raise exception instead of returning None

    Returns:
        API key string or None

    Raises:
        APIKeyNotFoundError: If key not found and raise_on_not_found=True
    """
    resolver = UnifiedApiKeyResolver(non_interactive=non_interactive)
    result = resolver.get_api_key(provider, explicit_key=api_key)

    if result.key is None and raise_on_not_found:
        raise APIKeyNotFoundError(
            provider=provider,
            sources_attempted=result.sources_attempted,
            non_interactive=result.non_interactive,
        )

    return result.key
