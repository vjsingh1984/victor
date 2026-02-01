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

"""Shared HTTP error handling for LLM providers.

This module provides a unified error handling mixin for all providers,
extracting common error detection and conversion logic.
"""

import logging
from abc import ABC
from typing import Optional

from victor.core.errors import (
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

logger = logging.getLogger(__name__)


# HTTP status codes commonly mapped to specific errors
AUTH_STATUS_CODES = {401, 403}
RATE_LIMIT_STATUS_CODES = {429}
TIMEOUT_STATUS_CODES = {408, 504}
CONNECTION_STATUS_CODES = {502, 503, 521, 522, 523, 524, 599}

# Error message patterns for detection
AUTH_PATTERNS = [
    "authentication",
    "unauthorized",
    "invalid api key",
    "api_key",
    "apikey",
    "bearer token",
    "credentials",
    "forbidden",
]

RATE_LIMIT_PATTERNS = [
    "rate limit",
    "rate_limit",
    "ratelimit",
    "too many requests",
    "quota exceeded",
    "429",
]

TIMEOUT_PATTERNS = [
    "timeout",
    "timed out",
    "request timed",
    "connection timed",
]

CONNECTION_PATTERNS = [
    "connection",
    "network",
    "dns",
    "hostname",
    "unreachable",
    "refused",
    "reset",
]


class HTTPErrorHandlerMixin(ABC):
    """Mixin providing unified HTTP error handling for providers.

    This mixin extracts common error handling patterns from providers like:
    - AnthropicProvider
    - OpenAIProvider
    - MoonshotProvider
    - GroqProvider
    - DeepSeekProvider

    Usage:
        class MyProvider(BaseProvider, HTTPErrorHandlerMixin):
            async def chat(self, messages, **kwargs):
                try:
                    # ... API call ...
                except httpx.HTTPStatusError as e:
                    raise self._handle_http_error(e, "myprovider")
                except Exception as e:
                    raise self._handle_error(e, "myprovider")
    """

    @staticmethod
    def _format_provider_name(provider_name: str) -> str:
        """Format provider name for display, preserving known acronyms.

        Args:
            provider_name: Lowercase provider name (e.g., "openai", "anthropic")

        Returns:
            Properly formatted provider name (e.g., "OpenAI", "Anthropic")
        """
        # Known provider name mappings
        name_mappings = {
            "anthropic": "Anthropic",
            "google": "Google",
            "vertex": "Vertex AI",
            "azure": "Azure OpenAI",
            "azure_openai": "Azure OpenAI",
            "groq": "Groq",
            "deepseek": "DeepSeek",
            "together": "Together AI",
            "moonshot": "Moonshot",
            "mistral": "Mistral",
            "cohere": "Cohere",
            "xai": "xAI",
            "zai": "Z AI",
        }
        return name_mappings.get(provider_name, provider_name.capitalize())

    @staticmethod
    def _append_status_code(message: str, status_code: Optional[int]) -> str:
        """Append HTTP status code to message when available."""
        if status_code is None:
            return message
        status_str = str(status_code)
        if status_str in message:
            return message
        return f"{message} (HTTP {status_code})"

    def _handle_http_error(
        self,
        error: Exception,
        provider_name: str,
    ) -> ProviderError:
        """Handle HTTP-specific errors with proper categorization.

        Args:
            error: The HTTP exception (httpx.HTTPStatusError, etc.)
            provider_name: Name of the provider for error messages

        Returns:
            ProviderError: Appropriate error subclass

        Raises:
            ProviderError: Always raises after converting
        """
        error_msg = str(error).lower()
        status_code: Optional[int] = None

        # Extract status code if available
        if hasattr(error, "response"):
            response = getattr(error, "response", None)
            if response and hasattr(response, "status_code"):
                status_code = response.status_code

        # Try to get error body for more context
        error_body = ""
        if hasattr(error, "response"):
            response = getattr(error, "response", None)
            if response:
                # Try text first
                if hasattr(response, "text"):
                    try:
                        error_body = response.text[:500]
                    except Exception:
                        pass

                # Also try JSON for more detailed error info
                # (check even if text exists, as JSON might have better structure)
                if hasattr(response, "json"):
                    try:
                        error_data = response.json()
                        if isinstance(error_data, dict):
                            # Try common error fields
                            json_message = error_data.get("error", {}).get(
                                "message", ""
                            ) or error_data.get("message", "")
                            # Use JSON message if available, otherwise use text
                            if json_message:
                                error_body = json_message
                            elif not error_body:
                                # Fall back to stringified JSON if no text and no message
                                error_body = str(error_data)[:500]
                    except Exception:
                        pass

        # Format provider name for display
        display_name = self._format_provider_name(provider_name)

        # Categorize by status code first (more reliable)
        if status_code:
            if status_code in AUTH_STATUS_CODES:
                message = self._append_status_code(
                    f"Authentication failed for {display_name}: {error_body or error}",
                    status_code,
                )
                return ProviderAuthError(
                    message=message,
                    provider=provider_name,
                    status_code=status_code,
                    raw_error=error,
                )

            if status_code in RATE_LIMIT_STATUS_CODES:
                # Try to extract retry_after from response
                retry_after = self._extract_retry_after(error)
                message = self._append_status_code(
                    f"Rate limit exceeded for {display_name}: {error_body or error}",
                    status_code,
                )
                return ProviderRateLimitError(
                    message=message,
                    provider=provider_name,
                    status_code=status_code,
                    retry_after=retry_after,
                    raw_error=error,
                )

            if status_code in TIMEOUT_STATUS_CODES:
                timeout = self._extract_timeout_value(error)
                message = self._append_status_code(
                    f"Request timed out for {display_name}: {error_body or error}",
                    status_code,
                )
                return ProviderTimeoutError(
                    message=message,
                    provider=provider_name,
                    status_code=status_code,
                    timeout=timeout,
                    raw_error=error,
                )

            if status_code in CONNECTION_STATUS_CODES:
                message = self._append_status_code(
                    f"Connection error for {display_name}: {error_body or error}",
                    status_code,
                )
                return ProviderConnectionError(
                    message=message,
                    provider=provider_name,
                    status_code=status_code,
                    raw_error=error,
                )

        # Fallback to message pattern matching
        combined_msg = f"{error_msg} {error_body}".lower()

        if self._matches_any_pattern(combined_msg, AUTH_PATTERNS):
            message = self._append_status_code(
                f"Authentication failed for {display_name}: {error_body or error}",
                status_code,
            )
            return ProviderAuthError(
                message=message,
                provider=provider_name,
                status_code=status_code,
                raw_error=error,
            )

        if self._matches_any_pattern(combined_msg, RATE_LIMIT_PATTERNS):
            retry_after = self._extract_retry_after(error)
            message = self._append_status_code(
                f"Rate limit exceeded for {display_name}: {error_body or error}",
                status_code,
            )
            return ProviderRateLimitError(
                message=message,
                provider=provider_name,
                status_code=status_code,
                retry_after=retry_after,
                raw_error=error,
            )

        if self._matches_any_pattern(combined_msg, TIMEOUT_PATTERNS):
            timeout = self._extract_timeout_value(error)
            message = self._append_status_code(
                f"Request timed out for {display_name}: {error_body or error}",
                status_code,
            )
            return ProviderTimeoutError(
                message=message,
                provider=provider_name,
                status_code=status_code,
                timeout=timeout,
                raw_error=error,
            )

        if self._matches_any_pattern(combined_msg, CONNECTION_PATTERNS):
            message = self._append_status_code(
                f"Connection error for {display_name}: {error_body or error}",
                status_code,
            )
            return ProviderConnectionError(
                message=message,
                provider=provider_name,
                status_code=status_code,
                raw_error=error,
            )

        # Generic provider error
        message = self._append_status_code(
            f"{display_name} API error: {error_body or error}",
            status_code,
        )
        return ProviderError(
            message=message,
            provider=provider_name,
            status_code=status_code,
            raw_error=error,
        )

    def _handle_error(
        self,
        error: Exception,
        provider_name: str,
    ) -> ProviderError:
        """Handle generic errors with pattern-based categorization.

        This is a catch-all for errors that aren't HTTPStatusError.
        It uses message patterns to categorize the error.

        Args:
            error: The exception to handle
            provider_name: Name of the provider for error messages

        Returns:
            ProviderError: Appropriate error subclass

        Raises:
            ProviderError: Always raises after converting
        """
        error_msg = str(error).lower()
        display_name = self._format_provider_name(provider_name)

        # Check for timeout
        if self._matches_any_pattern(error_msg, TIMEOUT_PATTERNS):
            timeout = self._extract_timeout_value(error)
            return ProviderTimeoutError(
                message=f"Request timed out for {display_name}: {error}",
                provider=provider_name,
                timeout=timeout,
                raw_error=error,
            )

        # Check for auth errors
        if self._matches_any_pattern(error_msg, AUTH_PATTERNS):
            return ProviderAuthError(
                message=f"Authentication failed for {display_name}: {error}",
                provider=provider_name,
                raw_error=error,
            )

        # Check for rate limit
        if self._matches_any_pattern(error_msg, RATE_LIMIT_PATTERNS):
            message = self._append_status_code(
                f"Rate limit exceeded for {display_name}: {error}",
                429,
            )
            return ProviderRateLimitError(
                message=message,
                provider=provider_name,
                status_code=429,
                raw_error=error,
            )

        # Check for connection errors
        if self._matches_any_pattern(error_msg, CONNECTION_PATTERNS):
            return ProviderConnectionError(
                message=f"Connection error for {display_name}: {error}",
                provider=provider_name,
                raw_error=error,
            )

        # Generic provider error
        message = self._append_status_code(
            f"{display_name} API error: {error}",
            None,
        )
        return ProviderError(
            message=message,
            provider=provider_name,
            raw_error=error,
        )

    @staticmethod
    def _matches_any_pattern(text: str, patterns: list[str]) -> bool:
        """Check if text matches any of the given patterns.

        Args:
            text: Text to search in (case-insensitive)
            patterns: List of patterns to search for

        Returns:
            True if any pattern is found in text
        """
        text_lower = text.lower()
        return any(pattern.lower() in text_lower for pattern in patterns)

    @staticmethod
    def _extract_retry_after(error: Exception) -> Optional[int]:
        """Extract retry-after value from error response.

        Args:
            error: The HTTP error exception

        Returns:
            Retry-after seconds, or None if not found
        """
        if hasattr(error, "response"):
            response = getattr(error, "response", None)
            if response:
                # Try headers first
                if hasattr(response, "headers"):
                    retry_after = response.headers.get("retry-after")
                    if retry_after:
                        try:
                            return int(retry_after)
                        except ValueError:
                            pass

                # Try JSON body
                if hasattr(response, "json"):
                    try:
                        error_data = response.json()
                        if isinstance(error_data, dict):
                            retry_after = error_data.get("error", {}).get(
                                "retry_after"
                            ) or error_data.get("retry_after")
                            if retry_after:
                                return int(retry_after)
                    except Exception:
                        pass

        return None

    @staticmethod
    def _extract_timeout_value(error: Exception) -> Optional[int]:
        """Extract timeout value from error.

        Args:
            error: The error exception

        Returns:
            Timeout in seconds, or None if not found
        """
        # Try to extract from error message
        error_str = str(error)
        import re

        # Match "timeout" or "timed out" followed by optional non-digits and a number
        timeout_match = re.search(r"timed?\s*out?\D*(\d+)", error_str, re.IGNORECASE)
        if timeout_match:
            try:
                return int(timeout_match.group(1))
            except ValueError:
                pass

        # Check for timeout attribute
        if hasattr(error, "timeout"):
            timeout = error.timeout
            if isinstance(timeout, (int, float)):
                return int(timeout)

        return None


def handle_provider_error(
    error: Exception,
    provider_name: str,
) -> ProviderError:
    """Standalone function for handling provider errors.

    This is a convenience function that can be used without
    inheriting from HTTPErrorHandlerMixin.

    Args:
        error: The exception to handle
        provider_name: Name of the provider

    Returns:
        ProviderError: Appropriate error subclass

    Raises:
        ProviderError: Always raises after converting

    Example:
        try:
            response = await client.chat.completions.create(...)
        except Exception as e:
            raise handle_provider_error(e, "openai")
    """
    handler = HTTPErrorHandlerMixin()
    # Try HTTP-specific handling first
    if hasattr(error, "response") and hasattr(error, "status_code"):
        return handler._handle_http_error(error, provider_name)
    return handler._handle_error(error, provider_name)
