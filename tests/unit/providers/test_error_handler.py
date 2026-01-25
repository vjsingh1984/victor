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

"""Tests for HTTP error handler mixin."""

import pytest
from unittest.mock import MagicMock

from victor.providers.error_handler import (
    HTTPErrorHandlerMixin,
    handle_provider_error,
)
from victor.core.errors import (
    ProviderAuthError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)


# =============================================================================
# TEST HELPERS
# =============================================================================


class MockHTTPError(Exception):
    """Mock HTTP error for testing."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        response_text: str = "",
        headers: dict | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.text = response_text
        self.headers = headers or {}

        # Mock response object
        self.response = MagicMock()
        self.response.status_code = status_code
        self.response.text = response_text
        self.response.headers = headers or {}


class TestHandler(HTTPErrorHandlerMixin):
    """Concrete handler for testing the mixin."""

    pass


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def handler():
    """Create a test handler instance."""
    return TestHandler()


@pytest.fixture
def mock_response():
    """Create a mock response object."""
    response = MagicMock()
    response.status_code = 500
    response.text = "Internal server error"
    response.headers = {}
    response.json.return_value = {"error": {"message": "Error details"}}
    return response


# =============================================================================
# HTTP STATUS CODE TESTS
# =============================================================================


class TestHTTPStatusCodeHandling:
    """Tests for error handling based on HTTP status codes."""

    def test_401_status_code_returns_auth_error(self, handler):
        """Test that 401 status code is mapped to ProviderAuthError."""
        error = MockHTTPError("Unauthorized", status_code=401, response_text="Invalid API key")

        result = handler._handle_http_error(error, "testprovider")

        assert isinstance(result, ProviderAuthError)
        assert result.provider == "testprovider"
        assert result.status_code == 401

    def test_403_status_code_returns_auth_error(self, handler):
        """Test that 403 status code is mapped to ProviderAuthError."""
        error = MockHTTPError("Forbidden", status_code=403, response_text="Access denied")

        result = handler._handle_http_error(error, "testprovider")

        assert isinstance(result, ProviderAuthError)
        assert result.status_code == 403

    def test_429_status_code_returns_rate_limit_error(self, handler):
        """Test that 429 status code is mapped to ProviderRateLimitError."""
        error = MockHTTPError("Too many requests", status_code=429, response_text="Rate limit")

        result = handler._handle_http_error(error, "testprovider")

        assert isinstance(result, ProviderRateLimitError)
        assert result.status_code == 429

    def test_408_status_code_returns_timeout_error(self, handler):
        """Test that 408 status code is mapped to ProviderTimeoutError."""
        error = MockHTTPError("Request timeout", status_code=408)

        result = handler._handle_http_error(error, "testprovider")

        assert isinstance(result, ProviderTimeoutError)
        assert result.status_code == 408

    def test_504_status_code_returns_timeout_error(self, handler):
        """Test that 504 status code is mapped to ProviderTimeoutError."""
        error = MockHTTPError("Gateway timeout", status_code=504)

        result = handler._handle_http_error(error, "testprovider")

        assert isinstance(result, ProviderTimeoutError)
        assert result.status_code == 504

    def test_502_status_code_returns_connection_error(self, handler):
        """Test that 502 status code is mapped to ProviderConnectionError."""
        error = MockHTTPError("Bad gateway", status_code=502)

        result = handler._handle_http_error(error, "testprovider")

        assert isinstance(result, ProviderConnectionError)
        assert result.status_code == 502

    def test_503_status_code_returns_connection_error(self, handler):
        """Test that 503 status code is mapped to ProviderConnectionError."""
        error = MockHTTPError("Service unavailable", status_code=503)

        result = handler._handle_http_error(error, "testprovider")

        assert isinstance(result, ProviderConnectionError)
        assert result.status_code == 503


# =============================================================================
# ERROR MESSAGE PATTERN TESTS
# =============================================================================


class TestErrorMessagePatternMatching:
    """Tests for error categorization based on message patterns."""

    def test_authentication_pattern_in_message(self, handler):
        """Test that 'authentication' in message triggers ProviderAuthError."""
        error = Exception("authentication failed: invalid credentials")

        result = handler._handle_error(error, "testprovider")

        assert isinstance(result, ProviderAuthError)

    def test_api_key_pattern_in_message(self, handler):
        """Test that 'api_key' in message triggers ProviderAuthError."""
        error = Exception("Invalid api_key provided")

        result = handler._handle_error(error, "testprovider")

        assert isinstance(result, ProviderAuthError)

    def test_rate_limit_pattern_in_message(self, handler):
        """Test that 'rate limit' in message triggers ProviderRateLimitError."""
        error = Exception("rate limit exceeded, please retry later")

        result = handler._handle_error(error, "testprovider")

        assert isinstance(result, ProviderRateLimitError)

    def test_timeout_pattern_in_message(self, handler):
        """Test that 'timeout' in message triggers ProviderTimeoutError."""
        error = Exception("request timed out after 30 seconds")

        result = handler._handle_error(error, "testprovider")

        assert isinstance(result, ProviderTimeoutError)

    def test_connection_pattern_in_message(self, handler):
        """Test that 'connection' in message triggers ProviderConnectionError."""
        error = Exception("connection refused by remote host")

        result = handler._handle_error(error, "testprovider")

        assert isinstance(result, ProviderConnectionError)

    def test_generic_error_fallback(self, handler):
        """Test that unrecognized errors become generic ProviderError."""
        error = Exception("something unexpected happened")

        result = handler._handle_error(error, "testprovider")

        assert isinstance(result, ProviderError)
        assert not isinstance(result, (ProviderAuthError, ProviderRateLimitError))

    def test_case_insensitive_pattern_matching(self, handler):
        """Test that pattern matching is case-insensitive."""
        error = Exception("AUTHENTICATION FAILED")

        result = handler._handle_error(error, "testprovider")

        assert isinstance(result, ProviderAuthError)


# =============================================================================
# RETRY-AFTER EXTRACTION TESTS
# =============================================================================


class TestRetryAfterExtraction:
    """Tests for extracting retry-after values from errors."""

    def test_extract_retry_after_from_header(self, handler):
        """Test extracting retry-after from response headers."""
        error = MockHTTPError(
            "Rate limited",
            status_code=429,
            headers={"retry-after": "60"},
        )

        result = handler._handle_http_error(error, "testprovider")

        assert isinstance(result, ProviderRateLimitError)
        assert result.retry_after == 60

    def test_extract_retry_after_from_json_body(self, handler):
        """Test extracting retry-after from JSON response body."""
        error = MockHTTPError("Rate limited", status_code=429)
        error.response.json.return_value = {"error": {"retry_after": "120"}}

        result = handler._handle_http_error(error, "testprovider")

        assert isinstance(result, ProviderRateLimitError)
        assert result.retry_after == 120

    def test_no_retry_after_returns_none(self, handler):
        """Test that missing retry-after results in None."""
        error = MockHTTPError("Rate limited", status_code=429)
        error.response.json.return_value = {"error": {"message": "Too many requests"}}

        result = handler._handle_http_error(error, "testprovider")

        assert isinstance(result, ProviderRateLimitError)
        assert result.retry_after is None


# =============================================================================
# TIMEOUT EXTRACTION TESTS
# =============================================================================


class TestTimeoutExtraction:
    """Tests for extracting timeout values from errors."""

    def test_extract_timeout_from_message(self, handler):
        """Test extracting timeout value from error message."""
        error = Exception("Request timed out after 45 seconds")

        result = handler._handle_error(error, "testprovider")

        assert isinstance(result, ProviderTimeoutError)
        assert result.timeout == 45

    def test_timeout_attribute_extraction(self, handler):
        """Test extracting timeout from error attribute."""
        error = Exception("Timeout")
        error.timeout = 30

        result = handler._handle_error(error, "testprovider")

        assert isinstance(result, ProviderTimeoutError)
        assert result.timeout == 30

    def test_no_timeout_returns_none(self, handler):
        """Test that missing timeout results in None."""
        error = Exception("Request timed out")

        result = handler._handle_error(error, "testprovider")

        assert isinstance(result, ProviderTimeoutError)
        assert result.timeout is None


# =============================================================================
# ERROR BODY EXTRACTION TESTS
# =============================================================================


class TestErrorBodyExtraction:
    """Tests for extracting error details from HTTP responses."""

    def test_extract_text_from_response(self, handler):
        """Test extracting error text from response."""
        error = MockHTTPError(
            "Bad request",
            status_code=400,
            response_text="Invalid parameter: 'model'",
        )

        result = handler._handle_http_error(error, "testprovider")

        assert "Invalid parameter" in result.message

    def test_extract_json_error_message(self, handler):
        """Test extracting error message from JSON response."""
        error = MockHTTPError("Bad request", status_code=400)
        error.response.json.return_value = {"error": {"message": "Invalid API key format"}}

        result = handler._handle_http_error(error, "testprovider")

        assert "Invalid API key format" in result.message

    def test_extract_top_level_message(self, handler):
        """Test extracting top-level message from JSON response."""
        error = MockHTTPError("Bad request", status_code=400)
        error.response.json.return_value = {"message": "Missing required field"}

        result = handler._handle_http_error(error, "testprovider")

        assert "Missing required field" in result.message


# =============================================================================
# PROVIDER NAME TESTS
# =============================================================================


class TestProviderNameHandling:
    """Tests for provider name handling in errors."""

    def test_provider_name_in_error_message(self, handler):
        """Test that provider name is included in error message."""
        error = Exception("Something went wrong")

        result = handler._handle_error(error, "myprovider")

        assert "myprovider" in result.message.lower()

    def test_provider_name_capitalized(self, handler):
        """Test that provider name is capitalized in message."""
        error = Exception("Error")

        result = handler._handle_error(error, "openai")

        assert "Openai" in result.message


# =============================================================================
# STANDALONE FUNCTION TESTS
# =============================================================================


class TestStandaloneFunction:
    """Tests for the standalone handle_provider_error function."""

    def test_handle_provider_error_with_http_error(self):
        """Test standalone function with HTTP error."""
        error = MockHTTPError("Unauthorized", status_code=401)

        result = handle_provider_error(error, "testprovider")

        assert isinstance(result, ProviderAuthError)

    def test_handle_provider_error_with_generic_error(self):
        """Test standalone function with generic error."""
        error = Exception("rate limit exceeded")

        result = handle_provider_error(error, "testprovider")

        assert isinstance(result, ProviderRateLimitError)

    def test_handle_provider_error_without_response_attr(self):
        """Test standalone function with error lacking response attribute."""
        error = ValueError("Invalid parameter")

        result = handle_provider_error(error, "testprovider")

        assert isinstance(result, ProviderError)


# =============================================================================
# PATTERN MATCHING UTILITIES
# =============================================================================


class TestPatternMatchingUtilities:
    """Tests for pattern matching utility methods."""

    def test_matches_any_pattern_positive(self, handler):
        """Test _matches_any_pattern with matching pattern."""
        assert handler._matches_any_pattern("authentication failed", ["auth", "token"])

    def test_matches_any_pattern_negative(self, handler):
        """Test _matches_any_pattern with no matching patterns."""
        assert not handler._matches_any_pattern("success", ["error", "failed"])

    def test_matches_any_pattern_case_insensitive(self, handler):
        """Test _matches_any_pattern is case-insensitive."""
        assert handler._matches_any_pattern("AUTHENTICATION", ["authentication"])


# =============================================================================
# RAW ERROR PRESERVATION TESTS
# =============================================================================


class TestRawErrorPreservation:
    """Tests that original exceptions are preserved."""

    def test_raw_error_preserved_in_auth_error(self, handler):
        """Test that original error is preserved in ProviderAuthError."""
        original = Exception("authentication failed")

        result = handler._handle_error(original, "testprovider")

        assert result.raw_error is original

    def test_raw_error_preserved_in_rate_limit_error(self, handler):
        """Test that original error is preserved in ProviderRateLimitError."""
        original = MockHTTPError("Rate limited", status_code=429)

        result = handler._handle_http_error(original, "testprovider")

        assert result.raw_error is original


# =============================================================================
# ERROR DETAILS TESTS
# =============================================================================


class TestErrorDetails:
    """Tests for error details structure."""

    def test_error_includes_provider_details(self, handler):
        """Test that provider name is in error details."""
        error = Exception("Error occurred")

        result = handler._handle_error(error, "myprovider")

        assert result.provider == "myprovider"

    def test_error_includes_status_code(self, handler):
        """Test that status code is included when available."""
        error = MockHTTPError("Unauthorized", status_code=401)

        result = handler._handle_http_error(error, "testprovider")

        assert result.status_code == 401

    def test_error_with_no_status_code(self, handler):
        """Test error handling without status code."""
        error = Exception("Generic error")

        result = handler._handle_error(error, "testprovider")

        assert result.status_code is None
