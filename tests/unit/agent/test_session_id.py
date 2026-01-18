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

"""Tests for session ID generation and validation with improved error messages."""

import pytest

from victor.agent.session_id import (
    encode_base62,
    decode_base62,
    get_project_root_hash,
    generate_session_id,
    parse_session_id,
    validate_session_id,
)
from pathlib import Path


class TestBase62Encoding:
    """Tests for base62 encoding/decoding."""

    def test_encode_zero(self):
        """Test encoding zero."""
        assert encode_base62(0) == "0"

    def test_encode_positive_integer(self):
        """Test encoding positive integers."""
        assert encode_base62(123456789) == "8M0kX"

    def test_decode_roundtrip(self):
        """Test roundtrip encoding/decoding."""
        original = 1234567890
        encoded = encode_base62(original)
        decoded = decode_base62(encoded)
        assert decoded == original


class TestProjectRootHash:
    """Tests for project root hashing."""

    def test_long_alphanumeric_name(self):
        """Test hashing long alphanumeric directory names."""
        path = Path("/home/user/myproject")
        hash_val = get_project_root_hash(path)
        assert len(hash_val) == 6
        assert hash_val == "myproj"

    def test_short_name(self):
        """Test hashing short directory names."""
        path = Path("/home/user/abc")
        hash_val = get_project_root_hash(path)
        assert len(hash_val) == 6

    def test_special_characters(self):
        """Test hashing names with special characters."""
        path = Path("/home/user/victor-ai")
        hash_val = get_project_root_hash(path)
        assert len(hash_val) == 6


class TestGenerateSessionId:
    """Tests for session ID generation."""

    def test_generate_session_id_format(self):
        """Test that generated session IDs have correct format."""
        session_id = generate_session_id()
        parts = session_id.split("-")
        assert len(parts) == 2
        assert len(parts[0]) == 6  # project root hash
        assert len(parts[1]) > 0  # base62 timestamp

    def test_generate_session_id_with_project_root(self):
        """Test generating session ID with explicit project root."""
        project_root = Path("/home/user/testproject")
        session_id = generate_session_id(project_root)
        parts = session_id.split("-")
        assert len(parts) == 2


class TestParseSessionId:
    """Tests for session ID parsing with improved error messages."""

    def test_parse_valid_session_id(self):
        """Test parsing a valid session ID."""
        session_id = "myproj-9Kx7Z2"
        result = parse_session_id(session_id)

        assert result["project_root"] == "myproj"
        assert result["base62_timestamp"] == "9Kx7Z2"
        assert "timestamp_ms" in result
        assert isinstance(result["timestamp_ms"], int)
        assert "timestamp_iso" in result

    def test_parse_invalid_type(self):
        """Test parsing non-string session ID with detailed error."""
        with pytest.raises(ValueError) as exc_info:
            parse_session_id(12345)

        error_message = str(exc_info.value)
        # Check for improved error message components
        assert "Invalid session ID format" in error_message
        assert "expected string" in error_message
        assert "Expected format:" in error_message
        assert "Example:" in error_message
        assert "Correlation ID:" in error_message

    def test_parse_missing_parts(self):
        """Test parsing session ID with missing dash."""
        with pytest.raises(ValueError) as exc_info:
            parse_session_id("invalid")

        error_message = str(exc_info.value)
        # Check for improved error message
        assert "Invalid session ID format" in error_message
        assert "Expected format:" in error_message
        assert "Examples:" in error_message
        assert "Your session ID has" in error_message
        assert "expected 2" in error_message
        assert "Recovery:" in error_message
        assert "Correlation ID:" in error_message

    def test_parse_too_many_parts(self):
        """Test parsing session ID with too many parts."""
        with pytest.raises(ValueError) as exc_info:
            parse_session_id("a-b-c-d")

        error_message = str(exc_info.value)
        # Check for improved error message
        assert "Invalid session ID format" in error_message
        assert "Your session ID has 4 parts" in error_message
        assert "expected 2" in error_message
        assert "Recovery:" in error_message

    def test_parse_invalid_base62_characters(self):
        """Test parsing session ID with invalid base62 characters."""
        with pytest.raises(ValueError) as exc_info:
            parse_session_id("myproj-@#$%")

        error_message = str(exc_info.value)
        # Check for improved error message
        assert "Invalid base62 timestamp" in error_message
        assert "Base62 timestamp should only contain:" in error_message
        assert "0-9, A-Z, a-z" in error_message
        assert "Correlation ID:" in error_message

    def test_parse_empty_timestamp(self):
        """Test parsing session ID with empty timestamp."""
        with pytest.raises(ValueError) as exc_info:
            parse_session_id("myproj-")

        error_message = str(exc_info.value)
        # Check for improved error message
        assert "Invalid base62 timestamp" in error_message
        assert "Correlation ID:" in error_message


class TestValidateSessionId:
    """Tests for session ID validation."""

    def test_validate_valid_session_id(self):
        """Test validating a valid session ID."""
        assert validate_session_id("myproj-9Kx7Z2") is True

    def test_validate_invalid_session_id(self):
        """Test validating an invalid session ID."""
        assert validate_session_id("invalid") is False

    def test_validate_missing_dash(self):
        """Test validating session ID without dash."""
        assert validate_session_id("invalidid") is False

    def test_validate_empty_string(self):
        """Test validating empty string."""
        assert validate_session_id("") is False
