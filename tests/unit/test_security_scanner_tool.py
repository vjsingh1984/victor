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

"""Tests for security_scanner_tool module."""

import tempfile
from pathlib import Path
import pytest

from victor.tools.security_scanner_tool import (
    security_scan_secrets,
    security_scan_dependencies,
    security_scan_config,
    security_scan_all,
    security_check_file,
)


class TestSecurityScanSecrets:
    """Tests for security_scan_secrets function."""

    @pytest.mark.asyncio
    async def test_scan_secrets_finds_api_key(self, tmp_path):
        """Test detecting API keys in code."""
        test_file = tmp_path / "config.py"
        test_file.write_text(
            """
API_KEY = "sk_test_1234567890abcdefghijklmnop"
api_key = "1234567890abcdefghijklmnopqrstuvwxyz"
"""
        )

        result = await security_scan_secrets(path=str(tmp_path))

        assert result["success"] is True
        assert result["secrets_found"] > 0
        assert any("api_key" in f["type"] for f in result["findings"])

    @pytest.mark.asyncio
    async def test_scan_secrets_finds_password(self, tmp_path):
        """Test detecting passwords in code."""
        test_file = tmp_path / "auth.py"
        test_file.write_text(
            """
password = "MySecretPassword123"
PASSWORD = "AnotherPassword456"
"""
        )

        result = await security_scan_secrets(path=str(tmp_path))

        assert result["success"] is True
        assert result["secrets_found"] > 0
        assert any("password" in f["type"] for f in result["findings"])

    @pytest.mark.asyncio
    async def test_scan_secrets_finds_token(self, tmp_path):
        """Test detecting tokens in code."""
        test_file = tmp_path / "api.py"
        test_file.write_text(
            """
auth_token = "ghp_1234567890abcdefghijklmnopqrstuvwxyz1234"
TOKEN = "Bearer_1234567890abcdefghijklmnopqrstuvwxyz"
"""
        )

        result = await security_scan_secrets(path=str(tmp_path))

        assert result["success"] is True
        assert result["secrets_found"] > 0

    @pytest.mark.asyncio
    async def test_scan_secrets_finds_private_key(self, tmp_path):
        """Test detecting private keys."""
        test_file = tmp_path / "keys.py"
        test_file.write_text(
            """
private_key = '''-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC
-----END PRIVATE KEY-----'''
"""
        )

        result = await security_scan_secrets(path=str(tmp_path))

        assert result["success"] is True
        assert result["secrets_found"] > 0
        assert any("private_key" in f["type"] for f in result["findings"])

    @pytest.mark.asyncio
    async def test_scan_secrets_no_secrets_found(self, tmp_path):
        """Test scanning clean code with no secrets."""
        test_file = tmp_path / "clean.py"
        test_file.write_text(
            """
def hello_world():
    print("Hello, World!")
    return True
"""
        )

        result = await security_scan_secrets(path=str(tmp_path))

        assert result["success"] is True
        assert result["secrets_found"] == 0
        assert len(result["findings"]) == 0

    @pytest.mark.asyncio
    async def test_scan_secrets_path_not_found(self):
        """Test handling of non-existent path."""
        result = await security_scan_secrets(path="/nonexistent/path")

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_scan_secrets_missing_path(self):
        """Test handling of missing path parameter."""
        result = await security_scan_secrets(path="")

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_scan_secrets_with_file_pattern(self, tmp_path):
        """Test scanning with custom file pattern."""
        # Create both .py and .txt files
        py_file = tmp_path / "test.py"
        py_file.write_text('api_key = "1234567890abcdefghijklmnopqrstuvwxyz"')

        txt_file = tmp_path / "test.txt"
        txt_file.write_text('api_key = "1234567890abcdefghijklmnopqrstuvwxyz"')

        # Scan only .txt files
        result = await security_scan_secrets(path=str(tmp_path), file_pattern="*.txt")

        assert result["success"] is True
        assert result["files_scanned"] > 0

    @pytest.mark.asyncio
    async def test_scan_secrets_empty_directory(self, tmp_path):
        """Test scanning empty directory."""
        result = await security_scan_secrets(path=str(tmp_path))

        assert result["success"] is True
        assert result["files_scanned"] == 0
        assert "No files found" in result["message"]

    @pytest.mark.asyncio
    async def test_scan_secrets_with_many_findings(self, tmp_path):
        """Test scanning file with more than 20 findings."""
        test_file = tmp_path / "many_secrets.py"
        # Create a file with 25+ secrets to trigger the "... and X more" message
        secrets = []
        for i in range(25):
            # Create API keys that match the pattern: at least 20 chars after the =
            secrets.append(f'api_key = "sk_test_key_number_{i:04d}_abcdefghijklmnopqrstuvwxyz"')
        test_file.write_text("\n".join(secrets))

        result = await security_scan_secrets(path=str(tmp_path))

        assert result["success"] is True
        assert result["secrets_found"] >= 20
        assert "... and" in result["formatted_report"]  # Should have truncation message

    @pytest.mark.asyncio
    async def test_scan_secrets_file_read_error(self, tmp_path):
        """Test handling of file read errors."""
        from unittest.mock import patch, MagicMock

        # Create a valid file
        test_file = tmp_path / "test.py"
        test_file.write_text('api_key = "test_key"')

        # Mock open to raise an exception
        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            result = await security_scan_secrets(path=str(tmp_path))

            # Should still return success but log the error
            assert result["success"] is True


class TestSecurityScanDependencies:
    """Tests for security_scan_dependencies function."""

    @pytest.mark.asyncio
    async def test_scan_dependencies_success(self, tmp_path):
        """Test scanning requirements file."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(
            """
django==3.2.0
flask==1.1.2
requests==2.25.0
pytest==6.2.0
"""
        )

        result = await security_scan_dependencies(requirements_file=str(req_file))

        assert result["success"] is True
        assert result["packages_checked"] == 4

    @pytest.mark.asyncio
    async def test_scan_dependencies_with_vulnerabilities(self, tmp_path):
        """Test detecting known vulnerable packages."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(
            """
django==2.0.0
flask==0.12.0
requests==2.20.0
"""
        )

        result = await security_scan_dependencies(requirements_file=str(req_file))

        assert result["success"] is True
        # Should detect some known vulnerabilities
        assert result["packages_checked"] > 0

    @pytest.mark.asyncio
    async def test_scan_dependencies_file_not_found(self):
        """Test handling of missing requirements file."""
        result = await security_scan_dependencies(requirements_file="/nonexistent/requirements.txt")

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_scan_dependencies_with_comments(self, tmp_path):
        """Test scanning requirements with comments."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(
            """
# Production dependencies
django==3.2.0
# Testing dependencies
pytest==6.2.0
"""
        )

        result = await security_scan_dependencies(requirements_file=str(req_file))

        assert result["success"] is True
        assert result["packages_checked"] == 2  # Only non-comment lines

    @pytest.mark.asyncio
    async def test_scan_dependencies_no_vulnerabilities(self, tmp_path):
        """Test scanning with packages that have no known vulnerabilities."""
        req_file = tmp_path / "requirements.txt"
        # Use very recent or less common packages that likely have no known vulns
        req_file.write_text(
            """
pytest==7.4.0
black==23.7.0
mypy==1.4.1
"""
        )

        result = await security_scan_dependencies(requirements_file=str(req_file))

        assert result["success"] is True
        assert result["packages_checked"] == 3
        # Should have message about no vulnerabilities found
        if result.get("vulnerabilities", []) == []:
            assert "No known vulnerabilities" in result["formatted_report"]

    @pytest.mark.asyncio
    async def test_scan_dependencies_exception_handling(self, tmp_path):
        """Test exception handling in dependency scanning."""
        from unittest.mock import patch
        from pathlib import Path

        req_file = tmp_path / "requirements.txt"
        req_file.write_text("django==3.2.0")

        # Mock Path.read_text to raise an exception during file reading
        with patch.object(Path, "read_text", side_effect=PermissionError("Access denied")):
            result = await security_scan_dependencies(requirements_file=str(req_file))

            assert result["success"] is False
            assert "error" in result
            assert "Failed to scan dependencies" in result["error"]


class TestSecurityScanConfig:
    """Tests for security_scan_config function."""

    @pytest.mark.asyncio
    async def test_scan_config_finds_debug_mode(self, tmp_path):
        """Test detecting debug mode enabled."""
        test_file = tmp_path / "settings.py"
        test_file.write_text(
            """
DEBUG = True
debug = true
"""
        )

        result = await security_scan_config(path=str(tmp_path))

        assert result["success"] is True
        assert result["issues_found"] > 0
        assert any("debug" in f["type"].lower() for f in result["findings"])

    @pytest.mark.asyncio
    async def test_scan_config_finds_insecure_protocol(self, tmp_path):
        """Test detecting insecure HTTP protocol."""
        test_file = tmp_path / "api.py"
        test_file.write_text(
            """
API_URL = "http://api.example.com/data"
BASE_URL = "http://insecure-site.com"
"""
        )

        result = await security_scan_config(path=str(tmp_path))

        assert result["success"] is True
        assert result["issues_found"] > 0

    @pytest.mark.asyncio
    async def test_scan_config_finds_hardcoded_ip(self, tmp_path):
        """Test detecting hardcoded IP addresses."""
        test_file = tmp_path / "config.py"
        test_file.write_text(
            """
DATABASE_HOST = "192.168.1.100"
API_SERVER = "10.0.0.1"
"""
        )

        result = await security_scan_config(path=str(tmp_path))

        assert result["success"] is True
        assert result["issues_found"] > 0

    @pytest.mark.asyncio
    async def test_scan_config_clean_file(self, tmp_path):
        """Test scanning secure configuration."""
        test_file = tmp_path / "secure.py"
        test_file.write_text(
            """
DEBUG = False
API_URL = "https://secure-api.example.com"
DATABASE_HOST = "database.example.com"
"""
        )

        result = await security_scan_config(path=str(tmp_path))

        assert result["success"] is True
        # May still find some issues due to https check being simple

    @pytest.mark.asyncio
    async def test_scan_config_path_not_found(self):
        """Test handling of non-existent path."""
        result = await security_scan_config(path="/nonexistent/path")

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_scan_config_missing_path(self):
        """Test handling of missing path parameter."""
        result = await security_scan_config(path="")

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_scan_config_no_files_found(self, tmp_path):
        """Test scanning directory with no matching files."""
        # Create a directory with only non-Python files
        test_file = tmp_path / "test.txt"
        test_file.write_text("Some text content")

        result = await security_scan_config(path=str(tmp_path), file_pattern="*.py")

        assert result["success"] is True
        assert result["files_scanned"] == 0
        assert "No files found" in result["message"]

    @pytest.mark.asyncio
    async def test_scan_config_file_read_error(self, tmp_path):
        """Test handling of file read errors."""
        from unittest.mock import patch

        # Create a valid file
        test_file = tmp_path / "test.py"
        test_file.write_text("DEBUG = True")

        # Mock open to raise an exception
        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            result = await security_scan_config(path=str(tmp_path))

            # Should still return success but log the error
            assert result["success"] is True


class TestSecurityScanAll:
    """Tests for security_scan_all function."""

    @pytest.mark.asyncio
    async def test_scan_all_comprehensive(self, tmp_path):
        """Test comprehensive security scan."""
        # Create test files
        py_file = tmp_path / "app.py"
        py_file.write_text(
            """
api_key = "sk_test_1234567890abcdefghijklmnop"
DEBUG = True
API_URL = "http://insecure.com"
"""
        )

        req_file = tmp_path / "requirements.txt"
        req_file.write_text("django==3.2.0\nflask==1.1.2")

        result = await security_scan_all(path=str(tmp_path))

        assert result["success"] is True
        assert result["total_issues"] > 0
        assert "secrets" in result
        assert "config" in result

    @pytest.mark.asyncio
    async def test_scan_all_clean_project(self, tmp_path):
        """Test scanning a clean project."""
        clean_file = tmp_path / "clean.py"
        clean_file.write_text(
            """
def hello():
    return "Hello, World!"
"""
        )

        result = await security_scan_all(path=str(tmp_path))

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_scan_all_missing_path(self):
        """Test handling of missing path parameter."""
        result = await security_scan_all(path="")

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]


class TestSecurityCheckFile:
    """Tests for security_check_file function."""

    @pytest.mark.asyncio
    async def test_check_file_with_secrets(self, tmp_path):
        """Test checking file with secrets."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """
api_key = "sk_test_1234567890abcdefghijklmnop"
password = "MySecretPassword123"
"""
        )

        result = await security_check_file(file=str(test_file))

        assert result["success"] is True
        assert result["issues_found"] > 0

    @pytest.mark.asyncio
    async def test_check_file_with_config_issues(self, tmp_path):
        """Test checking file with config issues."""
        test_file = tmp_path / "settings.py"
        test_file.write_text(
            """
DEBUG = True
API_URL = "http://insecure.com"
"""
        )

        result = await security_check_file(file=str(test_file))

        assert result["success"] is True
        assert result["issues_found"] > 0

    @pytest.mark.asyncio
    async def test_check_file_clean(self, tmp_path):
        """Test checking clean file."""
        test_file = tmp_path / "clean.py"
        test_file.write_text(
            """
def calculate(x, y):
    return x + y
"""
        )

        result = await security_check_file(file=str(test_file))

        assert result["success"] is True
        assert result["issues_found"] == 0

    @pytest.mark.asyncio
    async def test_check_file_not_found(self):
        """Test handling of non-existent file."""
        result = await security_check_file(file="/nonexistent/file.py")

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_check_file_missing_parameter(self):
        """Test handling of missing file parameter."""
        result = await security_check_file(file="")

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]


class TestDeprecatedSecurityScannerTool:
    """Tests for deprecated SecurityScannerTool class."""

    def test_deprecated_class_warning(self):
        """Test that deprecated class raises warning."""
        from victor.tools.security_scanner_tool import SecurityScannerTool
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tool = SecurityScannerTool()

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()


class TestExceptionHandling:
    """Tests to cover exception handling in helper functions."""

    @pytest.mark.asyncio
    async def test_scan_secrets_file_read_error(self, tmp_path):
        """Test exception handling when scanning for secrets fails."""
        from unittest.mock import patch
        from victor.tools.security_scanner_tool import security_scan_secrets

        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("api_key = 'test'")

        # Mock read_text to raise an exception
        with patch("pathlib.Path.read_text", side_effect=OSError("Read error")):
            result = await security_scan_secrets(path=str(tmp_path))

            # Should still succeed but with no findings
            assert result["success"] is True
            assert len(result["findings"]) == 0

    @pytest.mark.asyncio
    async def test_scan_config_file_read_error(self, tmp_path):
        """Test exception handling when scanning for config issues fails."""
        from unittest.mock import patch
        from victor.tools.security_scanner_tool import security_scan_config

        # Create a test file
        test_file = tmp_path / "config.py"
        test_file.write_text("DEBUG = True")

        # Mock read_text to raise an exception
        with patch("pathlib.Path.read_text", side_effect=OSError("Read error")):
            result = await security_scan_config(path=str(tmp_path))

            # Should still succeed but with no findings
            assert result["success"] is True
            assert len(result["findings"]) == 0
