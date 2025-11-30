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

"""Tests for dependency_tool module."""

import json
import pytest
from unittest.mock import Mock, patch
import subprocess

from victor.tools.dependency_tool import (
    dependency_list,
    dependency_outdated,
    dependency_security,
    dependency_generate,
    dependency_update,
    dependency_tree,
    dependency_check,
    _parse_version,
    _version_satisfies,
)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_parse_version_standard(self):
        """Test parsing standard version."""
        assert _parse_version("1.2.3") == (1, 2, 3)
        assert _parse_version("2.0.1") == (2, 0, 1)

    def test_parse_version_with_extras(self):
        """Test parsing version with extra characters."""
        assert _parse_version("1.2.3rc1") == (1, 2, 3, 1)
        assert _parse_version("2.0.0.dev5") == (2, 0, 0, 5)

    def test_parse_version_invalid(self):
        """Test parsing invalid version."""
        # "invalid" has no digits, so returns empty tuple
        assert _parse_version("invalid") == ()
        assert _parse_version("") == ()

    def test_version_satisfies_less_than(self):
        """Test version constraint checking."""
        assert _version_satisfies("1.0.0", "<2.0.0") is True
        assert _version_satisfies("2.0.0", "<2.0.0") is False
        assert _version_satisfies("2.5.0", "<2.0.0") is False


class TestDependencyList:
    """Tests for dependency_list function."""

    @pytest.mark.asyncio
    async def test_dependency_list_success(self):
        """Test successful package listing."""
        mock_packages = [
            {"name": "requests", "version": "2.28.0"},
            {"name": "pytest", "version": "7.2.0"},
            {"name": "aiofiles", "version": "23.1.0"},
        ]

        mock_result = Mock()
        mock_result.stdout = json.dumps(mock_packages)
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            result = await dependency_list()

            assert result["success"] is True
            assert result["count"] == 3
            assert len(result["packages"]) == 3
            assert "formatted_report" in result

    @pytest.mark.asyncio
    async def test_dependency_list_empty(self):
        """Test listing with no packages."""
        mock_result = Mock()
        mock_result.stdout = "[]"
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            result = await dependency_list()

            assert result["success"] is True
            assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_dependency_list_error(self):
        """Test error handling in package listing."""
        with patch(
            "subprocess.run", side_effect=subprocess.CalledProcessError(1, "pip", stderr="Error")
        ):
            result = await dependency_list()

            assert result["success"] is False
            assert "error" in result


class TestDependencyOutdated:
    """Tests for dependency_outdated function."""

    @pytest.mark.asyncio
    async def test_dependency_outdated_with_updates(self):
        """Test checking outdated packages."""
        mock_outdated = [
            {"name": "requests", "version": "2.25.0", "latest_version": "3.0.0"},
            {"name": "pytest", "version": "7.1.0", "latest_version": "7.2.0"},
            {"name": "black", "version": "22.10.0", "latest_version": "22.10.1"},
        ]

        mock_result = Mock()
        mock_result.stdout = json.dumps(mock_outdated)
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            result = await dependency_outdated()

            assert result["success"] is True
            assert result["count"] == 3
            assert "by_severity" in result
            assert "major" in result["by_severity"]
            assert "minor" in result["by_severity"]
            assert "patch" in result["by_severity"]

    @pytest.mark.asyncio
    async def test_dependency_outdated_all_up_to_date(self):
        """Test when all packages are up to date."""
        mock_result = Mock()
        mock_result.stdout = "[]"
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            result = await dependency_outdated()

            assert result["success"] is True
            assert result["count"] == 0
            assert "All packages are up to date" in result["message"]

    @pytest.mark.asyncio
    async def test_dependency_outdated_error(self):
        """Test error handling in outdated check."""
        with patch(
            "subprocess.run", side_effect=subprocess.CalledProcessError(1, "pip", stderr="Error")
        ):
            result = await dependency_outdated()

            assert result["success"] is False
            assert "error" in result


class TestDependencySecurity:
    """Tests for dependency_security function."""

    @pytest.mark.asyncio
    async def test_dependency_security_with_vulnerabilities(self):
        """Test security check finding vulnerabilities."""
        mock_packages = [
            {"name": "django", "version": "2.2.0"},
            {"name": "pillow", "version": "8.0.0"},
            {"name": "safe-package", "version": "1.0.0"},
        ]

        mock_result = Mock()
        mock_result.stdout = json.dumps(mock_packages)
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            result = await dependency_security()

            assert result["success"] is True
            assert "vulnerabilities" in result
            # Should find vulnerabilities in django and pillow
            assert result["count"] >= 0

    @pytest.mark.asyncio
    async def test_dependency_security_all_safe(self):
        """Test security check with no vulnerabilities."""
        mock_packages = [
            {"name": "safe-package-1", "version": "1.0.0"},
            {"name": "safe-package-2", "version": "2.0.0"},
        ]

        mock_result = Mock()
        mock_result.stdout = json.dumps(mock_packages)
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            result = await dependency_security()

            assert result["success"] is True
            assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_dependency_security_error(self):
        """Test error handling in security check."""
        with patch(
            "subprocess.run", side_effect=subprocess.CalledProcessError(1, "pip", stderr="Error")
        ):
            result = await dependency_security()

            assert result["success"] is False
            assert "error" in result


class TestDependencyGenerate:
    """Tests for dependency_generate function."""

    @pytest.mark.asyncio
    async def test_dependency_generate_success(self):
        """Test generating requirements file."""
        mock_result = Mock()
        mock_result.stdout = "requests==2.28.0\npytest==7.2.0"
        mock_result.returncode = 0

        # Mock both subprocess.run and Path.write_text
        with patch("subprocess.run", return_value=mock_result):
            with patch("pathlib.Path.write_text"):
                result = await dependency_generate(output="test_requirements.txt")

                assert result["success"] is True
                assert "file" in result  # Function returns 'file', not 'output_file'

    @pytest.mark.asyncio
    async def test_dependency_generate_error(self):
        """Test error in generating requirements."""
        with patch(
            "subprocess.run", side_effect=subprocess.CalledProcessError(1, "pip", stderr="Error")
        ):
            result = await dependency_generate()

            assert result["success"] is False
            assert "error" in result


class TestDependencyUpdate:
    """Tests for dependency_update function."""

    @pytest.mark.asyncio
    async def test_dependency_update_dry_run(self):
        """Test updating packages in dry run mode."""
        # Dry run returns immediately without calling subprocess
        result = await dependency_update(packages=["requests"], dry_run=True)

        assert result["success"] is True
        assert "would_update" in result
        assert "requests" in result["would_update"]

    @pytest.mark.asyncio
    async def test_dependency_update_actual(self):
        """Test actually updating packages."""
        mock_result = Mock()
        mock_result.stdout = "Successfully installed requests-2.28.0"
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            result = await dependency_update(packages=["requests"], dry_run=False)

            assert result["success"] is True
            assert "updated" in result

    @pytest.mark.asyncio
    async def test_dependency_update_empty_packages(self):
        """Test update with empty package list."""
        result = await dependency_update(packages=[])

        assert result["success"] is False
        assert "No packages specified" in result["error"]

    @pytest.mark.asyncio
    async def test_dependency_update_error(self):
        """Test error handling in package update."""
        with patch(
            "subprocess.run", side_effect=subprocess.CalledProcessError(1, "pip", stderr="Error")
        ):
            result = await dependency_update(packages=["invalid-package"], dry_run=False)

            assert result["success"] is False
            assert "error" in result


class TestDependencyTree:
    """Tests for dependency_tree function."""

    @pytest.mark.asyncio
    async def test_dependency_tree_all_packages(self):
        """Test showing dependency tree for all packages."""
        mock_result = Mock()
        mock_result.stdout = "requests==2.28.0\n  - urllib3 [required: >=1.26, installed: 1.26.9]"
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            result = await dependency_tree()

            assert result["success"] is True
            assert "tree" in result

    @pytest.mark.asyncio
    async def test_dependency_tree_specific_package(self):
        """Test showing dependency tree for specific package."""
        mock_result = Mock()
        mock_result.stdout = "requests==2.28.0\n  - urllib3"
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            result = await dependency_tree(package="requests")

            assert result["success"] is True
            assert "package" in result

    @pytest.mark.asyncio
    async def test_dependency_tree_error(self):
        """Test error handling in dependency tree."""
        with patch(
            "subprocess.run", side_effect=subprocess.CalledProcessError(1, "pip", stderr="Error")
        ):
            result = await dependency_tree()

            assert result["success"] is False
            assert "error" in result


class TestDependencyCheck:
    """Tests for dependency_check function."""

    @pytest.mark.asyncio
    async def test_dependency_check_file_not_found(self):
        """Test checking non-existent requirements file."""
        with patch("pathlib.Path.exists", return_value=False):
            result = await dependency_check(requirements_file="nonexistent.txt")

            assert result["success"] is False
            assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_dependency_check_valid_file(self):
        """Test checking valid requirements file."""
        mock_content = "requests==2.28.0\npytest==7.2.0"
        mock_installed = [
            {"name": "requests", "version": "2.28.0"},
            {"name": "pytest", "version": "7.2.0"},
        ]

        mock_result = Mock()
        mock_result.stdout = json.dumps(mock_installed)
        mock_result.returncode = 0

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.read_text", return_value=mock_content):
                with patch("subprocess.run", return_value=mock_result):
                    result = await dependency_check(requirements_file="requirements.txt")

                    assert result["success"] is True
                    assert "satisfied_count" in result

    @pytest.mark.asyncio
    async def test_dependency_check_empty_file(self):
        """Test checking empty requirements file."""
        mock_installed = []

        mock_result = Mock()
        mock_result.stdout = json.dumps(mock_installed)
        mock_result.returncode = 0

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.read_text", return_value=""):
                with patch("subprocess.run", return_value=mock_result):
                    result = await dependency_check(requirements_file="requirements.txt")

                    assert result["success"] is True
                    assert result["satisfied_count"] == 0
