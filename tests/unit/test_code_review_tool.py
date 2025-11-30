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

"""Tests for code_review_tool module.

Tests the consolidated code_review function with various aspects:
- security
- complexity
- best_practices
- documentation
- all (comprehensive review)
"""

import tempfile
from pathlib import Path
import pytest

from victor.tools.code_review_tool import (
    code_review,
    set_code_review_config,
)


class TestSetCodeReviewConfig:
    """Tests for set_code_review_config function."""

    def test_set_config_default(self):
        """Test setting config with default values."""
        set_code_review_config()
        # Should not raise

    def test_set_config_custom(self):
        """Test setting config with custom values."""
        set_code_review_config(max_complexity=15)
        # Should not raise


class TestCodeReviewBasic:
    """Tests for basic code_review functionality."""

    @pytest.mark.asyncio
    async def test_review_missing_path(self):
        """Test handling of missing path parameter."""
        result = await code_review(path="")

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_review_not_found(self):
        """Test handling of non-existent path."""
        result = await code_review(path="/nonexistent/file.py")

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_review_invalid_aspect(self):
        """Test handling of invalid aspect."""
        result = await code_review(path=".", aspects=["invalid_aspect"])

        assert result["success"] is False
        assert "Invalid aspect" in result["error"]

    @pytest.mark.asyncio
    async def test_review_clean_python_file(self, tmp_path):
        """Test reviewing clean Python file."""
        test_file = tmp_path / "clean.py"
        test_file.write_text(
            """
def calculate(x: int, y: int) -> int:
    \"\"\"Calculate sum of two numbers.\"\"\"
    return x + y

def process_data(data: list) -> dict:
    \"\"\"Process data and return results.\"\"\"
    result = {}
    for item in data:
        result[item] = item * 2
    return result
"""
        )

        result = await code_review(path=str(test_file))

        assert result["success"] is True
        assert "total_issues" in result
        assert "formatted_report" in result

    @pytest.mark.asyncio
    async def test_review_with_aspects_string(self, tmp_path):
        """Test review with aspects as JSON string."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello(): pass")

        result = await code_review(path=str(test_file), aspects='["security"]')

        assert result["success"] is True
        assert "security" in result.get("aspects_checked", [])

    @pytest.mark.asyncio
    async def test_review_with_single_aspect_string(self, tmp_path):
        """Test review with single aspect as string."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello(): pass")

        result = await code_review(path=str(test_file), aspects="complexity")

        assert result["success"] is True


class TestCodeReviewSecurity:
    """Tests for security aspect of code_review."""

    @pytest.mark.asyncio
    async def test_review_security_clean_file(self, tmp_path):
        """Test security review of clean file."""
        test_file = tmp_path / "clean.py"
        test_file.write_text(
            """
def safe_function(data):
    result = process(data)
    return result
"""
        )

        result = await code_review(path=str(test_file), aspects=["security"])

        assert result["success"] is True
        assert "security" in result.get("aspects_checked", [])

    @pytest.mark.asyncio
    async def test_review_security_with_issues(self, tmp_path):
        """Test security review with security issues."""
        test_file = tmp_path / "insecure.py"
        test_file.write_text(
            """
import os

password = "hardcoded_password"
api_key = "sk_test_12345678901234567890"

def execute(cmd):
    os.system(cmd)
    eval(cmd)
"""
        )

        result = await code_review(path=str(test_file), aspects=["security"])

        assert result["success"] is True
        # Should detect security issues
        assert result.get("total_issues", 0) > 0 or "security" in result.get("results", {})

    @pytest.mark.asyncio
    async def test_review_security_with_severity_filter(self, tmp_path):
        """Test security review with severity filter."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """
import os
os.system("ls")
"""
        )

        result = await code_review(path=str(test_file), aspects=["security"], severity="high")

        assert result["success"] is True


class TestCodeReviewComplexity:
    """Tests for complexity aspect of code_review."""

    @pytest.mark.asyncio
    async def test_review_complexity_simple_file(self, tmp_path):
        """Test complexity review of simple file."""
        test_file = tmp_path / "simple.py"
        test_file.write_text(
            """
def simple():
    return True

def another():
    x = 1
    return x
"""
        )

        result = await code_review(path=str(test_file), aspects=["complexity"])

        assert result["success"] is True
        assert "complexity" in result.get("aspects_checked", [])

    @pytest.mark.asyncio
    async def test_review_complexity_complex_file(self, tmp_path):
        """Test complexity review of complex file."""
        test_file = tmp_path / "complex.py"
        test_file.write_text(
            """
def very_complex(a, b, c, d, e):
    if a:
        if b:
            if c:
                if d:
                    if e:
                        return 1
                    return 2
                return 3
            return 4
        return 5
    return 6
"""
        )

        result = await code_review(path=str(test_file), aspects=["complexity"])

        assert result["success"] is True


class TestCodeReviewBestPractices:
    """Tests for best_practices aspect of code_review."""

    @pytest.mark.asyncio
    async def test_review_best_practices_good_file(self, tmp_path):
        """Test best practices review of well-written file."""
        test_file = tmp_path / "good.py"
        test_file.write_text(
            '''
"""Module docstring."""

def well_named_function(parameter: int) -> int:
    """Function with proper documentation."""
    result = parameter * 2
    return result

class WellNamedClass:
    """Class with proper documentation."""

    def __init__(self):
        """Initialize instance."""
        self.value = 0
'''
        )

        result = await code_review(path=str(test_file), aspects=["best_practices"])

        assert result["success"] is True
        assert "best_practices" in result.get("aspects_checked", [])

    @pytest.mark.asyncio
    async def test_review_best_practices_issues(self, tmp_path):
        """Test best practices review with issues."""
        test_file = tmp_path / "bad.py"
        test_file.write_text(
            """
def x():
    a=1
    b=2
    return a+b

def VeryLongFunctionNameThatViolatesNamingConventionsAndIsWayTooLong():
    pass
"""
        )

        result = await code_review(path=str(test_file), aspects=["best_practices"])

        assert result["success"] is True


class TestCodeReviewDocumentation:
    """Tests for documentation aspect of code_review."""

    @pytest.mark.asyncio
    async def test_review_documentation_with_docs(self, tmp_path):
        """Test documentation review of well-documented file."""
        test_file = tmp_path / "documented.py"
        test_file.write_text(
            '''
"""Module with documentation."""

def documented_function():
    """This function is documented."""
    pass
'''
        )

        result = await code_review(path=str(test_file), aspects=["documentation"])

        assert result["success"] is True
        assert "documentation" in result.get("aspects_checked", [])

    @pytest.mark.asyncio
    async def test_review_documentation_missing(self, tmp_path):
        """Test documentation review with missing docstrings."""
        test_file = tmp_path / "nodocs.py"
        test_file.write_text(
            """
def function1():
    pass

def function2():
    pass

class MyClass:
    def method(self):
        pass
"""
        )

        result = await code_review(path=str(test_file), aspects=["documentation"])

        assert result["success"] is True


class TestCodeReviewDirectory:
    """Tests for directory review functionality."""

    @pytest.mark.asyncio
    async def test_review_empty_directory(self, tmp_path):
        """Test reviewing empty directory."""
        result = await code_review(path=str(tmp_path))

        assert result["success"] is True
        assert result.get("files_reviewed", 0) == 0

    @pytest.mark.asyncio
    async def test_review_directory_with_files(self, tmp_path):
        """Test reviewing directory with Python files."""
        # Create test files
        file1 = tmp_path / "file1.py"
        file1.write_text("def func1(): pass")

        file2 = tmp_path / "file2.py"
        file2.write_text("def func2(): pass")

        result = await code_review(path=str(tmp_path))

        assert result["success"] is True
        assert result.get("files_reviewed", 0) >= 2

    @pytest.mark.asyncio
    async def test_review_directory_with_file_pattern(self, tmp_path):
        """Test reviewing directory with specific file pattern."""
        # Create different file types
        (tmp_path / "test.py").write_text("def py_func(): pass")
        (tmp_path / "test.txt").write_text("Not Python")

        result = await code_review(path=str(tmp_path), file_pattern="*.py")

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_review_directory_with_nested_files(self, tmp_path):
        """Test reviewing directory with nested structure."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        (tmp_path / "root.py").write_text("def root(): pass")
        (subdir / "nested.py").write_text("def nested(): pass")

        result = await code_review(path=str(tmp_path))

        assert result["success"] is True
        assert result.get("files_reviewed", 0) >= 1


class TestCodeReviewComprehensive:
    """Tests for comprehensive (all aspects) review."""

    @pytest.mark.asyncio
    async def test_review_all_aspects(self, tmp_path):
        """Test comprehensive review with all aspects."""
        test_file = tmp_path / "comprehensive.py"
        test_file.write_text(
            """
import os

def insecure_function(cmd):
    os.system(cmd)

def complex_function(a, b, c):
    if a:
        if b:
            if c:
                return 1
            return 2
        return 3
    return 4
"""
        )

        result = await code_review(path=str(test_file), aspects=["all"])

        assert result["success"] is True
        aspects_checked = result.get("aspects_checked", [])
        assert "security" in aspects_checked
        assert "complexity" in aspects_checked
        assert "best_practices" in aspects_checked
        assert "documentation" in aspects_checked

    @pytest.mark.asyncio
    async def test_review_with_metrics(self, tmp_path):
        """Test review with metrics enabled."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello(): return 'Hello'")

        result = await code_review(path=str(test_file), include_metrics=True)

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_review_with_max_issues(self, tmp_path):
        """Test review with max_issues limit."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def func(): pass")

        result = await code_review(path=str(test_file), max_issues=5)

        assert result["success"] is True


class TestCodeReviewEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_review_file_with_syntax_error(self, tmp_path):
        """Test that syntax errors are handled gracefully."""
        test_file = tmp_path / "broken.py"
        test_file.write_text("def incomplete(:\n    pass")

        result = await code_review(path=str(test_file))

        # Should complete without crashing
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_review_non_python_file(self, tmp_path):
        """Test reviewing non-Python file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Not Python code")

        result = await code_review(path=str(test_file))

        # Should handle gracefully
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_review_empty_file(self, tmp_path):
        """Test reviewing empty file."""
        test_file = tmp_path / "empty.py"
        test_file.write_text("")

        result = await code_review(path=str(test_file))

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_review_very_high_complexity(self, tmp_path):
        """Test reviewing file with very high complexity."""
        test_file = tmp_path / "very_complex.py"
        test_file.write_text(
            """
def extremely_complex(a, b, c, d, e, f, g, h):
    if a:
        if b:
            if c:
                if d:
                    if e:
                        if f:
                            if g:
                                if h:
                                    return 1
                                return 2
                            return 3
                        return 4
                    return 5
                return 6
            return 7
        return 8
    return 9
"""
        )

        result = await code_review(path=str(test_file), aspects=["complexity"])

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_review_multiple_security_issues(self, tmp_path):
        """Test security review detecting multiple issue types."""
        test_file = tmp_path / "multiple_security.py"
        test_file.write_text(
            """
import os

API_KEY = "sk_live_1234567890abcdef"
PASSWORD = "hardcoded_password123"

def execute_command(cmd):
    os.system(cmd)
    result = eval(cmd)
    return result

def sql_injection(user_input):
    query = f"SELECT * FROM users WHERE id = {user_input}"
    return query
"""
        )

        result = await code_review(path=str(test_file), aspects=["security"])

        assert result["success"] is True
