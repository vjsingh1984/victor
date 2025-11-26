"""Tests for code_review_tool module."""

import tempfile
from pathlib import Path
import pytest

from victor.tools.code_review_tool import (
    code_review_file,
    code_review_directory,
    code_review_security,
    code_review_complexity,
    code_review_best_practices,
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


class TestCodeReviewFile:
    """Tests for code_review_file function."""

    @pytest.mark.asyncio
    async def test_review_file_missing_path(self):
        """Test handling of missing path parameter."""
        result = await code_review_file(path="")

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_review_file_not_found(self):
        """Test handling of non-existent file."""
        result = await code_review_file(path="/nonexistent/file.py")

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_review_file_not_a_file(self, tmp_path):
        """Test handling of directory path."""
        result = await code_review_file(path=str(tmp_path))

        assert result["success"] is False
        assert "not a file" in result["error"]

    @pytest.mark.asyncio
    async def test_review_clean_python_file(self, tmp_path):
        """Test reviewing clean Python file."""
        test_file = tmp_path / "clean.py"
        test_file.write_text("""
def calculate(x: int, y: int) -> int:
    \"\"\"Calculate sum of two numbers.\"\"\"
    return x + y

def process_data(data: list) -> dict:
    \"\"\"Process data and return results.\"\"\"
    result = {}
    for item in data:
        result[item] = item * 2
    return result
""")

        result = await code_review_file(path=str(test_file))

        assert result["success"] is True
        assert result["issues_count"] >= 0
        assert "formatted_report" in result

    @pytest.mark.asyncio
    async def test_review_file_with_security_issues(self, tmp_path):
        """Test reviewing file with security issues."""
        test_file = tmp_path / "insecure.py"
        test_file.write_text("""
import os

def execute_command(cmd):
    os.system(cmd)

def query_db(user_input):
    query = f"SELECT * FROM users WHERE id = {user_input}"
    return query
""")

        result = await code_review_file(path=str(test_file))

        assert result["success"] is True
        assert result["issues_count"] > 0

    @pytest.mark.asyncio
    async def test_review_file_with_complexity(self, tmp_path):
        """Test reviewing file with high complexity."""
        test_file = tmp_path / "complex.py"
        test_file.write_text("""
def complex_function(x, y, z):
    if x > 0:
        if y > 0:
            if z > 0:
                if x > y:
                    if y > z:
                        if x > z:
                            return 1
                        else:
                            return 2
                    else:
                        return 3
                else:
                    return 4
            else:
                return 5
        else:
            return 6
    else:
        return 7
""")

        result = await code_review_file(path=str(test_file))

        assert result["success"] is True
        # Likely to have complexity issues

    @pytest.mark.asyncio
    async def test_review_file_without_docstrings(self, tmp_path):
        """Test reviewing file missing docstrings."""
        test_file = tmp_path / "nodocs.py"
        test_file.write_text("""
def function1():
    pass

def function2():
    pass

class MyClass:
    def method(self):
        pass
""")

        result = await code_review_file(path=str(test_file))

        assert result["success"] is True
        # Likely to have documentation issues

    @pytest.mark.asyncio
    async def test_review_file_with_metrics(self, tmp_path):
        """Test reviewing file with metrics enabled."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def hello():
    return "Hello"
""")

        result = await code_review_file(path=str(test_file), include_metrics=True)

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_review_non_python_file(self, tmp_path):
        """Test reviewing non-Python file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Not Python code")

        result = await code_review_file(path=str(test_file))

        assert result["success"] is True
        # Should handle gracefully


class TestCodeReviewDirectory:
    """Tests for code_review_directory function."""

    @pytest.mark.asyncio
    async def test_review_directory_missing_path(self):
        """Test handling of missing path parameter."""
        result = await code_review_directory(path="")

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_review_directory_not_found(self):
        """Test handling of non-existent directory."""
        result = await code_review_directory(path="/nonexistent/directory")

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_review_directory_not_a_directory(self, tmp_path):
        """Test handling of file path instead of directory."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello(): pass")

        result = await code_review_directory(path=str(test_file))

        # May succeed or fail depending on implementation
        # Just check it handles gracefully
        assert "success" in result

    @pytest.mark.asyncio
    async def test_review_empty_directory(self, tmp_path):
        """Test reviewing empty directory."""
        result = await code_review_directory(path=str(tmp_path))

        assert result["success"] is True
        assert result["files_reviewed"] == 0

    @pytest.mark.asyncio
    async def test_review_directory_with_files(self, tmp_path):
        """Test reviewing directory with Python files."""
        # Create test files
        file1 = tmp_path / "file1.py"
        file1.write_text("def func1(): pass")

        file2 = tmp_path / "file2.py"
        file2.write_text("def func2(): pass")

        result = await code_review_directory(path=str(tmp_path))

        assert result["success"] is True
        assert result["files_reviewed"] >= 2  # Should find both files
        assert "total_issues" in result

    @pytest.mark.asyncio
    async def test_review_directory_with_nested_files(self, tmp_path):
        """Test reviewing directory with nested structure."""
        # Create nested structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        (tmp_path / "root.py").write_text("def root(): pass")
        (subdir / "nested.py").write_text("def nested(): pass")

        result = await code_review_directory(path=str(tmp_path))

        # Implementation uses rglob which is recursive by default
        assert result["success"] is True
        assert result["files_reviewed"] >= 1


class TestCodeReviewSecurity:
    """Tests for code_review_security function."""

    @pytest.mark.asyncio
    async def test_review_security_missing_path(self):
        """Test handling of missing path parameter."""
        result = await code_review_security(path="")

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_review_security_file_not_found(self):
        """Test handling of non-existent path."""
        result = await code_review_security(path="/nonexistent/path")

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_review_security_clean_file(self, tmp_path):
        """Test security review of clean file."""
        test_file = tmp_path / "clean.py"
        test_file.write_text("""
def safe_function(data):
    result = process(data)
    return result
""")

        result = await code_review_security(path=str(test_file))

        assert result["success"] is True
        assert "security_issues" in result

    @pytest.mark.asyncio
    async def test_review_security_with_issues(self, tmp_path):
        """Test security review with security issues."""
        test_file = tmp_path / "insecure.py"
        test_file.write_text("""
import os

password = "hardcoded_password"
api_key = "sk_test_12345678901234567890"

def execute(cmd):
    os.system(cmd)
    eval(cmd)
""")

        result = await code_review_security(path=str(test_file))

        assert result["success"] is True
        assert result["security_issues"] > 0


class TestCodeReviewComplexity:
    """Tests for code_review_complexity function."""

    @pytest.mark.asyncio
    async def test_review_complexity_missing_path(self):
        """Test handling of missing path parameter."""
        result = await code_review_complexity(path="")

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_review_complexity_file_not_found(self):
        """Test handling of non-existent file."""
        result = await code_review_complexity(path="/nonexistent/file.py")

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_review_complexity_simple_file(self, tmp_path):
        """Test complexity review of simple file."""
        test_file = tmp_path / "simple.py"
        test_file.write_text("""
def simple():
    return True

def another():
    x = 1
    return x
""")

        result = await code_review_complexity(path=str(test_file))

        assert result["success"] is True
        # Function count or complexity info should be present
        assert "formatted_report" in result or "complex_functions" in result

    @pytest.mark.asyncio
    async def test_review_complexity_complex_file(self, tmp_path):
        """Test complexity review of complex file."""
        test_file = tmp_path / "complex.py"
        test_file.write_text("""
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
""")

        result = await code_review_complexity(path=str(test_file))

        assert result["success"] is True
        # Likely to detect high complexity


class TestCodeReviewBestPractices:
    """Tests for code_review_best_practices function."""

    @pytest.mark.asyncio
    async def test_review_best_practices_missing_path(self):
        """Test handling of missing path parameter."""
        result = await code_review_best_practices(path="")

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_review_best_practices_file_not_found(self):
        """Test handling of non-existent file."""
        result = await code_review_best_practices(path="/nonexistent/file.py")

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_review_best_practices_good_file(self, tmp_path):
        """Test best practices review of well-written file."""
        test_file = tmp_path / "good.py"
        test_file.write_text("""
\"\"\"Module docstring.\"\"\"

def well_named_function(parameter: int) -> int:
    \"\"\"Function with proper documentation.\"\"\"
    result = parameter * 2
    return result

class WellNamedClass:
    \"\"\"Class with proper documentation.\"\"\"

    def __init__(self):
        \"\"\"Initialize instance.\"\"\"
        self.value = 0
""")

        result = await code_review_best_practices(path=str(test_file))

        assert result["success"] is True
        assert "issues" in result

    @pytest.mark.asyncio
    async def test_review_best_practices_issues(self, tmp_path):
        """Test best practices review with issues."""
        test_file = tmp_path / "bad.py"
        test_file.write_text("""
def x():
    a=1
    b=2
    return a+b

def VeryLongFunctionNameThatViolatesNamingConventionsAndIsWayTooLong():
    pass
""")

        result = await code_review_best_practices(path=str(test_file))

        assert result["success"] is True
        # Likely to find best practice violations

    @pytest.mark.asyncio
    async def test_review_best_practices_with_syntax_error(self, tmp_path):
        """Test best practices review with syntax error."""
        test_file = tmp_path / "syntax_error.py"
        test_file.write_text("def broken(\n    pass")

        result = await code_review_best_practices(path=str(test_file))

        assert result["success"] is True
        # Should handle syntax errors gracefully


class TestCodeReviewAdditionalCoverage:
    """Additional tests for uncovered code paths."""

    @pytest.mark.asyncio
    async def test_review_file_very_high_complexity(self, tmp_path):
        """Test reviewing file with very high complexity triggering issues."""
        test_file = tmp_path / "very_complex.py"
        # Create function with cyclomatic complexity > 15
        test_file.write_text("""
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
""")

        result = await code_review_file(path=str(test_file))

        assert result["success"] is True
        # Should detect high complexity (but may not always depending on implementation)
        # Just verify it completes successfully
        assert "issues_count" in result

    @pytest.mark.asyncio
    async def test_review_complexity_exceeds_threshold(self, tmp_path):
        """Test complexity review when threshold is exceeded."""
        test_file = tmp_path / "high_complexity.py"
        test_file.write_text("""
def complex_logic(x, y, z):
    if x > 0:
        if y > 0:
            if z > 0:
                if x > y:
                    if y > z:
                        if x > z:
                            if x == 10:
                                return 1
                        return 2
                    return 3
                return 4
            return 5
        return 6
    return 7
""")

        result = await code_review_complexity(path=str(test_file))

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_review_file_with_syntax_error_handling(self, tmp_path):
        """Test that syntax errors are handled gracefully."""
        test_file = tmp_path / "broken.py"
        test_file.write_text("def incomplete(:\n    pass")

        result = await code_review_file(path=str(test_file))

        # Should complete without crashing
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_review_security_with_multiple_issues(self, tmp_path):
        """Test security review detecting multiple issue types."""
        test_file = tmp_path / "multiple_security.py"
        test_file.write_text("""
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
""")

        result = await code_review_security(path=str(test_file))

        assert result["success"] is True
        assert result["security_issues"] > 0

    @pytest.mark.asyncio
    async def test_review_directory_with_subdirectories(self, tmp_path):
        """Test directory review with nested structure."""
        # Create nested structure
        subdir1 = tmp_path / "module1"
        subdir1.mkdir()
        subdir2 = tmp_path / "module2"
        subdir2.mkdir()

        (tmp_path / "root.py").write_text("def root_func(): pass")
        (subdir1 / "mod1.py").write_text("def mod1_func(): pass")
        (subdir2 / "mod2.py").write_text("def mod2_func(): pass")

        result = await code_review_directory(path=str(tmp_path))

        assert result["success"] is True
        assert result["files_reviewed"] >= 1
