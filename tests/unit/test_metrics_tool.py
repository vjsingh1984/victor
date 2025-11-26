"""Tests for metrics_tool module."""

import tempfile
from pathlib import Path
import pytest

from victor.tools.metrics_tool import (
    metrics_complexity,
    metrics_maintainability,
    metrics_debt,
    metrics_profile,
    metrics_analyze,
    metrics_report,
)


class TestMetricsComplexity:
    """Tests for metrics_complexity function."""

    @pytest.mark.asyncio
    async def test_complexity_missing_file(self):
        """Test handling of missing file parameter."""
        result = await metrics_complexity(file="")

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_complexity_file_not_found(self):
        """Test handling of non-existent file."""
        result = await metrics_complexity(file="/nonexistent/file.py")

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_complexity_simple_function(self, tmp_path):
        """Test complexity of simple function."""
        test_file = tmp_path / "simple.py"
        test_file.write_text("def foo():\n    if True:\n        pass\n")

        result = await metrics_complexity(file=str(test_file), threshold=10)

        assert result["success"] is True
        assert "complexity" in result
        assert result["complexity"] >= 1

    @pytest.mark.asyncio
    async def test_complexity_with_threshold(self, tmp_path):
        """Test complexity with different threshold."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def simple():\n    return 42\n")

        result = await metrics_complexity(file=str(test_file), threshold=5)

        assert result["success"] is True
        assert "complexity" in result

    @pytest.mark.asyncio
    async def test_complexity_complex_function(self, tmp_path):
        """Test complexity of complex function."""
        test_file = tmp_path / "complex.py"
        test_file.write_text("""
def complex_func(a, b, c):
    if a:
        if b:
            if c:
                return 1
            return 2
        return 3
    return 4
""")

        result = await metrics_complexity(file=str(test_file))

        assert result["success"] is True
        assert result["complexity"] > 1

    @pytest.mark.asyncio
    async def test_complexity_empty_file(self, tmp_path):
        """Test complexity of empty file."""
        test_file = tmp_path / "empty.py"
        test_file.write_text("")

        result = await metrics_complexity(file=str(test_file))

        assert result["success"] is True


class TestMetricsMaintainability:
    """Tests for metrics_maintainability function."""

    @pytest.mark.asyncio
    async def test_maintainability_missing_file(self):
        """Test handling of missing file parameter."""
        result = await metrics_maintainability(file="")

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_maintainability_file_not_found(self):
        """Test handling of non-existent file."""
        result = await metrics_maintainability(file="/nonexistent/file.py")

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_maintainability_simple_function(self, tmp_path):
        """Test maintainability of simple function."""
        test_file = tmp_path / "simple.py"
        test_file.write_text("def foo():\n    return 42\n")

        result = await metrics_maintainability(file=str(test_file))

        assert result["success"] is True
        assert "maintainability_index" in result
        assert 0 <= result["maintainability_index"] <= 100

    @pytest.mark.asyncio
    async def test_maintainability_empty_file(self, tmp_path):
        """Test maintainability of empty file."""
        test_file = tmp_path / "empty.py"
        test_file.write_text("")

        result = await metrics_maintainability(file=str(test_file))

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_maintainability_well_documented(self, tmp_path):
        """Test maintainability of well-documented code."""
        test_file = tmp_path / "documented.py"
        test_file.write_text('''
"""Module docstring."""

def documented_func():
    """Well documented function."""
    return 42
''')

        result = await metrics_maintainability(file=str(test_file))

        assert result["success"] is True
        assert result["maintainability_index"] >= 0


class TestMetricsDebt:
    """Tests for metrics_debt function."""

    @pytest.mark.asyncio
    async def test_debt_missing_file(self):
        """Test handling of missing file parameter."""
        result = await metrics_debt(file="")

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_debt_file_not_found(self):
        """Test handling of non-existent file."""
        result = await metrics_debt(file="/nonexistent/file.py")

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_debt_simple_function(self, tmp_path):
        """Test debt estimation for simple function."""
        test_file = tmp_path / "simple.py"
        test_file.write_text("def foo():\n    pass\n")

        result = await metrics_debt(file=str(test_file))

        assert result["success"] is True
        assert "debt_hours" in result
        assert "debt_level" in result

    @pytest.mark.asyncio
    async def test_debt_empty_file(self, tmp_path):
        """Test debt estimation for empty file."""
        test_file = tmp_path / "empty.py"
        test_file.write_text("")

        result = await metrics_debt(file=str(test_file))

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_debt_complex_code(self, tmp_path):
        """Test debt estimation for complex code."""
        test_file = tmp_path / "complex.py"
        test_file.write_text("""
def complex_function(a, b, c, d):
    if a:
        if b:
            if c:
                if d:
                    return 1
    return 0

def another_complex():
    pass
""")

        result = await metrics_debt(file=str(test_file))

        assert result["success"] is True
        assert result["debt_hours"] >= 0


class TestMetricsProfile:
    """Tests for metrics_profile function."""

    @pytest.mark.asyncio
    async def test_profile_missing_file(self):
        """Test handling of missing file parameter."""
        result = await metrics_profile(file="")

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_profile_file_not_found(self):
        """Test handling of non-existent file."""
        result = await metrics_profile(file="/nonexistent/file.py")

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_profile_simple_code(self, tmp_path):
        """Test profiling simple code."""
        test_file = tmp_path / "simple.py"
        test_file.write_text("class Foo:\n    def bar(self):\n        pass\n")

        result = await metrics_profile(file=str(test_file))

        assert result["success"] is True
        assert "lines" in result
        assert "functions" in result
        assert "classes" in result

    @pytest.mark.asyncio
    async def test_profile_empty_file(self, tmp_path):
        """Test profiling empty file."""
        test_file = tmp_path / "empty.py"
        test_file.write_text("")

        result = await metrics_profile(file=str(test_file))

        assert result["success"] is True
        assert result["lines"] >= 0

    @pytest.mark.asyncio
    async def test_profile_multiple_items(self, tmp_path):
        """Test profiling file with multiple functions and classes."""
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def func1():
    pass

def func2():
    pass

class Class1:
    def method1(self):
        pass

    def method2(self):
        pass

class Class2:
    pass
""")

        result = await metrics_profile(file=str(test_file))

        assert result["success"] is True
        assert result["functions"] >= 2
        assert result["classes"] >= 2


class TestMetricsAnalyze:
    """Tests for metrics_analyze function."""

    @pytest.mark.asyncio
    async def test_analyze_missing_file(self):
        """Test handling of missing file parameter."""
        result = await metrics_analyze(file="")

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_analyze_file_not_found(self):
        """Test handling of non-existent file."""
        result = await metrics_analyze(file="/nonexistent/file.py")

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_analyze_simple_file(self, tmp_path):
        """Test comprehensive analysis of simple file."""
        test_file = tmp_path / "simple.py"
        test_file.write_text("def foo():\n    return 42\n")

        result = await metrics_analyze(file=str(test_file))

        assert result["success"] is True
        assert "complexity" in result
        assert "maintainability" in result
        assert "debt" in result
        assert "profile" in result

    @pytest.mark.asyncio
    async def test_analyze_empty_file(self, tmp_path):
        """Test comprehensive analysis of empty file."""
        test_file = tmp_path / "empty.py"
        test_file.write_text("")

        result = await metrics_analyze(file=str(test_file))

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_analyze_complete_module(self, tmp_path):
        """Test comprehensive analysis of complete module."""
        test_file = tmp_path / "module.py"
        test_file.write_text('''
"""Module docstring."""

def function1(x, y):
    """Function 1."""
    if x > y:
        return x
    return y

class MyClass:
    """My class."""

    def __init__(self):
        self.value = 0

    def method(self, data):
        """Method."""
        return data * 2
''')

        result = await metrics_analyze(file=str(test_file))

        assert result["success"] is True
        assert all(k in result for k in ["complexity", "maintainability", "debt", "profile"])


class TestMetricsReport:
    """Tests for metrics_report function."""

    @pytest.mark.asyncio
    async def test_report_missing_file(self):
        """Test handling of missing file parameter."""
        result = await metrics_report(file="")

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_report_file_not_found(self):
        """Test handling of non-existent file."""
        result = await metrics_report(file="/nonexistent/file.py")

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_report_simple_file(self, tmp_path):
        """Test quality report for simple file."""
        test_file = tmp_path / "simple.py"
        test_file.write_text("def foo():\n    return 42\n")

        result = await metrics_report(file=str(test_file))

        assert result["success"] is True
        assert "formatted_report" in result

    @pytest.mark.asyncio
    async def test_report_empty_file(self, tmp_path):
        """Test quality report for empty file."""
        test_file = tmp_path / "empty.py"
        test_file.write_text("")

        result = await metrics_report(file=str(test_file))

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_report_complex_file(self, tmp_path):
        """Test quality report for complex file."""
        test_file = tmp_path / "complex.py"
        test_file.write_text('''
def complex_function(a, b, c):
    if a > 0:
        if b > 0:
            if c > 0:
                return a + b + c
            return a + b
        return a
    return 0

class Calculator:
    def add(self, x, y):
        return x + y

    def multiply(self, x, y):
        return x * y
''')

        result = await metrics_report(file=str(test_file))

        assert result["success"] is True
        assert "formatted_report" in result
        assert len(result["formatted_report"]) > 0
