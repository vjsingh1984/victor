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

"""Tests for documentation_tool module."""

import pytest
import ast
import tempfile
from pathlib import Path

from victor.tools.documentation_tool import (
    _generate_function_docstring,
    _generate_class_docstring,
    _extract_api_info,
    _build_markdown_docs,
    docs,
    docs_coverage,
)


class TestGenerateFunctionDocstring:
    """Tests for _generate_function_docstring function."""

    def test_simple_function(self):
        """Test docstring for simple function."""
        code = "def hello(): pass"
        tree = ast.parse(code)
        node = tree.body[0]
        docstring = _generate_function_docstring(node, "google")
        assert "Hello" in docstring or "hello" in docstring

    def test_function_with_args(self):
        """Test docstring for function with arguments."""
        code = "def greet(name, age): pass"
        tree = ast.parse(code)
        node = tree.body[0]
        docstring = _generate_function_docstring(node, "google")
        assert "Args:" in docstring
        assert "name" in docstring
        assert "age" in docstring

    def test_function_with_return(self):
        """Test docstring for function with return."""
        code = "def get_value():\n    return 42"
        tree = ast.parse(code)
        node = tree.body[0]
        docstring = _generate_function_docstring(node, "google")
        assert "Returns:" in docstring

    def test_function_with_raise(self):
        """Test docstring for function with raise."""
        code = "def risky():\n    raise ValueError('error')"
        tree = ast.parse(code)
        node = tree.body[0]
        docstring = _generate_function_docstring(node, "google")
        assert "Raises:" in docstring


class TestGenerateClassDocstring:
    """Tests for _generate_class_docstring function."""

    def test_simple_class(self):
        """Test docstring for simple class."""
        code = "class MyClass:\n    pass"
        tree = ast.parse(code)
        node = tree.body[0]
        docstring = _generate_class_docstring(node, "google")
        assert "MyClass" in docstring

    def test_class_with_init(self):
        """Test docstring for class with __init__."""
        code = """
class MyClass:
    def __init__(self):
        self.name = "test"
        self.value = 42
"""
        tree = ast.parse(code)
        node = tree.body[0]
        docstring = _generate_class_docstring(node, "google")
        # Should detect attributes
        assert "MyClass" in docstring


class TestExtractApiInfo:
    """Tests for _extract_api_info function."""

    def test_extract_api_info(self):
        """Test extracting API info from AST."""
        code = """
def hello():
    pass

class MyClass:
    def method(self):
        pass
"""
        tree = ast.parse(code)
        info = _extract_api_info(tree, "test_module")
        assert "module" in info  # Returns "module", not "module_name"
        assert "functions" in info
        assert "classes" in info


class TestBuildMarkdownDocs:
    """Tests for _build_markdown_docs function."""

    def test_build_markdown_docs(self):
        """Test building markdown from API info."""
        api_info = {
            "module": "test",  # Uses "module", not "module_name"
            "functions": [{"name": "hello", "args": [], "docstring": "A test function"}],
            "classes": [],
        }
        md = _build_markdown_docs(api_info)
        assert "test" in md or "hello" in md


class TestGenerateDocs:
    """Tests for generate_docs function."""

    @pytest.mark.asyncio
    async def test_generate_docs_file(self):
        """Test generating docs for a single file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("""
def hello():
    print("hello")

class MyClass:
    def method(self):
        pass
""")
            temp_path = f.name

        try:
            result = await docs(path=temp_path)
            assert result["success"] is True
        finally:
            Path(temp_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_generate_docs_nonexistent_file(self):
        """Test generating docs for nonexistent file."""
        result = await docs(path="/nonexistent/file.py")
        assert result["success"] is False


class TestAnalyzeDocs:
    """Tests for analyze_docs function."""

    @pytest.mark.asyncio
    async def test_analyze_docs(self):
        """Test analyzing documentation coverage."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write('''
def documented():
    """This function is documented."""
    pass

def undocumented():
    pass
''')
            temp_path = f.name

        try:
            result = await docs_coverage(path=temp_path)
            assert result["success"] is True
        finally:
            Path(temp_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_analyze_docs_nonexistent(self):
        """Test analyzing nonexistent file."""
        result = await docs_coverage(path="/nonexistent/file.py")
        assert result["success"] is False
