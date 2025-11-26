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

import ast
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from victor.tools.documentation_tool import (
    docs_generate_docstrings,
    docs_generate_api,
    docs_generate_readme,
    docs_add_type_hints,
    docs_analyze_coverage,
    _generate_function_docstring,
    _generate_class_docstring,
    _extract_api_info,
    _build_markdown_docs,
    _build_rst_docs,
    _get_installation_template,
    _get_usage_template,
    _get_contributing_template,
    _get_features_template,
    _get_api_template,
)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_generate_function_docstring_with_args(self):
        """Test generating function docstring with arguments."""
        code = """
def example_function(arg1, arg2):
    return arg1 + arg2
"""
        tree = ast.parse(code)
        func_node = tree.body[0]

        docstring = _generate_function_docstring(func_node, "google")

        assert "Example Function" in docstring
        assert "arg1" in docstring
        assert "arg2" in docstring
        assert "Return value description" in docstring

    def test_generate_function_docstring_no_args(self):
        """Test generating function docstring without arguments."""
        code = """
def simple_function():
    pass
"""
        tree = ast.parse(code)
        func_node = tree.body[0]

        docstring = _generate_function_docstring(func_node, "google")

        assert "Simple Function" in docstring
        assert "None" in docstring  # No args section

    def test_generate_function_docstring_with_return(self):
        """Test generating function docstring with return statement."""
        code = """
def returning_function(x):
    return x * 2
"""
        tree = ast.parse(code)
        func_node = tree.body[0]

        docstring = _generate_function_docstring(func_node, "google")

        assert "Return value description" in docstring

    def test_generate_function_docstring_with_raises(self):
        """Test generating function docstring with raise statement."""
        code = """
def raising_function():
    raise ValueError("Error")
"""
        tree = ast.parse(code)
        func_node = tree.body[0]

        docstring = _generate_function_docstring(func_node, "google")

        assert "Exception: Description" in docstring

    def test_generate_class_docstring_basic(self):
        """Test generating class docstring."""
        code = """
class ExampleClass:
    pass
"""
        tree = ast.parse(code)
        class_node = tree.body[0]

        docstring = _generate_class_docstring(class_node, "google")

        assert "ExampleClass class" in docstring
        assert "Attributes:" in docstring

    def test_generate_class_docstring_with_init(self):
        """Test generating class docstring with __init__ attributes."""
        code = """
class MyClass:
    def __init__(self):
        self.attr1 = "value1"
        self.attr2 = 42
"""
        tree = ast.parse(code)
        class_node = tree.body[0]

        docstring = _generate_class_docstring(class_node, "google")

        assert "MyClass class" in docstring
        assert "attr1" in docstring
        assert "attr2" in docstring

    def test_extract_api_info_functions(self):
        """Test extracting API info for functions."""
        code = """
def public_function(arg1, arg2):
    \"\"\"Public function docstring.\"\"\"
    pass

def _private_function():
    \"\"\"Private function.\"\"\"
    pass
"""
        tree = ast.parse(code)
        api_info = _extract_api_info(tree, "test_module")

        assert api_info["module"] == "test_module"
        assert len(api_info["functions"]) == 1  # Only public
        assert api_info["functions"][0]["name"] == "public_function"
        assert "Public function docstring" in api_info["functions"][0]["docstring"]
        assert "arg1" in api_info["functions"][0]["args"]

    def test_extract_api_info_classes(self):
        """Test extracting API info for classes."""
        code = """
class TestClass:
    \"\"\"Test class docstring.\"\"\"

    def public_method(self, param):
        \"\"\"Public method.\"\"\"
        pass

    def _private_method(self):
        \"\"\"Private method.\"\"\"
        pass

    def __init__(self):
        \"\"\"Constructor.\"\"\"
        pass
"""
        tree = ast.parse(code)
        api_info = _extract_api_info(tree, "test_module")

        assert len(api_info["classes"]) == 1
        assert api_info["classes"][0]["name"] == "TestClass"
        # Should include __init__ and public methods
        method_names = [m["name"] for m in api_info["classes"][0]["methods"]]
        assert "public_method" in method_names
        assert "__init__" in method_names
        assert "_private_method" not in method_names

    def test_build_markdown_docs_with_functions(self):
        """Test building markdown documentation with functions."""
        api_info = {
            "module": "test_module",
            "functions": [
                {
                    "name": "test_func",
                    "docstring": "Test function description",
                    "args": ["arg1", "arg2"]
                }
            ],
            "classes": []
        }

        docs = _build_markdown_docs(api_info)

        assert "# test_module API Documentation" in docs
        assert "## Functions" in docs
        assert "### `test_func(arg1, arg2)`" in docs
        assert "Test function description" in docs
        assert "**Parameters:**" in docs

    def test_build_markdown_docs_with_classes(self):
        """Test building markdown documentation with classes."""
        api_info = {
            "module": "test_module",
            "functions": [],
            "classes": [
                {
                    "name": "TestClass",
                    "docstring": "Test class description",
                    "methods": [
                        {
                            "name": "test_method",
                            "docstring": "Method description",
                            "args": ["param"]
                        }
                    ]
                }
            ]
        }

        docs = _build_markdown_docs(api_info)

        assert "## Classes" in docs
        assert "### `class TestClass`" in docs
        assert "**Methods:**" in docs
        assert "#### `test_method(param)`" in docs

    def test_build_rst_docs(self):
        """Test building RST documentation."""
        api_info = {
            "module": "test_module",
            "functions": [],
            "classes": []
        }

        docs = _build_rst_docs(api_info)

        # Currently simplified to return markdown
        assert "test_module" in docs

    def test_get_installation_template(self):
        """Test getting installation template."""
        template = _get_installation_template()

        assert "## Installation" in template
        assert "pip install" in template
        assert "From PyPI" in template

    def test_get_usage_template(self):
        """Test getting usage template."""
        template = _get_usage_template()

        assert "## Usage" in template
        assert "Quick Start" in template
        assert "from your_package import" in template

    def test_get_contributing_template(self):
        """Test getting contributing template."""
        template = _get_contributing_template()

        assert "## Contributing" in template
        assert "Fork the repository" in template
        assert "pytest" in template

    def test_get_features_template(self):
        """Test getting features template."""
        template = _get_features_template()

        assert "## Features" in template
        assert "Feature 1" in template

    def test_get_api_template(self):
        """Test getting API template."""
        template = _get_api_template()

        assert "## API Reference" in template
        assert "Main Classes" in template


class TestDocsGenerateDocstrings:
    """Tests for docs_generate_docstrings function."""

    @pytest.mark.asyncio
    async def test_generate_docstrings_missing_file(self):
        """Test with missing file parameter."""
        result = await docs_generate_docstrings(file="")

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_generate_docstrings_file_not_found(self):
        """Test with non-existent file."""
        result = await docs_generate_docstrings(file="/nonexistent/file.py")

        assert result["success"] is False
        assert "File not found" in result["error"]

    @pytest.mark.asyncio
    async def test_generate_docstrings_syntax_error(self, tmp_path):
        """Test with file containing syntax errors."""
        test_file = tmp_path / "bad_syntax.py"
        test_file.write_text("def bad function(:")

        result = await docs_generate_docstrings(file=str(test_file))

        assert result["success"] is False
        assert "Syntax error" in result["error"]

    @pytest.mark.asyncio
    async def test_generate_docstrings_already_documented(self, tmp_path):
        """Test with fully documented file."""
        test_file = tmp_path / "documented.py"
        test_file.write_text('''
def my_function():
    """Already has docstring."""
    pass
''')

        result = await docs_generate_docstrings(file=str(test_file))

        assert result["success"] is True
        assert result["generated"] == 0
        assert "already have docstrings" in result["message"]

    @pytest.mark.asyncio
    async def test_generate_docstrings_success(self, tmp_path):
        """Test successful docstring generation."""
        test_file = tmp_path / "undocumented.py"
        test_file.write_text('''
def function_without_docs(arg1, arg2):
    return arg1 + arg2

class ClassWithoutDocs:
    pass
''')

        result = await docs_generate_docstrings(file=str(test_file))

        assert result["success"] is True
        assert result["generated"] == 2  # Function and class
        assert len(result["items"]) == 2
        assert "formatted_report" in result

        # Verify file was modified
        content = test_file.read_text()
        assert '"""' in content

    @pytest.mark.asyncio
    async def test_generate_docstrings_format_google(self, tmp_path):
        """Test docstring generation with google format."""
        test_file = tmp_path / "test.py"
        test_file.write_text('def test_func(): pass')

        result = await docs_generate_docstrings(file=str(test_file), format="google")

        assert result["success"] is True
        assert result["generated"] == 1


class TestDocsGenerateApi:
    """Tests for docs_generate_api function."""

    @pytest.mark.asyncio
    async def test_generate_api_missing_file(self):
        """Test with missing file parameter."""
        result = await docs_generate_api(file="")

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_generate_api_file_not_found(self):
        """Test with non-existent file."""
        result = await docs_generate_api(file="/nonexistent/file.py")

        assert result["success"] is False
        assert "File not found" in result["error"]

    @pytest.mark.asyncio
    async def test_generate_api_syntax_error(self, tmp_path):
        """Test with file containing syntax errors."""
        test_file = tmp_path / "bad_syntax.py"
        test_file.write_text("def bad function(:")

        result = await docs_generate_api(file=str(test_file))

        assert result["success"] is False
        assert "Syntax error" in result["error"]

    @pytest.mark.asyncio
    async def test_generate_api_success_markdown(self, tmp_path):
        """Test successful API generation with markdown format."""
        test_file = tmp_path / "test_module.py"
        test_file.write_text('''
def public_function(arg1):
    """Public function."""
    pass

class TestClass:
    """Test class."""
    def method(self):
        """Method."""
        pass
''')

        # Create docs directory
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        result = await docs_generate_api(
            file=str(test_file),
            output=str(docs_dir / "api.md"),
            format="markdown"
        )

        assert result["success"] is True
        # ast.walk finds both the standalone function and class method
        assert result["functions_count"] >= 1
        assert result["classes_count"] == 1
        assert "preview" in result
        assert "formatted_report" in result

        # Verify output file was created
        assert (docs_dir / "api.md").exists()

    @pytest.mark.asyncio
    async def test_generate_api_default_output(self, tmp_path):
        """Test API generation with default output path."""
        test_file = tmp_path / "test_module.py"
        test_file.write_text('def func(): pass')

        # Change to tmp directory
        import os
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = await docs_generate_api(file=str(test_file))

            assert result["success"] is True
            assert "test_module_api.md" in result["output_file"]
        finally:
            os.chdir(original_dir)

    @pytest.mark.asyncio
    async def test_generate_api_rst_format(self, tmp_path):
        """Test API generation with RST format."""
        test_file = tmp_path / "test.py"
        test_file.write_text('def func(): pass')

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        result = await docs_generate_api(
            file=str(test_file),
            output=str(docs_dir / "api.rst"),
            format="rst"
        )

        assert result["success"] is True


class TestDocsGenerateReadme:
    """Tests for docs_generate_readme function."""

    @pytest.mark.asyncio
    async def test_generate_readme_installation(self):
        """Test generating installation section."""
        result = await docs_generate_readme(section="installation")

        assert result["success"] is True
        assert result["section"] == "installation"
        assert "## Installation" in result["content"]
        assert "pip install" in result["content"]

    @pytest.mark.asyncio
    async def test_generate_readme_usage(self):
        """Test generating usage section."""
        result = await docs_generate_readme(section="usage")

        assert result["success"] is True
        assert result["section"] == "usage"
        assert "## Usage" in result["content"]

    @pytest.mark.asyncio
    async def test_generate_readme_contributing(self):
        """Test generating contributing section."""
        result = await docs_generate_readme(section="contributing")

        assert result["success"] is True
        assert "## Contributing" in result["content"]

    @pytest.mark.asyncio
    async def test_generate_readme_features(self):
        """Test generating features section."""
        result = await docs_generate_readme(section="features")

        assert result["success"] is True
        assert "## Features" in result["content"]

    @pytest.mark.asyncio
    async def test_generate_readme_api(self):
        """Test generating API section."""
        result = await docs_generate_readme(section="api")

        assert result["success"] is True
        assert "## API Reference" in result["content"]

    @pytest.mark.asyncio
    async def test_generate_readme_unknown_section(self):
        """Test with unknown section."""
        result = await docs_generate_readme(section="unknown")

        assert result["success"] is False
        assert "Unknown section" in result["error"]
        assert "Available:" in result["error"]


class TestDocsAddTypeHints:
    """Tests for docs_add_type_hints function."""

    @pytest.mark.asyncio
    async def test_add_type_hints_missing_file(self):
        """Test with missing file parameter."""
        result = await docs_add_type_hints(file="")

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_add_type_hints_file_not_found(self):
        """Test with non-existent file."""
        result = await docs_add_type_hints(file="/nonexistent/file.py")

        assert result["success"] is False
        assert "File not found" in result["error"]

    @pytest.mark.asyncio
    async def test_add_type_hints_success(self, tmp_path):
        """Test successful type hints analysis."""
        test_file = tmp_path / "test.py"
        test_file.write_text('def func(x): return x')

        result = await docs_add_type_hints(file=str(test_file))

        assert result["success"] is True
        assert "suggestions" in result
        assert len(result["suggestions"]) > 0
        assert "formatted_report" in result


class TestDocsAnalyzeCoverage:
    """Tests for docs_analyze_coverage function."""

    @pytest.mark.asyncio
    async def test_analyze_coverage_missing_file(self):
        """Test with missing file parameter."""
        result = await docs_analyze_coverage(file="")

        assert result["success"] is False
        assert "Missing required parameter" in result["error"]

    @pytest.mark.asyncio
    async def test_analyze_coverage_file_not_found(self):
        """Test with non-existent file."""
        result = await docs_analyze_coverage(file="/nonexistent/file.py")

        assert result["success"] is False
        assert "File not found" in result["error"]

    @pytest.mark.asyncio
    async def test_analyze_coverage_syntax_error(self, tmp_path):
        """Test with file containing syntax errors."""
        test_file = tmp_path / "bad_syntax.py"
        test_file.write_text("def bad function(:")

        result = await docs_analyze_coverage(file=str(test_file))

        assert result["success"] is False
        assert "Syntax error" in result["error"]

    @pytest.mark.asyncio
    async def test_analyze_coverage_fully_documented(self, tmp_path):
        """Test with fully documented file."""
        test_file = tmp_path / "documented.py"
        test_file.write_text('''
def my_function():
    """Function docstring."""
    pass

class MyClass:
    """Class docstring."""
    pass
''')

        result = await docs_analyze_coverage(file=str(test_file))

        assert result["success"] is True
        assert result["coverage"] == 100.0
        assert result["total_items"] == 2
        assert result["documented_items"] == 2
        assert len(result["missing"]) == 0

    @pytest.mark.asyncio
    async def test_analyze_coverage_partially_documented(self, tmp_path):
        """Test with partially documented file."""
        test_file = tmp_path / "partial.py"
        test_file.write_text('''
def documented_function():
    """Has docstring."""
    pass

def undocumented_function():
    pass

class DocumentedClass:
    """Has docstring."""
    pass

class UndocumentedClass:
    pass
''')

        result = await docs_analyze_coverage(file=str(test_file))

        assert result["success"] is True
        assert result["coverage"] == 50.0
        assert result["total_items"] == 4
        assert result["documented_items"] == 2
        assert len(result["missing"]) == 2
        assert "recommendations" in result

    @pytest.mark.asyncio
    async def test_analyze_coverage_low_coverage(self, tmp_path):
        """Test with low documentation coverage."""
        test_file = tmp_path / "low_coverage.py"
        test_file.write_text('''
def func1(): pass
def func2(): pass
def func3(): pass
def func4(): pass
class Class1: pass
''')

        result = await docs_analyze_coverage(file=str(test_file))

        assert result["success"] is True
        assert result["coverage"] == 0.0
        assert "Low coverage" in result["recommendations"][0]

    @pytest.mark.asyncio
    async def test_analyze_coverage_private_functions_skipped(self, tmp_path):
        """Test that private functions are skipped."""
        test_file = tmp_path / "private.py"
        test_file.write_text('''
def public_function():
    """Public."""
    pass

def _private_function():
    pass
''')

        result = await docs_analyze_coverage(file=str(test_file))

        assert result["success"] is True
        # Should only count public function
        assert result["total_items"] == 1
        assert result["coverage"] == 100.0

    @pytest.mark.asyncio
    async def test_analyze_coverage_many_missing(self, tmp_path):
        """Test with many missing items (truncation)."""
        test_file = tmp_path / "many_missing.py"
        # Create 20 undocumented functions
        code = "\n".join([f"def func{i}(): pass" for i in range(20)])
        test_file.write_text(code)

        result = await docs_analyze_coverage(file=str(test_file))

        assert result["success"] is True
        assert len(result["missing"]) == 20
        # Report should show truncation message
        assert "... and" in result["formatted_report"]


class TestDeprecatedDocumentationTool:
    """Tests for deprecated DocumentationTool class."""

    def test_deprecated_class_warning(self):
        """Test that deprecated class raises warning."""
        from victor.tools.documentation_tool import DocumentationTool
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tool = DocumentationTool()

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()


class TestEdgeCaseCoverage:
    """Tests to cover edge cases and reach 100% coverage."""

    @pytest.mark.asyncio
    async def test_generate_docstrings_more_than_ten(self, tmp_path):
        """Test generating more than 10 docstrings to trigger truncation message."""
        test_file = tmp_path / "many_undocumented.py"
        # Create a file with 12 undocumented functions
        functions = "\n\n".join([f"def func{i}(arg1, arg2):\n    return arg1 + arg2" for i in range(12)])
        test_file.write_text(functions)

        result = await docs_generate_docstrings(file=str(test_file))

        assert result["success"] is True
        assert result["generated"] == 12
        assert "... and 2 more" in result["formatted_report"]

    @pytest.mark.asyncio
    async def test_generate_api_long_docs(self, tmp_path):
        """Test generating API docs longer than 1000 chars to trigger ellipsis."""
        test_file = tmp_path / "large_api.py"
        # Create a file with many functions to generate long documentation
        functions = []
        for i in range(20):
            functions.append(f'''
def function_{i}(param1, param2, param3, param4):
    """
    This is function {i} with a detailed docstring.

    This function does something important and needs detailed documentation
    to explain all its parameters and return values.

    Args:
        param1: First parameter description
        param2: Second parameter description
        param3: Third parameter description
        param4: Fourth parameter description

    Returns:
        A complex result that needs explanation
    """
    return param1 + param2 + param3 + param4
''')
        test_file.write_text("\n".join(functions))

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        result = await docs_generate_api(
            file=str(test_file),
            output=str(docs_dir / "api.md"),
            format="markdown"
        )

        assert result["success"] is True
        # The report should contain "..." since docs are > 1000 chars
        assert "..." in result["formatted_report"]
