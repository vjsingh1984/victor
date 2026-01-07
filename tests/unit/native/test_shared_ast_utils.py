# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
# Licensed under the Apache License, Version 2.0

"""Tests for shared_ast_utils module."""


from victor.tools.shared_ast_utils import (
    calculate_cognitive_complexity,
    calculate_complexity,
    calculate_maintainability_index,
    count_classes,
    count_functions,
    find_classes,
    find_functions,
    find_imports,
    find_symbol,
    get_class_info,
    get_function_info,
    get_line_count,
    get_module_info,
    get_undocumented_classes,
    get_undocumented_functions,
    has_docstring,
    parse_code,
    parse_file,
)


class TestParseCode:
    """Tests for parse_code function."""

    def test_parse_valid_python(self):
        """Test parsing valid Python code."""
        code = "x = 1\ny = 2\n"
        result = parse_code(code)

        assert result is not None
        assert result.tree is not None

    def test_parse_invalid_python(self):
        """Test parsing invalid Python code."""
        code = "def foo(:\n"
        result = parse_code(code)

        assert result.tree is None
        assert result.error is not None


class TestParseFile:
    """Tests for parse_file function."""

    def test_parse_valid_file(self, tmp_path):
        """Test parsing a valid Python file."""
        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1\ny = 2\n")

        result = parse_file(test_file)

        assert result.tree is not None
        assert result.error is None

    def test_parse_nonexistent_file(self, tmp_path):
        """Test parsing a nonexistent file."""
        result = parse_file(tmp_path / "nonexistent.py")

        assert result.tree is None
        assert result.error is not None


class TestFindFunctions:
    """Tests for find_functions function."""

    def test_find_simple_function(self):
        """Test finding a simple function."""
        code = """
def hello():
    pass
"""
        result = parse_code(code)
        functions = list(find_functions(result.tree))

        assert len(functions) == 1
        assert functions[0].name == "hello"

    def test_find_multiple_functions(self):
        """Test finding multiple functions."""
        code = """
def foo():
    pass

def bar():
    pass
"""
        result = parse_code(code)
        functions = list(find_functions(result.tree))

        assert len(functions) == 2
        names = [f.name for f in functions]
        assert "foo" in names
        assert "bar" in names


class TestFindClasses:
    """Tests for find_classes function."""

    def test_find_simple_class(self):
        """Test finding a simple class."""
        code = """
class MyClass:
    pass
"""
        result = parse_code(code)
        classes = list(find_classes(result.tree))

        assert len(classes) == 1
        assert classes[0].name == "MyClass"

    def test_find_class_with_methods(self):
        """Test finding a class with methods."""
        code = """
class MyClass:
    def method1(self):
        pass

    def method2(self):
        pass
"""
        result = parse_code(code)
        classes = list(find_classes(result.tree))

        assert len(classes) == 1
        # Methods are found separately
        functions = list(find_functions(result.tree, include_methods=True))
        assert len(functions) == 2


class TestFindImports:
    """Tests for find_imports function."""

    def test_find_simple_import(self):
        """Test finding simple imports."""
        code = "import os\n"
        result = parse_code(code)
        imports = find_imports(result.tree)

        assert len(imports) > 0
        assert "os" in imports

    def test_find_from_import(self):
        """Test finding from imports."""
        code = "from pathlib import Path\n"
        result = parse_code(code)
        imports = find_imports(result.tree)

        assert len(imports) > 0
        # Returns full module path like 'pathlib.Path'
        assert any("pathlib" in imp for imp in imports)


class TestFindSymbol:
    """Tests for find_symbol function."""

    def test_find_function_symbol(self):
        """Test finding a function by name."""
        code = """
def hello():
    pass
"""
        result = parse_code(code)
        symbol = find_symbol(result.tree, "hello")

        assert symbol is not None
        assert symbol.name == "hello"

    def test_find_nonexistent_symbol(self):
        """Test finding a nonexistent symbol."""
        code = "def hello(): pass\n"
        result = parse_code(code)
        symbol = find_symbol(result.tree, "nonexistent")

        assert symbol is None


class TestGetFunctionInfo:
    """Tests for get_function_info function."""

    def test_get_function_with_docstring(self):
        """Test getting function info with docstring."""
        code = '''
def hello():
    """A hello function."""
    pass
'''
        result = parse_code(code)
        func_node = find_symbol(result.tree, "hello")
        info = get_function_info(func_node)

        assert info is not None
        assert info.name == "hello"
        assert info.docstring == "A hello function."

    def test_get_function_with_args(self):
        """Test getting function info with arguments."""
        code = """
def greet(name: str, count: int = 1):
    pass
"""
        result = parse_code(code)
        func_node = find_symbol(result.tree, "greet")
        info = get_function_info(func_node)

        assert info is not None
        assert len(info.args) == 2


class TestGetClassInfo:
    """Tests for get_class_info function."""

    def test_get_class_with_methods(self):
        """Test getting class info with methods."""
        code = '''
class MyClass:
    """A test class."""

    def __init__(self):
        pass

    def method(self):
        pass
'''
        result = parse_code(code)
        class_node = find_symbol(result.tree, "MyClass")
        info = get_class_info(class_node)

        assert info is not None
        assert info.name == "MyClass"
        assert info.docstring == "A test class."
        assert len(info.methods) >= 1


class TestGetModuleInfo:
    """Tests for get_module_info function."""

    def test_get_module_info(self):
        """Test getting module info."""
        code = '''"""Module docstring."""

import os

class MyClass:
    pass

def my_function():
    pass
'''

        info = get_module_info(code)

        assert info is not None
        assert info.docstring == "Module docstring."
        # Uses 'functions' and 'classes' attributes
        assert len(info.functions) >= 1
        assert len(info.classes) >= 1


class TestCalculateComplexity:
    """Tests for calculate_complexity function."""

    def test_simple_function_complexity(self):
        """Test complexity of a simple function."""
        code = """
def simple():
    return 1
"""
        result = parse_code(code)
        func_node = find_symbol(result.tree, "simple")
        complexity = calculate_complexity(func_node)

        assert complexity >= 1

    def test_complex_function_complexity(self):
        """Test complexity of a complex function."""
        code = """
def complex_func(x):
    if x > 0:
        if x > 10:
            return "big"
        return "positive"
    elif x < 0:
        return "negative"
    return "zero"
"""
        result = parse_code(code)
        func_node = find_symbol(result.tree, "complex_func")
        complexity = calculate_complexity(func_node)

        assert complexity > 1


class TestCalculateCognitiveComplexity:
    """Tests for calculate_cognitive_complexity function."""

    def test_cognitive_complexity(self):
        """Test cognitive complexity calculation."""
        code = """
def nested():
    for i in range(10):
        if i > 5:
            while True:
                break
"""
        result = parse_code(code)
        func_node = find_symbol(result.tree, "nested")
        complexity = calculate_cognitive_complexity(func_node)

        assert complexity >= 1


class TestCalculateMaintainabilityIndex:
    """Tests for calculate_maintainability_index function."""

    def test_maintainability_index(self):
        """Test maintainability index calculation."""
        code = """
def simple():
    return 1
"""

        index = calculate_maintainability_index(code)

        assert index > 0


class TestHasDocstring:
    """Tests for has_docstring function."""

    def test_function_with_docstring(self):
        """Test function with docstring."""
        code = '''
def hello():
    """A docstring."""
    pass
'''
        result = parse_code(code)
        func_node = find_symbol(result.tree, "hello")

        assert has_docstring(func_node) is True

    def test_function_without_docstring(self):
        """Test function without docstring."""
        code = "def hello(): pass\n"
        result = parse_code(code)
        func_node = find_symbol(result.tree, "hello")

        assert has_docstring(func_node) is False


class TestGetUndocumentedFunctions:
    """Tests for get_undocumented_functions function."""

    def test_find_undocumented_functions(self):
        """Test finding undocumented functions."""
        code = '''
def documented():
    """Has docstring."""
    pass

def undocumented():
    pass
'''
        result = parse_code(code)
        undoc = get_undocumented_functions(result.tree)

        # Returns list of tuples (name, line)
        assert len(undoc) == 1
        assert undoc[0][0] == "undocumented"


class TestGetUndocumentedClasses:
    """Tests for get_undocumented_classes function."""

    def test_find_undocumented_classes(self):
        """Test finding undocumented classes."""
        code = '''
class Documented:
    """Has docstring."""
    pass

class Undocumented:
    pass
'''
        result = parse_code(code)
        undoc = get_undocumented_classes(result.tree)

        # Returns list of tuples (name, line)
        assert len(undoc) == 1
        assert undoc[0][0] == "Undocumented"


class TestCountFunctions:
    """Tests for count_functions function."""

    def test_count_functions(self):
        """Test counting functions."""
        code = """
def foo(): pass
def bar(): pass
def baz(): pass
"""
        result = parse_code(code)
        count = count_functions(result.tree)

        assert count == 3


class TestCountClasses:
    """Tests for count_classes function."""

    def test_count_classes(self):
        """Test counting classes."""
        code = """
class A: pass
class B: pass
"""
        result = parse_code(code)
        count = count_classes(result.tree)

        assert count == 2


class TestGetLineCount:
    """Tests for get_line_count function."""

    def test_get_line_count(self):
        """Test getting line count."""
        code = "line1\nline2\nline3\n"

        # Returns a dict with line statistics
        counts = get_line_count(code)

        assert isinstance(counts, dict)
        assert "total" in counts or len(counts) > 0
