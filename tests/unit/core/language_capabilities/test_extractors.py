"""Tests for code extractors."""

from pathlib import Path

import pytest

from victor.core.language_capabilities import (
    LanguageCapabilityRegistry,
    UnifiedLanguageExtractor,
)
from victor.core.language_capabilities.extractors import (
    PythonASTExtractor,
    TreeSitterExtractor,
)


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset registry singleton between tests."""
    LanguageCapabilityRegistry.reset_instance()
    yield
    LanguageCapabilityRegistry.reset_instance()


class TestPythonASTExtractor:
    """Tests for Python AST extractor."""

    @pytest.fixture
    def extractor(self):
        return PythonASTExtractor()

    def test_extract_function(self, extractor):
        """Should extract function definitions."""
        code = '''
def hello(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}!"
'''
        symbols = extractor.extract(code, Path("test.py"))

        assert len(symbols) == 1
        assert symbols[0].name == "hello"
        assert symbols[0].symbol_type == "function"
        assert symbols[0].return_type == "str"
        assert "name: str" in symbols[0].parameters
        assert symbols[0].docstring == "Say hello."

    def test_extract_class(self, extractor):
        """Should extract class definitions."""
        code = '''
class Greeter:
    """A greeting class."""

    def greet(self, name: str) -> None:
        print(f"Hello, {name}!")
'''
        symbols = extractor.extract(code, Path("test.py"))

        assert len(symbols) == 2

        greeter = next(s for s in symbols if s.name == "Greeter")
        assert greeter.symbol_type == "class"
        assert greeter.docstring == "A greeting class."

        greet = next(s for s in symbols if s.name == "greet")
        assert greet.symbol_type == "method"
        assert greet.parent_symbol == "Greeter"

    def test_extract_async_function(self, extractor):
        """Should detect async functions."""
        code = """
async def fetch_data(url: str) -> dict:
    return {}
"""
        symbols = extractor.extract(code, Path("test.py"))

        assert len(symbols) == 1
        assert symbols[0].name == "fetch_data"
        assert symbols[0].is_async

    def test_extract_decorated_function(self, extractor):
        """Should extract decorators."""
        code = """
@staticmethod
@cache
def compute(x: int) -> int:
    return x * 2
"""
        symbols = extractor.extract(code, Path("test.py"))

        assert len(symbols) == 1
        assert "staticmethod" in symbols[0].decorators
        assert "cache" in symbols[0].decorators

    def test_extract_class_inheritance(self, extractor):
        """Should extract class inheritance."""
        code = """
class MyClass(BaseClass, Mixin):
    pass
"""
        symbols = extractor.extract(code, Path("test.py"))

        assert len(symbols) == 1
        assert symbols[0].name == "MyClass"
        assert symbols[0].metadata.get("bases") == ["BaseClass", "Mixin"]

    def test_extract_visibility(self, extractor):
        """Should detect visibility."""
        code = """
def public_func(): pass
def _protected_func(): pass
def __private_func(): pass
"""
        symbols = extractor.extract(code, Path("test.py"))

        public = next(s for s in symbols if s.name == "public_func")
        assert public.visibility == "public"

        protected = next(s for s in symbols if s.name == "_protected_func")
        assert protected.visibility == "protected"

        private = next(s for s in symbols if s.name == "__private_func")
        assert private.visibility == "private"

    def test_extract_nested_class(self, extractor):
        """Should extract nested classes."""
        code = """
class Outer:
    class Inner:
        def method(self): pass
"""
        symbols = extractor.extract(code, Path("test.py"))

        assert len(symbols) == 3

        inner = next(s for s in symbols if s.name == "Inner")
        assert inner.parent_symbol == "Outer"

        method = next(s for s in symbols if s.name == "method")
        assert method.parent_symbol == "Inner"

    def test_handle_syntax_error(self, extractor):
        """Should handle syntax errors gracefully."""
        code = "def foo(:"  # Invalid syntax

        symbols = extractor.extract(code, Path("test.py"))

        assert symbols == []

    def test_can_parse(self, extractor):
        """Should check if code can be parsed."""
        assert extractor.can_parse("def foo(): pass")
        assert not extractor.can_parse("def foo(:")


class TestTreeSitterExtractor:
    """Tests for tree-sitter extractor."""

    @pytest.fixture
    def extractor(self):
        return TreeSitterExtractor()

    def test_is_available(self, extractor):
        """Should check availability."""
        is_available = extractor.is_available()
        assert isinstance(is_available, bool)

    @pytest.mark.skipif(
        not TreeSitterExtractor().is_available(), reason="tree-sitter not available"
    )
    def test_extract_python(self, extractor):
        """Should extract Python symbols."""
        code = """
def hello():
    pass

class Greeter:
    def greet(self):
        pass
"""
        symbols = extractor.extract(code, Path("test.py"), "python")

        names = [s.name for s in symbols]
        assert "hello" in names
        assert "Greeter" in names
        # greet might or might not be extracted depending on query

    @pytest.mark.skipif(
        not TreeSitterExtractor().is_available(), reason="tree-sitter not available"
    )
    def test_has_syntax_errors(self, extractor):
        """Should detect syntax errors."""
        assert extractor.has_syntax_errors("def foo(:", "python")
        assert not extractor.has_syntax_errors("def foo(): pass", "python")

    @pytest.mark.skipif(
        not TreeSitterExtractor().is_available(), reason="tree-sitter not available"
    )
    def test_get_error_locations(self, extractor):
        """Should get error locations."""
        errors = extractor.get_error_locations("def foo(:", "python")

        assert len(errors) > 0
        assert "line" in errors[0]
        assert "column" in errors[0]

    def test_get_supported_languages(self, extractor):
        """Should list supported languages."""
        languages = extractor.get_supported_languages()

        assert "python" in languages
        assert "javascript" in languages
        assert "typescript" in languages


class TestUnifiedLanguageExtractor:
    """Tests for unified language extractor."""

    @pytest.fixture
    def extractor(self):
        return UnifiedLanguageExtractor()

    def test_extract_python(self, extractor):
        """Should extract Python symbols."""
        code = """
def hello(name: str) -> str:
    return f"Hello, {name}!"
"""
        symbols = extractor.extract_symbols(code, Path("test.py"))

        assert len(symbols) == 1
        assert symbols[0].name == "hello"
        assert symbols[0].symbol_type == "function"

    def test_extract_with_language_override(self, extractor):
        """Should respect language override."""
        code = "def hello(): pass"
        symbols = extractor.extract_symbols(code, Path("unknown.xyz"), language="python")

        assert len(symbols) == 1
        assert symbols[0].name == "hello"

    def test_can_extract(self, extractor):
        """Should check if extraction is available."""
        assert extractor.can_extract(Path("test.py"))
        assert not extractor.can_extract(Path("unknown.xyz"))

    def test_get_extraction_method(self, extractor):
        """Should get extraction method."""
        from victor.core.language_capabilities import ASTAccessMethod

        method = extractor.get_extraction_method(Path("test.py"))
        assert method == ASTAccessMethod.NATIVE

    def test_unknown_file_type(self, extractor):
        """Should handle unknown file types."""
        symbols = extractor.extract_symbols("content", Path("unknown.xyz"))
        assert symbols == []
