"""Tests for code validators."""

from pathlib import Path

import pytest

from victor.core.language_capabilities import (
    LanguageCapabilityRegistry,
    ValidationConfig,
    ValidationSeverity,
)
from victor.core.language_capabilities.validators import (
    PythonASTValidator,
    TreeSitterValidator,
    UnifiedLanguageValidator,
)


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset registry singleton between tests."""
    LanguageCapabilityRegistry.reset_instance()
    yield
    LanguageCapabilityRegistry.reset_instance()


class TestPythonASTValidator:
    """Tests for Python AST validator."""

    @pytest.fixture
    def validator(self):
        return PythonASTValidator()

    def test_valid_python_code(self, validator):
        """Should validate correct Python code."""
        code = """
def hello(name: str) -> str:
    return f"Hello, {name}!"

class Greeter:
    def greet(self, name: str) -> None:
        print(hello(name))
"""
        result = validator.validate(code, Path("test.py"))

        assert result.is_valid
        assert result.language == "python"
        assert len(result.errors) == 0

    def test_invalid_python_syntax(self, validator):
        """Should detect Python syntax errors."""
        code = "def foo(:"  # Missing parameter and body

        result = validator.validate(code, Path("test.py"))

        assert not result.is_valid
        assert len(result.errors) == 1
        assert result.errors[0].severity == ValidationSeverity.ERROR
        assert result.errors[0].line == 1

    def test_unclosed_parenthesis(self, validator):
        """Should detect unclosed parenthesis."""
        code = """
def foo():
    print("hello"
"""
        result = validator.validate(code, Path("test.py"))

        assert not result.is_valid
        assert len(result.errors) == 1

    def test_invalid_indentation(self, validator):
        """Should detect indentation errors."""
        code = """
def foo():
print("hello")
"""
        result = validator.validate(code, Path("test.py"))

        assert not result.is_valid
        assert len(result.errors) == 1

    def test_validate_expression(self, validator):
        """Should validate Python expressions."""
        # Valid expression
        result = validator.validate_expression("x + 1")
        assert result.is_valid

        # Invalid expression
        result = validator.validate_expression("x +")
        assert not result.is_valid

    def test_validate_statement(self, validator):
        """Should validate Python statements."""
        # Valid statement
        result = validator.validate_statement("x = 1")
        assert result.is_valid

        # Invalid statement
        result = validator.validate_statement("x = ")
        assert not result.is_valid

    def test_skip_validation_when_disabled(self, validator):
        """Should skip validation when syntax check disabled."""
        code = "def foo(:"  # Invalid syntax

        config = ValidationConfig(check_syntax=False)
        result = validator.validate(code, Path("test.py"), config)

        # Should pass because syntax check is disabled
        assert result.is_valid

    def test_get_syntax_errors(self, validator):
        """Should get syntax errors list."""
        errors = validator.get_syntax_errors("def foo(:")

        assert len(errors) == 1
        assert errors[0]["line"] == 1
        assert "message" in errors[0]


class TestUnifiedLanguageValidator:
    """Tests for unified language validator."""

    @pytest.fixture
    def validator(self):
        return UnifiedLanguageValidator()

    def test_validate_python(self, validator):
        """Should validate Python code."""
        code = "def hello(): return 'world'"
        result = validator.validate(code, Path("test.py"))

        assert result.is_valid
        assert result.language == "python"
        assert "python_ast" in result.validators_used

    def test_validate_python_error(self, validator):
        """Should detect Python errors."""
        code = "def foo(:"
        result = validator.validate(code, Path("test.py"))

        assert not result.is_valid
        assert len(result.errors) > 0

    def test_validate_unknown_language(self, validator):
        """Should handle unknown languages."""
        code = "some unknown content"
        result = validator.validate(code, Path("unknown.xyz"))

        # Unknown languages are allowed with warning
        assert result.is_valid
        assert result.language == "unknown"
        assert len(result.warnings) > 0

    def test_can_validate(self, validator):
        """Should check if validation is available."""
        assert validator.can_validate(Path("test.py"))
        assert not validator.can_validate(Path("unknown.xyz"))

    def test_get_validation_method(self, validator):
        """Should get validation method for file."""
        from victor.core.language_capabilities import ASTAccessMethod

        method = validator.get_validation_method(Path("test.py"))
        assert method == ASTAccessMethod.NATIVE

    def test_quick_validate(self, validator):
        """Should do quick syntax validation."""
        # Valid code
        assert validator.quick_validate("def foo(): pass", Path("test.py"))

        # Invalid code
        assert not validator.quick_validate("def foo(:", Path("test.py"))

        # Unknown file type (allows)
        assert validator.quick_validate("unknown", Path("test.xyz"))


class TestTreeSitterValidator:
    """Tests for tree-sitter validator."""

    @pytest.fixture
    def validator(self):
        from victor.core.language_capabilities.validators import TreeSitterValidator

        return TreeSitterValidator()

    def test_is_available(self, validator):
        """Should check if tree-sitter is available."""
        # This depends on whether tree-sitter is installed
        # Just check it doesn't raise
        is_available = validator.is_available()
        assert isinstance(is_available, bool)

    @pytest.mark.skipif(
        not TreeSitterValidator().is_available(), reason="tree-sitter not available"
    )
    def test_validate_python_with_tree_sitter(self, validator):
        """Should validate Python with tree-sitter."""
        code = "def hello(): return 'world'"
        result = validator.validate(code, Path("test.py"), "python")

        assert result.is_valid
        assert "tree_sitter" in result.validators_used

    @pytest.mark.skipif(
        not TreeSitterValidator().is_available(), reason="tree-sitter not available"
    )
    def test_validate_python_error_with_tree_sitter(self, validator):
        """Should detect errors with tree-sitter."""
        code = "def foo(:"
        result = validator.validate(code, Path("test.py"), "python")

        assert not result.is_valid
        assert len(result.errors) > 0

    def test_get_supported_languages(self, validator):
        """Should list supported languages."""
        languages = validator.get_supported_languages()

        assert "python" in languages
        assert "javascript" in languages
        assert "typescript" in languages
