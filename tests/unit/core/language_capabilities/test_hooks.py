"""Tests for code grounding hooks."""

from pathlib import Path

import pytest

from victor.core.language_capabilities import (
    CodeGroundingHook,
    LanguageCapabilityRegistry,
    validate_code_before_write,
)


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singletons between tests."""
    LanguageCapabilityRegistry.reset_instance()
    CodeGroundingHook.reset_instance()
    yield
    LanguageCapabilityRegistry.reset_instance()
    CodeGroundingHook.reset_instance()


class TestCodeGroundingHook:
    """Tests for CodeGroundingHook."""

    @pytest.fixture
    def hook(self):
        return CodeGroundingHook.instance()

    def test_singleton_instance(self):
        """Hook should be a singleton."""
        hook1 = CodeGroundingHook.instance()
        hook2 = CodeGroundingHook.instance()
        assert hook1 is hook2

    def test_validate_valid_python(self, hook):
        """Should pass valid Python code."""
        code = "def hello(): return 'world'"
        should_proceed, result = hook.validate_before_write_sync(
            code, Path("test.py")
        )

        assert should_proceed
        assert result.is_valid

    def test_validate_invalid_python(self, hook):
        """Should fail invalid Python code."""
        code = "def foo(:"
        should_proceed, result = hook.validate_before_write_sync(
            code, Path("test.py")
        )

        # Default is not strict, so should proceed with warnings
        assert should_proceed
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_validate_invalid_python_strict(self, hook):
        """Should block invalid Python code in strict mode."""
        code = "def foo(:"
        should_proceed, result = hook.validate_before_write_sync(
            code, Path("test.py"), strict=True
        )

        assert not should_proceed
        assert not result.is_valid

    def test_validate_unknown_file_type(self, hook):
        """Should allow unknown file types with warning."""
        code = "some content"
        should_proceed, result = hook.validate_before_write_sync(
            code, Path("unknown.xyz")
        )

        assert should_proceed
        assert result.is_valid
        assert len(result.warnings) > 0

    def test_quick_validate(self, hook):
        """Should do quick validation."""
        # Valid
        assert hook.quick_validate("def foo(): pass", Path("test.py"))

        # Invalid
        assert not hook.quick_validate("def foo(:", Path("test.py"))

    def test_can_validate(self, hook):
        """Should check if validation is available."""
        assert hook.can_validate(Path("test.py"))
        assert not hook.can_validate(Path("unknown.xyz"))

    def test_string_path_conversion(self, hook):
        """Should handle string paths."""
        code = "def hello(): return 'world'"
        should_proceed, result = hook.validate_before_write_sync(
            code, Path("test.py")
        )

        assert should_proceed
        assert result.is_valid


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_validate_code_before_write(self):
        """Should validate using convenience function."""
        should_proceed, result = validate_code_before_write(
            "def hello(): pass",
            Path("test.py"),
        )

        assert should_proceed
        assert result.is_valid

    def test_validate_code_before_write_strict(self):
        """Should validate in strict mode."""
        should_proceed, result = validate_code_before_write(
            "def foo(:",
            Path("test.py"),
            strict=True,
        )

        assert not should_proceed
        assert not result.is_valid


@pytest.mark.asyncio
class TestAsyncValidation:
    """Tests for async validation."""

    @pytest.fixture
    def hook(self):
        return CodeGroundingHook.instance()

    async def test_async_validate_valid_python(self, hook):
        """Should validate async."""
        code = "async def hello(): return 'world'"
        should_proceed, result = await hook.validate_before_write(
            code, Path("test.py")
        )

        assert should_proceed
        assert result.is_valid

    async def test_async_validate_invalid_python(self, hook):
        """Should validate async with errors."""
        code = "def foo(:"
        should_proceed, result = await hook.validate_before_write(
            code, Path("test.py"), strict=True
        )

        assert not should_proceed
        assert not result.is_valid
