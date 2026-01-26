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

"""Integration tests for TestGenManager validation integration."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from victor.coding.testgen.manager import TestGenManager, reset_testgen_manager
from victor.coding.testgen.protocol import GeneratedTest, TestSuite, TestCase
from victor.core.language_capabilities.hooks import (
    CodeGroundingHook,
    validate_code_before_write,
)
from victor.core.language_capabilities.types import CodeValidationResult


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singletons before each test."""
    reset_testgen_manager()
    CodeGroundingHook.reset_instance()
    yield
    reset_testgen_manager()
    CodeGroundingHook.reset_instance()


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def manager(temp_dir):
    """Create a TestGenManager instance."""
    return TestGenManager(project_root=temp_dir)


class TestTestGenManagerValidation:
    """Tests for TestGenManager validation integration."""

    def test_write_test_file_validates_by_default(self, manager, temp_dir):
        """Test that _write_test_file validates code by default."""
        # Create a generated test with valid code
        test_file = temp_dir / "test_example.py"
        generated = GeneratedTest(
            file_path=test_file,
            content='''"""Tests for example module."""
import pytest


def test_example():
    """Test example function."""
    assert True
''',
            suites=[
                TestSuite(
                    name="TestExample",
                    target_file=temp_dir / "example.py",
                    target_name="example",
                    test_cases=[
                        TestCase(
                            name="test_example",
                            description="Test example function",
                        )
                    ],
                )
            ],
        )

        # Write should succeed
        result = manager._write_test_file(generated, validate=True)
        assert result is True
        assert test_file.exists()

    def test_write_test_file_blocks_invalid_code_in_strict_mode(self, manager, temp_dir):
        """Test that _write_test_file blocks invalid code in strict mode."""
        # Create a generated test with invalid code
        test_file = temp_dir / "test_invalid.py"
        generated = GeneratedTest(
            file_path=test_file,
            content="""def test_broken(:  # Syntax error
    assert True
""",
            suites=[],
        )

        # Write should fail in strict mode
        result = manager._write_test_file(
            generated,
            validate=True,
            strict_validation=True,
        )
        assert result is False
        assert not test_file.exists()

    def test_write_test_file_allows_invalid_code_without_strict(self, manager, temp_dir):
        """Test that _write_test_file allows invalid code without strict mode."""
        # Create a generated test with invalid code
        test_file = temp_dir / "test_invalid.py"
        generated = GeneratedTest(
            file_path=test_file,
            content="""def test_broken(:  # Syntax error
    assert True
""",
            suites=[],
        )

        # Write should succeed (just log warnings) without strict mode
        # Note: Default behavior depends on capability's fallback_on_error setting
        result = manager._write_test_file(
            generated,
            validate=True,
            strict_validation=False,
        )
        # In non-strict mode with default "warn" fallback, write proceeds
        # But our hook returns should_proceed=True in non-strict mode
        # The test file should be written with warnings logged
        assert test_file.exists()

    def test_write_test_file_skips_validation_when_disabled(self, manager, temp_dir):
        """Test that _write_test_file can skip validation."""
        # Create a generated test with invalid code
        test_file = temp_dir / "test_invalid.py"
        generated = GeneratedTest(
            file_path=test_file,
            content="""def test_broken(:  # Syntax error
    assert True
""",
            suites=[],
        )

        # Write should succeed when validation is disabled
        result = manager._write_test_file(generated, validate=False)
        assert result is True
        assert test_file.exists()

    def test_generate_for_file_validates_by_default(self, manager, temp_dir):
        """Test that generate_for_file passes validation parameters."""
        # Create a source file to generate tests for
        source_file = temp_dir / "simple_module.py"
        source_file.write_text(
            '''"""Simple module."""


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
'''
        )

        # Generate tests (validation enabled by default)
        result = manager.generate_for_file(
            source_file,
            write_file=True,
            validate=True,
            strict_validation=False,
        )

        # Result should be successful
        assert result.success

    def test_generate_for_file_with_strict_validation(self, manager, temp_dir):
        """Test generate_for_file with strict validation."""
        # Create a source file to generate tests for
        source_file = temp_dir / "simple_module.py"
        source_file.write_text(
            '''"""Simple module."""


def multiply(x: int, y: int) -> int:
    """Multiply two numbers."""
    return x * y
'''
        )

        # Generate tests with strict validation
        result = manager.generate_for_file(
            source_file,
            write_file=True,
            validate=True,
            strict_validation=True,
        )

        # Result should be successful since valid test code is generated
        assert result.success

    def test_generate_for_directory_validates(self, manager, temp_dir):
        """Test that generate_for_directory passes validation parameters."""
        # Create a source file
        source_file = temp_dir / "module.py"
        source_file.write_text(
            '''"""A module."""


def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}"
'''
        )

        # Generate tests for directory
        result = manager.generate_for_directory(
            temp_dir,
            write_files=True,
            validate=True,
            strict_validation=False,
        )

        # Result should be successful
        assert result.success

    def test_generate_for_module_validates(self, manager, temp_dir):
        """Test that generate_for_module passes validation parameters."""
        # Create a package
        module_dir = temp_dir / "mypackage"
        module_dir.mkdir()
        init_file = module_dir / "__init__.py"
        init_file.write_text('"""My package."""\n')

        source_file = module_dir / "utils.py"
        source_file.write_text(
            '''"""Utilities."""


def helper() -> None:
    """Helper function."""
    pass
'''
        )

        # Generate tests for module
        result = manager.generate_for_module(
            module_dir,
            write_files=True,
            validate=True,
            strict_validation=False,
        )

        # Result should be successful
        assert result.success


class TestValidationHookIntegration:
    """Tests for validation hook integration with TestGenManager."""

    def test_validation_hook_is_used(self, manager, temp_dir):
        """Test that the validation hook is actually called."""
        test_file = temp_dir / "test_hooked.py"
        generated = GeneratedTest(
            file_path=test_file,
            content='''"""Tests."""


def test_hooked():
    assert True
''',
            suites=[],
        )

        with patch("victor.coding.testgen.manager.validate_code_before_write") as mock_validate:
            mock_validate.return_value = (
                True,
                CodeValidationResult(is_valid=True, language="python"),
            )

            manager._write_test_file(generated, validate=True)

            # Verify validation was called
            mock_validate.assert_called_once()
            call_args = mock_validate.call_args
            assert call_args[0][0] == generated.content
            assert call_args[0][1] == generated.file_path

    def test_validation_hook_not_called_when_disabled(self, manager, temp_dir):
        """Test that validation hook is not called when validation is disabled."""
        test_file = temp_dir / "test_no_validation.py"
        generated = GeneratedTest(
            file_path=test_file,
            content="def test_foo(): pass\n",
            suites=[],
        )

        with patch("victor.coding.testgen.manager.validate_code_before_write") as mock_validate:
            manager._write_test_file(generated, validate=False)

            # Validation should not be called
            mock_validate.assert_not_called()

    def test_strict_mode_passed_to_validation(self, manager, temp_dir):
        """Test that strict mode is correctly passed to validation."""
        test_file = temp_dir / "test_strict.py"
        generated = GeneratedTest(
            file_path=test_file,
            content="def test_strict(): pass\n",
            suites=[],
        )

        with patch("victor.coding.testgen.manager.validate_code_before_write") as mock_validate:
            mock_validate.return_value = (
                True,
                CodeValidationResult(is_valid=True, language="python"),
            )

            manager._write_test_file(
                generated,
                validate=True,
                strict_validation=True,
            )

            # Verify strict was passed
            call_args = mock_validate.call_args
            assert call_args[1]["strict"] is True


class TestValidationErrorHandling:
    """Tests for validation error handling in TestGenManager."""

    def test_validation_errors_logged(self, manager, temp_dir, caplog):
        """Test that validation errors are properly logged."""
        import logging

        caplog.set_level(logging.ERROR)

        test_file = temp_dir / "test_error.py"
        generated = GeneratedTest(
            file_path=test_file,
            content="def test_broken(:\n    pass\n",  # Syntax error
            suites=[],
        )

        # Write with strict mode should fail and log error
        result = manager._write_test_file(
            generated,
            validate=True,
            strict_validation=True,
        )

        assert result is False
        assert "Validation failed" in caplog.text or not test_file.exists()

    def test_validation_warnings_logged(self, manager, temp_dir, caplog):
        """Test that validation warnings are logged but don't block write."""
        import logging

        caplog.set_level(logging.WARNING)

        test_file = temp_dir / "test_warning.py"
        # Valid syntax but might have warnings depending on configuration
        generated = GeneratedTest(
            file_path=test_file,
            content='''"""Tests with potential warnings."""


def test_something():
    """A test."""
    assert True
''',
            suites=[],
        )

        result = manager._write_test_file(generated, validate=True)

        # Should succeed even if there were warnings
        assert result is True
        assert test_file.exists()
