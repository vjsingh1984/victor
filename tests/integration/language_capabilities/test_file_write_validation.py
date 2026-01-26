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

"""Integration tests for file write validation.

Tests the end-to-end validation flow when writing files through
FileEditor and file_editor_tool.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from victor.core.language_capabilities import (
    CodeGroundingHook,
    LanguageCapabilityRegistry,
    ValidationSeverity,
)


class TestFileWriteValidation:
    """Test file write validation integration."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def hook(self):
        """Get the code grounding hook."""
        return CodeGroundingHook.instance()

    @pytest.fixture
    def registry(self):
        """Get the language capability registry."""
        return LanguageCapabilityRegistry.instance()

    def test_valid_python_passes_validation(self, hook, temp_dir):
        """Test that valid Python code passes validation."""
        file_path = temp_dir / "valid.py"
        content = '''
def greet(name: str) -> str:
    """Return a greeting message."""
    return f"Hello, {name}!"

class Calculator:
    def add(self, a: int, b: int) -> int:
        return a + b
'''
        should_proceed, result = hook.validate_before_write_sync(content, file_path, strict=False)

        assert should_proceed is True
        assert result.is_valid is True
        assert result.language == "python"

    def test_invalid_python_blocked_in_strict_mode(self, hook, temp_dir):
        """Test that invalid Python code is blocked in strict mode."""
        file_path = temp_dir / "invalid.py"
        content = "def foo(:"  # Syntax error

        should_proceed, result = hook.validate_before_write_sync(content, file_path, strict=True)

        assert should_proceed is False
        assert result.is_valid is False
        assert result.language == "python"
        assert len(result.issues) > 0
        assert any(issue.severity == ValidationSeverity.ERROR for issue in result.issues)

    def test_invalid_python_warns_in_non_strict_mode(self, hook, temp_dir):
        """Test that invalid Python code generates warnings in non-strict mode."""
        file_path = temp_dir / "invalid.py"
        content = "def foo(:"  # Syntax error

        should_proceed, result = hook.validate_before_write_sync(content, file_path, strict=False)

        # In non-strict mode, validation still fails but write may proceed
        # depending on fallback_on_error setting
        assert result.is_valid is False
        assert result.language == "python"

    def test_unknown_file_type_allowed(self, hook, temp_dir):
        """Test that unknown file types are allowed with warning."""
        file_path = temp_dir / "unknown.xyz"
        content = "some content that cannot be validated"

        should_proceed, result = hook.validate_before_write_sync(content, file_path, strict=False)

        assert should_proceed is True
        assert result.language == "unknown"
        assert len(result.warnings) > 0

    def test_typescript_validation(self, hook, temp_dir):
        """Test TypeScript validation (tree-sitter based)."""
        file_path = temp_dir / "test.ts"
        content = """
interface User {
    name: string;
    age: number;
}

function greet(user: User): string {
    return `Hello, ${user.name}!`;
}
"""
        should_proceed, result = hook.validate_before_write_sync(content, file_path, strict=False)

        assert should_proceed is True
        assert result.language == "typescript"

    def test_javascript_validation(self, hook, temp_dir):
        """Test JavaScript validation (tree-sitter based)."""
        file_path = temp_dir / "test.js"
        content = """
function greet(name) {
    return `Hello, ${name}!`;
}

const calculator = {
    add: (a, b) => a + b,
    subtract: (a, b) => a - b,
};
"""
        should_proceed, result = hook.validate_before_write_sync(content, file_path, strict=False)

        assert should_proceed is True
        assert result.language == "javascript"

    def test_go_validation(self, hook, temp_dir):
        """Test Go validation (tree-sitter based)."""
        file_path = temp_dir / "main.go"
        content = """
package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}

type Calculator struct {
    value int
}

func (c *Calculator) Add(n int) int {
    return c.value + n
}
"""
        should_proceed, result = hook.validate_before_write_sync(content, file_path, strict=False)

        assert should_proceed is True
        assert result.language == "go"

    def test_rust_validation(self, hook, temp_dir):
        """Test Rust validation (tree-sitter based)."""
        file_path = temp_dir / "main.rs"
        content = """
fn main() {
    println!("Hello, World!");
}

struct Calculator {
    value: i32,
}

impl Calculator {
    fn new(value: i32) -> Self {
        Calculator { value }
    }

    fn add(&self, n: i32) -> i32 {
        self.value + n
    }
}
"""
        should_proceed, result = hook.validate_before_write_sync(content, file_path, strict=False)

        assert should_proceed is True
        assert result.language == "rust"

    def test_java_validation(self, hook, temp_dir):
        """Test Java validation (tree-sitter based)."""
        file_path = temp_dir / "Main.java"
        content = """
public class Main {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }

    public int add(int a, int b) {
        return a + b;
    }
}
"""
        should_proceed, result = hook.validate_before_write_sync(content, file_path, strict=False)

        assert should_proceed is True
        assert result.language == "java"

    def test_quick_validate_valid_code(self, hook, temp_dir):
        """Test quick_validate with valid code."""
        file_path = temp_dir / "test.py"
        content = "def foo(): pass"

        is_valid = hook.quick_validate(content, file_path)
        assert is_valid is True

    def test_quick_validate_invalid_code(self, hook, temp_dir):
        """Test quick_validate with invalid code."""
        file_path = temp_dir / "test.py"
        content = "def foo(:"

        is_valid = hook.quick_validate(content, file_path)
        assert is_valid is False

    def test_registry_used_for_language_detection(self, hook, registry, temp_dir):
        """Test that the registry is used for language detection."""
        file_path = temp_dir / "test.py"

        # Verify Python is in registry
        cap = registry.get_for_file(file_path)
        assert cap is not None
        assert cap.name == "python"

        # Validate using the hook
        content = "x = 1"
        should_proceed, result = hook.validate_before_write_sync(content, file_path, strict=False)
        assert result.language == "python"


class TestFileEditorValidationIntegration:
    """Test FileEditor validation integration."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_editor(self):
        """Create a mock file editor."""
        editor = MagicMock()
        editor.current_transaction = MagicMock()
        return editor

    def test_validation_enabled_by_default(self):
        """Test that validation is enabled by default in FileEditor."""
        # Import here to avoid circular imports
        from victor.coding.editing.editor import FileEditor

        # Check the default signature
        import inspect

        sig = inspect.signature(FileEditor.commit)
        params = sig.parameters

        # Verify validate parameter defaults to True
        assert "validate" in params
        assert params["validate"].default is True

    def test_strict_validation_disabled_by_default(self):
        """Test that strict_validation is disabled by default."""
        from victor.coding.editing.editor import FileEditor

        import inspect

        sig = inspect.signature(FileEditor.commit)
        params = sig.parameters

        assert "strict_validation" in params
        assert params["strict_validation"].default is False


class TestFileEditorToolValidation:
    """Test file_editor_tool.py validation integration."""

    def test_validation_parameters_in_edit_function(self):
        """Test that validation parameters exist in edit function."""
        from victor.tools.file_editor_tool import edit

        import inspect

        sig = inspect.signature(edit)
        params = sig.parameters

        # Check validate parameter
        assert "validate" in params
        assert params["validate"].default is True

        # Check strict_validation parameter
        assert "strict_validation" in params
        assert params["strict_validation"].default is False


class TestValidationWithFeatureFlags:
    """Test validation behavior with feature flags."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_validation_respects_feature_flags(self, temp_dir):
        """Test that validation respects feature flag settings."""
        from victor.core.language_capabilities import (
            FeatureFlagManager,
            GlobalFeatureFlags,
        )

        # Create a manager with validation disabled
        manager = FeatureFlagManager()
        manager._global = GlobalFeatureFlags(validation_enabled=False)

        # Validation should be disabled
        assert manager.is_validation_enabled("python") is False

    def test_per_language_override(self, temp_dir):
        """Test per-language feature flag overrides."""
        from victor.core.language_capabilities import (
            FeatureFlagManager,
            GlobalFeatureFlags,
            LanguageFeatureFlags,
        )

        manager = FeatureFlagManager()
        manager._global = GlobalFeatureFlags(
            validation_enabled=True,
            language_overrides={"python": LanguageFeatureFlags(validation_enabled=False)},
        )

        # Python should have validation disabled
        assert manager.is_validation_enabled("python") is False

        # Other languages should have validation enabled
        assert manager.is_validation_enabled("typescript") is True


class TestValidationPipeline:
    """Test the validation pipeline end-to-end."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def validator(self):
        """Get the unified validator."""
        from victor.core.language_capabilities import UnifiedLanguageValidator

        return UnifiedLanguageValidator()

    def test_validation_pipeline(self, validator, temp_dir):
        """Test the validation pipeline."""
        file_path = temp_dir / "test.py"
        content = "def greet(name): return f'Hello, {name}!'"

        result = validator.validate(content, file_path)

        assert result.is_valid is True
        assert result.language == "python"

    def test_validation_with_errors(self, validator, temp_dir):
        """Test validation with syntax errors."""
        file_path = temp_dir / "test.py"
        content = "def foo(\n"  # Missing closing parenthesis

        result = validator.validate(content, file_path)

        assert result.is_valid is False
        assert result.language == "python"
        assert len(result.issues) > 0

    def test_multiple_errors(self, validator, temp_dir):
        """Test validation with multiple errors."""
        file_path = temp_dir / "test.py"
        content = """
def foo(:
    pass

def bar(
    pass
"""
        result = validator.validate(content, file_path)

        assert result.is_valid is False
        # Should have at least one error for the first syntax error
        assert len(result.issues) > 0

    def test_empty_file_validation(self, validator, temp_dir):
        """Test validation of empty files."""
        file_path = temp_dir / "empty.py"
        content = ""

        result = validator.validate(content, file_path)

        assert result.is_valid is True
        assert result.language == "python"

    def test_whitespace_only_file(self, validator, temp_dir):
        """Test validation of whitespace-only files."""
        file_path = temp_dir / "whitespace.py"
        content = "   \n\n   \t\t   \n"

        result = validator.validate(content, file_path)

        assert result.is_valid is True
        assert result.language == "python"
