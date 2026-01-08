# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for code validators and self-correction system.

This module tests:
- Python validator (AST-based validation)
- JavaScript validator
- Go validator
- Rust validator
- Java validator
- Generic validator (fallback)
- Registry auto-discovery
- Language detection
- SelfCorrector orchestrator
- FeedbackGenerator
"""

import pytest

from victor.evaluation.correction import (
    Language,
    CodeValidationResult,
    CorrectionFeedback,
    create_self_corrector,
    CodeValidatorRegistry,
    detect_language,
    get_feedback_generator,
)
from victor.evaluation.correction.validators import (
    PythonCodeValidator,
    JavaScriptCodeValidator,
    GoCodeValidator,
    RustCodeValidator,
    JavaCodeValidator,
    GenericCodeValidator,
)
from victor.evaluation.correction.base import ValidatorCapabilities


class TestPythonCodeValidator:
    """Tests for Python code validator."""

    @pytest.fixture
    def validator(self):
        return PythonCodeValidator()

    def test_supported_languages(self, validator):
        """Test that validator supports Python."""
        assert Language.PYTHON in validator.supported_languages

    def test_validate_valid_code(self, validator):
        """Test validation of valid Python code."""
        code = """
def add(a, b):
    return a + b

result = add(1, 2)
"""
        result = validator.validate(code)
        assert result.valid is True
        assert result.syntax_valid is True
        assert len(result.errors) == 0

    def test_validate_syntax_error(self, validator):
        """Test detection of syntax errors."""
        code = """
def broken(
    return x
"""
        result = validator.validate(code)
        assert result.valid is False
        assert result.syntax_valid is False
        assert len(result.errors) > 0

    def test_validate_missing_colon(self, validator):
        """Test detection of missing colon in function def."""
        code = """
def missing_colon()
    return 42
"""
        result = validator.validate(code)
        assert result.valid is False
        assert result.syntax_valid is False

    def test_validate_unbalanced_parens(self, validator):
        """Test detection of unbalanced parentheses."""
        code = """
def unbalanced(x):
    return (x + 1
"""
        result = validator.validate(code)
        assert result.valid is False
        assert result.syntax_valid is False

    def test_clean_markdown_python_block(self, validator):
        """Test markdown code block cleaning."""
        code = """```python
def hello():
    return "world"
```"""
        cleaned = validator.clean_markdown(code)
        assert "```" not in cleaned
        assert "def hello():" in cleaned

    def test_clean_markdown_generic_block(self, validator):
        """Test cleaning of generic code blocks."""
        code = """```
def hello():
    return "world"
```"""
        cleaned = validator.clean_markdown(code)
        assert "```" not in cleaned

    def test_fix_missing_imports(self, validator):
        """Test auto-fix for missing standard library imports."""
        code = """
def calculate():
    return math.sqrt(16)
"""
        result = validator.validate(code)
        fixed = validator.fix(code, result)
        # After fix, should have math import
        assert "import math" in fixed or result.valid

    def test_validate_class_definition(self, validator):
        """Test validation of class definitions."""
        code = """
class MyClass:
    def __init__(self, value):
        self.value = value

    def get_value(self):
        return self.value
"""
        result = validator.validate(code)
        assert result.valid is True
        assert result.syntax_valid is True

    def test_validate_async_function(self, validator):
        """Test validation of async functions."""
        code = """
async def fetch_data():
    await some_async_call()
    return data
"""
        result = validator.validate(code)
        # Should parse correctly even if some_async_call undefined
        assert result.syntax_valid is True

    def test_validate_type_hints(self, validator):
        """Test validation of type hints."""
        code = """
from typing import List, Optional

def process(items: List[int]) -> Optional[int]:
    if items:
        return sum(items)
    return None
"""
        result = validator.validate(code)
        assert result.valid is True
        assert result.syntax_valid is True

    def test_capabilities(self, validator):
        """Test validator capabilities."""
        assert validator.capabilities.has_ast_parsing is True
        assert validator.capabilities.has_import_detection is True
        assert validator.capabilities.has_auto_fix is True


class TestJavaScriptCodeValidator:
    """Tests for JavaScript code validator."""

    @pytest.fixture
    def validator(self):
        return JavaScriptCodeValidator()

    def test_supported_languages(self, validator):
        """Test supported languages."""
        assert Language.JAVASCRIPT in validator.supported_languages
        assert Language.TYPESCRIPT in validator.supported_languages

    def test_validate_valid_code(self, validator):
        """Test validation of valid JS code."""
        code = """
function add(a, b) {
    return a + b;
}

const result = add(1, 2);
"""
        result = validator.validate(code)
        assert result.syntax_valid is True

    def test_validate_arrow_function(self, validator):
        """Test validation of arrow functions."""
        code = """
const multiply = (a, b) => a * b;
const square = x => x * x;
"""
        result = validator.validate(code)
        assert result.syntax_valid is True

    def test_validate_unbalanced_braces(self, validator):
        """Test detection of unbalanced braces."""
        code = """
function broken() {
    return {
        key: "value"
    // missing closing brace
"""
        result = validator.validate(code)
        assert result.valid is False

    def test_validate_async_await(self, validator):
        """Test validation of async/await syntax."""
        code = """
async function fetchData() {
    const response = await fetch(url);
    return response.json();
}
"""
        result = validator.validate(code)
        assert result.syntax_valid is True

    def test_clean_markdown(self, validator):
        """Test markdown cleaning."""
        code = """```javascript
const x = 1;
```"""
        cleaned = validator.clean_markdown(code)
        assert "```" not in cleaned


class TestGoCodeValidator:
    """Tests for Go code validator."""

    @pytest.fixture
    def validator(self):
        return GoCodeValidator()

    def test_supported_languages(self, validator):
        """Test that validator supports Go."""
        assert Language.GO in validator.supported_languages

    def test_validate_valid_code(self, validator):
        """Test validation of valid Go code."""
        code = """
package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}
"""
        result = validator.validate(code)
        assert result.syntax_valid is True

    def test_validate_function_signature(self, validator):
        """Test validation of Go function signatures."""
        code = """
func add(a, b int) int {
    return a + b
}
"""
        result = validator.validate(code)
        assert result.syntax_valid is True

    def test_validate_struct(self, validator):
        """Test validation of Go structs."""
        code = """
type Person struct {
    Name string
    Age  int
}
"""
        result = validator.validate(code)
        assert result.syntax_valid is True

    def test_validate_unbalanced_braces(self, validator):
        """Test detection of unbalanced braces."""
        code = """
func broken() {
    return 1
"""
        result = validator.validate(code)
        assert result.valid is False


class TestRustCodeValidator:
    """Tests for Rust code validator."""

    @pytest.fixture
    def validator(self):
        return RustCodeValidator()

    def test_supported_languages(self, validator):
        """Test that validator supports Rust."""
        assert Language.RUST in validator.supported_languages

    def test_validate_valid_code(self, validator):
        """Test validation of valid Rust code."""
        code = """
fn main() {
    let x = 5;
    println!("x = {}", x);
}
"""
        result = validator.validate(code)
        assert result.syntax_valid is True

    def test_validate_function_with_return(self, validator):
        """Test validation of Rust functions with return types."""
        code = """
fn add(a: i32, b: i32) -> i32 {
    a + b
}
"""
        result = validator.validate(code)
        assert result.syntax_valid is True

    def test_validate_struct(self, validator):
        """Test validation of Rust structs."""
        code = """
struct Point {
    x: f64,
    y: f64,
}
"""
        result = validator.validate(code)
        assert result.syntax_valid is True


class TestJavaCodeValidator:
    """Tests for Java code validator."""

    @pytest.fixture
    def validator(self):
        return JavaCodeValidator()

    def test_supported_languages(self, validator):
        """Test that validator supports Java."""
        assert Language.JAVA in validator.supported_languages

    def test_validate_valid_code(self, validator):
        """Test validation of valid Java code."""
        code = """
public class Hello {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
"""
        result = validator.validate(code)
        assert result.syntax_valid is True

    def test_validate_method(self, validator):
        """Test validation of Java methods."""
        code = """
public int add(int a, int b) {
    return a + b;
}
"""
        result = validator.validate(code)
        assert result.syntax_valid is True


class TestGenericCodeValidator:
    """Tests for generic fallback validator."""

    @pytest.fixture
    def validator(self):
        return GenericCodeValidator()

    def test_supported_languages(self, validator):
        """Test that generic validator supports UNKNOWN."""
        assert Language.UNKNOWN in validator.supported_languages

    def test_validate_any_code(self, validator):
        """Test that generic validator accepts most code."""
        code = "some arbitrary code that should pass basic validation"
        result = validator.validate(code)
        # Generic validator should be lenient
        assert result.syntax_valid is True

    def test_clean_markdown(self, validator):
        """Test markdown cleaning in generic validator."""
        code = "```\ncode here\n```"
        cleaned = validator.clean_markdown(code)
        assert "```" not in cleaned


class TestCodeValidatorRegistry:
    """Tests for validator registry."""

    @pytest.fixture
    def registry(self):
        # Create fresh registry for each test
        CodeValidatorRegistry.reset_singleton()
        r = CodeValidatorRegistry()
        r.reset()
        return r

    def test_singleton_pattern(self):
        """Test that registry is a singleton."""
        CodeValidatorRegistry.reset_singleton()
        r1 = CodeValidatorRegistry()
        r2 = CodeValidatorRegistry()
        assert r1 is r2

    def test_discover_validators(self, registry):
        """Test auto-discovery of validators."""
        count = registry.discover_validators()
        assert count >= 5  # At least Python, JS, Go, Rust, Java, Generic

    def test_get_validator_python(self, registry):
        """Test getting Python validator."""
        registry.discover_validators()
        validator = registry.get_validator(Language.PYTHON)
        assert isinstance(validator, PythonCodeValidator)

    def test_get_validator_fallback(self, registry):
        """Test fallback to generic validator."""
        registry.discover_validators()
        # Request unknown language - should get generic
        validator = registry.get_validator(Language.UNKNOWN)
        assert isinstance(validator, GenericCodeValidator)

    def test_register_custom_validator(self, registry):
        """Test manual registration."""
        validator = PythonCodeValidator()
        registry.register(validator)
        assert Language.PYTHON in registry.registered_languages

    def test_unregister_validator(self, registry):
        """Test unregistering a validator."""
        registry.discover_validators()
        registry.unregister(Language.PYTHON)
        assert Language.PYTHON not in registry.registered_languages

    def test_has_validator(self, registry):
        """Test checking for validator existence."""
        registry.discover_validators()
        assert registry.has_validator(Language.PYTHON)


class TestLanguageDetection:
    """Tests for language detection."""

    def test_detect_python_by_extension(self):
        """Test detection from filename."""
        lang = detect_language("", filename="script.py")
        assert lang == Language.PYTHON

    def test_detect_javascript_by_extension(self):
        """Test detection of JavaScript."""
        lang = detect_language("", filename="app.js")
        assert lang == Language.JAVASCRIPT

    def test_detect_typescript_by_extension(self):
        """Test detection of TypeScript."""
        lang = detect_language("", filename="app.ts")
        assert lang == Language.TYPESCRIPT

    def test_detect_go_by_extension(self):
        """Test detection of Go."""
        lang = detect_language("", filename="main.go")
        assert lang == Language.GO

    def test_detect_rust_by_extension(self):
        """Test detection of Rust."""
        lang = detect_language("", filename="main.rs")
        assert lang == Language.RUST

    def test_detect_java_by_extension(self):
        """Test detection of Java."""
        lang = detect_language("", filename="Main.java")
        assert lang == Language.JAVA

    def test_detect_python_by_content(self):
        """Test detection from Python code patterns."""
        code = """
def hello():
    print("Hello, world!")
"""
        lang = detect_language(code)
        assert lang == Language.PYTHON

    def test_detect_go_by_content(self):
        """Test detection from Go code patterns."""
        code = """
package main

func main() {
    fmt.Println("Hello")
}
"""
        lang = detect_language(code)
        assert lang == Language.GO

    def test_detect_rust_by_content(self):
        """Test detection from Rust code patterns."""
        # Must match at least 2 Rust patterns: fn main(), let mut, and ->
        code = """
fn main() -> i32 {
    let mut x = 5;
    println!("{}", x);
    x
}
"""
        lang = detect_language(code)
        assert lang == Language.RUST

    def test_detect_unknown(self):
        """Test fallback to UNKNOWN."""
        lang = detect_language("random text without code patterns")
        assert lang == Language.UNKNOWN


class TestSelfCorrector:
    """Tests for SelfCorrector orchestrator."""

    @pytest.fixture
    def corrector(self):
        # Reset registry to ensure fresh state
        CodeValidatorRegistry.reset_singleton()
        return create_self_corrector()

    def test_validate_valid_python(self, corrector):
        """Test validation of valid Python code."""
        code = """
def add(a, b):
    return a + b
"""
        fixed, result = corrector.validate_and_fix(code, Language.PYTHON)
        assert result.valid is True
        assert result.syntax_valid is True

    def test_validate_and_fix_syntax_error(self, corrector):
        """Test fixing code with syntax errors."""
        code = """```python
def broken(
    return 1
```"""
        fixed, result = corrector.validate_and_fix(code, Language.PYTHON)
        # At minimum, markdown should be cleaned
        assert "```" not in fixed

    def test_validate_with_auto_detection(self, corrector):
        """Test validation with auto language detection."""
        code = """
def hello():
    return "world"
"""
        fixed, result = corrector.validate_and_fix(code)
        # Should auto-detect as Python
        assert result.language == Language.PYTHON

    def test_generate_feedback(self, corrector):
        """Test feedback generation for invalid code."""
        code = "def broken("
        fixed, result = corrector.validate_and_fix(code, Language.PYTHON)
        feedback = corrector.generate_feedback(code, result)
        assert isinstance(feedback, CorrectionFeedback)


class TestFeedbackGenerator:
    """Tests for FeedbackGenerator."""

    @pytest.fixture
    def generator(self):
        return get_feedback_generator()

    def test_generate_feedback_for_syntax_error(self, generator):
        """Test feedback for syntax errors."""
        result = CodeValidationResult(
            valid=False,
            language=Language.PYTHON,
            syntax_valid=False,
            imports_valid=True,
            errors=("SyntaxError: invalid syntax at line 2",),
            warnings=(),
        )
        feedback = generator.generate(code="def broken(", validation=result)
        assert feedback is not None
        assert feedback.has_issues is True
        assert feedback.syntax_feedback is not None

    def test_generate_feedback_for_import_error(self, generator):
        """Test feedback for missing imports."""
        result = CodeValidationResult(
            valid=False,
            language=Language.PYTHON,
            syntax_valid=True,
            imports_valid=False,
            errors=(),
            warnings=(),
            missing_imports=("numpy", "pandas"),
        )
        feedback = generator.generate(code="import numpy", validation=result)
        assert feedback is not None

    def test_feedback_to_prompt(self, generator):
        """Test converting feedback to prompt string."""
        result = CodeValidationResult(
            valid=False,
            language=Language.PYTHON,
            syntax_valid=False,
            imports_valid=True,
            errors=("SyntaxError at line 1",),
            warnings=(),
        )
        feedback = generator.generate(code="code", validation=result)
        prompt = feedback.to_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0


class TestBaseCodeValidator:
    """Tests for base validator functionality."""

    def test_clean_markdown_preserves_plain_code(self):
        """Test that plain code is preserved."""
        validator = GenericCodeValidator()
        code = "def hello():\n    return 'world'"
        result = validator.clean_markdown(code)
        assert result == code

    def test_preprocess_calls_clean_markdown(self):
        """Test that preprocess uses clean_markdown."""
        validator = GenericCodeValidator()
        code = "```\ndef test():\n    pass\n```"
        result = validator.preprocess(code)
        assert "```" not in result


class TestValidatorCapabilities:
    """Tests for ValidatorCapabilities."""

    def test_default_capabilities(self):
        """Test default capability values."""
        caps = ValidatorCapabilities()
        assert caps.has_ast_parsing is False
        assert caps.has_import_detection is False
        assert caps.has_type_checking is False
        assert caps.has_auto_fix is True

    def test_custom_capabilities(self):
        """Test custom capability values."""
        caps = ValidatorCapabilities(
            has_ast_parsing=True,
            has_import_detection=True,
            has_type_checking=True,
            has_auto_fix=False,
        )
        assert caps.has_ast_parsing is True
        assert caps.has_import_detection is True
        assert caps.has_type_checking is True
        assert caps.has_auto_fix is False

    def test_repr(self):
        """Test string representation."""
        caps = ValidatorCapabilities(has_ast_parsing=True)
        repr_str = repr(caps)
        assert "ast=True" in repr_str
