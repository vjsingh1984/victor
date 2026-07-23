# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for code correction middleware.

This module tests:
- CodeCorrectionConfig default values
- CodeCorrectionMiddleware tool filtering
- Code argument detection
- Validation and correction flow
- Correction application
- Error formatting
- Retry prompt generation
- Singleton pattern
"""

import pytest

from victor.agent.code_correction_middleware import (
    CodeCorrectionConfig,
    CodeCorrectionMiddleware,
    CorrectionResult,
    get_code_correction_middleware,
    reset_middleware,
)
from victor.evaluation.correction import (
    Language,
    CodeValidationResult,
    CorrectionFeedback,
    CorrectionMetricsCollector,
    CodeValidatorRegistry,
)


class TestCodeCorrectionConfig:
    """Tests for CodeCorrectionConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CodeCorrectionConfig()

        assert config.enabled is True
        assert config.auto_fix is True
        assert config.max_iterations == 1
        assert config.collect_metrics is True

    def test_default_executable_code_tools(self):
        """Executable-code tools (run generated code immediately) are correctable."""
        config = CodeCorrectionConfig()

        # Code-execution tools qualify by name
        assert "code_executor" in config.executable_code_tools
        assert "execute_code" in config.executable_code_tools
        assert "run_code" in config.executable_code_tools
        # File-authoring tools are NEVER correctable (content is a document)
        assert "write" not in config.executable_code_tools
        assert "edit" not in config.executable_code_tools
        assert "file_editor" not in config.executable_code_tools

    def test_default_code_argument_names(self):
        """Only executable-code argument names are correctable."""
        config = CodeCorrectionConfig()

        assert "code" in config.code_argument_names
        assert "source" in config.code_argument_names
        assert "script" in config.code_argument_names
        # File-content argument names are never treated as correctable code
        assert "content" not in config.code_argument_names
        assert "new_content" not in config.code_argument_names
        assert "file_content" not in config.code_argument_names

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CodeCorrectionConfig(
            enabled=False,
            auto_fix=False,
            max_iterations=3,
            executable_code_tools={"my_custom_tool"},
            code_argument_names={"my_code_arg"},
        )

        assert config.enabled is False
        assert config.auto_fix is False
        assert config.max_iterations == 3
        assert "my_custom_tool" in config.executable_code_tools
        assert "my_code_arg" in config.code_argument_names


class TestCodeCorrectionMiddleware:
    """Tests for CodeCorrectionMiddleware class."""

    @pytest.fixture
    def middleware(self):
        """Create a fresh middleware instance."""
        # Reset registry to ensure fresh state
        CodeValidatorRegistry.reset_singleton()
        return CodeCorrectionMiddleware()

    @pytest.fixture
    def disabled_middleware(self):
        """Create a disabled middleware instance."""
        config = CodeCorrectionConfig(enabled=False)
        return CodeCorrectionMiddleware(config=config)

    def test_should_validate_code_executor(self, middleware):
        """Executable-code tools should be validated."""
        assert middleware.should_validate("code_executor") is True

    def test_should_not_validate_write_tool(self, middleware):
        """File-authoring tools (write) must NOT be validated — content is a document."""
        assert middleware.should_validate("write") is False

    def test_should_not_validate_write_file_alias(self, middleware):
        """Legacy write alias must not be validated either."""
        assert middleware.should_validate("write_file") is False

    def test_should_not_validate_edit_file_alias(self, middleware):
        """Legacy edit alias must not be validated either."""
        assert middleware.should_validate("edit_file") is False

    def test_should_not_validate_read_file(self, middleware):
        """Test that read_file tool should not be validated."""
        assert middleware.should_validate("read_file") is False

    def test_should_not_validate_random_tool(self, middleware):
        """Test that unknown tools are not validated (fail-safe)."""
        assert middleware.should_validate("random_tool") is False

    def test_should_validate_executes_trait_tool(self, middleware):
        """A tool whose contract declares access_mode=EXECUTE qualifies by trait."""

        class _ExecTool:
            access_mode = "execute"

        class _WriteTool:
            access_mode = "write"

        assert middleware.should_validate("custom_runner", tool=_ExecTool()) is True
        assert middleware.should_validate("custom_writer", tool=_WriteTool()) is False

    def test_should_not_validate_when_disabled(self, disabled_middleware):
        """Test that disabled middleware doesn't validate any tools."""
        assert disabled_middleware.should_validate("code_executor") is False
        assert disabled_middleware.should_validate("write_file") is False


class TestFindCodeArgument:
    """Tests for find_code_argument method."""

    @pytest.fixture
    def middleware(self):
        return CodeCorrectionMiddleware()

    def test_find_code_argument_with_code(self, middleware):
        """Test finding 'code' argument."""
        args = {"code": "print('hello')", "other": "value"}
        result = middleware.find_code_argument(args)

        assert result is not None
        assert result[0] == "code"
        assert result[1] == "print('hello')"

    def test_find_code_argument_ignores_content(self, middleware):
        """File-content args ('content') are not treated as correctable code."""
        args = {"content": "def foo(): pass", "path": "/tmp/test.py"}
        result = middleware.find_code_argument(args)

        assert result is None

    def test_find_code_argument_with_source(self, middleware):
        """Test finding 'source' argument."""
        args = {"source": "class MyClass: pass"}
        result = middleware.find_code_argument(args)

        assert result is not None
        assert result[0] == "source"

    def test_find_code_argument_empty_string(self, middleware):
        """Test that empty strings are not found."""
        args = {"code": "", "other": "value"}
        result = middleware.find_code_argument(args)

        assert result is None

    def test_find_code_argument_whitespace_only(self, middleware):
        """Test that whitespace-only strings are not found."""
        args = {"code": "   \n\t  ", "other": "value"}
        result = middleware.find_code_argument(args)

        assert result is None

    def test_find_code_argument_no_code(self, middleware):
        """Test when no code argument exists."""
        args = {"path": "/tmp/test.py", "mode": "w"}
        result = middleware.find_code_argument(args)

        assert result is None

    def test_find_code_argument_non_string(self, middleware):
        """Test that non-string values are ignored."""
        args = {"code": 12345, "content": ["list", "of", "items"]}
        result = middleware.find_code_argument(args)

        assert result is None


class TestValidateAndFix:
    """Tests for validate_and_fix method."""

    @pytest.fixture
    def middleware(self):
        CodeValidatorRegistry.reset_singleton()
        return CodeCorrectionMiddleware()

    def test_validate_valid_python_code(self, middleware):
        """Test validation of valid Python code."""
        args = {"code": "def hello():\n    return 'world'"}
        result = middleware.validate_and_fix("code_executor", args)

        assert result.validation.valid is True
        assert result.validation.syntax_valid is True
        assert result.was_corrected is False

    def test_validate_no_code_argument(self, middleware):
        """Test validation when no code argument present."""
        args = {"path": "/tmp/test.py"}
        result = middleware.validate_and_fix("write_file", args)

        # Should return valid result (nothing to validate)
        assert result.validation.valid is True
        assert result.original_code == ""
        assert result.corrected_code == ""
        assert result.was_corrected is False

    def test_validate_with_markdown_cleanup(self, middleware):
        """Test that markdown blocks are cleaned up."""
        code_with_markdown = """```python
def hello():
    return 'world'
```"""
        args = {"code": code_with_markdown}
        result = middleware.validate_and_fix("code_executor", args)

        # Markdown should be cleaned
        assert "```" not in result.corrected_code
        assert result.was_corrected is True

    def test_validate_syntax_error(self, middleware):
        """Test validation of code with syntax error."""
        args = {"code": "def broken(\n    return 1"}
        result = middleware.validate_and_fix("code_executor", args)

        assert result.validation.valid is False
        assert result.validation.syntax_valid is False
        # Feedback should be generated for invalid code
        assert result.feedback is not None

    def test_validate_with_language_hint(self, middleware):
        """Test validation with explicit language hint."""
        args = {"code": "function hello() { return 'world'; }"}
        result = middleware.validate_and_fix("write_file", args, language_hint="js")

        # Should detect as JavaScript
        assert result.validation.language in {Language.JAVASCRIPT, Language.UNKNOWN}

    def test_validate_python_tool_uses_python_language(self, middleware):
        """Test that code_executor tools default to Python."""
        args = {"code": "x = 1"}
        result = middleware.validate_and_fix("code_executor", args)

        # Should be detected as Python for code_executor
        assert result.validation.language == Language.PYTHON


class TestApplyCorrection:
    """Tests for apply_correction method."""

    @pytest.fixture
    def middleware(self):
        return CodeCorrectionMiddleware()

    def test_apply_correction_when_corrected(self, middleware):
        """Test applying correction when code was corrected."""
        original_args = {"code": "original code", "other": "value"}
        result = CorrectionResult(
            original_code="original code",
            corrected_code="fixed code",
            validation=CodeValidationResult(
                valid=True, language=Language.PYTHON, errors=(), warnings=()
            ),
            was_corrected=True,
        )

        new_args = middleware.apply_correction(original_args, result)

        # Should update code but not modify original
        assert new_args["code"] == "fixed code"
        assert new_args["other"] == "value"
        assert original_args["code"] == "original code"  # Original unchanged

    def test_apply_correction_when_not_corrected(self, middleware):
        """Test applying correction when no changes made."""
        original_args = {"code": "original code"}
        result = CorrectionResult(
            original_code="original code",
            corrected_code="original code",
            validation=CodeValidationResult(
                valid=True, language=Language.PYTHON, errors=(), warnings=()
            ),
            was_corrected=False,
        )

        new_args = middleware.apply_correction(original_args, result)

        # Should return same arguments
        assert new_args is original_args


class TestFileContentNotCorrected:
    """Regression: file-authoring tools' content must never be auto-corrected.

    Root cause of the sandhi-c3966e22 stuck loop: a markdown document written via
    ``write`` was silently truncated to a single code block by the corrector's
    destructive ``clean_markdown``. These guard the whole class.
    """

    @pytest.fixture
    def middleware(self):
        CodeValidatorRegistry.reset_singleton()
        return CodeCorrectionMiddleware()

    @staticmethod
    def _markdown_doc() -> str:
        # Multi-section markdown with prose and several fenced code blocks — the exact
        # shape that was truncated to one ~20-byte block.
        doc = "# TD\n\n## Context\n\nIntro prose here.\n\n## Scope\n\n"
        for i in range(1, 4):
            doc += (
                f"### Pillar {i}\n\nNarrative about pillar {i}.\n\n"
                f"```rust\nfn h{i}() -> i32 {{ {i} }}\n```\n\n"
            )
        doc += "```text\nsandhi keys add <provider>\nsandhi usage --by key\n```\n"
        return doc

    def test_process_write_leaves_markdown_untouched(self, middleware):
        """process() on write returns content byte-identical and no correction result."""
        doc = self._markdown_doc()
        args = {"path": "x.md", "content": doc, "force": True}
        out_args, result = middleware.process("write", args)
        assert result is None  # gate not entered (write is not executable-code)
        assert out_args["content"] == doc

    def test_process_write_file_alias_leaves_markdown_untouched(self, middleware):
        doc = self._markdown_doc()
        args = {"path": "x.md", "content": doc}
        out_args, result = middleware.process("write_file", args)
        assert result is None
        assert out_args["content"] == doc

    def test_process_edit_leaves_markdown_untouched(self, middleware):
        doc = self._markdown_doc()
        args = {"ops": [{"type": "create", "path": "x.md", "content": doc}]}
        out_args, result = middleware.process("edit", args)
        assert result is None
        assert out_args == args

    def test_process_code_executor_still_corrects_fenced_code(self, middleware):
        """Executable-code tools still get fenced code unwrapped and corrected."""
        args = {"code": "```python\ndef f():\n    return 1\n```"}
        out_args, result = middleware.process("code_executor", args)
        assert result is not None
        assert result.was_corrected is True
        assert "```" not in out_args["code"]
        assert out_args["code"].strip() == "def f():\n    return 1"


class TestFormatValidationError:
    """Tests for format_validation_error method."""

    @pytest.fixture
    def middleware(self):
        return CodeCorrectionMiddleware()

    def test_format_valid_result(self, middleware):
        """Test formatting valid result returns empty string."""
        result = CorrectionResult(
            original_code="code",
            corrected_code="code",
            validation=CodeValidationResult(
                valid=True, language=Language.PYTHON, errors=(), warnings=()
            ),
            was_corrected=False,
        )

        output = middleware.format_validation_error(result)
        assert output == ""

    def test_format_syntax_error(self, middleware):
        """Test formatting syntax error result."""
        result = CorrectionResult(
            original_code="def broken(",
            corrected_code="def broken(",
            validation=CodeValidationResult(
                valid=False,
                language=Language.PYTHON,
                syntax_valid=False,
                imports_valid=True,
                errors=("SyntaxError: unexpected EOF",),
                warnings=(),
            ),
            was_corrected=False,
        )

        output = middleware.format_validation_error(result)

        assert "validation failed" in output.lower()
        assert "Syntax" in output

    def test_format_import_error(self, middleware):
        """Test formatting import error result."""
        result = CorrectionResult(
            original_code="import nonexistent",
            corrected_code="import nonexistent",
            validation=CodeValidationResult(
                valid=False,
                language=Language.PYTHON,
                syntax_valid=True,
                imports_valid=False,
                errors=(),
                warnings=(),
                missing_imports=("nonexistent",),
            ),
            was_corrected=False,
        )

        output = middleware.format_validation_error(result)

        assert "Import" in output
        assert "nonexistent" in output

    def test_format_multiple_errors(self, middleware):
        """Test formatting multiple errors."""
        result = CorrectionResult(
            original_code="broken",
            corrected_code="broken",
            validation=CodeValidationResult(
                valid=False,
                language=Language.PYTHON,
                syntax_valid=False,
                imports_valid=True,
                errors=("Error 1", "Error 2", "Error 3"),
                warnings=(),
            ),
            was_corrected=False,
        )

        output = middleware.format_validation_error(result)

        assert "Error 1" in output
        assert "Error 2" in output
        assert "Error 3" in output


class TestGetRetryPrompt:
    """Tests for get_retry_prompt method."""

    @pytest.fixture
    def middleware(self):
        CodeValidatorRegistry.reset_singleton()
        return CodeCorrectionMiddleware()

    def test_get_retry_prompt_with_feedback(self, middleware):
        """Test generating retry prompt with feedback."""
        feedback = CorrectionFeedback(
            has_issues=True,
            language=Language.PYTHON,
            syntax_feedback="Missing closing parenthesis",
        )
        result = CorrectionResult(
            original_code="def broken(",
            corrected_code="def broken(",
            validation=CodeValidationResult(
                valid=False,
                language=Language.PYTHON,
                syntax_valid=False,
                imports_valid=True,
                errors=("SyntaxError",),
                warnings=(),
            ),
            was_corrected=False,
            feedback=feedback,
        )

        prompt = middleware.get_retry_prompt(result, "Write a function")

        assert len(prompt) > 0
        assert isinstance(prompt, str)

    def test_get_retry_prompt_without_feedback(self, middleware):
        """Test generating retry prompt without feedback."""
        result = CorrectionResult(
            original_code="def broken(",
            corrected_code="def broken(",
            validation=CodeValidationResult(
                valid=False,
                language=Language.PYTHON,
                syntax_valid=False,
                imports_valid=True,
                errors=("SyntaxError",),
                warnings=(),
            ),
            was_corrected=False,
            feedback=None,
        )

        prompt = middleware.get_retry_prompt(result)

        # Should fall back to format_validation_error
        assert "validation failed" in prompt.lower()


class TestCorrectionResult:
    """Tests for CorrectionResult dataclass."""

    def test_correction_result_creation(self):
        """Test creating a CorrectionResult."""
        validation = CodeValidationResult(
            valid=True, language=Language.PYTHON, errors=(), warnings=()
        )
        result = CorrectionResult(
            original_code="original",
            corrected_code="corrected",
            validation=validation,
            was_corrected=True,
            feedback=None,
        )

        assert result.original_code == "original"
        assert result.corrected_code == "corrected"
        assert result.validation is validation
        assert result.was_corrected is True
        assert result.feedback is None

    def test_correction_result_with_feedback(self):
        """Test CorrectionResult with feedback."""
        feedback = CorrectionFeedback(
            has_issues=True,
            language=Language.PYTHON,
            syntax_feedback="Fix required",
        )
        result = CorrectionResult(
            original_code="broken",
            corrected_code="broken",
            validation=CodeValidationResult(
                valid=False, language=Language.PYTHON, errors=(), warnings=()
            ),
            was_corrected=False,
            feedback=feedback,
        )

        assert result.feedback is feedback
        assert result.feedback.has_issues is True


class TestMetricsCollection:
    """Tests for metrics collection in middleware."""

    def test_metrics_collected_on_validation(self):
        """Test that metrics are collected during validation."""
        CodeValidatorRegistry.reset_singleton()
        collector = CorrectionMetricsCollector()
        middleware = CodeCorrectionMiddleware(metrics_collector=collector)

        args = {"code": "def hello(): return 'world'"}
        middleware.validate_and_fix("code_executor", args)

        # Metrics should be recorded
        assert collector.metrics.total_validations == 1

    def test_metrics_track_language(self):
        """Test that language is tracked in metrics."""
        CodeValidatorRegistry.reset_singleton()
        collector = CorrectionMetricsCollector()
        middleware = CodeCorrectionMiddleware(metrics_collector=collector)

        args = {"code": "print('hello')"}
        middleware.validate_and_fix("code_executor", args)

        # Should have Python validation recorded
        assert Language.PYTHON in collector.metrics.language_validations


class TestGlobalMiddleware:
    """Tests for global middleware singleton."""

    def test_get_middleware_returns_instance(self):
        """Test that get_code_correction_middleware returns an instance."""
        reset_middleware()
        middleware = get_code_correction_middleware()

        assert middleware is not None
        assert isinstance(middleware, CodeCorrectionMiddleware)

    def test_get_middleware_is_singleton(self):
        """Test that get_code_correction_middleware returns same instance."""
        reset_middleware()
        m1 = get_code_correction_middleware()
        m2 = get_code_correction_middleware()

        assert m1 is m2

    def test_reset_middleware_creates_new_instance(self):
        """Test that reset_middleware allows new instance creation."""
        m1 = get_code_correction_middleware()
        reset_middleware()
        m2 = get_code_correction_middleware()

        assert m1 is not m2


class TestLazyCorrectorLoading:
    """Tests for lazy corrector loading."""

    def test_corrector_not_loaded_on_init(self):
        """Test that corrector is not loaded on initialization."""
        middleware = CodeCorrectionMiddleware()

        # _corrector should be None initially
        assert middleware._corrector is None

    def test_corrector_loaded_on_access(self):
        """Test that corrector is loaded when accessed."""
        CodeValidatorRegistry.reset_singleton()
        middleware = CodeCorrectionMiddleware()

        # Access the corrector property
        corrector = middleware.corrector

        assert corrector is not None
        assert middleware._corrector is not None

    def test_corrector_loaded_only_once(self):
        """Test that corrector is loaded only once."""
        CodeValidatorRegistry.reset_singleton()
        middleware = CodeCorrectionMiddleware()

        c1 = middleware.corrector
        c2 = middleware.corrector

        assert c1 is c2
