# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Code Correction Middleware for Tool Pipeline.

This middleware integrates the self-correction system into Victor's
tool execution pipeline, providing automatic code validation and
fixing for tools that accept code as input.

Design Pattern: Middleware/Interceptor Pattern
- Intercepts tool calls before execution
- Validates and auto-fixes code arguments
- Provides feedback for failed validations

Integration Points:
- ToolPipeline: Called during argument processing
- code_executor: Validates Python code before execution
- file_editor: Validates code in file content
- write_file: Validates code in file content

Usage:
    from victor.agent.code_correction_middleware import CodeCorrectionMiddleware

    middleware = CodeCorrectionMiddleware(enabled=True)

    # In tool pipeline, before execution:
    if middleware.should_validate(tool_name):
        validated_args, validation = middleware.validate_and_fix(
            tool_name, arguments
        )
        if not validation.valid and not middleware.auto_fix:
            # Return feedback to LLM for retry
            feedback = middleware.get_feedback(validation)

Enterprise Integration Pattern: Intercepting Filter
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from victor.evaluation.correction import (
    SelfCorrector,
    CodeValidationResult,
    CorrectionFeedback,
    Language,
    detect_language,
    create_self_corrector,
    CorrectionMetricsCollector,
)

logger = logging.getLogger(__name__)


@dataclass
class CodeCorrectionConfig:
    """Configuration for code correction middleware."""

    enabled: bool = True
    auto_fix: bool = True
    max_iterations: int = 1
    collect_metrics: bool = True

    # Use frozensets for better performance
    code_tools: set[str] = field(
        default_factory=lambda: {
            "code_executor",
            "execute_code",
            "run_code",
            "write_file",
            "file_editor",
            "edit_file",
            "create_file",
        }
    )

    code_argument_names: set[str] = field(
        default_factory=lambda: {
            "code",
            "python_code",
            "content",
            "source",
            "script",
            "new_content",
            "file_content",
        }
    )


@dataclass
class CorrectionResult:
    """Result of code correction attempt."""

    original_code: str
    corrected_code: str
    validation: CodeValidationResult
    was_corrected: bool
    feedback: Optional[CorrectionFeedback] = None


class CodeCorrectionMiddleware:
    """Middleware for validating and correcting code in tool arguments.

    This middleware integrates with the tool pipeline to provide
    automatic code validation and fixing before tool execution.

    Example:
        middleware = CodeCorrectionMiddleware()

        # Check if tool needs code validation
        if middleware.should_validate("code_executor"):
            # Validate and optionally fix the code
            result = middleware.validate_and_fix(
                tool_name="code_executor",
                arguments={"code": "print('hello')"},
            )

            if result.was_corrected:
                arguments["code"] = result.corrected_code

            if not result.validation.valid:
                # Generate feedback for LLM retry
                feedback = middleware.generate_feedback(result)
    """

    def __init__(
        self,
        config: Optional[CodeCorrectionConfig] = None,
        metrics_collector: Optional[CorrectionMetricsCollector] = None,
    ):
        """Initialize the middleware.

        Args:
            config: Configuration options
            metrics_collector: Optional collector for correction metrics
        """
        self.config = config or CodeCorrectionConfig()
        self.metrics_collector = metrics_collector
        self._corrector: Optional[SelfCorrector] = None

    @property
    def corrector(self) -> SelfCorrector:
        """Lazy-load the self-corrector."""
        if self._corrector is None:
            self._corrector = create_self_corrector(
                max_iterations=self.config.max_iterations,
                auto_fix=self.config.auto_fix,
            )
        return self._corrector

    def should_validate(self, tool_name: str) -> bool:
        """Check if this tool should have its code validated.

        Args:
            tool_name: Name of the tool

        Returns:
            True if code validation should be applied
        """
        if not self.config.enabled:
            return False
        return tool_name in self.config.code_tools

    def find_code_argument(self, arguments: dict[str, Any]) -> Optional[tuple[str, str]]:
        """Find the code argument in tool arguments."""
        for arg_name in self.config.code_argument_names:
            if (value := arguments.get(arg_name)) and isinstance(value, str) and value.strip():
                return (arg_name, value)
        return None

    def validate_and_fix(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        language_hint: Optional[str] = None,
    ) -> CorrectionResult:
        """Validate and optionally fix code in tool arguments.

        Args:
            tool_name: Name of the tool being called
            arguments: Tool arguments (may be modified if auto_fix enabled)
            language_hint: Optional language hint (e.g., "python", "javascript")

        Returns:
            CorrectionResult with validation status and corrected code
        """
        # Find the code argument
        code_arg = self.find_code_argument(arguments)

        if not code_arg:
            return CorrectionResult(
                "", "", CodeValidationResult(True, Language.UNKNOWN, True, True, (), ()), False
            )

        arg_name, code = code_arg

        # Detect language
        lang = (
            detect_language(code, filename=f"code.{language_hint}")
            if language_hint
            else (
                Language.PYTHON
                if tool_name in {"code_executor", "execute_code", "run_code"}
                else detect_language(code)
            )
        )

        # Validate and fix
        fixed_code, validation = self.corrector.validate_and_fix(code, language=lang)

        if self.metrics_collector:
            self.metrics_collector.record_validation(lang, validation)

        return CorrectionResult(
            code,
            fixed_code,
            validation,
            fixed_code != code,
            (
                self.corrector.generate_feedback(code=code, validation=validation)
                if not validation.valid
                else None
            ),
        )

    def apply_correction(
        self, arguments: dict[str, Any], result: CorrectionResult
    ) -> dict[str, Any]:
        """Apply correction to tool arguments."""
        if not result.was_corrected or not (code_arg := self.find_code_argument(arguments)):
            return arguments

        updated = arguments.copy()
        updated[code_arg[0]] = result.corrected_code
        return updated

    def format_validation_error(self, result: CorrectionResult) -> str:
        """Format validation errors for display."""
        if result.validation.valid:
            return ""

        lines = ["Code validation failed:"]

        if not result.validation.syntax_valid:
            lines.append("  - Syntax errors detected")

        if not result.validation.imports_valid:
            lines.append("  - Import issues detected")
            if hasattr(result.validation, "missing_imports") and result.validation.missing_imports:
                lines.extend(
                    f"    - Missing: {imp}" for imp in result.validation.missing_imports[:5]
                )

        lines.extend(f"  - {error}" for error in result.validation.errors[:5])

        if result.feedback:
            # Use general_feedback instead of suggestions attribute
            if result.feedback.general_feedback:
                lines.append("\nSuggestions:")
                lines.append(f"  - {result.feedback.general_feedback}")

        return "\n".join(lines)

    def get_retry_prompt(
        self, result: CorrectionResult, original_prompt: Optional[str] = None
    ) -> str:
        """Generate a retry prompt for the LLM."""
        return (
            self.format_validation_error(result)
            if not result.feedback
            else self.corrector.build_retry_prompt(
                original_prompt or "", result.original_code, result.feedback, 1
            )
        )


# Singleton instance for shared use
_middleware_instance: Optional[CodeCorrectionMiddleware] = None


def get_code_correction_middleware() -> CodeCorrectionMiddleware:
    """Get the global code correction middleware instance.

    Returns:
        Shared middleware instance
    """
    global _middleware_instance
    if _middleware_instance is None:
        _middleware_instance = CodeCorrectionMiddleware()
    return _middleware_instance


def reset_middleware() -> None:
    """Reset the global middleware instance."""
    global _middleware_instance
    _middleware_instance = None
