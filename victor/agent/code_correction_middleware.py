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
- ToolPipeline / ToolExecutor: the single ``process()`` entry (argument processing)
- code_executor / execute_code / run_code: validate + fix executable code before it runs

Scope (important): correction (mutation) applies ONLY to executable-code tools — tools
whose declared contract is ``access_mode == EXECUTE`` (code that runs immediately, where an
auto-fix is useful). File-authoring tools (write/edit/file_editor, ``access_mode == WRITE``)
are NEVER auto-corrected: their ``content``/``new_content`` is an authored document
(markdown/prose/config/mixed-language), not a single code blob, and silently mutating it is
destructive (it truncated markdown docs to one code block → agent rewrite loops). File
content gets read-only LSP diagnostics via ``victor.tools.lsp_write_enhancer`` instead.

Enterprise Integration Pattern: Intercepting Filter
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, Tuple

from victor.evaluation.correction import (
    SelfCorrector,
    CodeValidationResult,
    CorrectionFeedback,
    Language,
    detect_language,
    create_self_corrector,
    CorrectionMetricsCollector,
)
from victor.tools.tool_names import get_canonical_name

logger = logging.getLogger(__name__)


def _trait_value(value: Any) -> str:
    """Normalize a tool-trait value (Enum or str) to a comparable lowercase string."""
    if isinstance(value, enum.Enum):
        value = value.value
    return str(value).lower() if value is not None else ""


@dataclass
class CodeCorrectionConfig:
    """Configuration for code correction middleware."""

    enabled: bool = True
    auto_fix: bool = True
    max_iterations: int = 1
    collect_metrics: bool = True

    # Tools that EXECUTE generated code immediately (their code argument runs right now),
    # so an auto-fix is safe and useful. File-authoring tools (write/edit/file_editor) are
    # deliberately ABSENT: their content is an authored document, not executable code, and
    # auto-"correcting" it is destructive (see module docstring). New code-execution tools
    # are also picked up automatically by the access_mode trait gate in ``should_validate``.
    executable_code_tools: Set[str] = field(
        default_factory=lambda: frozenset(
            {
                "code_executor",
                "execute_code",
                "run_code",
            }
        )
    )

    # Argument names that carry executable code (for execution tools only). File-content
    # argument names (``content``/``new_content``/``file_content``) are intentionally
    # excluded — they are never treated as correctable code.
    code_argument_names: Set[str] = field(
        default_factory=lambda: frozenset(
            {
                "code",
                "python_code",
                "source",
                "script",
            }
        )
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

    def should_validate(self, tool_name: str, *, tool: Any = None) -> bool:
        """Check if this tool should have its code validated and auto-fixed.

        Correction is gated on the tool's *semantic contract*, not a name list: only
        executable-code tools qualify. A tool qualifies iff (a) its canonical name is in
        ``executable_code_tools``, OR (b) its resolved contract declares
        ``access_mode == EXECUTE`` / ``execution_category in {COMPUTE, EXECUTE}``. File
        tools (``access_mode == WRITE``) and unknown tools never qualify (fail-safe).

        Args:
            tool_name: Name of the tool (canonicalized; aliases resolve to canonical).
            tool: Optional tool object whose ``access_mode``/``execution_category``
                contract traits can be inspected. When omitted, only the name allowlist
                decides.

        Returns:
            True if code correction should be applied to this tool's arguments.
        """
        if not self.config.enabled:
            return False
        if get_canonical_name(tool_name) in self.config.executable_code_tools:
            return True
        if tool is not None:
            access_mode = _trait_value(getattr(tool, "access_mode", None))
            execution_category = _trait_value(getattr(tool, "execution_category", None))
            if access_mode == "execute" or execution_category in ("compute", "execute"):
                return True
        return False

    def process(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        *,
        tool: Any = None,
    ) -> Tuple[Dict[str, Any], Optional[CorrectionResult]]:
        """Single pipeline entry: gate → validate_and_fix → apply_correction → audit.

        Both tool-execution sites (ToolPipeline and ToolExecutor) call this so the logic
        lives in one place and cannot drift. Returns the (possibly mutated) arguments and
        the CorrectionResult (or ``None`` when correction did not apply). Callers map the
        result onto ``ToolCallResult.code_corrected`` / ``code_validation_errors``.

        Args:
            tool_name: Name of the tool being executed.
            arguments: Normalized tool arguments (not mutated; a copy is returned if changed).
            tool: Optional tool object for contract-trait gating (see ``should_validate``).

        Returns:
            ``(arguments, correction_result)`` — arguments unchanged if not corrected.
        """
        if not self.should_validate(tool_name, tool=tool):
            return arguments, None
        try:
            result = self.validate_and_fix(tool_name, arguments)
        except (ValueError, TypeError, KeyError) as e:
            logger.warning("code-correction failed for %s: %s", tool_name, e)
            return arguments, None
        if not result.was_corrected:
            return arguments, result
        new_arguments = self.apply_correction(arguments, result)
        code_arg = self.find_code_argument(arguments)
        logger.info(
            "code-corrected tool=%s arg=%s lang=%s before=%d after=%d",
            tool_name,
            code_arg[0] if code_arg else "?",
            _trait_value(result.validation.language),
            len(result.original_code),
            len(result.corrected_code),
        )
        return new_arguments, result

    def find_code_argument(self, arguments: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        """Find the code argument in tool arguments."""
        for arg_name in self.config.code_argument_names:
            if (value := arguments.get(arg_name)) and isinstance(value, str) and value.strip():
                return (arg_name, value)
        return None

    def validate_and_fix(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
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
                "",
                "",
                CodeValidationResult(True, Language.UNKNOWN, True, True, (), ()),
                False,
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
        self, arguments: Dict[str, Any], result: CorrectionResult
    ) -> Dict[str, Any]:
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

        if result.feedback and result.feedback.suggestions:
            lines.append("\nSuggestions:")
            lines.extend(f"  - {suggestion}" for suggestion in result.feedback.suggestions[:3])

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
