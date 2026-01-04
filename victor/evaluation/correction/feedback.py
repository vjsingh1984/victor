# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Feedback generation for code correction.

This module generates structured, actionable feedback from validation
results and test failures. The feedback is designed to guide LLMs
in fixing issues with their generated code.

Design Pattern: Builder Pattern (builds complex feedback objects)
"""

from typing import Optional

from .types import CorrectionFeedback, Language, CodeValidationResult


class FeedbackGenerator:
    """Generates structured feedback for code correction.

    Creates clear, actionable feedback from:
    - Validation errors (syntax, imports)
    - Test failures (stdout/stderr)
    - Static analysis results

    Language-agnostic design - works with any CodeValidationResult.

    Usage:
        generator = FeedbackGenerator()
        feedback = generator.generate(
            code=code,
            validation=validation_result,
            test_stderr=error_output,
        )
    """

    def __init__(self, max_error_length: int = 500):
        """Initialize feedback generator.

        Args:
            max_error_length: Maximum length for error excerpts
        """
        self.max_error_length = max_error_length

    def generate(
        self,
        code: str,
        validation: Optional[CodeValidationResult] = None,
        test_stdout: Optional[str] = None,
        test_stderr: Optional[str] = None,
        test_passed: Optional[int] = None,
        test_total: Optional[int] = None,
    ) -> CorrectionFeedback:
        """Generate correction feedback from various error sources.

        Args:
            code: The code that was validated/tested
            validation: Validation result (if validation was performed)
            test_stdout: Standard output from test execution
            test_stderr: Standard error from test execution
            test_passed: Number of tests that passed
            test_total: Total number of tests

        Returns:
            CorrectionFeedback with structured feedback for retry
        """
        language = validation.language if validation else Language.UNKNOWN

        syntax_feedback = self._build_syntax_feedback(validation)
        import_feedback = self._build_import_feedback(validation)
        test_feedback = self._build_test_feedback(test_stdout, test_stderr, test_passed, test_total)

        has_issues = any([syntax_feedback, import_feedback, test_feedback])

        return CorrectionFeedback(
            has_issues=has_issues,
            language=language,
            syntax_feedback=syntax_feedback,
            import_feedback=import_feedback,
            test_feedback=test_feedback,
        )

    def _build_syntax_feedback(self, validation: Optional[CodeValidationResult]) -> Optional[str]:
        """Build syntax error feedback."""
        if not validation or validation.syntax_valid:
            return None

        return (
            f"Your code has a syntax error: {validation.syntax_error}\n"
            "Please fix the syntax and ensure the code is valid."
        )

    def _build_import_feedback(self, validation: Optional[CodeValidationResult]) -> Optional[str]:
        """Build import issue feedback."""
        if not validation or validation.imports_valid:
            return None

        issues = [e for e in validation.errors if "import" in e.lower()]
        if not issues:
            return None

        issues_text = "\n".join(f"- {e}" for e in issues)
        return (
            f"Import issues detected:\n{issues_text}\n"
            "Please add the necessary import statements at the top of your code."
        )

    def _build_test_feedback(
        self,
        test_stdout: Optional[str],
        test_stderr: Optional[str],
        test_passed: Optional[int],
        test_total: Optional[int],
    ) -> Optional[str]:
        """Build test failure feedback."""
        has_failures = test_stderr or (
            test_passed is not None and test_total is not None and test_passed < test_total
        )

        if not has_failures:
            return None

        parts = []

        if test_passed is not None and test_total is not None:
            parts.append(f"Tests: {test_passed}/{test_total} passed")

        if test_stderr:
            error_info = self._extract_error_info(test_stderr)
            parts.append(f"Error output:\n{error_info}")

        return "\n".join(parts)

    def _extract_error_info(self, stderr: str) -> str:
        """Extract meaningful error information from stderr.

        Focuses on the most relevant error lines (assertions,
        exceptions, expected values) and truncates if too long.

        Args:
            stderr: Full stderr output

        Returns:
            Extracted and truncated error info
        """
        if not stderr:
            return ""

        lines = stderr.strip().split("\n")

        # Keywords that indicate important error lines
        error_keywords = [
            "Error",
            "Exception",
            "AssertionError",
            "assert",
            "expected",
            "failed",
            "got",
            "Traceback",
            "raise",
            "!=",
            "==",
        ]

        # Find most relevant lines
        relevant_lines = []
        for i, line in enumerate(lines):
            if any(keyword in line for keyword in error_keywords):
                # Get context around the error
                start = max(0, i - 1)
                end = min(len(lines), i + 3)
                relevant_lines.extend(lines[start:end])
                break

        if relevant_lines:
            result = "\n".join(dict.fromkeys(relevant_lines))  # Dedup
        else:
            # Fallback: last few lines
            result = "\n".join(lines[-5:])

        # Truncate if too long
        if len(result) > self.max_error_length:
            result = result[: self.max_error_length] + "\n... (truncated)"

        return result


class RetryPromptBuilder:
    """Builds retry prompts with feedback for LLM correction.

    Creates structured prompts that include:
    - The original task
    - The previous attempt
    - Specific feedback on what went wrong
    - Clear instructions for correction
    """

    # Language-specific markdown markers for code blocks
    LANGUAGE_MARKERS: dict[Language, str] = {
        Language.PYTHON: "python",
        Language.JAVASCRIPT: "javascript",
        Language.TYPESCRIPT: "typescript",
        Language.GO: "go",
        Language.RUST: "rust",
        Language.JAVA: "java",
        Language.CPP: "cpp",
        Language.C: "c",
        Language.RUBY: "ruby",
        Language.PHP: "php",
        Language.SWIFT: "swift",
        Language.KOTLIN: "kotlin",
        Language.CSHARP: "csharp",
        Language.SCALA: "scala",
    }

    def build(
        self,
        original_prompt: str,
        previous_code: str,
        feedback: CorrectionFeedback,
        iteration: int,
    ) -> str:
        """Build a retry prompt with feedback.

        Args:
            original_prompt: The original task description
            previous_code: The code from the previous attempt
            feedback: Feedback on what went wrong
            iteration: Current iteration number (for context)

        Returns:
            Complete retry prompt for the LLM
        """
        feedback_text = feedback.to_prompt()
        lang_marker = self.LANGUAGE_MARKERS.get(feedback.language, "")

        return f"""Your previous solution had issues that need to be fixed.

{feedback_text}

ORIGINAL TASK:
{original_prompt}

YOUR PREVIOUS CODE:
```{lang_marker}
{previous_code}
```

Please provide a corrected solution that addresses the issues above.
Respond with ONLY the complete code, no explanations or markdown."""


# Singleton instances for convenience
_feedback_generator: Optional[FeedbackGenerator] = None
_prompt_builder: Optional[RetryPromptBuilder] = None


def get_feedback_generator() -> FeedbackGenerator:
    """Get the global FeedbackGenerator instance."""
    global _feedback_generator
    if _feedback_generator is None:
        _feedback_generator = FeedbackGenerator()
    return _feedback_generator


def get_prompt_builder() -> RetryPromptBuilder:
    """Get the global RetryPromptBuilder instance."""
    global _prompt_builder
    if _prompt_builder is None:
        _prompt_builder = RetryPromptBuilder()
    return _prompt_builder
