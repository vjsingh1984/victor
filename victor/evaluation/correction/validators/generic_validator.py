# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Generic code validator for unsupported languages.

This validator serves as the Null Object / fallback when no specialized
validator exists for a language. It provides basic validation that
works for ANY language without false positives.

Design Pattern: Null Object Pattern
- Provides safe fallback behavior
- Never fails due to missing language support
- Can still catch obvious issues (empty code, markdown artifacts)
"""

from ..base import BaseCodeValidator
from ..types import Language, CodeValidationResult


class GenericCodeValidator(BaseCodeValidator):
    """Language-agnostic validator for unsupported languages.

    This validator provides minimal, safe validation that works
    for any programming language. It only catches issues that are
    universally problematic (empty code, truncation, markdown).

    Used as fallback when:
    - Language cannot be detected
    - No specialized validator is registered
    - Explicit UNKNOWN language is specified
    """

    @property
    def supported_languages(self) -> set[Language]:
        """Handles UNKNOWN and acts as fallback for all languages."""
        return {Language.UNKNOWN}

    def validate(self, code: str) -> CodeValidationResult:
        """Basic validation without language-specific logic.

        Only catches universally problematic patterns.
        Designed to minimize false positives.

        Args:
            code: Source code string

        Returns:
            CodeValidationResult (almost always valid unless obviously broken)
        """
        warnings: list[str] = []

        # Check for empty code
        if not code or not code.strip():
            return CodeValidationResult.failure(
                errors=["Code is empty"],
                language=Language.UNKNOWN,
            )

        # Check for obvious truncation
        stripped = code.rstrip()
        if stripped.endswith("..."):
            warnings.append("Code may be truncated (ends with '...')")

        # Check for markdown artifacts (not fixed yet)
        if "```" in code:
            warnings.append("Code contains markdown code block markers")

        # Check for common incomplete patterns
        if stripped.endswith("# TODO") or stripped.endswith("// TODO"):
            warnings.append("Code ends with TODO comment (may be incomplete)")

        # Build result
        result = CodeValidationResult.success(
            language=Language.UNKNOWN,
            used_ast=False,
        )

        if warnings:
            result = result.with_warnings(warnings)

        return result

    def fix(self, code: str, validation: CodeValidationResult) -> str:
        """Apply generic fixes that work for any language.

        Only applies safe transformations:
        - Remove markdown code blocks
        - Trim whitespace

        Args:
            code: Original code
            validation: Validation result

        Returns:
            Cleaned code
        """
        # Remove markdown (safe for all languages)
        cleaned = self.clean_markdown(code)

        # Normalize line endings
        cleaned = cleaned.replace("\r\n", "\n")

        # Remove trailing whitespace on lines
        lines = cleaned.split("\n")
        lines = [line.rstrip() for line in lines]
        cleaned = "\n".join(lines)

        # Ensure single trailing newline
        cleaned = cleaned.rstrip() + "\n"

        return cleaned
