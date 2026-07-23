# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Base validator interface and contracts.

This module defines the abstract base class that all language validators
must implement. Follows the Strategy Pattern for interchangeable validators.

Design Patterns:
- Strategy Pattern: Interchangeable validation algorithms
- Template Method: Common markdown cleanup in base class
"""

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .types import Language, CodeValidationResult


class BaseCodeValidator(ABC):
    """Abstract base class for language-specific code validators.

    All language validators must implement this interface. The registry
    uses this contract to dispatch validation to the appropriate handler.

    To add a new language validator:
    1. Create a new class inheriting from BaseCodeValidator
    2. Implement supported_languages, validate(), and fix()
    3. Place in validators/ directory (auto-discovered)

    Example:
        class RubyCodeValidator(BaseCodeValidator):
            @property
            def supported_languages(self) -> set[Language]:
                return {Language.RUBY}

            def validate(self, code: str) -> CodeValidationResult:
                # Ruby-specific validation
                ...

            def fix(self, code: str, validation: CodeValidationResult) -> str:
                # Ruby-specific fixes
                ...
    """

    # Language markers for markdown code block detection
    MARKDOWN_LANGUAGE_MARKERS = frozenset(
        [
            "python",
            "javascript",
            "typescript",
            "go",
            "rust",
            "java",
            "cpp",
            "c",
            "ruby",
            "php",
            "swift",
            "kotlin",
            "csharp",
            "scala",
            "bash",
            "sh",
            "sql",
            "json",
            "yaml",
        ]
    )

    @property
    @abstractmethod
    def supported_languages(self) -> "set[Language]":
        """Languages this validator can handle.

        Returns:
            Set of Language enum values this validator supports.
            A validator can support multiple related languages
            (e.g., JavaScript and TypeScript).
        """
        pass

    @abstractmethod
    def validate(self, code: str) -> "CodeValidationResult":
        """Validate code and return structured result.

        Args:
            code: Source code string to validate

        Returns:
            CodeValidationResult with validation status and details
        """
        pass

    @abstractmethod
    def fix(self, code: str, validation: "CodeValidationResult") -> str:
        """Attempt to auto-fix issues found during validation.

        Args:
            code: Original source code
            validation: Validation result with identified issues

        Returns:
            Fixed code string (may be unchanged if no fixes applicable)
        """
        pass

    def clean_markdown(self, code: str) -> str:
        """Non-destructively unwrap a SINGLE wrapping markdown code fence.

        Acts ONLY when the entire content is one fenced block (the fence wraps the
        whole document) — the common case where a model wraps one executable-code
        snippet in a fence. It never extracts a block out of a prose document, never
        drops prose, and never merges or skips multiple blocks. Markdown documents
        (prose with embedded code blocks) are returned verbatim.

        This is critical: the previous unanchored ``re.search(r"```...```")`` returned
        only the FIRST fenced block's interior, silently destroying markdown documents
        (e.g. a design doc with several code blocks collapsed to a single block) which
        made agents loop rewriting the "mangled" file.

        Args:
            code: Code possibly wrapped in a single markdown fence.

        Returns:
            The unwrapped code if the whole input was one fenced block; else the
            input unchanged.
        """
        if "```" not in code:
            return code
        stripped = code.strip()
        # Only unwrap when the ENTIRE content is a single fenced block: exactly one
        # fence pair (count == 2) AND the fence wraps the whole document. Anything else
        # (multiple blocks, or a fence embedded in prose) is returned unchanged.
        if stripped.count("```") != 2:
            return code
        # Whole-document, language-tagged fence:  ```lang\n ...\n```
        match = re.match(r"^```[A-Za-z0-9+#.\-]*\n(.*?)\n?```\s*\Z", stripped, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Whole-document, generic fence:  ```\n? ...\n```
        match = re.match(r"^```\n?(.*?)\n?```\s*\Z", stripped, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fence embedded in a document, multiple blocks, or prose+code → UNCHANGED.
        return code

    def preprocess(self, code: str) -> str:
        """Standard preprocessing applied before validation.

        Override in subclasses for language-specific preprocessing.

        Args:
            code: Raw code string

        Returns:
            Preprocessed code ready for validation
        """
        return self.clean_markdown(code)

    # =========================================================================
    # Common Validation Methods (Template Methods)
    # =========================================================================
    # These methods are shared across multiple validators to reduce duplication.
    # Subclasses can override for language-specific behavior.

    def _check_brackets(self, code: str) -> Optional[str]:
        """Check for balanced brackets, braces, and parentheses.

        This is a universal bracket matching algorithm used by Python, Rust,
        JavaScript, Go, and Java validators. Language-specific validators can
        override for additional bracket types (e.g., angle brackets in Rust/Java).

        Args:
            code: Source code to check

        Returns:
            Error message if brackets are unbalanced, None otherwise
        """
        cleaned = self._remove_strings_and_comments(code)

        bracket_pairs = {"(": ")", "[": "]", "{": "}"}
        stack: list[tuple[str, int]] = []

        for i, char in enumerate(cleaned):
            if char in bracket_pairs:
                stack.append((char, i))
            elif char in bracket_pairs.values():
                if not stack:
                    line = code[:i].count("\n") + 1
                    return f"Unmatched closing '{char}' at line {line}"

                open_bracket, _ = stack.pop()
                if bracket_pairs[open_bracket] != char:
                    line = code[:i].count("\n") + 1
                    return f"Mismatched brackets: expected '{bracket_pairs[open_bracket]}' but found '{char}' at line {line}"

        if stack:
            open_bracket, pos = stack[-1]
            line = code[:pos].count("\n") + 1
            return f"Unclosed '{open_bracket}' starting at line {line}"

        return None

    def _remove_strings_and_comments(
        self,
        code: str,
        *,
        has_raw_strings: bool = False,
        has_template_strings: bool = False,
        has_char_literals: bool = True,
    ) -> str:
        """Remove string contents and comments for structural analysis.

        This allows bracket matching and other structural checks to work without
        false positives from string contents or comments.

        Args:
            code: Source code to clean
            has_raw_strings: Language has raw string literals (r"...") - Rust, Go
            has_template_strings: Language has template strings (`...`) - JS/TS
            has_char_literals: Language has char literals ('...')

        Returns:
            Code with strings replaced by empty placeholders and comments removed
        """
        result = code

        # Remove single-line comments
        result = re.sub(r"//.*$", "", result, flags=re.MULTILINE)

        # Remove multi-line comments
        result = re.sub(r"/\*.*?\*/", "", result, flags=re.DOTALL)

        # Remove string contents (keep quotes for structure)
        result = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', '""', result)

        # Remove single-quote strings
        if has_char_literals:
            result = re.sub(r"'[^'\\]*(?:\\.[^'\\]*)*'", "''", result)

        # Remove raw strings (Rust r#"..."#, Go `...`)
        if has_raw_strings:
            result = re.sub(r'r#*".*?"#*', '""', result)
            result = re.sub(r"`[^`]*`", "``", result)

        # Remove template strings (JS/TS)
        if has_template_strings:
            result = re.sub(r"`[^`\\]*(?:\\.[^`\\]*)*`", "``", result)

        return result

    def _check_incomplete_code(
        self,
        code: str,
        *,
        extra_patterns: list[str] | None = None,
        check_brackets: bool = True,
    ) -> Optional[str]:
        """Check for obviously incomplete/truncated code.

        Looks for patterns that suggest code was cut off mid-statement.

        Args:
            code: Source code to check
            extra_patterns: Additional regex patterns for language-specific truncation
            check_brackets: If True, also verify brackets are balanced

        Returns:
            Error message if code appears incomplete, None otherwise
        """
        stripped = code.strip()

        if not stripped:
            return "Empty code"

        # Common truncation patterns across languages
        truncation_patterns = [
            r"=\s*$",  # Ends with assignment
            r"{\s*$",  # Ends with open brace
            r"\(\s*$",  # Ends with open paren
        ]

        # Add language-specific patterns
        if extra_patterns:
            truncation_patterns.extend(extra_patterns)

        for pattern in truncation_patterns:
            if re.search(pattern, stripped):
                # Only flag if also has unclosed brackets (when requested)
                if check_brackets:
                    bracket_error = self._check_brackets(stripped)
                    if bracket_error:
                        return "Code appears to be truncated"
                else:
                    return "Code appears to be truncated"

        return None


class ValidatorCapabilities:
    """Descriptor for validator capabilities.

    Used for introspection and capability discovery.
    """

    def __init__(
        self,
        has_ast_parsing: bool = False,
        has_import_detection: bool = False,
        has_type_checking: bool = False,
        has_auto_fix: bool = True,
    ):
        self.has_ast_parsing = has_ast_parsing
        self.has_import_detection = has_import_detection
        self.has_type_checking = has_type_checking
        self.has_auto_fix = has_auto_fix

    def __repr__(self) -> str:
        return (
            f"ValidatorCapabilities(ast={self.has_ast_parsing}, "
            f"imports={self.has_import_detection}, "
            f"types={self.has_type_checking})"
        )
