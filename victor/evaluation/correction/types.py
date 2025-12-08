# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Core types for the code correction system.

This module defines the fundamental data structures used across the
correction system. Types are kept minimal and language-agnostic.

Design Patterns:
- Value Object: Immutable dataclasses for validation results
- Enumeration: Type-safe language identification
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class Language(Enum):
    """Programming language enumeration.

    Used for language detection and validator dispatch.
    UNKNOWN serves as the fallback for unsupported languages.
    """

    # Tier 1: Full AST validation support
    PYTHON = auto()

    # Tier 2: Pattern-based validation
    JAVASCRIPT = auto()
    TYPESCRIPT = auto()
    GO = auto()
    RUST = auto()

    # Tier 3: Basic validation only
    JAVA = auto()
    CPP = auto()
    C = auto()
    RUBY = auto()
    PHP = auto()
    SWIFT = auto()
    KOTLIN = auto()
    CSHARP = auto()
    SCALA = auto()

    # Fallback
    UNKNOWN = auto()

    @classmethod
    def from_string(cls, name: str) -> "Language":
        """Convert string to Language enum (case-insensitive)."""
        name_upper = name.upper().replace("#", "SHARP").replace("++", "PP")
        try:
            return cls[name_upper]
        except KeyError:
            return cls.UNKNOWN


@dataclass(frozen=True)
class ValidationResult:
    """Immutable result of code validation.

    Attributes:
        valid: Overall validation passed
        language: Detected or specified language
        syntax_valid: No syntax errors detected
        imports_valid: All imports are resolved
        errors: List of error messages
        warnings: List of warning messages
        missing_imports: Imports that need to be added
        syntax_error: Detailed syntax error message
        used_ast_validation: Whether AST parsing was used
    """

    valid: bool
    language: Language = Language.UNKNOWN
    syntax_valid: bool = True
    imports_valid: bool = True
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    missing_imports: tuple[str, ...] = ()
    syntax_error: Optional[str] = None
    used_ast_validation: bool = False

    @classmethod
    def success(
        cls, language: Language = Language.UNKNOWN, used_ast: bool = False
    ) -> "ValidationResult":
        """Factory for successful validation."""
        return cls(valid=True, language=language, used_ast_validation=used_ast)

    @classmethod
    def failure(
        cls,
        errors: list[str],
        language: Language = Language.UNKNOWN,
        syntax_error: Optional[str] = None,
    ) -> "ValidationResult":
        """Factory for failed validation."""
        return cls(
            valid=False,
            language=language,
            syntax_valid=syntax_error is None,
            errors=tuple(errors),
            syntax_error=syntax_error,
        )

    def with_warnings(self, warnings: list[str]) -> "ValidationResult":
        """Return new result with added warnings."""
        return ValidationResult(
            valid=self.valid,
            language=self.language,
            syntax_valid=self.syntax_valid,
            imports_valid=self.imports_valid,
            errors=self.errors,
            warnings=tuple(warnings),
            missing_imports=self.missing_imports,
            syntax_error=self.syntax_error,
            used_ast_validation=self.used_ast_validation,
        )


@dataclass(frozen=True)
class CorrectionFeedback:
    """Structured feedback for code correction.

    Provides clear, actionable feedback to guide the LLM
    in fixing issues with generated code.
    """

    has_issues: bool
    language: Language = Language.UNKNOWN
    syntax_feedback: Optional[str] = None
    import_feedback: Optional[str] = None
    test_feedback: Optional[str] = None
    general_feedback: Optional[str] = None

    def to_prompt(self) -> str:
        """Convert feedback to a prompt string for the LLM."""
        parts = []

        if self.syntax_feedback:
            parts.append(f"SYNTAX ERROR:\n{self.syntax_feedback}")

        if self.import_feedback:
            parts.append(f"IMPORT ISSUES:\n{self.import_feedback}")

        if self.test_feedback:
            parts.append(f"TEST FAILURES:\n{self.test_feedback}")

        if self.general_feedback:
            parts.append(f"ISSUES:\n{self.general_feedback}")

        return "\n\n".join(parts) if parts else ""

    @classmethod
    def no_issues(cls, language: Language = Language.UNKNOWN) -> "CorrectionFeedback":
        """Factory for feedback with no issues."""
        return cls(has_issues=False, language=language)
