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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import Language, ValidationResult


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

            def validate(self, code: str) -> ValidationResult:
                # Ruby-specific validation
                ...

            def fix(self, code: str, validation: ValidationResult) -> str:
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
    def validate(self, code: str) -> "ValidationResult":
        """Validate code and return structured result.

        Args:
            code: Source code string to validate

        Returns:
            ValidationResult with validation status and details
        """
        pass

    @abstractmethod
    def fix(self, code: str, validation: "ValidationResult") -> str:
        """Attempt to auto-fix issues found during validation.

        Args:
            code: Original source code
            validation: Validation result with identified issues

        Returns:
            Fixed code string (may be unchanged if no fixes applicable)
        """
        pass

    def clean_markdown(self, code: str) -> str:
        """Remove markdown code blocks - universal for all languages.

        This is a Template Method - shared implementation that all
        validators can use for consistent markdown handling.

        Args:
            code: Code potentially wrapped in markdown blocks

        Returns:
            Cleaned code without markdown formatting
        """
        # Try language-specific blocks first
        for lang_marker in self.MARKDOWN_LANGUAGE_MARKERS:
            pattern = rf"```{lang_marker}\n(.*?)```"
            match = re.search(pattern, code, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Try generic code block
        if "```" in code:
            match = re.search(r"```\n?(.*?)```", code, re.DOTALL)
            if match:
                return match.group(1).strip()

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
