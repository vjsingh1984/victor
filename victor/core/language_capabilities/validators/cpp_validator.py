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

"""C/C++ language validator using libclang.

This validator uses libclang (Python bindings for Clang) when available,
falling back to tree-sitter when libclang is not installed.

libclang provides:
- Accurate syntax error detection
- Semantic error detection (type mismatches, undefined symbols, etc.)
- Warning detection

Install libclang with: pip install libclang
"""

import logging
from pathlib import Path
from typing import List, Optional

from ..types import CodeValidationResult, ValidationConfig, ValidationIssue, ValidationSeverity
from .tree_sitter_validator import TreeSitterValidator

logger = logging.getLogger(__name__)

# Check if libclang is available
try:
    import clang.cindex as cindex

    LIBCLANG_AVAILABLE = True
except ImportError:
    LIBCLANG_AVAILABLE = False
    cindex = None
    logger.debug("libclang not available, C/C++ validation will use tree-sitter")


class CppValidator:
    """C/C++ language validator using libclang.

    Falls back to tree-sitter when libclang is not available.

    libclang provides superior validation including:
    - Syntax error detection
    - Semantic error detection (type mismatches, etc.)
    - Warning detection
    - Fix-it suggestions
    """

    def __init__(
        self,
        tree_sitter_validator: Optional[TreeSitterValidator] = None,
    ):
        """Initialize the C/C++ validator.

        Args:
            tree_sitter_validator: Optional tree-sitter validator for fallback
        """
        self._ts_validator = tree_sitter_validator or TreeSitterValidator()
        self._index = None

        if LIBCLANG_AVAILABLE:
            self._init_libclang()

    def _init_libclang(self) -> None:
        """Initialize libclang index."""
        if not LIBCLANG_AVAILABLE:
            return

        try:
            self._index = cindex.Index.create()
        except Exception as e:
            logger.warning(f"Failed to initialize libclang: {e}")
            self._index = None

    def is_available(self) -> bool:
        """Check if native C/C++ validation is available."""
        return LIBCLANG_AVAILABLE and self._index is not None

    def validate(
        self,
        code: str,
        file_path: Path,
        language: Optional[str] = None,
        config: Optional[ValidationConfig] = None,
    ) -> CodeValidationResult:
        """Validate C/C++ code syntax and semantics.

        Args:
            code: C/C++ source code to validate
            file_path: Path to the source file
            language: 'c' or 'cpp' (auto-detected from extension if not provided)
            config: Optional validation configuration

        Returns:
            CodeValidationResult with validation status and issues
        """
        # Determine language from extension if not provided
        if language is None:
            ext = file_path.suffix.lower()
            language = "cpp" if ext in [".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"] else "c"

        if config and not config.check_syntax:
            return CodeValidationResult(
                is_valid=True,
                language=language,
                validators_used=["cpp_validator"],
            )

        if not self.is_available():
            # Fallback to tree-sitter
            return self._ts_validator.validate(code, file_path, language, config)

        try:
            return self._validate_with_libclang(code, file_path, language)
        except Exception as e:
            logger.warning(f"libclang validation failed for {file_path}: {e}")
            # Fallback to tree-sitter on error
            return self._ts_validator.validate(code, file_path, language, config)

    def _validate_with_libclang(
        self,
        code: str,
        file_path: Path,
        language: str,
    ) -> CodeValidationResult:
        """Validate C/C++ code using libclang.

        Args:
            code: C/C++ source code
            file_path: Path to the source file
            language: 'c' or 'cpp'

        Returns:
            CodeValidationResult with validation status
        """
        issues: List[ValidationIssue] = []
        warnings: List[ValidationIssue] = []

        if self._index is None:
            raise RuntimeError("libclang index not initialized")

        try:
            # Parse options
            args = ["-x", "c++" if language == "cpp" else "c"]

            # Parse the code
            tu = self._index.parse(
                str(file_path),
                args=args,
                unsaved_files=[(str(file_path), code)],
            )

            # Map clang severity to our severity
            severity_map = {
                cindex.Diagnostic.Ignored: ValidationSeverity.INFO,
                cindex.Diagnostic.Note: ValidationSeverity.INFO,
                cindex.Diagnostic.Warning: ValidationSeverity.WARNING,
                cindex.Diagnostic.Error: ValidationSeverity.ERROR,
                cindex.Diagnostic.Fatal: ValidationSeverity.ERROR,
            }

            has_errors = False

            for diag in tu.diagnostics:
                severity = severity_map.get(diag.severity, ValidationSeverity.WARNING)

                issue = ValidationIssue(
                    line=diag.location.line,
                    column=diag.location.column,
                    message=diag.spelling,
                    severity=severity,
                    source="libclang",
                )

                if severity == ValidationSeverity.ERROR:
                    issues.append(issue)
                    has_errors = True
                elif severity == ValidationSeverity.WARNING:
                    warnings.append(issue)

            if has_errors:
                logger.debug(
                    f"{language.upper()} syntax validation failed for {file_path}: "
                    f"{len(issues)} errors"
                )
                return CodeValidationResult(
                    is_valid=False,
                    language=language,
                    validators_used=["libclang"],
                    issues=issues,
                    warnings=warnings,
                )

            logger.debug(f"{language.upper()} syntax validation passed for {file_path}")
            return CodeValidationResult(
                is_valid=True,
                language=language,
                validators_used=["libclang"],
                warnings=warnings,
            )

        except Exception as e:
            logger.debug(f"{language.upper()} parsing error for {file_path}: {e}")

            issues.append(
                ValidationIssue(
                    line=1,
                    column=0,
                    message=str(e),
                    severity=ValidationSeverity.ERROR,
                    source="libclang",
                )
            )

            return CodeValidationResult(
                is_valid=False,
                language=language,
                validators_used=["libclang"],
                issues=issues,
            )

    def has_errors(self, code: str, language: str = "cpp") -> bool:
        """Check if C/C++ code has syntax errors.

        Args:
            code: C/C++ source code
            language: 'c' or 'cpp'

        Returns:
            True if code has syntax errors
        """
        if not self.is_available():
            return self._ts_validator.has_errors(code, language)

        if self._index is None:
            return False

        try:
            args = ["-x", "c++" if language == "cpp" else "c"]

            tu = self._index.parse(
                "temp.cpp" if language == "cpp" else "temp.c",
                args=args,
                unsaved_files=[("temp.cpp" if language == "cpp" else "temp.c", code)],
            )

            # Check for errors in diagnostics
            for diag in tu.diagnostics:
                if diag.severity >= cindex.Diagnostic.Error:
                    return True

            return False
        except Exception:
            return True
