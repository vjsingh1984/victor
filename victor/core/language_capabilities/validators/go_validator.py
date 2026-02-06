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

"""Go language validator using gopygo library.

This validator uses gopygo (Pure Python Go parser) when available,
falling back to tree-sitter when gopygo is not installed.
"""

import logging
from pathlib import Path
from typing import Optional

from ..types import CodeValidationResult, ValidationConfig, ValidationIssue, ValidationSeverity
from .tree_sitter_validator import TreeSitterValidator

logger = logging.getLogger(__name__)

# Check if gopygo is available
try:
    import gopygo

    GOPYGO_AVAILABLE = True
except ImportError:
    GOPYGO_AVAILABLE = False
    logger.debug("gopygo not available, Go validation will use tree-sitter")


class GoValidator:
    """Go language validator using gopygo.

    Falls back to tree-sitter when gopygo is not available.
    """

    def __init__(
        self,
        tree_sitter_validator: Optional[TreeSitterValidator] = None,
    ):
        """Initialize the Go validator.

        Args:
            tree_sitter_validator: Optional tree-sitter validator for fallback
        """
        self._ts_validator = tree_sitter_validator or TreeSitterValidator()

    def is_available(self) -> bool:
        """Check if native Go validation is available."""
        return GOPYGO_AVAILABLE

    def validate(
        self,
        code: str,
        file_path: Path,
        config: Optional[ValidationConfig] = None,
    ) -> CodeValidationResult:
        """Validate Go code syntax.

        Args:
            code: Go source code to validate
            file_path: Path to the source file
            config: Optional validation configuration

        Returns:
            CodeValidationResult with validation status and issues
        """
        if config and not config.check_syntax:
            return CodeValidationResult(
                is_valid=True,
                language="go",
                validators_used=["go_validator"],
            )

        if not GOPYGO_AVAILABLE:
            # Fallback to tree-sitter
            return self._ts_validator.validate(code, file_path, "go", config)

        try:
            return self._validate_with_gopygo(code, file_path)
        except Exception as e:
            logger.warning(f"gopygo validation failed for {file_path}: {e}")
            # Fallback to tree-sitter on error
            return self._ts_validator.validate(code, file_path, "go", config)

    def _validate_with_gopygo(
        self,
        code: str,
        file_path: Path,
    ) -> CodeValidationResult:
        """Validate Go code using gopygo.

        Args:
            code: Go source code
            file_path: Path to the source file

        Returns:
            CodeValidationResult with validation status
        """
        issues = []

        try:
            gopygo.parse(code)
            logger.debug(f"Go syntax validation passed for {file_path}")
            return CodeValidationResult(
                is_valid=True,
                language="go",
                validators_used=["gopygo"],
            )
        except Exception as e:
            error_msg = str(e)
            logger.debug(f"Go syntax validation failed for {file_path}: {error_msg}")

            # Try to extract line number from error
            line = 1
            column = 0

            issues.append(
                ValidationIssue(
                    line=line,
                    column=column,
                    message=error_msg,
                    severity=ValidationSeverity.ERROR,
                    source="gopygo",
                )
            )

            return CodeValidationResult(
                is_valid=False,
                language="go",
                validators_used=["gopygo"],
                issues=issues,
            )

    def has_errors(self, code: str) -> bool:
        """Check if Go code has syntax errors.

        Args:
            code: Go source code

        Returns:
            True if code has syntax errors
        """
        if not GOPYGO_AVAILABLE:
            return self._ts_validator.has_errors(code, "go")

        try:
            gopygo.parse(code)
            return False
        except Exception:
            return True
