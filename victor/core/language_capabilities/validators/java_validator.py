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

"""Java language validator using javalang library.

This validator uses javalang (Pure Python Java parser) when available,
falling back to tree-sitter when javalang is not installed.

Note: javalang only supports Java 8 syntax. For Java 9+ features,
tree-sitter is used automatically.
"""

import logging
from pathlib import Path
from typing import Optional

from ..types import CodeValidationResult, ValidationConfig, ValidationIssue, ValidationSeverity
from .tree_sitter_validator import TreeSitterValidator

logger = logging.getLogger(__name__)

# Check if javalang is available
try:
    import javalang  
    import javalang.parser  

    JAVALANG_AVAILABLE = True
except ImportError:
    JAVALANG_AVAILABLE = False
    logger.debug("javalang not available, Java validation will use tree-sitter")


class JavaValidator:
    """Java language validator using javalang.

    Falls back to tree-sitter when javalang is not available
    or when parsing fails (e.g., Java 9+ code).
    """

    def __init__(
        self,
        tree_sitter_validator: Optional[TreeSitterValidator] = None,
    ):
        """Initialize the Java validator.

        Args:
            tree_sitter_validator: Optional tree-sitter validator for fallback
        """
        self._ts_validator = tree_sitter_validator or TreeSitterValidator()

    def is_available(self) -> bool:
        """Check if native Java validation is available."""
        return JAVALANG_AVAILABLE

    def validate(
        self,
        code: str,
        file_path: Path,
        config: Optional[ValidationConfig] = None,
    ) -> CodeValidationResult:
        """Validate Java code syntax.

        Args:
            code: Java source code to validate
            file_path: Path to the source file
            config: Optional validation configuration

        Returns:
            CodeValidationResult with validation status and issues
        """
        if config and not config.check_syntax:
            return CodeValidationResult(
                is_valid=True,
                language="java",
                validators_used=["java_validator"],
            )

        if not JAVALANG_AVAILABLE:
            # Fallback to tree-sitter
            return self._ts_validator.validate(code, file_path, "java", config)

        try:
            return self._validate_with_javalang(code, file_path)
        except Exception as e:
            logger.warning(f"javalang validation failed for {file_path}: {e}")
            # Fallback to tree-sitter on error (e.g., Java 9+ features)
            return self._ts_validator.validate(code, file_path, "java", config)

    def _validate_with_javalang(
        self,
        code: str,
        file_path: Path,
    ) -> CodeValidationResult:
        """Validate Java code using javalang.

        Args:
            code: Java source code
            file_path: Path to the source file

        Returns:
            CodeValidationResult with validation status
        """
        issues = []

        try:
            javalang.parse.parse(code)
            logger.debug(f"Java syntax validation passed for {file_path}")
            return CodeValidationResult(
                is_valid=True,
                language="java",
                validators_used=["javalang"],
            )
        except javalang.parser.JavaSyntaxError as e:
            error_msg = str(e)
            logger.debug(f"Java syntax validation failed for {file_path}: {error_msg}")

            # Try to extract position from error
            line = 1
            column = 0
            if hasattr(e, "at") and e.at:
                if hasattr(e.at, "position") and e.at.position:
                    line = e.at.position[0]
                    column = e.at.position[1]

            issues.append(
                ValidationIssue(
                    line=line,
                    column=column,
                    message=error_msg,
                    severity=ValidationSeverity.ERROR,
                    source="javalang",
                )
            )

            return CodeValidationResult(
                is_valid=False,
                language="java",
                validators_used=["javalang"],
                issues=issues,
            )
        except Exception as e:
            # Other parsing errors
            error_msg = str(e)
            logger.debug(f"Java parsing error for {file_path}: {error_msg}")

            issues.append(
                ValidationIssue(
                    line=1,
                    column=0,
                    message=error_msg,
                    severity=ValidationSeverity.ERROR,
                    source="javalang",
                )
            )

            return CodeValidationResult(
                is_valid=False,
                language="java",
                validators_used=["javalang"],
                issues=issues,
            )

    def has_errors(self, code: str) -> bool:
        """Check if Java code has syntax errors.

        Args:
            code: Java source code

        Returns:
            True if code has syntax errors
        """
        if not JAVALANG_AVAILABLE:
            return self._ts_validator.has_errors(code, "java")

        try:
            javalang.parse.parse(code)
            return False
        except Exception:
            return True
