"""
Python AST validator using the built-in ast module.

Validates Python code syntax using ast.parse().
"""

import ast
import logging
from pathlib import Path
from typing import Optional

from ..types import (
    CodeValidationResult,
    LanguageTier,
    ValidationConfig,
    ValidationIssue,
    ValidationSeverity,
)

logger = logging.getLogger(__name__)


class PythonASTValidator:
    """
    Python syntax validator using the built-in ast module.

    Validates Python code can be parsed without syntax errors.
    """

    def validate(
        self,
        code: str,
        file_path: Path,
        config: Optional[ValidationConfig] = None,
    ) -> CodeValidationResult:
        """
        Validate Python code syntax.

        Args:
            code: Python source code to validate
            file_path: Path to the source file (for error messages)
            config: Optional validation configuration

        Returns:
            CodeValidationResult with validation status and any errors
        """
        config = config or ValidationConfig()
        result = CodeValidationResult(
            is_valid=True,
            language="python",
            tier=LanguageTier.TIER_1,
            validators_used=["python_ast"],
        )

        if not config.check_syntax:
            return result

        try:
            # Try to parse the code
            ast.parse(code)
            logger.debug(f"Python syntax validation passed for {file_path}")

        except SyntaxError as e:
            result.is_valid = False
            issue = ValidationIssue(
                line=e.lineno or 1,
                column=e.offset or 0,
                message=str(e.msg) if e.msg else "Syntax error",
                severity=ValidationSeverity.ERROR,
                source="python_ast",
                code="E0001",
            )
            result.add_issue(issue)

            # Add context if available
            if e.text:
                result.metadata["error_text"] = e.text.rstrip()
            if e.end_lineno:
                issue.end_line = e.end_lineno
            if e.end_offset:
                issue.end_column = e.end_offset

            logger.debug(
                f"Python syntax validation failed for {file_path}: "
                f"line {e.lineno}, {e.msg}"
            )

        except Exception as e:
            # Unexpected error during parsing
            result.is_valid = False
            result.add_issue(
                ValidationIssue(
                    line=1,
                    column=0,
                    message=f"Parse error: {e}",
                    severity=ValidationSeverity.ERROR,
                    source="python_ast",
                    code="E0000",
                )
            )
            logger.warning(f"Unexpected error validating {file_path}: {e}")

        return result

    def validate_expression(self, expression: str) -> CodeValidationResult:
        """
        Validate a Python expression.

        Args:
            expression: Python expression to validate

        Returns:
            CodeValidationResult for the expression
        """
        result = CodeValidationResult(
            is_valid=True,
            language="python",
            tier=LanguageTier.TIER_1,
            validators_used=["python_ast_expression"],
        )

        try:
            ast.parse(expression, mode="eval")
        except SyntaxError as e:
            result.is_valid = False
            result.add_issue(
                ValidationIssue(
                    line=e.lineno or 1,
                    column=e.offset or 0,
                    message=str(e.msg) if e.msg else "Invalid expression",
                    severity=ValidationSeverity.ERROR,
                    source="python_ast",
                )
            )

        return result

    def validate_statement(self, statement: str) -> CodeValidationResult:
        """
        Validate a single Python statement.

        Args:
            statement: Python statement to validate

        Returns:
            CodeValidationResult for the statement
        """
        result = CodeValidationResult(
            is_valid=True,
            language="python",
            tier=LanguageTier.TIER_1,
            validators_used=["python_ast_statement"],
        )

        try:
            ast.parse(statement, mode="single")
        except SyntaxError as e:
            result.is_valid = False
            result.add_issue(
                ValidationIssue(
                    line=e.lineno or 1,
                    column=e.offset or 0,
                    message=str(e.msg) if e.msg else "Invalid statement",
                    severity=ValidationSeverity.ERROR,
                    source="python_ast",
                )
            )

        return result

    def get_syntax_errors(self, code: str) -> list:
        """
        Get all syntax errors from code (for compatibility).

        Args:
            code: Python source code

        Returns:
            List of error dictionaries
        """
        try:
            ast.parse(code)
            return []
        except SyntaxError as e:
            return [{
                "line": e.lineno or 1,
                "column": e.offset or 0,
                "message": str(e.msg) if e.msg else "Syntax error",
                "text": e.text,
            }]
