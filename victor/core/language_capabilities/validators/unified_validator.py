"""
Unified language validator for code grounding/enforcement.

Uses the capability registry to select the best validation method
for each language, falling back through the strategy chain.
"""

import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from ..types import (
    ASTAccessMethod,
    CodeValidationResult,
    LanguageTier,
    ValidationConfig,
    ValidationIssue,
    ValidationSeverity,
)
from ..extractors.base import BaseLanguageProcessor
from .python_validator import PythonASTValidator
from .tree_sitter_validator import TreeSitterValidator
from .config_validators import (
    JsonValidator,
    YamlValidator,
    TomlValidator,
    HoconValidator,
    XmlValidator,
    MarkdownValidator,
    get_config_validator,
)

if TYPE_CHECKING:
    from ..registry import LanguageCapabilityRegistry

logger = logging.getLogger(__name__)


class UnifiedLanguageValidator(BaseLanguageProcessor):
    """
    Unified validator for code grounding/enforcement.

    Uses the capability registry to select the best validation method
    for each language. Falls back through the strategy chain if the
    preferred method is unavailable.

    Validation Strategy:
    1. Try native AST validation (Python, etc.)
    2. Fall back to tree-sitter validation
    3. Optionally use LSP diagnostics
    """

    def __init__(
        self,
        registry: Optional["LanguageCapabilityRegistry"] = None,
        python_validator: Optional[PythonASTValidator] = None,
        tree_sitter_validator: Optional[TreeSitterValidator] = None,
    ) -> None:
        """
        Initialize the unified validator.

        Args:
            registry: Language capability registry (uses singleton if None)
            python_validator: Python AST validator (creates one if None)
            tree_sitter_validator: Tree-sitter validator (creates one if None)
        """
        super().__init__(registry)
        self._python_validator = python_validator or PythonASTValidator()
        self._tree_sitter = tree_sitter_validator or TreeSitterValidator()

    def process(
        self,
        code: str,
        file_path: Path,
        language: Optional[str] = None,
    ) -> CodeValidationResult:
        """
        Process code and validate it.

        Args:
            code: Source code to validate
            file_path: Path to the source file
            language: Optional language override

        Returns:
            CodeValidationResult with validation status
        """
        return self.validate(code, file_path, language)

    def validate(
        self,
        code: str,
        file_path: Path,
        language: Optional[str] = None,
        config: Optional[ValidationConfig] = None,
    ) -> CodeValidationResult:
        """
        Validate code using best available method.

        Args:
            code: Source code to validate
            file_path: Path to the source file
            language: Optional language override
            config: Optional validation configuration

        Returns:
            CodeValidationResult with validation status and any errors
        """
        config = config or ValidationConfig()
        cap = self._get_capability(file_path, language)

        # Unknown language
        if not cap:
            return CodeValidationResult(
                is_valid=True,
                language="unknown",
                validators_used=[],
                warnings=[
                    ValidationIssue(
                        line=1,
                        column=0,
                        message=f"Unknown language for {file_path.name}",
                        severity=ValidationSeverity.WARNING,
                        source="registry",
                    )
                ],
            )

        # Validation disabled
        if not cap.validation_enabled:
            return CodeValidationResult(
                is_valid=True,
                language=cap.name,
                tier=cap.tier,
                validators_used=[],
            )

        # Merge result from all validators in strategy order
        result = CodeValidationResult(
            is_valid=True,
            language=cap.name,
            tier=cap.tier,
        )

        ran_any_validator = False

        for method in cap.validation_strategy:
            if not cap._method_available(method):
                continue

            validator_result = self._run_validator(
                method, code, file_path, cap.name, config
            )

            if validator_result:
                ran_any_validator = True
                result.validators_used.extend(validator_result.validators_used)
                result.issues.extend(validator_result.issues)
                result.warnings.extend(validator_result.warnings)

                if not validator_result.is_valid:
                    result.is_valid = False
                    # If we found errors, don't run more validators
                    # (unless config says to continue)
                    if validator_result.errors:
                        break

        if not ran_any_validator:
            # No validators could run - use fallback behavior
            if cap.fallback_on_unavailable == "block":
                result.is_valid = False
                result.add_issue(
                    ValidationIssue(
                        line=1,
                        column=0,
                        message=f"No validator available for {cap.name}",
                        severity=ValidationSeverity.ERROR,
                        source="registry",
                    )
                )
            elif cap.fallback_on_unavailable == "warn":
                result.warnings.append(
                    ValidationIssue(
                        line=1,
                        column=0,
                        message=f"No validator available for {cap.name}",
                        severity=ValidationSeverity.WARNING,
                        source="registry",
                    )
                )
            # "allow" is default - no message needed

        return result

    def _run_validator(
        self,
        method: ASTAccessMethod,
        code: str,
        file_path: Path,
        language: str,
        config: ValidationConfig,
    ) -> Optional[CodeValidationResult]:
        """
        Run a specific validator method.

        Args:
            method: Validation method to use
            code: Source code to validate
            file_path: Path to source file
            language: Language name
            config: Validation configuration

        Returns:
            ValidationResult or None if validator not available
        """
        try:
            if method == ASTAccessMethod.NATIVE:
                return self._validate_native(code, file_path, language, config)
            elif method == ASTAccessMethod.PYTHON_LIB:
                return self._validate_python_lib(code, file_path, language, config)
            elif method == ASTAccessMethod.TREE_SITTER:
                return self._validate_tree_sitter(code, file_path, language, config)
            elif method == ASTAccessMethod.LSP:
                return self._validate_lsp(code, file_path, language, config)
            else:
                logger.debug(f"Validation method {method.value} not implemented")
                return None

        except Exception as e:
            logger.warning(f"Validator {method.value} failed for {language}: {e}")
            return CodeValidationResult(
                is_valid=False,
                language=language,
                validators_used=[method.value],
                issues=[
                    ValidationIssue(
                        line=1,
                        column=0,
                        message=f"Validator error: {e}",
                        severity=ValidationSeverity.ERROR,
                        source=method.value,
                    )
                ],
            )

    def _validate_native(
        self,
        code: str,
        file_path: Path,
        language: str,
        config: ValidationConfig,
    ) -> Optional[CodeValidationResult]:
        """Validate using native AST."""
        if language == "python":
            return self._python_validator.validate(code, file_path, config)

        # Try config file validators (json, yaml, toml, hocon, xml, markdown)
        config_validator = get_config_validator(language, self._tree_sitter)
        if config_validator and config_validator.is_available():
            return config_validator.validate(code, file_path, config)

        return None

    def _validate_python_lib(
        self,
        code: str,
        file_path: Path,
        language: str,
        config: ValidationConfig,
    ) -> Optional[CodeValidationResult]:
        """Validate using pure Python library."""
        # TODO: Implement for Go (gopygo), Java (javalang), etc.
        logger.debug(f"Python lib validation not implemented for {language}")
        return None

    def _validate_tree_sitter(
        self,
        code: str,
        file_path: Path,
        language: str,
        config: ValidationConfig,
    ) -> Optional[CodeValidationResult]:
        """Validate using tree-sitter."""
        if not self._tree_sitter.is_available():
            return None
        return self._tree_sitter.validate(code, file_path, language, config)

    def _validate_lsp(
        self,
        code: str,
        file_path: Path,
        language: str,
        config: ValidationConfig,
    ) -> Optional[CodeValidationResult]:
        """Validate using LSP diagnostics."""
        # TODO: Implement LSP validation
        logger.debug(f"LSP validation not implemented for {language}")
        return None

    def can_validate(self, file_path: Path, language: Optional[str] = None) -> bool:
        """Check if validation is supported for a file."""
        cap = self._get_capability(file_path, language)
        if not cap or not cap.validation_enabled:
            return False
        return cap.get_best_validation_method() is not None

    def get_validation_method(
        self,
        file_path: Path,
        language: Optional[str] = None,
    ) -> Optional[ASTAccessMethod]:
        """Get the validation method that would be used for a file."""
        cap = self._get_capability(file_path, language)
        if not cap or not cap.validation_enabled:
            return None
        return cap.get_best_validation_method()

    def quick_validate(
        self,
        code: str,
        file_path: Path,
        language: Optional[str] = None,
    ) -> bool:
        """
        Quick syntax validation (for performance).

        Args:
            code: Source code to validate
            file_path: Path to source file
            language: Optional language override

        Returns:
            True if code has valid syntax, False otherwise
        """
        cap = self._get_capability(file_path, language)
        if not cap or not cap.validation_enabled:
            return True  # Unknown/disabled = allow

        lang = cap.name

        # Try native first (faster)
        if lang == "python":
            try:
                import ast
                ast.parse(code)
                return True
            except SyntaxError:
                return False

        # Fall back to tree-sitter
        if self._tree_sitter.is_available():
            return not self._tree_sitter.has_errors(code, lang)

        # No validator available - allow
        return True
