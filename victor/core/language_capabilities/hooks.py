"""
Integration hooks for code validation before file writes.

Provides CodeGroundingHook for integrating validation with:
- FileEditor.commit()
- TestGenManager._write_test_file()
- RefactorManager.apply()
- file_editor_tool.edit()
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from .types import (
    CodeValidationResult,
    ValidationConfig,
    ValidationIssue,
    ValidationSeverity,
)
from .validators.unified_validator import UnifiedLanguageValidator

if TYPE_CHECKING:
    from .registry import LanguageCapabilityRegistry

logger = logging.getLogger(__name__)


class CodeGroundingHook:
    """
    Integration hook for code validation before file writes.

    Used to validate code before it's written to disk, preventing
    the creation of syntactically invalid files.

    Usage:
        hook = CodeGroundingHook.instance()
        should_proceed, result = await hook.validate_before_write(
            content="def foo(:",
            file_path=Path("example.py"),
        )
        if not should_proceed:
            print("Validation failed:", result.issues)
    """

    _instance: Optional["CodeGroundingHook"] = None

    def __init__(
        self,
        validator: Optional[UnifiedLanguageValidator] = None,
        registry: Optional["LanguageCapabilityRegistry"] = None,
    ) -> None:
        """
        Initialize the code grounding hook.

        Args:
            validator: Unified validator (creates one if None)
            registry: Language capability registry (uses singleton if None)
        """
        if registry is None:
            from .registry import LanguageCapabilityRegistry

            registry = LanguageCapabilityRegistry.instance()

        self._validator = validator or UnifiedLanguageValidator(registry=registry)
        self._registry = registry

    @classmethod
    def instance(cls) -> "CodeGroundingHook":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None

    async def validate_before_write(
        self,
        content: str,
        file_path: str | Path,
        strict: bool = False,
        config: Optional[ValidationConfig] = None,
    ) -> tuple[bool, CodeValidationResult]:
        """
        Validate code before writing to disk.

        Args:
            content: Code content to validate
            file_path: Target file path
            strict: Block on any validation failure (overrides capability setting)
            config: Optional validation configuration

        Returns:
            Tuple of (should_proceed, validation_result)
            - should_proceed: True if write should proceed
            - validation_result: Full validation result with issues
        """
        # Convert str to Path for consistency
        if isinstance(file_path, str):
            file_path = Path(file_path)

        cap = self._registry.get_for_file(file_path)

        # Unknown language - allow with warning
        if not cap:
            return True, CodeValidationResult(
                is_valid=True,
                language="unknown",
                warnings=[
                    ValidationIssue(
                        line=1,
                        column=0,
                        message=f"Cannot validate unknown file type: {file_path.suffix}",
                        severity=ValidationSeverity.WARNING,
                        source="hook",
                    )
                ],
            )

        # Validation disabled for this language
        if not cap.validation_enabled:
            return True, CodeValidationResult(
                is_valid=True,
                language=cap.name,
                tier=cap.tier,
            )

        # Run validation
        result = self._validator.validate(content, file_path, config=config)

        # Determine if write should proceed
        if strict or cap.fallback_on_error == "block":
            should_proceed = result.is_valid
        elif cap.fallback_on_error == "warn":
            should_proceed = True  # Proceed but log warnings
            if not result.is_valid:
                for issue in result.errors:
                    logger.warning(f"Validation warning for {file_path}: {issue}")
        else:  # "allow"
            should_proceed = True

        return should_proceed, result

    def validate_before_write_sync(
        self,
        content: str,
        file_path: str | Path,
        strict: bool = False,
        config: Optional[ValidationConfig] = None,
    ) -> tuple[bool, CodeValidationResult]:
        """
        Synchronous version for non-async contexts.

        Args:
            content: Code content to validate
            file_path: Target file path
            strict: Block on any validation failure
            config: Optional validation configuration

        Returns:
            Tuple of (should_proceed, validation_result)
        """
        # Convert str to Path for consistency
        if isinstance(file_path, str):
            file_path = Path(file_path)

        try:
            asyncio.get_running_loop()
            # We're in an async context, need to use run_in_executor or similar
            # For now, use synchronous validation
            return self._validate_sync(content, file_path, strict, config)
        except RuntimeError:
            # No running loop, can use asyncio.run
            return asyncio.run(self.validate_before_write(content, file_path, strict, config))

    def _validate_sync(
        self,
        content: str,
        file_path: Path,
        strict: bool,
        config: Optional[ValidationConfig],
    ) -> tuple[bool, CodeValidationResult]:
        """
        Pure synchronous validation (no async).

        Uses quick_validate for fast path, then full validation if needed.
        """
        # No need to check isinstance since file_path is already Path from calling method

        cap = self._registry.get_for_file(file_path)

        # Unknown language
        if not cap:
            return True, CodeValidationResult(
                is_valid=True,
                language="unknown",
                warnings=[
                    ValidationIssue(
                        line=1,
                        column=0,
                        message=f"Cannot validate unknown file type: {file_path.suffix}",
                        severity=ValidationSeverity.WARNING,
                        source="hook",
                    )
                ],
            )

        # Validation disabled
        if not cap.validation_enabled:
            return True, CodeValidationResult(
                is_valid=True,
                language=cap.name,
                tier=cap.tier,
            )

        # Run validation synchronously
        result = self._validator.validate(content, file_path, config=config)

        # Determine if write should proceed
        if strict or cap.fallback_on_error == "block":
            should_proceed = result.is_valid
        else:
            should_proceed = True
            if not result.is_valid and cap.fallback_on_error == "warn":
                for issue in result.errors:
                    logger.warning(f"Validation warning for {file_path}: {issue}")

        return should_proceed, result

    def quick_validate(
        self,
        content: str,
        file_path: Path,
    ) -> bool:
        """
        Quick syntax validation (for performance-critical paths).

        Args:
            content: Code content to validate
            file_path: Target file path

        Returns:
            True if code appears valid, False otherwise
        """
        return self._validator.quick_validate(content, file_path)

    def can_validate(self, file_path: Path) -> bool:
        """
        Check if validation is available for a file.

        Args:
            file_path: Target file path

        Returns:
            True if validation is available
        """
        return self._validator.can_validate(file_path)

    @property
    def validator(self) -> UnifiedLanguageValidator:
        """Get the underlying validator."""
        return self._validator

    @property
    def registry(self) -> "LanguageCapabilityRegistry":
        """Get the capability registry."""
        return self._registry


def validate_code_before_write(
    content: str,
    file_path: Path,
    strict: bool = False,
) -> tuple[bool, CodeValidationResult]:
    """
    Convenience function for code validation.

    Args:
        content: Code content to validate
        file_path: Target file path
        strict: Block on any validation failure

    Returns:
        Tuple of (should_proceed, validation_result)
    """
    hook = CodeGroundingHook.instance()
    return hook.validate_before_write_sync(content, file_path, strict)


async def validate_code_before_write_async(
    content: str,
    file_path: Path,
    strict: bool = False,
) -> tuple[bool, CodeValidationResult]:
    """
    Async convenience function for code validation.

    Args:
        content: Code content to validate
        file_path: Target file path
        strict: Block on any validation failure

    Returns:
        Tuple of (should_proceed, validation_result)
    """
    hook = CodeGroundingHook.instance()
    return await hook.validate_before_write(content, file_path, strict)
