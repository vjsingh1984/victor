# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Self-correction orchestrator for iterative code refinement.

This module provides the main entry point for the self-correction system.
It coordinates language detection, validation, feedback generation,
and retry prompt building.

Design Patterns:
- Facade Pattern: Simple interface hiding complex subsystem
- Dependency Injection: Components can be injected for testing
- Strategy Pattern: Different validators for different languages

Enterprise Integration Patterns:
- Message Router: Routes code to appropriate validator
- Content-Based Router: Language detection drives routing
"""

import logging
from typing import Optional, Protocol

from .detector import LanguageDetector, get_detector
from .feedback import (
    FeedbackGenerator,
    RetryPromptBuilder,
    get_feedback_generator,
    get_prompt_builder,
)
from .registry import CodeValidatorRegistry, get_registry
from .types import CorrectionFeedback, Language, CodeValidationResult

logger = logging.getLogger(__name__)


class ValidatorProvider(Protocol):
    """Protocol for validator lookup."""

    def get_validator(self, language: Language): ...
    def has_validator(self, language: Language) -> bool: ...


class SelfCorrector:
    """Orchestrates the self-correction loop for code generation.

    This is the main entry point for the correction system. It provides
    a simple interface for:
    - Validating and fixing code
    - Generating feedback for failed attempts
    - Building retry prompts

    The orchestrator uses dependency injection for all components,
    making it fully testable and configurable.

    Usage:
        corrector = SelfCorrector()

        # Validate and optionally fix code
        fixed_code, validation = corrector.validate_and_fix(code)

        # Check if retry is needed
        if corrector.should_retry(iteration, validation, test_passed, test_total):
            feedback = corrector.generate_feedback(code, validation, stderr=error)
            retry_prompt = corrector.build_retry_prompt(original, code, feedback, iteration)

    Configuration:
        corrector = SelfCorrector(
            max_iterations=5,
            auto_fix=True,
            validate_before_test=True,
        )

    Dependency Injection:
        corrector = SelfCorrector(
            registry=custom_registry,
            detector=custom_detector,
            feedback_generator=custom_generator,
        )
    """

    def __init__(
        self,
        max_iterations: int = 3,
        auto_fix: bool = True,
        validate_before_test: bool = True,
        # Dependency injection
        registry: Optional[CodeValidatorRegistry] = None,
        detector: Optional[LanguageDetector] = None,
        feedback_generator: Optional[FeedbackGenerator] = None,
        prompt_builder: Optional[RetryPromptBuilder] = None,
    ):
        """Initialize self-corrector.

        Args:
            max_iterations: Maximum correction attempts
            auto_fix: Whether to auto-fix common issues
            validate_before_test: Whether to validate before running tests
            registry: Optional custom validator registry
            detector: Optional custom language detector
            feedback_generator: Optional custom feedback generator
            prompt_builder: Optional custom prompt builder
        """
        self.max_iterations = max_iterations
        self.auto_fix = auto_fix
        self.validate_before_test = validate_before_test

        # Use injected dependencies or defaults
        self._registry = registry or get_registry()
        self._detector = detector or get_detector()
        self._feedback_generator = feedback_generator or get_feedback_generator()
        self._prompt_builder = prompt_builder or get_prompt_builder()

        # Ensure validators are discovered
        self._registry.discover_validators()

        logger.debug(
            f"SelfCorrector initialized: max_iterations={max_iterations}, "
            f"auto_fix={auto_fix}, validators={len(self._registry.all_validators)}"
        )

    def validate_and_fix(
        self,
        code: str,
        language: Optional[Language] = None,
        filename: Optional[str] = None,
    ) -> tuple[str, CodeValidationResult]:
        """Validate code and optionally apply fixes.

        Args:
            code: Generated code string
            language: Optional language hint (auto-detected if not provided)
            filename: Optional filename for language detection

        Returns:
            Tuple of (possibly fixed code, validation result)
        """
        # Detect language if not provided
        if language is None:
            language = self._detector.detect(code, filename)
            logger.debug(f"Detected language: {language.name}")

        # Get appropriate validator
        validator = self._registry.get_validator(language)
        logger.debug(f"Using validator: {type(validator).__name__}")

        # Preprocess (clean markdown, etc.)
        cleaned_code = validator.preprocess(code)

        # Validate
        validation = validator.validate(cleaned_code)

        # Auto-fix if enabled and there are fixable issues
        if self.auto_fix and not validation.valid:
            fixed_code = validator.fix(cleaned_code, validation)

            # Re-validate after fixes
            if fixed_code != cleaned_code:
                logger.debug("Applied auto-fixes, re-validating")
                validation = validator.validate(fixed_code)
                cleaned_code = fixed_code

        return cleaned_code, validation

    def should_retry(
        self,
        iteration: int,
        validation: CodeValidationResult,
        test_passed: Optional[int] = None,
        test_total: Optional[int] = None,
    ) -> bool:
        """Determine if we should retry based on current state.

        Args:
            iteration: Current iteration number (0-indexed)
            validation: Validation result from current attempt
            test_passed: Number of tests passed (if available)
            test_total: Total number of tests (if available)

        Returns:
            True if should retry, False otherwise
        """
        # Check iteration limit
        if iteration >= self.max_iterations:
            logger.debug(f"Max iterations ({self.max_iterations}) reached")
            return False

        # Retry if validation failed
        if not validation.valid:
            logger.debug("Validation failed, should retry")
            return True

        # Retry if tests failed
        if test_passed is not None and test_total is not None:
            if test_passed < test_total:
                logger.debug(f"Tests failed ({test_passed}/{test_total}), should retry")
                return True

        return False

    def generate_feedback(
        self,
        code: str,
        validation: CodeValidationResult,
        test_stdout: Optional[str] = None,
        test_stderr: Optional[str] = None,
        test_passed: Optional[int] = None,
        test_total: Optional[int] = None,
    ) -> CorrectionFeedback:
        """Generate feedback for the next iteration.

        Args:
            code: The code that was attempted
            validation: Validation result
            test_stdout: Test standard output
            test_stderr: Test standard error
            test_passed: Number of tests passed
            test_total: Total number of tests

        Returns:
            CorrectionFeedback with structured feedback
        """
        return self._feedback_generator.generate(
            code=code,
            validation=validation,
            test_stdout=test_stdout,
            test_stderr=test_stderr,
            test_passed=test_passed,
            test_total=test_total,
        )

    def build_retry_prompt(
        self,
        original_prompt: str,
        previous_code: str,
        feedback: CorrectionFeedback,
        iteration: int,
    ) -> str:
        """Build a prompt for retry with feedback.

        Args:
            original_prompt: The original task description
            previous_code: Code from the failed attempt
            feedback: Feedback on what went wrong
            iteration: Current iteration number

        Returns:
            Complete retry prompt for the LLM
        """
        return self._prompt_builder.build(
            original_prompt=original_prompt,
            previous_code=previous_code,
            feedback=feedback,
            iteration=iteration,
        )

    def has_ast_support(self, language: Language) -> bool:
        """Check if language has specialized validator support.

        Args:
            language: Language to check

        Returns:
            True if specialized validator exists
        """
        return self._registry.has_validator(language)

    @property
    def supported_languages(self) -> set[Language]:
        """Languages with specialized validator support."""
        return self._registry.registered_languages

    @property
    def registry(self) -> CodeValidatorRegistry:
        """Access to the validator registry (for advanced use)."""
        return self._registry

    @property
    def detector(self) -> LanguageDetector:
        """Access to the language detector (for advanced use)."""
        return self._detector


# Factory function for easy creation
def create_self_corrector(
    max_iterations: int = 3,
    auto_fix: bool = True,
) -> SelfCorrector:
    """Create a configured SelfCorrector instance.

    Convenience factory function for common use cases.

    Args:
        max_iterations: Maximum correction attempts
        auto_fix: Whether to auto-fix common issues

    Returns:
        Configured SelfCorrector instance
    """
    return SelfCorrector(
        max_iterations=max_iterations,
        auto_fix=auto_fix,
        validate_before_test=True,
    )
