# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Code correction and self-improvement system.

This package provides a modular, extensible system for validating
and correcting LLM-generated code. It supports multiple programming
languages through a plugin-based validator architecture.

Architecture Overview:
----------------------

    ┌─────────────────────────────────────────────────────────────┐
    │                     SelfCorrector                            │
    │                    (Orchestrator/Facade)                     │
    └──────────────┬────────────────────────────────┬─────────────┘
                   │                                │
    ┌──────────────▼──────────────┐  ┌─────────────▼─────────────┐
    │     LanguageDetector        │  │    FeedbackGenerator      │
    │   (Chain of Responsibility) │  │      (Builder Pattern)    │
    └──────────────┬──────────────┘  └───────────────────────────┘
                   │
    ┌──────────────▼──────────────┐
    │   CodeValidatorRegistry     │
    │    (Service Locator)        │
    └──────────────┬──────────────┘
                   │ auto-discovers
    ┌──────────────▼──────────────┐
    │    validators/              │
    │  ├── python_validator.py    │ ◄── AST-based (Tier 1)
    │  ├── js_validator.py        │ ◄── Pattern-based (Tier 2)
    │  └── generic_validator.py   │ ◄── Fallback (Tier 3)
    └─────────────────────────────┘

Design Patterns Used:
--------------------
- Facade Pattern: SelfCorrector provides simple interface
- Strategy Pattern: Validators are interchangeable
- Registry Pattern: CodeValidatorRegistry for lookup
- Plugin Pattern: Auto-discovery of validators
- Null Object Pattern: GenericCodeValidator as fallback
- Template Method: BaseCodeValidator.clean_markdown()
- Builder Pattern: FeedbackGenerator builds feedback objects
- Chain of Responsibility: LanguageDetector tries multiple strategies

Enterprise Integration Patterns:
--------------------------------
- Service Locator: Registry provides validator lookup
- Content-Based Router: Language determines validator
- Message Router: Routes code to appropriate handler

Usage Examples:
--------------

    # Basic usage
    from victor.evaluation.correction import SelfCorrector

    corrector = SelfCorrector()
    fixed_code, validation = corrector.validate_and_fix(code)

    if not validation.valid:
        feedback = corrector.generate_feedback(code, validation)
        print(feedback.to_prompt())

    # With dependency injection (for testing)
    from victor.evaluation.correction import (
        SelfCorrector,
        CodeValidatorRegistry,
        LanguageDetector,
    )

    registry = CodeValidatorRegistry()
    registry.register(MyCustomValidator())

    corrector = SelfCorrector(registry=registry)

    # Language detection
    from victor.evaluation.correction import detect_language, Language

    lang = detect_language(code, filename="example.py")
    if lang == Language.PYTHON:
        print("Python code detected")

Adding New Language Validators:
------------------------------

    # 1. Create validators/rust_validator.py

    from ..base import BaseCodeValidator
    from ..types import Language, ValidationResult

    class RustCodeValidator(BaseCodeValidator):
        @property
        def supported_languages(self) -> set[Language]:
            return {Language.RUST}

        def validate(self, code: str) -> ValidationResult:
            # Rust-specific validation
            ...

        def fix(self, code: str, validation: ValidationResult) -> str:
            # Rust-specific fixes
            ...

    # 2. The registry auto-discovers it on next startup
"""

# Core types
from .types import (
    Language,
    ValidationResult,
    CorrectionFeedback,
)

# Base classes
from .base import (
    BaseCodeValidator,
    ValidatorCapabilities,
)

# Language detection
from .detector import (
    LanguageDetector,
    detect_language,
    get_detector,
)

# Registry
from .registry import (
    CodeValidatorRegistry,
    get_registry,
)

# Feedback
from .feedback import (
    FeedbackGenerator,
    RetryPromptBuilder,
    get_feedback_generator,
    get_prompt_builder,
)

# Orchestrator (main entry point)
from .orchestrator import (
    SelfCorrector,
    create_self_corrector,
)

# Metrics and observability
from .metrics import (
    CorrectionAttempt,
    CorrectionMetrics,
    CorrectionMetricsCollector,
    CorrectionTracker,
    get_metrics_collector,
    reset_metrics,
)

# Validators (optional direct import)
from .validators import (
    GenericCodeValidator,
    PythonCodeValidator,
)

__all__ = [
    # Types
    "Language",
    "ValidationResult",
    "CorrectionFeedback",
    # Base
    "BaseCodeValidator",
    "ValidatorCapabilities",
    # Detection
    "LanguageDetector",
    "detect_language",
    "get_detector",
    # Registry
    "CodeValidatorRegistry",
    "get_registry",
    # Feedback
    "FeedbackGenerator",
    "RetryPromptBuilder",
    "get_feedback_generator",
    "get_prompt_builder",
    # Orchestrator
    "SelfCorrector",
    "create_self_corrector",
    # Metrics
    "CorrectionAttempt",
    "CorrectionMetrics",
    "CorrectionMetricsCollector",
    "CorrectionTracker",
    "get_metrics_collector",
    "reset_metrics",
    # Validators
    "GenericCodeValidator",
    "PythonCodeValidator",
]
