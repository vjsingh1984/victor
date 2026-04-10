# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Validator registry with plugin-based auto-discovery.

This module implements the Registry Pattern with automatic discovery
of validator plugins. New validators are found automatically without
modifying existing code (Open/Closed Principle).

Design Patterns:
- Registry Pattern: Central lookup for validators
- Plugin Pattern: Auto-discovery of validator implementations
- Null Object Pattern: GenericCodeValidator as fallback

Enterprise Integration Patterns:
- Service Locator: Central registry for service lookup
- Plugin Architecture: Extensible without modification
"""

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .base import BaseCodeValidator
    from .types import Language

logger = logging.getLogger(__name__)


class CodeValidatorRegistry:
    """Registry for language-specific validators with auto-discovery.

    This registry implements the Service Locator pattern with plugin
    auto-discovery. Validators are discovered from the validators/
    directory automatically.

    Features:
    - Automatic discovery of validators in validators/ package
    - Fallback to generic validator for unsupported languages
    - Thread-safe singleton pattern
    - Lazy loading of validators

    Usage:
        registry = CodeValidatorRegistry()
        registry.discover_validators()  # Auto-load from validators/

        validator = registry.get_validator(Language.PYTHON)
        result = validator.validate(code)

    To add a new language:
    1. Create validators/my_language_validator.py
    2. Implement class inheriting BaseCodeValidator
    3. Registry auto-discovers on next discover_validators() call
    """

    _instance: Optional["CodeValidatorRegistry"] = None
    _initialized: bool = False

    def __new__(cls) -> "CodeValidatorRegistry":
        """Singleton pattern for registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize registry (only runs once due to singleton)."""
        if CodeValidatorRegistry._initialized:
            return

        from .types import Language

        self._validators: dict[Language, "BaseCodeValidator"] = {}
        self._fallback: Optional["BaseCodeValidator"] = None
        self._discovered: bool = False

        CodeValidatorRegistry._initialized = True
        logger.debug("CodeValidatorRegistry initialized")

    def register(self, validator: "BaseCodeValidator") -> None:
        """Manually register a validator.

        Args:
            validator: Validator instance to register
        """
        for lang in validator.supported_languages:
            self._validators[lang] = validator
            logger.debug(f"Registered validator for {lang.name}: {type(validator).__name__}")

    def unregister(self, language: "Language") -> None:
        """Remove a validator for a language.

        Args:
            language: Language to unregister
        """
        if language in self._validators:
            del self._validators[language]
            logger.debug(f"Unregistered validator for {language.name}")

    def get_validator(self, language: "Language") -> "BaseCodeValidator":
        """Get validator for language, with fallback to generic.

        Args:
            language: Target language

        Returns:
            Appropriate validator (never None)
        """
        # Ensure discovery has run
        if not self._discovered:
            self.discover_validators()

        validator = self._validators.get(language)
        if validator:
            return validator

        # Fallback to generic validator
        if self._fallback is None:
            self._fallback = self._create_fallback()

        return self._fallback

    def has_validator(self, language: "Language") -> bool:
        """Check if a specialized validator exists for language.

        Args:
            language: Language to check

        Returns:
            True if specialized validator registered
        """
        return language in self._validators

    def discover_validators(self) -> int:
        """Auto-discover and register validators from validators/ package.

        Scans the validators/ package for modules containing classes
        that inherit from BaseCodeValidator.

        Returns:
            Number of validators discovered
        """
        from .base import BaseCodeValidator

        validators_package = "victor.evaluation.correction.validators"
        validators_path = Path(__file__).parent / "validators"

        if not validators_path.exists():
            logger.warning(f"Validators directory not found: {validators_path}")
            self._discovered = True
            return 0

        count = 0
        for _, module_name, _ in pkgutil.iter_modules([str(validators_path)]):
            if module_name.startswith("_"):
                continue

            try:
                module = importlib.import_module(f"{validators_package}.{module_name}")

                # Find validator classes in module
                for attr_name in dir(module):
                    if attr_name.startswith("_"):
                        continue

                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, BaseCodeValidator)
                        and attr is not BaseCodeValidator
                    ):
                        try:
                            validator = attr()
                            self.register(validator)
                            count += 1
                            logger.info(
                                f"Discovered validator: {attr.__name__} "
                                f"for {[lang.name for lang in validator.supported_languages]}"
                            )
                        except Exception as e:
                            logger.warning(f"Failed to instantiate {attr.__name__}: {e}")

            except Exception as e:
                logger.warning(f"Failed to import validator module {module_name}: {e}")

        self._discovered = True
        logger.info(f"Discovered {count} validators")
        return count

    def _create_fallback(self) -> "BaseCodeValidator":
        """Create the generic fallback validator."""
        from .validators.generic_validator import GenericCodeValidator

        return GenericCodeValidator()

    @property
    def registered_languages(self) -> "set[Language]":
        """Languages with registered validators."""
        return set(self._validators.keys())

    @property
    def all_validators(self) -> "list[BaseCodeValidator]":
        """All registered validators (unique instances)."""
        return list(set(self._validators.values()))

    def reset(self) -> None:
        """Reset registry state (mainly for testing)."""
        self._validators.clear()
        self._fallback = None
        self._discovered = False

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset singleton instance (for testing only)."""
        cls._instance = None
        cls._initialized = False


def get_registry() -> CodeValidatorRegistry:
    """Get the global validator registry instance.

    Convenience function for accessing the singleton registry.

    Returns:
        The global CodeValidatorRegistry instance
    """
    return CodeValidatorRegistry()
