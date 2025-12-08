# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Language-specific code validators.

This package contains validators for different programming languages.
Validators are auto-discovered by the registry at runtime.

To add a new language validator:
1. Create a new file: my_language_validator.py
2. Implement a class inheriting from BaseCodeValidator
3. The registry will auto-discover it

Example:
    # rust_validator.py
    from ..base import BaseCodeValidator
    from ..types import Language, ValidationResult

    class RustCodeValidator(BaseCodeValidator):
        @property
        def supported_languages(self) -> set[Language]:
            return {Language.RUST}

        def validate(self, code: str) -> ValidationResult:
            ...

        def fix(self, code: str, validation: ValidationResult) -> str:
            ...
"""

from .generic_validator import GenericCodeValidator
from .go_validator import GoCodeValidator
from .java_validator import JavaCodeValidator
from .javascript_validator import JavaScriptCodeValidator
from .python_validator import PythonCodeValidator
from .rust_validator import RustCodeValidator

__all__ = [
    "GenericCodeValidator",
    "GoCodeValidator",
    "JavaCodeValidator",
    "JavaScriptCodeValidator",
    "PythonCodeValidator",
    "RustCodeValidator",
]
