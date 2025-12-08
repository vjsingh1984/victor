# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Python-specific code validator with AST parsing.

This validator provides deep validation for Python code using the
built-in AST module. It can detect syntax errors, missing imports,
and type hint issues.

Design Pattern: Strategy Pattern (specialized algorithm for Python)
- Deep AST-based syntax validation
- Import dependency analysis
- Type hint verification
- Auto-fix for common issues
"""

import ast
import re
from typing import Optional

from ..base import BaseCodeValidator, ValidatorCapabilities
from ..types import Language, ValidationResult


class PythonCodeValidator(BaseCodeValidator):
    """Python-specific validator using AST parsing.

    Provides comprehensive validation for Python code:
    - Syntax validation via ast.parse()
    - Import detection and verification
    - Type hint validation
    - Missing import auto-fix

    This is the reference implementation for language-specific validators.
    """

    # Standard library modules commonly used in code generation
    STDLIB_MODULES = frozenset(
        {
            # Core
            "math",
            "re",
            "os",
            "sys",
            "json",
            "copy",
            "random",
            "string",
            # Collections and iteration
            "collections",
            "itertools",
            "functools",
            "operator",
            # Typing
            "typing",
            "types",
            # Data
            "datetime",
            "time",
            "calendar",
            "decimal",
            "fractions",
            "statistics",
            # Algorithms
            "heapq",
            "bisect",
            "hashlib",
            "base64",
            # I/O
            "io",
            "pathlib",
            # Testing (for benchmark tasks)
            "unittest",
            "doctest",
        }
    )

    # External packages that models sometimes incorrectly assume available
    EXTERNAL_PACKAGES = frozenset(
        {
            "numpy",
            "scipy",
            "pandas",
            "sklearn",
            "torch",
            "tensorflow",
            "requests",
            "aiohttp",
            "flask",
            "django",
            "fastapi",
            "matplotlib",
            "seaborn",
            "plotly",
        }
    )

    # Common typing imports
    TYPING_TYPES = frozenset(
        {
            "List",
            "Dict",
            "Set",
            "Tuple",
            "Optional",
            "Any",
            "Union",
            "Callable",
            "Iterable",
            "Iterator",
            "Sequence",
            "Mapping",
            "TypeVar",
            "Generic",
            "Protocol",
            "Literal",
            "Final",
        }
    )

    capabilities = ValidatorCapabilities(
        has_ast_parsing=True,
        has_import_detection=True,
        has_type_checking=True,
        has_auto_fix=True,
    )

    @property
    def supported_languages(self) -> set[Language]:
        """Only handles Python."""
        return {Language.PYTHON}

    def validate(self, code: str) -> ValidationResult:
        """Validate Python code using AST parsing.

        Args:
            code: Python source code

        Returns:
            ValidationResult with detailed validation info
        """
        errors: list[str] = []
        warnings: list[str] = []
        missing_imports: list[str] = []

        # 1. Syntax validation via AST
        syntax_error = self._check_syntax(code)
        if syntax_error:
            return ValidationResult(
                valid=False,
                language=Language.PYTHON,
                syntax_valid=False,
                errors=(syntax_error,),
                syntax_error=syntax_error,
                used_ast_validation=True,
            )

        # 2. Import analysis
        imports_used = self._extract_imports(code)
        modules_referenced = self._find_module_references(code)

        # Check for external packages
        for pkg in imports_used:
            if pkg in self.EXTERNAL_PACKAGES:
                warnings.append(f"Uses external package '{pkg}' - consider stdlib alternative")

        # Check for missing imports
        for module in modules_referenced:
            if module not in imports_used and module in self.STDLIB_MODULES:
                missing_imports.append(module)
                errors.append(f"Uses '{module}' module without importing it")

        # 3. Type hint validation
        type_hints_used = self._find_type_hint_usage(code)
        if type_hints_used:
            typing_imports = self._extract_typing_imports(code)
            missing_types = type_hints_used - typing_imports

            # Check if typing module is imported at all
            has_typing_import = "typing" in imports_used or any(
                "from typing" in line for line in code.split("\n")
            )

            if missing_types and not has_typing_import:
                missing_imports.append("typing")
                errors.append(f"Uses type hints {missing_types} without importing from typing")

        # Build result
        imports_valid = len(missing_imports) == 0

        return ValidationResult(
            valid=imports_valid,
            language=Language.PYTHON,
            syntax_valid=True,
            imports_valid=imports_valid,
            errors=tuple(errors),
            warnings=tuple(warnings),
            missing_imports=tuple(missing_imports),
            used_ast_validation=True,
        )

    def fix(self, code: str, validation: ValidationResult) -> str:
        """Auto-fix Python-specific issues.

        Args:
            code: Original code
            validation: Validation result with identified issues

        Returns:
            Fixed code with missing imports added
        """
        # First apply generic cleanup
        fixed_code = self.clean_markdown(code)

        # Add missing imports
        if validation.missing_imports:
            fixed_code = self._add_missing_imports(fixed_code, list(validation.missing_imports))

        return fixed_code

    def _check_syntax(self, code: str) -> Optional[str]:
        """Check syntax using AST parsing."""
        try:
            ast.parse(code)
            return None
        except SyntaxError as e:
            return f"Line {e.lineno}: {e.msg}"

    def _extract_imports(self, code: str) -> set[str]:
        """Extract imported module names using AST."""
        imports = set()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split(".")[0])
        except SyntaxError:
            # Fallback to regex for broken code
            for match in re.finditer(r"(?:from|import)\s+(\w+)", code):
                imports.add(match.group(1))
        return imports

    def _extract_typing_imports(self, code: str) -> set[str]:
        """Extract type names imported from typing module."""
        types_imported = set()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "typing":
                    for alias in node.names:
                        types_imported.add(alias.name)
        except SyntaxError:
            # Fallback to regex
            match = re.search(r"from typing import (.+)", code)
            if match:
                types = match.group(1).split(",")
                types_imported.update(t.strip() for t in types)
        return types_imported

    def _find_module_references(self, code: str) -> set[str]:
        """Find references to module.function() patterns."""
        modules = set()
        for match in re.finditer(r"\b(\w+)\.\w+\s*\(", code):
            module = match.group(1)
            if module in self.STDLIB_MODULES:
                modules.add(module)
        return modules

    def _find_type_hint_usage(self, code: str) -> set[str]:
        """Find usage of typing module types."""
        type_hints = set()
        for type_name in self.TYPING_TYPES:
            # Match Type[ or Type] patterns
            if re.search(rf"\b{type_name}\[", code):
                type_hints.add(type_name)
        return type_hints

    def _add_missing_imports(self, code: str, missing: list[str]) -> str:
        """Add missing import statements at the top."""
        imports_to_add = []

        for module in missing:
            if module == "typing":
                # Add specific typing imports
                types_used = self._find_type_hint_usage(code)
                if types_used:
                    imports_to_add.append(f"from typing import {', '.join(sorted(types_used))}")
            else:
                imports_to_add.append(f"import {module}")

        if imports_to_add:
            import_block = "\n".join(imports_to_add) + "\n\n"
            return import_block + code

        return code
