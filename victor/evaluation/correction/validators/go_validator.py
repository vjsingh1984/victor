# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Go code validator with pattern-based analysis.

This validator provides pattern-based validation for Go code.
It checks for common Go-specific issues like missing imports,
unbalanced braces, and structural problems.

Design Pattern: Strategy Pattern (specialized algorithm for Go)
- Pattern-based syntax validation
- Import detection
- Common error detection
- Auto-fix for package and import issues
"""

import re
from typing import Optional

from ..base import BaseCodeValidator, ValidatorCapabilities
from ..types import Language, CodeValidationResult


class GoCodeValidator(BaseCodeValidator):
    """Go code validator using pattern analysis.

    Provides heuristic validation for Go code:
    - Package declaration check
    - Import statement validation
    - Bracket and brace matching
    - Common syntax pattern detection
    - gofmt-style basic checks

    Go has strict syntax rules which makes pattern-based
    validation more effective than for dynamic languages.
    """

    # Standard library packages commonly used
    STDLIB_PACKAGES = frozenset(
        {
            # Core
            "fmt",
            "os",
            "io",
            "log",
            "errors",
            "context",
            # Data structures
            "strings",
            "strconv",
            "bytes",
            "unicode",
            "regexp",
            "sort",
            "container/heap",
            "container/list",
            "container/ring",
            # Math
            "math",
            "math/rand",
            "math/big",
            # I/O and filesystem
            "bufio",
            "io/ioutil",
            "path",
            "path/filepath",
            # Encoding
            "encoding/json",
            "encoding/xml",
            "encoding/base64",
            "encoding/binary",
            # Network
            "net",
            "net/http",
            "net/url",
            # Time
            "time",
            # Sync
            "sync",
            "sync/atomic",
            # Reflection
            "reflect",
            # Testing
            "testing",
        }
    )

    # Go keywords
    GO_KEYWORDS = frozenset(
        {
            "break",
            "case",
            "chan",
            "const",
            "continue",
            "default",
            "defer",
            "else",
            "fallthrough",
            "for",
            "func",
            "go",
            "goto",
            "if",
            "import",
            "interface",
            "map",
            "package",
            "range",
            "return",
            "select",
            "struct",
            "switch",
            "type",
            "var",
        }
    )

    # Go built-in functions
    GO_BUILTINS = frozenset(
        {
            "append",
            "cap",
            "close",
            "complex",
            "copy",
            "delete",
            "imag",
            "len",
            "make",
            "new",
            "panic",
            "print",
            "println",
            "real",
            "recover",
        }
    )

    # Go built-in types
    GO_TYPES = frozenset(
        {
            "bool",
            "string",
            "int",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "uintptr",
            "byte",
            "rune",
            "float32",
            "float64",
            "complex64",
            "complex128",
            "error",
        }
    )

    capabilities = ValidatorCapabilities(
        has_ast_parsing=False,
        has_import_detection=True,
        has_type_checking=False,
        has_auto_fix=True,
    )

    @property
    def supported_languages(self) -> set[Language]:
        """Handles Go language."""
        return {Language.GO}

    def validate(self, code: str) -> CodeValidationResult:
        """Validate Go code using pattern analysis.

        Args:
            code: Go source code

        Returns:
            CodeValidationResult with validation info
        """
        errors: list[str] = []
        warnings: list[str] = []
        missing_imports: list[str] = []

        # 1. Check for package declaration
        package_error = self._check_package_declaration(code)
        if package_error:
            errors.append(package_error)

        # 2. Check bracket/brace matching
        bracket_error = self._check_brackets(code)
        if bracket_error:
            return CodeValidationResult.failure(
                errors=[bracket_error],
                language=Language.GO,
                syntax_error=bracket_error,
            )

        # 3. Check imports
        import_errors, missing = self._check_imports(code)
        errors.extend(import_errors)
        missing_imports.extend(missing)

        # 4. Check for common syntax issues
        syntax_issues = self._check_syntax_patterns(code)
        errors.extend(syntax_issues)

        # 5. Check for incomplete code
        incomplete_error = self._check_incomplete_code(code)
        if incomplete_error:
            errors.append(incomplete_error)

        # 6. Check function structure
        func_errors = self._check_function_structure(code)
        warnings.extend(func_errors)

        return CodeValidationResult(
            valid=len(errors) == 0,
            language=Language.GO,
            syntax_valid=len([e for e in errors if "syntax" in e.lower()]) == 0,
            imports_valid=len(missing_imports) == 0,
            errors=tuple(errors),
            warnings=tuple(warnings),
            missing_imports=tuple(missing_imports),
            used_ast_validation=False,
        )

    def fix(self, code: str, validation: CodeValidationResult) -> str:
        """Auto-fix Go-specific issues.

        Args:
            code: Original code
            validation: Validation result with identified issues

        Returns:
            Fixed code with common issues resolved
        """
        # First apply generic cleanup
        fixed_code = self.clean_markdown(code)

        # Add package declaration if missing
        if not self._has_package_declaration(fixed_code):
            fixed_code = "package main\n\n" + fixed_code

        # Add missing imports
        if validation.missing_imports:
            fixed_code = self._add_missing_imports(fixed_code, list(validation.missing_imports))

        return fixed_code

    def _check_package_declaration(self, code: str) -> Optional[str]:
        """Check for package declaration."""
        if not self._has_package_declaration(code):
            return "Missing package declaration"
        return None

    def _has_package_declaration(self, code: str) -> bool:
        """Check if code has a package declaration."""
        # Skip comments and find first non-comment line
        for line in code.strip().split("\n"):
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue
            if stripped.startswith("/*"):
                continue
            return stripped.startswith("package ")
        return False

    def _check_brackets(self, code: str) -> Optional[str]:
        """Check for balanced brackets and braces."""
        # Remove string contents and comments
        cleaned = self._remove_strings_and_comments(code)

        bracket_pairs = {"(": ")", "[": "]", "{": "}"}
        stack: list[tuple[str, int]] = []

        for i, char in enumerate(cleaned):
            if char in bracket_pairs:
                stack.append((char, i))
            elif char in bracket_pairs.values():
                if not stack:
                    line = code[:i].count("\n") + 1
                    return f"Unmatched closing '{char}' at line {line}"

                open_bracket, _ = stack.pop()
                if bracket_pairs[open_bracket] != char:
                    line = code[:i].count("\n") + 1
                    return f"Mismatched brackets: expected '{bracket_pairs[open_bracket]}' but found '{char}' at line {line}"

        if stack:
            open_bracket, pos = stack[-1]
            line = code[:pos].count("\n") + 1
            return f"Unclosed '{open_bracket}' starting at line {line}"

        return None

    def _check_imports(self, code: str) -> tuple[list[str], list[str]]:
        """Check import statements and detect missing imports."""
        errors = []
        missing = []

        # Extract declared imports
        declared_imports = self._extract_imports(code)

        # Find used packages
        used_packages = self._find_used_packages(code)

        # Check for missing imports
        for pkg in used_packages:
            if pkg not in declared_imports:
                # Only report for known stdlib packages
                if pkg in self.STDLIB_PACKAGES or pkg.split("/")[0] in [
                    "fmt",
                    "os",
                    "io",
                    "net",
                    "encoding",
                    "math",
                    "sync",
                    "container",
                ]:
                    missing.append(pkg)
                    errors.append(f"Package '{pkg}' used but not imported")

        return errors, missing

    def _extract_imports(self, code: str) -> set[str]:
        """Extract imported package names."""
        imports = set()

        # Single import: import "fmt"
        for match in re.finditer(r'import\s+"([^"]+)"', code):
            imports.add(match.group(1))

        # Import block: import ( "fmt" "os" )
        import_block = re.search(r"import\s*\((.*?)\)", code, re.DOTALL)
        if import_block:
            block_content = import_block.group(1)
            for match in re.finditer(r'"([^"]+)"', block_content):
                imports.add(match.group(1))

        # Extract just the package name (last component)
        package_names = set()
        for imp in imports:
            parts = imp.split("/")
            package_names.add(parts[-1])
            package_names.add(imp)  # Also keep full path

        return package_names

    def _find_used_packages(self, code: str) -> set[str]:
        """Find packages referenced in the code."""
        packages = set()

        # Pattern: pkg.Something (but not methods on variables)
        for match in re.finditer(r"\b([a-z][a-z0-9]*)\.[A-Z]\w*", code):
            pkg = match.group(1)
            # Skip if it's a builtin or keyword
            if pkg not in self.GO_BUILTINS and pkg not in self.GO_KEYWORDS:
                packages.add(pkg)

        return packages

    def _check_syntax_patterns(self, code: str) -> list[str]:
        """Check for common Go syntax issues."""
        errors = []

        # Check for := without a value
        if re.search(r":=\s*$", code, re.MULTILINE):
            errors.append("Incomplete short variable declaration (:=)")

        # Check for dangling commas in function calls (invalid in Go)
        if re.search(r",\s*\)", code):
            # Actually, trailing commas ARE allowed in Go
            pass

        # Check for = instead of := for new variables (common mistake)
        # This is tricky without full parsing, skip for now

        return errors

    def _check_incomplete_code(self, code: str) -> Optional[str]:
        """Check for obviously incomplete code."""
        stripped = code.strip()

        if not stripped:
            return "Empty code"

        # Check for truncated code
        truncation_patterns = [
            r":=\s*$",  # Ends with assignment
            r"{\s*$",  # Ends with open brace (might be valid)
            r"\(\s*$",  # Ends with open paren
        ]

        for pattern in truncation_patterns:
            if re.search(pattern, stripped):
                # Only flag if also has unclosed brackets
                if self._check_brackets(stripped):
                    return "Code appears to be truncated"

        return None

    def _check_function_structure(self, code: str) -> list[str]:
        """Check function declarations for issues."""
        warnings = []

        # Check for func without return type on non-void functions
        # This is hard to detect reliably without parsing

        # Check for unexported functions in main package
        if "package main" in code:
            funcs = re.findall(r"func\s+([A-Z]\w*)\s*\(", code)
            if funcs:
                for func in funcs:
                    if func != "Main":  # Main is expected
                        warnings.append(f"Exported function '{func}' in main package")

        return warnings

    def _remove_strings_and_comments(self, code: str) -> str:
        """Remove string contents and comments for structural analysis."""
        # Remove single-line comments
        result = re.sub(r"//.*$", "", code, flags=re.MULTILINE)

        # Remove multi-line comments
        result = re.sub(r"/\*.*?\*/", "", result, flags=re.DOTALL)

        # Remove string contents
        result = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', '""', result)
        result = re.sub(r"`[^`]*`", "``", result)  # Raw strings

        return result

    def _add_missing_imports(self, code: str, missing: list[str]) -> str:
        """Add missing import statements."""
        if not missing:
            return code

        # Find where to insert imports
        import_match = re.search(r"import\s*\((.*?)\)", code, re.DOTALL)

        if import_match:
            # Add to existing import block
            block_start = import_match.start()
            block_end = import_match.end()
            existing_block = import_match.group(1)

            new_imports = "\n".join(f'\t"{pkg}"' for pkg in missing)
            new_block = f"import ({existing_block.rstrip()}\n{new_imports}\n)"

            return code[:block_start] + new_block + code[block_end:]

        # Check for single import
        single_import = re.search(r'import\s+"[^"]+"\n', code)
        if single_import:
            # Convert to import block
            existing_match = re.search(r'import\s+"([^"]+)"', code)
            if existing_match is None:
                return code
            existing = existing_match.group(1)
            all_imports = [existing] + missing
            import_block = "import (\n" + "\n".join(f'\t"{pkg}"' for pkg in all_imports) + "\n)\n"
            return code[: single_import.start()] + import_block + code[single_import.end() :]

        # No existing imports - add after package declaration
        package_match = re.search(r"package\s+\w+\n*", code)
        if package_match:
            insert_pos = package_match.end()
            import_block = "\nimport (\n" + "\n".join(f'\t"{pkg}"' for pkg in missing) + "\n)\n"
            return code[:insert_pos] + import_block + code[insert_pos:]

        return code
