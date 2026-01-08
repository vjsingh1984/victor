# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Java code validator with pattern-based analysis.

This validator provides pattern-based validation for Java code.
It checks for common Java-specific issues like missing imports,
class structure, and syntax validation.

Design Pattern: Strategy Pattern (specialized algorithm for Java)
- Pattern-based syntax validation
- Import detection
- Class structure validation
- Auto-fix for import issues
"""

import re
from typing import Optional

from ..base import BaseCodeValidator, ValidatorCapabilities
from ..types import Language, CodeValidationResult


class JavaCodeValidator(BaseCodeValidator):
    """Java code validator using pattern analysis.

    Provides heuristic validation for Java code:
    - Bracket and brace matching
    - Import statement validation
    - Class structure validation
    - Common syntax pattern detection

    Since we don't have a Java parser in Python, this validator
    uses pattern matching and heuristics.
    """

    # Common Java packages
    JAVA_STDLIB = frozenset(
        {
            # java.lang is auto-imported
            "java.lang",
            # Common imports
            "java.util",
            "java.io",
            "java.nio",
            "java.net",
            "java.time",
            "java.math",
            "java.text",
            "java.sql",
            "java.util.stream",
            "java.util.function",
            "java.util.concurrent",
            "java.util.regex",
            "java.util.logging",
        }
    )

    # Common types that require imports
    IMPORT_REQUIRED_TYPES = {
        # java.util types
        "List": "java.util.List",
        "ArrayList": "java.util.ArrayList",
        "Map": "java.util.Map",
        "HashMap": "java.util.HashMap",
        "Set": "java.util.Set",
        "HashSet": "java.util.HashSet",
        "Queue": "java.util.Queue",
        "LinkedList": "java.util.LinkedList",
        "Stack": "java.util.Stack",
        "Vector": "java.util.Vector",
        "Arrays": "java.util.Arrays",
        "Collections": "java.util.Collections",
        "Optional": "java.util.Optional",
        "Scanner": "java.util.Scanner",
        "Random": "java.util.Random",
        "Date": "java.util.Date",
        "Calendar": "java.util.Calendar",
        "Iterator": "java.util.Iterator",
        "Comparator": "java.util.Comparator",
        # java.io types
        "File": "java.io.File",
        "FileReader": "java.io.FileReader",
        "FileWriter": "java.io.FileWriter",
        "BufferedReader": "java.io.BufferedReader",
        "BufferedWriter": "java.io.BufferedWriter",
        "InputStream": "java.io.InputStream",
        "OutputStream": "java.io.OutputStream",
        "IOException": "java.io.IOException",
        # java.math types
        "BigInteger": "java.math.BigInteger",
        "BigDecimal": "java.math.BigDecimal",
    }

    # Java keywords
    JAVA_KEYWORDS = frozenset(
        {
            "abstract",
            "assert",
            "boolean",
            "break",
            "byte",
            "case",
            "catch",
            "char",
            "class",
            "const",
            "continue",
            "default",
            "do",
            "double",
            "else",
            "enum",
            "extends",
            "final",
            "finally",
            "float",
            "for",
            "goto",
            "if",
            "implements",
            "import",
            "instanceof",
            "int",
            "interface",
            "long",
            "native",
            "new",
            "package",
            "private",
            "protected",
            "public",
            "return",
            "short",
            "static",
            "strictfp",
            "super",
            "switch",
            "synchronized",
            "this",
            "throw",
            "throws",
            "transient",
            "try",
            "void",
            "volatile",
            "while",
            "true",
            "false",
            "null",
        }
    )

    # Java primitive types
    JAVA_PRIMITIVES = frozenset(
        {
            "byte",
            "short",
            "int",
            "long",
            "float",
            "double",
            "boolean",
            "char",
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
        """Handles Java language."""
        return {Language.JAVA}

    def validate(self, code: str) -> CodeValidationResult:
        """Validate Java code using pattern analysis.

        Args:
            code: Java source code

        Returns:
            CodeValidationResult with validation info
        """
        errors: list[str] = []
        warnings: list[str] = []
        missing_imports: list[str] = []

        # 1. Check bracket/brace matching
        bracket_error = self._check_brackets(code)
        if bracket_error:
            return CodeValidationResult.failure(
                errors=[bracket_error],
                language=Language.JAVA,
                syntax_error=bracket_error,
            )

        # 2. Check class structure
        class_error = self._check_class_structure(code)
        if class_error:
            errors.append(class_error)

        # 3. Check import statements
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

        # 6. Check method structure
        method_warnings = self._check_method_structure(code)
        warnings.extend(method_warnings)

        return CodeValidationResult(
            valid=len(errors) == 0,
            language=Language.JAVA,
            syntax_valid=len([e for e in errors if "syntax" in e.lower()]) == 0,
            imports_valid=len(missing_imports) == 0,
            errors=tuple(errors),
            warnings=tuple(warnings),
            missing_imports=tuple(missing_imports),
            used_ast_validation=False,
        )

    def fix(self, code: str, validation: CodeValidationResult) -> str:
        """Auto-fix Java-specific issues.

        Args:
            code: Original code
            validation: Validation result with identified issues

        Returns:
            Fixed code with common issues resolved
        """
        # First apply generic cleanup
        fixed_code = self.clean_markdown(code)

        # Add missing imports
        if validation.missing_imports:
            fixed_code = self._add_missing_imports(fixed_code, list(validation.missing_imports))

        return fixed_code

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

    def _check_class_structure(self, code: str) -> Optional[str]:
        """Check for valid class structure."""
        # Check if there's at least one class or interface
        if not re.search(r"\b(class|interface|enum)\s+\w+", code):
            # Might be just a method snippet - that's ok for benchmarks
            if re.search(r"\bpublic\s+static\s+void\s+main", code):
                return "main method found but no class declaration"

        return None

    def _check_imports(self, code: str) -> tuple[list[str], list[str]]:
        """Check import statements and detect missing imports."""
        errors = []
        missing = []

        # Extract declared imports
        declared_imports = self._extract_imports(code)

        # Find used types
        used_types = self._find_used_types(code)

        # Check for missing imports
        for type_name in used_types:
            if type_name in self.IMPORT_REQUIRED_TYPES:
                full_import = self.IMPORT_REQUIRED_TYPES[type_name]
                package = full_import.rsplit(".", 1)[0]

                # Check if already imported (full class or wildcard)
                is_imported = (
                    full_import in declared_imports
                    or f"{package}.*" in declared_imports
                    or type_name in declared_imports
                )

                if not is_imported:
                    if full_import not in missing:
                        missing.append(full_import)
                        errors.append(f"Type '{type_name}' used but not imported")

        return errors, missing

    def _extract_imports(self, code: str) -> set[str]:
        """Extract import statements."""
        imports = set()

        for match in re.finditer(r"import\s+(?:static\s+)?([\w.]+(?:\.\*)?);", code):
            imports.add(match.group(1))

        return imports

    def _find_used_types(self, code: str) -> set[str]:
        """Find type names used in the code."""
        types = set()

        # Find capitalized identifiers that look like types
        # Exclude keywords and common false positives
        for match in re.finditer(r"\b([A-Z][a-zA-Z0-9_]*)\b", code):
            type_name = match.group(1)
            if type_name in self.IMPORT_REQUIRED_TYPES:
                types.add(type_name)

        return types

    def _check_syntax_patterns(self, code: str) -> list[str]:
        """Check for common Java syntax issues."""
        errors = []

        # Check for missing semicolons (simple heuristic)
        # Look for lines that end with ) followed by next line starting with something
        # This is tricky and might have false positives

        # Check for double semicolons
        if re.search(r";;", code):
            errors.append("Double semicolon detected")

        # Check for = in if condition (common mistake)
        # But be careful about ==
        if re.search(r"\bif\s*\([^)]*[^=!<>]=[^=][^)]*\)", code):
            pass  # Could be intentional assignment in condition

        return errors

    def _check_incomplete_code(self, code: str) -> Optional[str]:
        """Check for obviously incomplete code."""
        stripped = code.strip()

        if not stripped:
            return "Empty code"

        # Check for truncated code
        truncation_patterns = [
            r"=\s*$",  # Ends with assignment
            r"{\s*$",  # Ends with open brace
            r"\(\s*$",  # Ends with open paren
        ]

        for pattern in truncation_patterns:
            if re.search(pattern, stripped):
                # Only flag if also has unclosed brackets
                if self._check_brackets(stripped):
                    return "Code appears to be truncated"

        return None

    def _check_method_structure(self, code: str) -> list[str]:
        """Check method declarations for issues."""
        warnings = []

        # Check for public static void main without proper signature
        main_match = re.search(r"public\s+static\s+void\s+main\s*\(([^)]*)\)", code)
        if main_match:
            params = main_match.group(1).strip()
            if not re.match(r"String\s*\[\s*\]\s*\w+|String\s+\w+\s*\[\s*\]", params):
                warnings.append("main method should have 'String[] args' parameter")

        return warnings

    def _remove_strings_and_comments(self, code: str) -> str:
        """Remove string contents and comments for structural analysis."""
        # Remove single-line comments
        result = re.sub(r"//.*$", "", code, flags=re.MULTILINE)

        # Remove multi-line comments
        result = re.sub(r"/\*.*?\*/", "", result, flags=re.DOTALL)

        # Remove string contents
        result = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', '""', result)

        # Remove char literals
        result = re.sub(r"'[^'\\]'|'\\.'", "''", result)

        return result

    def _add_missing_imports(self, code: str, missing: list[str]) -> str:
        """Add missing import statements."""
        if not missing:
            return code

        import_statements = "\n".join(f"import {imp};" for imp in sorted(missing)) + "\n"

        # Find where to insert imports
        # After package declaration, before class declaration

        # Check for package declaration
        package_match = re.search(r"^package\s+[\w.]+;\s*\n", code, re.MULTILINE)
        if package_match:
            insert_pos = package_match.end()
            return code[:insert_pos] + "\n" + import_statements + code[insert_pos:]

        # Check for existing imports
        last_import = None
        for match in re.finditer(r"^import\s+[\w.*]+;\s*$", code, re.MULTILINE):
            last_import = match.end()

        if last_import:
            return code[:last_import] + "\n" + import_statements + code[last_import:]

        # No package or imports, add at the beginning (before class)
        class_match = re.search(r"\b(public\s+)?(class|interface|enum)\s+", code)
        if class_match:
            insert_pos = class_match.start()
            return code[:insert_pos] + import_statements + "\n" + code[insert_pos:]

        # Fallback: add at top
        return import_statements + "\n" + code
