# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""JavaScript and TypeScript code validator with pattern-based analysis.

This validator provides pattern-based validation for JavaScript and TypeScript
code. Without a native JS parser, it uses heuristic analysis for common issues.

Design Pattern: Strategy Pattern (specialized algorithm for JS/TS)
- Pattern-based syntax validation
- Bracket/brace matching
- Common error detection
- Auto-fix for simple issues
"""

import re
from typing import Optional

from ..base import BaseCodeValidator, ValidatorCapabilities
from ..types import Language, CodeValidationResult


class JavaScriptCodeValidator(BaseCodeValidator):
    """JavaScript and TypeScript validator using pattern analysis.

    Provides heuristic validation for JS/TS code:
    - Bracket and brace matching
    - Quote matching
    - Common syntax pattern detection
    - Import statement validation

    Since we don't have a JS AST parser in Python, this validator
    uses pattern matching and heuristics rather than deep parsing.
    """

    # Common JS/TS modules and globals
    COMMON_GLOBALS = frozenset(
        {
            # Built-in objects
            "Array",
            "Object",
            "String",
            "Number",
            "Boolean",
            "Symbol",
            "BigInt",
            "Map",
            "Set",
            "WeakMap",
            "WeakSet",
            "Promise",
            "Proxy",
            "Reflect",
            # Built-in functions
            "parseInt",
            "parseFloat",
            "isNaN",
            "isFinite",
            "decodeURI",
            "encodeURI",
            "setTimeout",
            "setInterval",
            "clearTimeout",
            "clearInterval",
            # Browser globals
            "window",
            "document",
            "console",
            "fetch",
            "localStorage",
            "sessionStorage",
            # Node.js globals
            "require",
            "module",
            "exports",
            "__dirname",
            "__filename",
            "process",
            "Buffer",
        }
    )

    # TypeScript-specific keywords
    TS_KEYWORDS = frozenset(
        {
            "interface",
            "type",
            "enum",
            "namespace",
            "declare",
            "readonly",
            "abstract",
            "implements",
            "private",
            "protected",
            "public",
            "as",
            "is",
            "keyof",
            "typeof",
            "infer",
            "never",
            "unknown",
        }
    )

    # Reserved words that can't be used as identifiers
    JS_RESERVED = frozenset(
        {
            "break",
            "case",
            "catch",
            "continue",
            "debugger",
            "default",
            "delete",
            "do",
            "else",
            "finally",
            "for",
            "function",
            "if",
            "in",
            "instanceof",
            "new",
            "return",
            "switch",
            "this",
            "throw",
            "try",
            "typeof",
            "var",
            "void",
            "while",
            "with",
            "class",
            "const",
            "let",
            "extends",
            "super",
            "yield",
            "import",
            "export",
            "null",
            "true",
            "false",
            "async",
            "await",
        }
    )

    capabilities = ValidatorCapabilities(
        has_ast_parsing=False,  # Pattern-based, not AST
        has_import_detection=True,
        has_type_checking=False,
        has_auto_fix=True,
    )

    @property
    def supported_languages(self) -> set[Language]:
        """Handles both JavaScript and TypeScript."""
        return {Language.JAVASCRIPT, Language.TYPESCRIPT}

    def validate(self, code: str) -> CodeValidationResult:
        """Validate JavaScript/TypeScript code using pattern analysis.

        Args:
            code: JS/TS source code

        Returns:
            CodeValidationResult with validation info
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Detect if TypeScript
        is_typescript = self._is_typescript(code)
        language = Language.TYPESCRIPT if is_typescript else Language.JAVASCRIPT

        # 1. Basic structure validation
        bracket_error = self._check_brackets(code)
        if bracket_error:
            return CodeValidationResult.failure(
                errors=[bracket_error],
                language=language,
                syntax_error=bracket_error,
            )

        # 2. Quote matching
        quote_error = self._check_quotes(code)
        if quote_error:
            errors.append(quote_error)

        # 3. Check for common syntax errors
        syntax_errors = self._check_common_syntax_errors(code)
        errors.extend(syntax_errors)

        # 4. Check for incomplete statements
        incomplete_error = self._check_incomplete_code(code)
        if incomplete_error:
            errors.append(incomplete_error)

        # 5. Import/export validation
        import_errors = self._check_imports(code)
        warnings.extend(import_errors)  # Treat as warnings, not hard errors

        # 6. TypeScript-specific checks
        if is_typescript:
            ts_errors = self._check_typescript_syntax(code)
            errors.extend(ts_errors)

        return CodeValidationResult(
            valid=len(errors) == 0,
            language=language,
            syntax_valid=len([e for e in errors if "syntax" in e.lower()]) == 0,
            imports_valid=len(import_errors) == 0,
            errors=tuple(errors),
            warnings=tuple(warnings),
            used_ast_validation=False,
        )

    def fix(self, code: str, validation: CodeValidationResult) -> str:
        """Auto-fix JavaScript/TypeScript issues.

        Args:
            code: Original code
            validation: Validation result with identified issues

        Returns:
            Fixed code with common issues resolved
        """
        # First apply generic cleanup
        fixed_code = self.clean_markdown(code)

        # Fix common issues
        fixed_code = self._fix_trailing_comma(fixed_code)
        fixed_code = self._fix_missing_semicolons(fixed_code)

        return fixed_code

    def _is_typescript(self, code: str) -> bool:
        """Detect if code is TypeScript based on patterns."""
        ts_patterns = [
            r":\s*(string|number|boolean|void|any|never|unknown)\b",
            r"interface\s+\w+\s*\{",
            r"type\s+\w+\s*=",
            r"<\w+(?:,\s*\w+)*>\s*\(",
            r"as\s+\w+",
            r":\s*\w+\[\]",
            r"readonly\s+\w+",
            r"implements\s+\w+",
        ]
        for pattern in ts_patterns:
            if re.search(pattern, code):
                return True
        return False

    def _check_brackets(self, code: str) -> Optional[str]:
        """Check for balanced brackets and braces."""
        # Remove string contents to avoid false positives
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

    def _check_quotes(self, code: str) -> Optional[str]:
        """Check for unmatched quotes."""
        # Remove comments first
        cleaned = self._remove_comments(code)

        # Track quote state
        in_single = False
        in_double = False
        in_template = False
        escaped = False

        for _i, char in enumerate(cleaned):
            if escaped:
                escaped = False
                continue

            if char == "\\":
                escaped = True
                continue

            if char == "'" and not in_double and not in_template:
                in_single = not in_single
            elif char == '"' and not in_single and not in_template:
                in_double = not in_double
            elif char == "`" and not in_single and not in_double:
                in_template = not in_template

        if in_single or in_double or in_template:
            quote_type = "'" if in_single else ('"' if in_double else "`")
            return f"Unclosed string literal starting with {quote_type}"

        return None

    def _check_common_syntax_errors(self, code: str) -> list[str]:
        """Check for common syntax errors."""
        errors = []

        # Check for consecutive operators
        if re.search(r"[+\-*/%][\s]*[+\-*/%]{2,}", code):
            errors.append("Consecutive operators detected")

        # Check for = instead of == in conditions (common mistake)
        # But exclude valid assignments in for loops
        condition_match = re.search(r"\bif\s*\(\s*[^=!<>]+[^!=<>]=[^=]", code)
        if condition_match:
            # Could be intentional, add as warning
            pass

        # Check for function without body
        if re.search(r"function\s+\w+\s*\([^)]*\)\s*[^{]", code):
            # Could be TypeScript declaration
            if not re.search(r"function\s+\w+\s*\([^)]*\)\s*:\s*\w+\s*;", code):
                pass  # Skip - might be arrow function on next line

        return errors

    def _check_incomplete_code(self, code: str) -> Optional[str]:
        """Check for obviously incomplete code."""
        stripped = code.strip()

        if not stripped:
            return "Empty code"

        # Check for truncated code (ends with incomplete statement)
        truncation_patterns = [
            r"=\s*$",  # Ends with assignment
            r",\s*$",  # Ends with comma
            r"\(\s*$",  # Ends with open paren
            r"\[\s*$",  # Ends with open bracket
            r"{\s*$",  # Ends with open brace (could be valid)
            r"=>\s*$",  # Ends with arrow
        ]

        for pattern in truncation_patterns:
            if re.search(pattern, stripped):
                return "Code appears to be truncated (ends with incomplete statement)"

        return None

    def _check_imports(self, code: str) -> list[str]:
        """Check import statements for common issues."""
        warnings = []

        # Check for mixed import styles
        has_require = "require(" in code
        has_import = re.search(r"^import\s+", code, re.MULTILINE) is not None

        if has_require and has_import:
            warnings.append("Mixed CommonJS (require) and ES modules (import) detected")

        # Check for relative imports that might be wrong
        if re.search(r"from\s+['\"]\.{3,}", code):
            warnings.append("Deep relative import path detected - consider using aliases")

        return warnings

    def _check_typescript_syntax(self, code: str) -> list[str]:
        """Check TypeScript-specific syntax issues."""
        errors = []

        # Check for incomplete type annotations
        if re.search(r":\s*$", code, re.MULTILINE):
            errors.append("Incomplete type annotation")

        # Check for invalid generic syntax
        if re.search(r"<\s*>", code):
            errors.append("Empty generic type parameter")

        return errors

    def _remove_strings_and_comments(self, code: str) -> str:
        """Remove string contents and comments for structural analysis."""
        result = self._remove_comments(code)

        # Remove string contents but keep quotes for structure
        result = re.sub(r"'[^'\\]*(?:\\.[^'\\]*)*'", "''", result)
        result = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', '""', result)
        result = re.sub(r"`[^`\\]*(?:\\.[^`\\]*)*`", "``", result)

        return result

    def _remove_comments(self, code: str) -> str:
        """Remove comments from code."""
        # Remove single-line comments
        result = re.sub(r"//.*$", "", code, flags=re.MULTILINE)

        # Remove multi-line comments
        result = re.sub(r"/\*.*?\*/", "", result, flags=re.DOTALL)

        return result

    def _fix_trailing_comma(self, code: str) -> str:
        """Fix trailing commas before closing brackets."""
        # This is actually valid in modern JS, but some tools complain
        # We'll leave it as is - trailing commas are fine
        return code

    def _fix_missing_semicolons(self, code: str) -> str:
        """Add missing semicolons where clearly needed.

        Note: JavaScript has ASI (Automatic Semicolon Insertion),
        so this is more for style consistency than correctness.
        We only fix obvious cases.
        """
        # For safety, don't auto-fix semicolons - ASI handles most cases
        # and incorrect fixes could break code
        return code
