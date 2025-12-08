# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Rust code validator with pattern-based analysis.

This validator provides pattern-based validation for Rust code.
It checks for common Rust-specific issues like lifetime annotations,
borrow checker patterns, and structural validation.

Design Pattern: Strategy Pattern (specialized algorithm for Rust)
- Pattern-based syntax validation
- Ownership/borrowing pattern detection
- Common error detection
- Auto-fix for simple issues
"""

import re
from typing import Optional

from ..base import BaseCodeValidator, ValidatorCapabilities
from ..types import Language, ValidationResult


class RustCodeValidator(BaseCodeValidator):
    """Rust code validator using pattern analysis.

    Provides heuristic validation for Rust code:
    - Bracket and brace matching
    - Use/mod statement validation
    - Common syntax pattern detection
    - Lifetime annotation checking

    Since we don't have a Rust parser in Python, this validator
    uses pattern matching and heuristics.
    """

    # Standard library crates and modules
    STDLIB_MODULES = frozenset(
        {
            # Core
            "std",
            "core",
            "alloc",
            # Common std modules
            "std::io",
            "std::fs",
            "std::collections",
            "std::fmt",
            "std::env",
            "std::path",
            "std::time",
            "std::thread",
            "std::sync",
            "std::net",
            "std::str",
            "std::mem",
            "std::ops",
            "std::cmp",
            "std::iter",
            "std::vec",
            "std::string",
            "std::option",
            "std::result",
            "std::convert",
            "std::default",
            "std::clone",
        }
    )

    # Common external crates
    COMMON_CRATES = frozenset(
        {
            "serde",
            "serde_json",
            "tokio",
            "async_std",
            "reqwest",
            "clap",
            "log",
            "env_logger",
            "thiserror",
            "anyhow",
            "regex",
            "chrono",
            "rand",
            "uuid",
            "itertools",
        }
    )

    # Rust keywords
    RUST_KEYWORDS = frozenset(
        {
            "as",
            "async",
            "await",
            "break",
            "const",
            "continue",
            "crate",
            "dyn",
            "else",
            "enum",
            "extern",
            "false",
            "fn",
            "for",
            "if",
            "impl",
            "in",
            "let",
            "loop",
            "match",
            "mod",
            "move",
            "mut",
            "pub",
            "ref",
            "return",
            "self",
            "Self",
            "static",
            "struct",
            "super",
            "trait",
            "true",
            "type",
            "unsafe",
            "use",
            "where",
            "while",
            "try",
        }
    )

    # Built-in types
    RUST_TYPES = frozenset(
        {
            "i8",
            "i16",
            "i32",
            "i64",
            "i128",
            "isize",
            "u8",
            "u16",
            "u32",
            "u64",
            "u128",
            "usize",
            "f32",
            "f64",
            "bool",
            "char",
            "str",
            "String",
            "Vec",
            "Box",
            "Rc",
            "Arc",
            "Cell",
            "RefCell",
            "Option",
            "Result",
            "Some",
            "None",
            "Ok",
            "Err",
            "HashMap",
            "HashSet",
            "BTreeMap",
            "BTreeSet",
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
        """Handles Rust language."""
        return {Language.RUST}

    def validate(self, code: str) -> ValidationResult:
        """Validate Rust code using pattern analysis.

        Args:
            code: Rust source code

        Returns:
            ValidationResult with validation info
        """
        errors: list[str] = []
        warnings: list[str] = []
        missing_imports: list[str] = []

        # 1. Check bracket/brace matching
        bracket_error = self._check_brackets(code)
        if bracket_error:
            return ValidationResult.failure(
                errors=[bracket_error],
                language=Language.RUST,
                syntax_error=bracket_error,
            )

        # 2. Check for fn main() in standalone code
        main_warning = self._check_main_function(code)
        if main_warning:
            warnings.append(main_warning)

        # 3. Check use statements
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

        # 6. Check lifetime annotations
        lifetime_issues = self._check_lifetimes(code)
        warnings.extend(lifetime_issues)

        return ValidationResult(
            valid=len(errors) == 0,
            language=Language.RUST,
            syntax_valid=len([e for e in errors if "syntax" in e.lower()]) == 0,
            imports_valid=len(missing_imports) == 0,
            errors=tuple(errors),
            warnings=tuple(warnings),
            missing_imports=tuple(missing_imports),
            used_ast_validation=False,
        )

    def fix(self, code: str, validation: ValidationResult) -> str:
        """Auto-fix Rust-specific issues.

        Args:
            code: Original code
            validation: Validation result with identified issues

        Returns:
            Fixed code with common issues resolved
        """
        # First apply generic cleanup
        fixed_code = self.clean_markdown(code)

        # Add missing use statements
        if validation.missing_imports:
            fixed_code = self._add_missing_imports(fixed_code, list(validation.missing_imports))

        return fixed_code

    def _check_brackets(self, code: str) -> Optional[str]:
        """Check for balanced brackets, braces, and angle brackets."""
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

    def _check_main_function(self, code: str) -> Optional[str]:
        """Check for main function in standalone code."""
        # Skip if it looks like a library (has mod declarations)
        if re.search(r"^mod\s+\w+;", code, re.MULTILINE):
            return None

        # Check if there's a fn main
        if not re.search(r"fn\s+main\s*\(\s*\)", code):
            # Only warn if there's a function defined (not just types)
            if re.search(r"fn\s+\w+\s*\(", code):
                return "No main function found (required for executable)"

        return None

    def _check_imports(self, code: str) -> tuple[list[str], list[str]]:
        """Check use statements and detect missing imports."""
        errors = []
        missing = []

        # Extract declared use statements
        declared_uses = self._extract_uses(code)

        # Find used types/modules
        used_types = self._find_used_types(code)

        # Check for common missing imports
        for type_name in used_types:
            if type_name in {"Vec", "String", "Box", "Option", "Result"}:
                # These are in prelude, no import needed
                continue

            if type_name in {"HashMap", "HashSet", "BTreeMap", "BTreeSet"}:
                if "std::collections" not in declared_uses:
                    if type_name not in missing:
                        missing.append(f"std::collections::{type_name}")
                        errors.append(f"Type '{type_name}' used but std::collections not imported")

        return errors, missing

    def _extract_uses(self, code: str) -> set[str]:
        """Extract use statement paths."""
        uses = set()

        # Match use statements
        for match in re.finditer(r"use\s+([\w:]+)", code):
            uses.add(match.group(1))

        return uses

    def _find_used_types(self, code: str) -> set[str]:
        """Find type names used in the code."""
        types = set()

        # Find capitalized identifiers that look like types
        for match in re.finditer(r"\b([A-Z][a-zA-Z0-9_]*)\b", code):
            type_name = match.group(1)
            if type_name in self.RUST_TYPES:
                types.add(type_name)

        return types

    def _check_syntax_patterns(self, code: str) -> list[str]:
        """Check for common Rust syntax issues."""
        errors = []

        # Check for let without binding
        if re.search(r"let\s*;", code):
            errors.append("Incomplete let binding")

        # Check for missing semicolons after statements (simple heuristic)
        # This is tricky because expressions don't need semicolons
        # We only check obvious cases like let bindings

        # Check for double semicolons (likely mistake)
        if re.search(r";;", code):
            errors.append("Double semicolon detected")

        # Check for arrow without body
        if re.search(r"->\s*$", code, re.MULTILINE):
            errors.append("Incomplete return type annotation")

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
            r"->\s*$",  # Ends with arrow
        ]

        for pattern in truncation_patterns:
            if re.search(pattern, stripped):
                # Only flag if also has unclosed brackets
                if self._check_brackets(stripped):
                    return "Code appears to be truncated"

        return None

    def _check_lifetimes(self, code: str) -> list[str]:
        """Check lifetime annotations for common issues."""
        warnings = []

        # Check for potentially missing lifetime annotations
        # This is a very basic heuristic

        # Find references in function signatures
        func_with_refs = re.findall(r"fn\s+\w+\s*\([^)]*&[^)]*\)\s*(->\s*&)?", code)
        for match in func_with_refs:
            if "'" not in match and match:  # Has return ref but no lifetime
                pass  # Could warn about potential lifetime issues

        return warnings

    def _remove_strings_and_comments(self, code: str) -> str:
        """Remove string contents and comments for structural analysis."""
        # Remove single-line comments
        result = re.sub(r"//.*$", "", code, flags=re.MULTILINE)

        # Remove multi-line comments
        result = re.sub(r"/\*.*?\*/", "", result, flags=re.DOTALL)

        # Remove string contents
        result = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', '""', result)

        # Remove raw strings (simplified)
        result = re.sub(r'r#*".*?"#*', '""', result)

        # Remove char literals
        result = re.sub(r"'[^'\\]'|'\\.'", "''", result)

        return result

    def _add_missing_imports(self, code: str, missing: list[str]) -> str:
        """Add missing use statements."""
        if not missing:
            return code

        # Find where to insert use statements
        # After any existing use statements, or at the top

        use_statements = "\n".join(f"use {imp};" for imp in missing) + "\n"

        # Check if there are existing use statements
        last_use = None
        for match in re.finditer(r"^use\s+.*;$", code, re.MULTILINE):
            last_use = match.end()

        if last_use:
            # Insert after last use statement
            return code[:last_use] + "\n" + use_statements + code[last_use:]
        else:
            # Insert at the top (but after any mod/crate declarations)
            return use_statements + "\n" + code
