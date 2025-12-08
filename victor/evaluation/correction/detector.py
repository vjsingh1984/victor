# Copyright 2025 Vijaykumar Singh
# SPDX-License-Identifier: Apache-2.0
"""Language detection for code snippets.

This module provides intelligent language detection using multiple
heuristics: file extension, shebang, and content pattern analysis.

Design Pattern: Chain of Responsibility
- Multiple detection strategies tried in order
- First successful detection wins
- Graceful fallback to UNKNOWN
"""

import re
from typing import Optional

from .types import Language


class LanguageDetector:
    """Detects programming language from code content or file extension.

    Uses multiple heuristics in order of reliability:
    1. File extension (most reliable when available)
    2. Shebang line (#!/usr/bin/env python)
    3. Content pattern analysis (keywords, syntax)

    Thread-safe and stateless - can be shared across threads.

    Usage:
        detector = LanguageDetector()
        lang = detector.detect(code, filename="example.py")
        # Returns Language.PYTHON
    """

    # File extension to language mapping (immutable)
    EXTENSION_MAP: dict[str, Language] = {
        # Python
        ".py": Language.PYTHON,
        ".pyw": Language.PYTHON,
        ".pyi": Language.PYTHON,
        # JavaScript
        ".js": Language.JAVASCRIPT,
        ".mjs": Language.JAVASCRIPT,
        ".cjs": Language.JAVASCRIPT,
        ".jsx": Language.JAVASCRIPT,
        # TypeScript
        ".ts": Language.TYPESCRIPT,
        ".tsx": Language.TYPESCRIPT,
        ".mts": Language.TYPESCRIPT,
        # Go
        ".go": Language.GO,
        # Rust
        ".rs": Language.RUST,
        # Java
        ".java": Language.JAVA,
        # C/C++
        ".cpp": Language.CPP,
        ".cc": Language.CPP,
        ".cxx": Language.CPP,
        ".hpp": Language.CPP,
        ".c": Language.C,
        ".h": Language.C,
        # Ruby
        ".rb": Language.RUBY,
        ".rake": Language.RUBY,
        # PHP
        ".php": Language.PHP,
        # Swift
        ".swift": Language.SWIFT,
        # Kotlin
        ".kt": Language.KOTLIN,
        ".kts": Language.KOTLIN,
        # C#
        ".cs": Language.CSHARP,
        # Scala
        ".scala": Language.SCALA,
        ".sc": Language.SCALA,
    }

    # Shebang patterns (immutable)
    SHEBANG_PATTERNS: dict[str, Language] = {
        "python": Language.PYTHON,
        "python3": Language.PYTHON,
        "node": Language.JAVASCRIPT,
        "ruby": Language.RUBY,
        "php": Language.PHP,
        "bash": Language.UNKNOWN,  # Shell scripts
        "sh": Language.UNKNOWN,
    }

    # Language-specific patterns (ordered by specificity)
    # Each tuple: (Language, list of regex patterns, min_matches)
    LANGUAGE_SIGNATURES: list[tuple[Language, list[str], int]] = [
        # TypeScript (check before JavaScript - more specific)
        (
            Language.TYPESCRIPT,
            [
                r":\s*(string|number|boolean|void|any|never)\b",
                r"interface\s+\w+\s*\{",
                r"type\s+\w+\s*=",
                r"<\w+>\s*\(",
                r"as\s+(string|number|boolean)",
                r":\s*\w+\[\]",
            ],
            1,
        ),
        # Python (distinctive syntax)
        (
            Language.PYTHON,
            [
                r"^def\s+\w+\s*\(",
                r"^class\s+\w+.*:",
                r"^import\s+\w+",
                r"^from\s+\w+\s+import",
                r"^\s*if\s+__name__\s*==",
                r":\s*$",  # Colon at end of line
                r"^\s*@\w+",  # Decorators
            ],
            2,
        ),
        # Go (distinctive keywords)
        (
            Language.GO,
            [
                r"^package\s+\w+",
                r"^func\s+\w+\s*\(",
                r"^import\s+\(",
                r":=",
                r"^type\s+\w+\s+struct",
                r"fmt\.Print",
            ],
            2,
        ),
        # Rust (distinctive syntax)
        (
            Language.RUST,
            [
                r"^fn\s+\w+\s*\(",
                r"^use\s+\w+::",
                r"^mod\s+\w+",
                r"let\s+mut\s+",
                r"impl\s+\w+",
                r"pub\s+fn",
                r"->.*\{",
            ],
            2,
        ),
        # JavaScript (after TypeScript)
        (
            Language.JAVASCRIPT,
            [
                r"^const\s+\w+\s*=",
                r"^let\s+\w+\s*=",
                r"^var\s+\w+\s*=",
                r"function\s+\w+\s*\(",
                r"=>\s*\{",
                r"require\s*\(",
                r"module\.exports",
                r"console\.log",
            ],
            2,
        ),
        # Java
        (
            Language.JAVA,
            [
                r"^public\s+class\s+\w+",
                r"^import\s+java\.",
                r"public\s+static\s+void\s+main",
                r"System\.out\.print",
                r"^package\s+\w+\.\w+",
            ],
            2,
        ),
        # C++
        (
            Language.CPP,
            [
                r"#include\s*<",
                r"^int\s+main\s*\(",
                r"std::",
                r"cout\s*<<",
                r"nullptr",
                r"template\s*<",
            ],
            2,
        ),
        # C (after C++)
        (
            Language.C,
            [
                r"#include\s*<stdio\.h>",
                r"printf\s*\(",
                r"malloc\s*\(",
                r"^int\s+main\s*\(",
            ],
            2,
        ),
        # Ruby
        (
            Language.RUBY,
            [
                r"^require\s+['\"]",
                r"^def\s+\w+\s*$",
                r"\.each\s+do\s*\|",
                r"^end\s*$",
                r"puts\s+",
                r"attr_\w+",
            ],
            2,
        ),
        # PHP
        (
            Language.PHP,
            [
                r"<\?php",
                r"\$\w+\s*=",
                r"echo\s+",
                r"function\s+\w+\s*\(",
                r"->",
            ],
            2,
        ),
    ]

    def detect(self, code: str, filename: Optional[str] = None) -> Language:
        """Detect language from code content and optional filename.

        Args:
            code: Source code string
            filename: Optional filename with extension

        Returns:
            Detected Language enum value (UNKNOWN if undetectable)
        """
        # 1. Try filename extension (most reliable)
        if filename:
            lang = self._detect_from_extension(filename)
            if lang != Language.UNKNOWN:
                return lang

        # 2. Try shebang line
        lang = self._detect_from_shebang(code)
        if lang != Language.UNKNOWN:
            return lang

        # 3. Try content pattern analysis
        return self._detect_from_content(code)

    def _detect_from_extension(self, filename: str) -> Language:
        """Detect language from file extension."""
        lower_filename = filename.lower()
        for ext, lang in self.EXTENSION_MAP.items():
            if lower_filename.endswith(ext):
                return lang
        return Language.UNKNOWN

    def _detect_from_shebang(self, code: str) -> Language:
        """Detect language from shebang line."""
        if not code.strip():
            return Language.UNKNOWN

        first_line = code.strip().split("\n")[0]
        if first_line.startswith("#!"):
            for pattern, lang in self.SHEBANG_PATTERNS.items():
                if pattern in first_line:
                    return lang
        return Language.UNKNOWN

    def _detect_from_content(self, code: str) -> Language:
        """Detect language from code content patterns."""
        if not code.strip():
            return Language.UNKNOWN

        scores: dict[Language, int] = {}

        for lang, patterns, min_matches in self.LANGUAGE_SIGNATURES:
            score = 0
            for pattern in patterns:
                if re.search(pattern, code, re.MULTILINE):
                    score += 1

            if score >= min_matches:
                scores[lang] = score

        if scores:
            # Return language with highest score
            return max(scores, key=lambda k: scores[k])

        return Language.UNKNOWN

    @property
    def supported_extensions(self) -> set[str]:
        """All supported file extensions."""
        return set(self.EXTENSION_MAP.keys())

    @property
    def detectable_languages(self) -> set[Language]:
        """Languages that can be detected."""
        return set(self.EXTENSION_MAP.values())


# Singleton instance for convenience
_detector: Optional[LanguageDetector] = None


def get_detector() -> LanguageDetector:
    """Get the global LanguageDetector instance."""
    global _detector
    if _detector is None:
        _detector = LanguageDetector()
    return _detector


def detect_language(code: str, filename: Optional[str] = None) -> Language:
    """Convenience function to detect language.

    Args:
        code: Source code string
        filename: Optional filename

    Returns:
        Detected Language
    """
    return get_detector().detect(code, filename)
