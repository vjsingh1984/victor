# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
High-performance regex pattern matching for code analysis.

This module provides 10-20x faster regex pattern matching compared to
Python's re module, using Rust's regex crate with DFA optimization and
multi-pattern matching via RegexSet.

Example usage:
    >>> from victor.native.rust.regex_engine import compile_language_patterns
    >>> regex_set = compile_language_patterns("python")
    >>> code = "def my_function(): import os; return 42"
    >>> matches = regex_set.match_all(code)
    >>> for match in matches:
    ...     print(f"{match.pattern_name}: {match.matched_text}")
"""

try:
    from victor_native import (  # type: ignore[import-untyped]
        CompiledRegexSet,
        MatchResult,
        compile_language_patterns,
        list_supported_languages,
        get_language_categories,
    )
except ImportError:
    # Fallback if native module is not available
    CompiledRegexSet = None
    MatchResult = None
    compile_language_patterns = None
    list_supported_languages = None
    get_language_categories = None

__all__ = [
    "CompiledRegexSet",
    "MatchResult",
    "compile_language_patterns",
    "list_supported_languages",
    "get_language_categories",
]


def get_supported_languages() -> list[str]:
    """Get list of supported programming languages.

    Returns:
        List of language names supported by compile_language_patterns

    Example:
        >>> languages = get_supported_languages()
        >>> print(languages)
        ['python', 'javascript', 'typescript', 'go', 'rust', 'java', 'cpp']
    """
    if list_supported_languages is None:
        return []
    return list_supported_languages()


def get_categories(language: str) -> list[str]:
    """Get available pattern categories for a language.

    Args:
        language: Programming language name

    Returns:
        List of pattern categories available for the language

    Raises:
        ValueError: If language is not supported

    Example:
        >>> categories = get_categories("python")
        >>> print(categories)
        ['function', 'class', 'decorator', 'import', 'comment', 'string', 'documentation']
    """
    if get_language_categories is None:
        raise ImportError("Native module not available")
    return get_language_categories(language)


def create_regex_set(
    language: str,
    pattern_types: list[str] | None = None,
) -> "CompiledRegexSet":
    """Create a compiled regex set for code analysis.

    This is a convenience wrapper around compile_language_patterns that
    provides a more Pythonic interface.

    Args:
        language: Programming language name (python, javascript, typescript,
                  go, rust, java, cpp)
        pattern_types: Optional list of pattern categories to include
                       (e.g., ["function", "class", "import"]). If None,
                       includes all pattern types.

    Returns:
        CompiledRegexSet with language-specific patterns

    Raises:
        ValueError: If language is not supported
        ImportError: If native module is not available

    Example:
        >>> # Compile all Python patterns
        >>> regex_set = create_regex_set("python")
        >>>
        >>> # Compile only function and class patterns
        >>> regex_set = create_regex_set("python", ["function", "class"])
        >>>
        >>> # Match patterns in code
        >>> matches = regex_set.match_all(source_code)
        >>> for match in matches:
        ...     print(f"Found {match.pattern_name} at line {match.line_number}")
    """
    if compile_language_patterns is None:
        raise ImportError("Native module not available. Please build victor_native.")
    return compile_language_patterns(language, pattern_types)
