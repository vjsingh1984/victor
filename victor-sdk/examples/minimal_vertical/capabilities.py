"""Capability providers for the minimal vertical.

This module demonstrates how to provide additional capabilities
through the entry point system.
"""

from typing import Dict, Any, List


class MinimalSearchCapability:
    """Search capability for code navigation."""

    def __init__(self):
        """Initialize search capability."""
        self._index = {}

    def index_file(self, path: str, content: str) -> None:
        """Index a file for search.

        Args:
            path: File path
            content: File content
        """
        # Simple word indexing
        words = content.lower().split()
        for word in set(words):
            if word not in self._index:
                self._index[word] = []
            self._index[word].append(path)

    def search(self, query: str) -> List[str]:
        """Search for a query term.

        Args:
            query: Search query

        Returns:
            List of matching file paths.
        """
        return self._index.get(query.lower(), [])

    def get_capability_info(self) -> Dict[str, Any]:
        """Return information about this capability."""
        return {
            "name": "minimal_search",
            "version": "1.0.0",
            "features": ["word_indexing", "file_search"],
        }


class MinimalValidationCapability:
    """Validation capability for code checks."""

    def __init__(self):
        """Initialize validation capability."""
        self._rules = {
            "max_line_length": 100,
            "check_indentation": True,
            "check_syntax": True,
        }

    def validate_code(self, code: str, language: str = "python") -> List[str]:
        """Validate code and return list of issues.

        Args:
            code: Code to validate
            language: Programming language

        Returns:
            List of validation error messages.
        """
        issues = []

        # Check line length
        for i, line in enumerate(code.split("\n"), 1):
            if len(line) > self._rules["max_line_length"]:
                issues.append(f"Line {i}: exceeds max line length")

        # Check for basic syntax issues (very basic)
        if language == "python":
            if code.count("(") != code.count(")"):
                issues.append("Unmatched parentheses")
            if code.count("[") != code.count("]"):
                issues.append("Unmatched brackets")

        return issues

    def get_capability_info(self) -> Dict[str, Any]:
        """Return information about this capability."""
        return {
            "name": "minimal_validation",
            "version": "1.0.0",
            "rules": self._rules,
        }
