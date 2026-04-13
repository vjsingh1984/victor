"""Tests for program skeleton mode in code search (SE Conventions paper)."""

import pytest


class TestExtractSkeleton:
    """Test the skeleton extraction utility."""

    def test_function_exists(self):
        """extract_skeleton function exists in code_search_tool."""
        from victor.tools.code_search_tool import extract_skeleton

        assert callable(extract_skeleton)

    def test_extracts_function_signatures(self):
        """Skeleton includes function signatures."""
        from victor.tools.code_search_tool import extract_skeleton

        source = '''
def hello(name: str) -> str:
    """Greet someone."""
    greeting = f"Hello, {name}!"
    return greeting

def goodbye(name: str) -> None:
    """Say goodbye."""
    print(f"Bye, {name}")
'''
        skeleton = extract_skeleton(source, "python")
        assert "def hello(name: str) -> str:" in skeleton
        assert "def goodbye(name: str) -> None:" in skeleton
        # Docstrings included
        assert "Greet someone" in skeleton

    def test_extracts_class_signatures(self):
        """Skeleton includes class definitions with method signatures."""
        from victor.tools.code_search_tool import extract_skeleton

        source = '''
class UserService:
    """Manages user operations."""

    def __init__(self, db):
        self.db = db
        self.cache = {}

    def get_user(self, user_id: int) -> dict:
        """Fetch user by ID."""
        result = self.db.query(user_id)
        if result:
            self.cache[user_id] = result
        return result
'''
        skeleton = extract_skeleton(source, "python")
        assert "class UserService:" in skeleton
        assert "Manages user operations" in skeleton
        assert "def get_user(self, user_id: int) -> dict:" in skeleton
        # Implementation details NOT in skeleton
        assert "self.cache[user_id]" not in skeleton

    def test_skeleton_smaller_than_source(self):
        """Skeleton is shorter than original source."""
        from victor.tools.code_search_tool import extract_skeleton

        source = '''
def complex_function(a, b, c):
    """Do something complex."""
    result = a + b
    result *= c
    if result > 100:
        result = 100
    for i in range(result):
        print(i)
    return result
'''
        skeleton = extract_skeleton(source, "python")
        assert len(skeleton) < len(source)

    def test_imports_included(self):
        """Skeleton includes import statements."""
        from victor.tools.code_search_tool import extract_skeleton

        source = '''
import os
from pathlib import Path
from typing import Dict, List

def process(items: List[str]) -> Dict[str, int]:
    """Process items."""
    result = {}
    for item in items:
        result[item] = len(item)
    return result
'''
        skeleton = extract_skeleton(source, "python")
        assert "import os" in skeleton
        assert "from pathlib import Path" in skeleton

    def test_unknown_language_returns_truncated(self):
        """Unknown language falls back to first N lines."""
        from victor.tools.code_search_tool import extract_skeleton

        source = "line1\nline2\nline3\nline4\nline5\n" * 20
        skeleton = extract_skeleton(source, "unknown_lang")
        assert len(skeleton) < len(source)

    def test_empty_source_returns_empty(self):
        """Empty source returns empty skeleton."""
        from victor.tools.code_search_tool import extract_skeleton

        assert extract_skeleton("", "python") == ""
