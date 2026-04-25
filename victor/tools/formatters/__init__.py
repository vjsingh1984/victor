"""Tool output formatting system.

This module provides a unified formatter system for tool outputs, mirroring the
preview strategy pattern. Tools can format their output with Rich markup by calling
simple functions like format_test_results().

Usage:
    from victor.tools.formatters import format_test_results, format_search_results

    # In tool implementation
    test_data = {"summary": {...}, "failures": [...]}
    formatted = format_test_results(test_data)

    return {
        "raw_data": test_data,
        "formatted_output": formatted.content,
        "contains_markup": formatted.contains_markup,
    }
"""

# Import base classes and registry
from .base import ToolFormatter, FormattedOutput
from .registry import (
    FormatterRegistry,
    get_formatter_registry,
    format_tool_output,
)

# Import concrete formatters
from .test_results import TestResultsFormatter
from .search import SearchResultsFormatter
from .git import GitFormatter
from .http import HTTPFormatter
from .database import DatabaseFormatter
from .refactor import RefactorFormatter
from .docker import DockerFormatter
from .security import SecurityFormatter
from .generic import GenericFormatter

# Register all formatters with the registry
def _register_default_formatters() -> None:
    """Register all default formatters with the global registry."""
    registry = get_formatter_registry()
    registry.register("test", TestResultsFormatter())
    registry.register("pytest", TestResultsFormatter())
    registry.register("run_tests", TestResultsFormatter())
    registry.register("testing", TestResultsFormatter())

    registry.register("code_search", SearchResultsFormatter())
    registry.register("semantic_code_search", SearchResultsFormatter())
    registry.register("search", SearchResultsFormatter())

    registry.register("git", GitFormatter())

    registry.register("http", HTTPFormatter())
    registry.register("https", HTTPFormatter())
    registry.register("request", HTTPFormatter())

    registry.register("database", DatabaseFormatter())
    registry.register("db", DatabaseFormatter())
    registry.register("sql", DatabaseFormatter())

    registry.register("refactor", RefactorFormatter())
    registry.register("refactoring", RefactorFormatter())

    registry.register("docker", DockerFormatter())

    registry.register("security", SecurityFormatter())
    registry.register("security_scan", SecurityFormatter())

# Auto-register on import
_register_default_formatters()

# Convenience functions for common tools
def format_test_results(data: dict, **kwargs) -> FormattedOutput:
    """Format test results with Rich markup."""
    return format_tool_output("test", data, **kwargs)

def format_search_results(data: dict, **kwargs) -> FormattedOutput:
    """Format code search results with Rich markup."""
    return format_tool_output("code_search", data, **kwargs)

def format_git_output(data: dict, operation: str = "status", **kwargs) -> FormattedOutput:
    """Format git output with Rich markup."""
    return format_tool_output("git", data, operation=operation, **kwargs)

def format_http_response(data: dict, **kwargs) -> FormattedOutput:
    """Format HTTP response with Rich markup."""
    return format_tool_output("http", data, **kwargs)

def format_database_results(data: dict, **kwargs) -> FormattedOutput:
    """Format database query results with Rich markup."""
    return format_tool_output("database", data, **kwargs)

def format_refactor_operations(data: dict, **kwargs) -> FormattedOutput:
    """Format refactor operations with Rich markup."""
    return format_tool_output("refactor", data, **kwargs)

def format_docker_output(data: dict, operation: str = "ps", **kwargs) -> FormattedOutput:
    """Format Docker output with Rich markup."""
    return format_tool_output("docker", data, operation=operation, **kwargs)

def format_security_results(data: dict, **kwargs) -> FormattedOutput:
    """Format security scan results with Rich markup."""
    return format_tool_output("security", data, **kwargs)

__all__ = [
    # Base classes
    "ToolFormatter",
    "FormattedOutput",
    # Registry
    "FormatterRegistry",
    "get_formatter_registry",
    "format_tool_output",
    # Concrete formatters (for direct use if needed)
    "TestResultsFormatter",
    "SearchResultsFormatter",
    "GitFormatter",
    "HTTPFormatter",
    "DatabaseFormatter",
    "RefactorFormatter",
    "DockerFormatter",
    "SecurityFormatter",
    "GenericFormatter",
    # Convenience functions
    "format_test_results",
    "format_search_results",
    "format_git_output",
    "format_http_response",
    "format_database_results",
    "format_refactor_operations",
    "format_docker_output",
    "format_security_results",
]
