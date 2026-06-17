"""Validator functions for the minimal vertical.

This module demonstrates how to provide validation functions
through the entry point system.
"""

from typing import Dict, Any


def validate_file_path(path: str) -> Dict[str, Any]:
    """Validate a file path.

    Args:
        path: File path to validate

    Returns:
        Validation result with 'valid' boolean and 'errors' list.
    """
    errors = []

    # Check if path is empty
    if not path:
        errors.append("Path cannot be empty")

    # Check if path has extension
    if "." not in path:
        errors.append("Path should include file extension")

    # Check for suspicious paths
    suspicious = ["/etc", "/sys", "/proc", "~/.ssh"]
    for sus in suspicious:
        if sus in path:
            errors.append(f"Suspicious path detected: {sus}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }


def validate_code_content(code: str, max_length: int = 10000) -> Dict[str, Any]:
    """Validate code content.

    Args:
        code: Code content to validate
        max_length: Maximum allowed code length

    Returns:
        Validation result with 'valid' boolean and 'errors' list.
    """
    errors = []

    # Check length
    if len(code) > max_length:
        errors.append(f"Code exceeds maximum length of {max_length}")

    # Check for empty code
    if not code.strip():
        errors.append("Code cannot be empty")

    # Basic syntax checks
    if code.count("{") != code.count("}"):
        errors.append("Unmatched braces")
    if code.count("(") != code.count(")"):
        errors.append("Unmatched parentheses")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }


def get_all_validators() -> Dict[str, callable]:
    """Get all validator functions.

    Returns:
        Dictionary mapping validator names to validator functions.
    """
    return {
        "file_path": validate_file_path,
        "code_content": validate_code_content,
    }
