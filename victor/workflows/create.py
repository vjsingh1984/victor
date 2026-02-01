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

"""Plugin creation API for workflow compilers.

This module provides a unified entry point for creating workflow compilers
from various sources (YAML, JSON, S3, custom plugins, etc.).

Example:
    from victor.workflows.create import create_compiler

    # Create YAML compiler (default)
    compiler = create_compiler("workflow.yaml")

    # Create with explicit scheme
    compiler = create_compiler("yaml://workflow.yaml")

    # Create from S3
    compiler = create_compiler("s3://my-bucket/workflows/deep_research.yaml")

    # Create with custom options
    compiler = create_compiler(
        "yaml://",
        enable_caching=True,
        cache_ttl=7200,
    )

    # Compile and execute
    compiled = compiler.compile("workflow.yaml")
    result = await compiled.invoke({"input": "data"})
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

from victor.workflows.compiler_registry import WorkflowCompilerRegistry
from victor.workflows.unified_compiler import UnifiedWorkflowCompiler

logger = logging.getLogger(__name__)


# =============================================================================
# Scheme Constants
# =============================================================================

# Built-in schemes that map to UnifiedWorkflowCompiler
BUILTIN_SCHEMES = {
    "yaml": "yaml",
    "yml": "yaml",
    "file": "yaml",
    "json": "json",
}

# Default scheme when none is specified
DEFAULT_SCHEME = "yaml"


# =============================================================================
# Plugin Creation API
# =============================================================================


def create_compiler(
    source: str,
    *,
    enable_caching: bool = True,
    cache_ttl: int = 3600,
    **options: Any,
) -> Any:
    """Create a workflow compiler based on source scheme.

    This is the main entry point for creating workflow compilers.
    It parses the source URI scheme and returns the appropriate compiler:

    1. If source has a scheme (e.g., "s3://", "yaml://", "custom://"):
       - Check if a plugin is registered for that scheme
       - If plugin exists, return plugin instance
       - If no plugin, check if it's a built-in scheme
       - If built-in, return UnifiedWorkflowCompiler
       - Otherwise, raise ValueError

    2. If source has no scheme:
       - Use DEFAULT_SCHEME (yaml)
       - Return UnifiedWorkflowCompiler

    Args:
        source: Source URI or file path
            - With scheme: "yaml://file.yaml", "s3://bucket/key", "custom://source"
            - Without scheme: "workflow.yaml", "file.json"
        enable_caching: Enable compilation caching (default: True)
        cache_ttl: Cache time-to-live in seconds (default: 3600)
        **options: Additional compiler-specific options
            - For UnifiedWorkflowCompiler: max_cache_entries, validate_before_compile, etc.
            - For plugins: plugin-specific options (bucket, region, etc.)

    Returns:
        Compiler instance (UnifiedWorkflowCompiler or plugin instance)

    Raises:
        ValueError: If scheme is unknown and no plugin is registered

    Example:
        # Default YAML compiler
        compiler = create_compiler("workflow.yaml")

        # Explicit scheme
        compiler = create_compiler("yaml://workflow.yaml")

        # S3 plugin (if registered)
        compiler = create_compiler(
            "s3://my-bucket/workflows/deep_research.yaml",
            bucket="my-bucket",
            region="us-west-2",
        )

        # With custom options
        compiler = create_compiler(
            "yaml://",
            enable_caching=True,
            cache_ttl=7200,
            max_cache_entries=1000,
        )

        # Compile and execute
        compiled = compiler.compile("workflow.yaml")
        result = await compiled.invoke({"input": "data"})
    """
    # Parse source to extract scheme
    scheme = _parse_scheme(source)

    # Check if a plugin is registered for this scheme
    registry = WorkflowCompilerRegistry.get_instance()

    if registry.is_compiler_registered(scheme):
        logger.debug(f"Using registered plugin compiler for scheme: {scheme}")
        # Return plugin instance
        return registry.get_compiler(
            scheme,
            enable_caching=enable_caching,
            cache_ttl=cache_ttl,
            **options,
        )

    # Check if it's a built-in scheme
    if scheme in BUILTIN_SCHEMES:
        logger.debug(f"Using built-in UnifiedWorkflowCompiler for scheme: {scheme}")
        # Return UnifiedWorkflowCompiler
        return UnifiedWorkflowCompiler(
            enable_caching=enable_caching,
            cache_ttl=cache_ttl,
            **options,
        )

    # Unknown scheme - error
    registered = ", ".join(registry.list_compilers())
    builtins = ", ".join(BUILTIN_SCHEMES.keys())
    raise ValueError(
        f"Unknown compiler scheme: '{scheme}'. "
        f"Registered plugins: {registered or '(none)'}. "
        f"Built-in schemes: {builtins}. "
        f"Register a plugin or use a built-in scheme."
    )


def _parse_scheme(source: str) -> str:
    """Parse scheme from source URI.

    Args:
        source: Source URI or file path

    Returns:
        Scheme name (lowercase)

    Examples:
        "yaml://file.yaml" -> "yaml"
        "s3://bucket/key" -> "s3"
        "workflow.yaml" -> "yaml" (default)
        "file.json" -> "json" (extension-based)
    """
    # Check for explicit scheme (://)
    if "://" in source:
        # Parse URL and extract scheme
        parsed = urlparse(source)
        scheme = parsed.scheme.lower()

        # Handle compound schemes like "s3+https"
        if "+" in scheme:
            # Split and take first part
            scheme = scheme.split("+")[0]

        return scheme

    # No explicit scheme - use file extension or default
    # Get file extension
    if "." in source:
        # Extract extension
        ext = source.rsplit(".", 1)[-1].lower()

        # Check if extension maps to a built-in scheme
        if ext in BUILTIN_SCHEMES:
            return BUILTIN_SCHEMES[ext]

    # Use default scheme
    return DEFAULT_SCHEME


def register_scheme_alias(alias: str, target_scheme: str) -> None:
    """Register a scheme alias.

    Allows custom schemes to map to built-in schemes.

    Args:
        alias: New alias to register (e.g., "yml")
        target_scheme: Existing scheme to alias (e.g., "yaml")

    Example:
        register_scheme_alias("yml", "yaml")
        compiler = create_compiler("workflow.yml")  # Uses YAML compiler
    """
    if target_scheme not in BUILTIN_SCHEMES.values():
        raise ValueError(f"Target scheme '{target_scheme}' is not a built-in scheme")

    BUILTIN_SCHEMES[alias] = target_scheme
    logger.info(f"Registered scheme alias: {alias} -> {target_scheme}")


def list_supported_schemes() -> list[str]:
    """List all supported compiler schemes.

    Returns:
        List of scheme names (built-in + registered plugins)

    Example:
        schemes = list_supported_schemes()
        print(f"Supported schemes: {', '.join(schemes)}")
    """
    registry = WorkflowCompilerRegistry.get_instance()

    # Combine built-in schemes with registered plugins
    builtin = set(BUILTIN_SCHEMES.keys())
    plugins = set(registry.list_compilers())

    return sorted(builtin | plugins)


def is_scheme_supported(scheme: str) -> bool:
    """Check if a scheme is supported.

    Args:
        scheme: Scheme name to check

    Returns:
        True if scheme is supported (built-in or registered plugin)

    Example:
        if is_scheme_supported("s3"):
            compiler = create_compiler("s3://bucket/key")
    """
    # Check built-in schemes
    if scheme in BUILTIN_SCHEMES:
        return True

    # Check registered plugins
    registry = WorkflowCompilerRegistry.get_instance()
    return registry.is_compiler_registered(scheme)


def get_default_scheme() -> str:
    """Get the default compiler scheme.

    Returns:
        Default scheme name (typically "yaml")
    """
    return DEFAULT_SCHEME


def set_default_scheme(scheme: str) -> None:
    """Set the default compiler scheme.

    Args:
        scheme: New default scheme

    Raises:
        ValueError: If scheme is not supported

    Example:
        # Change default to JSON
        set_default_scheme("json")
        compiler = create_compiler("workflow")  # Uses JSON compiler
    """
    global DEFAULT_SCHEME

    if not is_scheme_supported(scheme):
        raise ValueError(f"Scheme '{scheme}' is not supported")

    DEFAULT_SCHEME = scheme
    logger.info(f"Default scheme changed to: {scheme}")


__all__ = [
    # Main API
    "create_compiler",
    # Scheme management
    "register_scheme_alias",
    "list_supported_schemes",
    "is_scheme_supported",
    "get_default_scheme",
    "set_default_scheme",
]
