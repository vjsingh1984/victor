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

"""Simple registry for third-party workflow compiler plugins.

Purpose:
    Allow third-party verticals to register custom workflow compilers
    without modifying Victor's core code.

Example (Third-party package):
    from victor.workflows.plugins import WorkflowCompilerPlugin
    from victor.workflows.compiler_registry import register_compiler

    class SecurityCompilerPlugin(WorkflowCompilerPlugin):
        def compile(self, source, *, workflow_name=None, validate=True):
            # Security-specific compilation
            return self._compile_security_workflow(source)

    # Register the plugin
    register_compiler("security", SecurityCompilerPlugin)

    # Later, use the plugin
    from victor.workflows.compiler_registry import get_compiler
    compiler = get_compiler("security")
    result = compiler.compile("security_workflow.yaml")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Global registry: name -> compiler class
_compilers: Dict[str, type] = {}


def register_compiler(name: str, compiler_class: type) -> None:
    """Register a third-party workflow compiler plugin.

    Args:
        name: Unique name for the compiler (e.g., "security", "ml-pipeline")
        compiler_class: Plugin class implementing WorkflowCompilerPlugin protocol

    Example:
        from victor.workflows.compiler_registry import register_compiler
        from victor.workflows.plugins import WorkflowCompilerPlugin

        class MyCustomPlugin(WorkflowCompilerPlugin):
            def compile(self, source, *, workflow_name=None, validate=True):
                # Custom compilation logic
                pass

        register_compiler("my_custom", MyCustomPlugin)
    """
    if name in _compilers:
        logger.warning(f"Compiler plugin '{name}' already registered, overwriting")

    _compilers[name] = compiler_class
    logger.info(f"Registered compiler plugin: {name} -> {compiler_class.__name__}")


def unregister_compiler(name: str) -> None:
    """Unregister a compiler plugin.

    Args:
        name: Name of the compiler to unregister

    Raises:
        KeyError: If name is not registered
    """
    if name not in _compilers:
        raise KeyError(f"No compiler plugin registered for: {name}")

    del _compilers[name]
    logger.info(f"Unregistered compiler plugin: {name}")


def get_compiler(name: str, **options: Any) -> Any:
    """Get an instance of a registered compiler plugin.

    Args:
        name: Name of the registered compiler
        **options: Options to pass to the compiler constructor

    Returns:
        Compiler instance

    Raises:
        ValueError: If name is not registered

    Example:
        from victor.workflows.compiler_registry import get_compiler

        compiler = get_compiler("security", enable_caching=True)
        compiled = compiler.compile("workflow.yaml")
    """
    if name not in _compilers:
        registered = ", ".join(_compilers.keys())
        raise ValueError(
            f"Unknown compiler plugin: '{name}'. "
            f"Registered compilers: {registered or '(none)'}"
        )

    compiler_class = _compilers[name]
    return compiler_class(**options)


def is_registered(name: str) -> bool:
    """Check if a compiler plugin is registered.

    Args:
        name: Name of the compiler to check

    Returns:
        True if registered, False otherwise
    """
    return name in _compilers


def list_compilers() -> list[str]:
    """List all registered compiler plugin names.

    Returns:
        List of registered compiler names
    """
    return list(_compilers.keys())


__all__ = [
    "register_compiler",
    "unregister_compiler",
    "get_compiler",
    "is_registered",
    "list_compilers",
]
