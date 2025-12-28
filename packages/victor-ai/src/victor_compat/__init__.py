"""
Victor AI Compatibility Layer

This package provides backward compatibility for imports from the
original victor-ai package structure. New code should import directly
from victor (victor-core) or victor_coding.

Deprecated import paths (will be removed in 0.5.0):
- victor.codebase -> victor_coding.codebase
- victor.lsp -> victor_coding.lsp
- victor.languages -> victor_coding.languages
- victor.editing -> victor_coding.editing
- victor.testgen -> victor_coding.testgen
- victor.refactor -> victor_coding.refactor
- victor.review -> victor_coding.review
- victor.security -> victor_coding.security

Migration Guide:
    # Old (deprecated)
    from victor.codebase import CodebaseIndexer

    # New (recommended)
    from victor_coding.codebase import CodebaseIndexer

    # Or if you need both core and coding
    import victor  # Core framework
    import victor_coding  # Coding vertical
"""

import warnings
import sys
from typing import Any

__version__ = "0.3.0"

# Mapping of deprecated module paths to new locations
_DEPRECATED_PATHS = {
    "victor.codebase": "victor_coding.codebase",
    "victor.lsp": "victor_coding.lsp",
    "victor.languages": "victor_coding.languages",
    "victor.editing": "victor_coding.editing",
    "victor.testgen": "victor_coding.testgen",
    "victor.refactor": "victor_coding.refactor",
    "victor.review": "victor_coding.review",
    "victor.security": "victor_coding.security",
    "victor.docgen": "victor_coding.docgen",
    "victor.completion": "victor_coding.completion",
    "victor.deps": "victor_coding.deps",
    "victor.coverage": "victor_coding.coverage",
}


def _deprecation_warning(old_path: str, new_path: str) -> None:
    """Emit deprecation warning for old import paths."""
    warnings.warn(
        f"Importing from '{old_path}' is deprecated. "
        f"Please use '{new_path}' instead. "
        "This compatibility shim will be removed in version 0.5.0.",
        DeprecationWarning,
        stacklevel=3,
    )


def check_deprecated_import(module_name: str) -> None:
    """Check if a module import is deprecated and warn.

    Call this from module __init__.py files that are being migrated.

    Args:
        module_name: The module being imported (e.g., 'victor.codebase')
    """
    if module_name in _DEPRECATED_PATHS:
        new_path = _DEPRECATED_PATHS[module_name]
        _deprecation_warning(module_name, new_path)


class CompatibilityFinder:
    """Import hook to intercept deprecated module imports.

    This allows us to emit deprecation warnings when old import paths
    are used, while still allowing the imports to work.
    """

    def find_module(self, fullname: str, path: Any = None):
        """Check if module is deprecated."""
        if fullname in _DEPRECATED_PATHS:
            return self
        return None

    def load_module(self, fullname: str):
        """Load module with deprecation warning."""
        if fullname in sys.modules:
            return sys.modules[fullname]

        new_path = _DEPRECATED_PATHS.get(fullname)
        if new_path:
            _deprecation_warning(fullname, new_path)

        # Let the normal import mechanism handle it
        return None


def install_compatibility_hook():
    """Install the import hook for deprecation warnings.

    Call this early in application startup to enable deprecation
    warnings for old import paths.
    """
    finder = CompatibilityFinder()
    if finder not in sys.meta_path:
        sys.meta_path.insert(0, finder)


__all__ = [
    "__version__",
    "_deprecation_warning",
    "check_deprecated_import",
    "install_compatibility_hook",
    "_DEPRECATED_PATHS",
]
