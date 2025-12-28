"""
Victor AI Coding Vertical

Code intelligence with Tree-sitter, LSP, and 25 coding tools.
Install with: pip install victor-coding

This package provides:
- Tree-sitter based code analysis
- LSP (Language Server Protocol) integration
- 25 coding-specific tools (search, review, refactor, etc.)
- Language-specific handlers (Python, JavaScript, Go, Rust, etc.)

Requires: victor-core>=0.3.0
"""

__version__ = "0.1.0"

# Coding submodules are now part of this package
_CODING_SUBMODULES = [
    "codebase",
    "lsp",
    "languages",
    "editing",
    "testgen",
    "refactor",
    "review",
    "security",
    "docgen",
    "completion",
    "deps",
    "coverage",
    "tools",
]


def __getattr__(name: str):
    """Lazy import for coding modules and classes."""
    if name == "CodingVertical":
        from victor_coding.vertical import CodingVertical
        return CodingVertical

    # Check if it's a coding submodule
    if name in _CODING_SUBMODULES:
        import importlib
        try:
            module = importlib.import_module(f"victor_coding.{name}")
            return module
        except ImportError as e:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}: {e}"
            ) from e

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "CodingVertical",
    # Submodules (lazy loaded)
    "codebase",
    "lsp",
    "languages",
    "editing",
    "testgen",
    "refactor",
    "review",
    "security",
    "docgen",
    "completion",
    "deps",
    "coverage",
    "tools",
]
