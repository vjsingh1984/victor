"""Framework safety base classes for vertical consolidation.

Exports safety rule factories for git operations and file protection.
These are domain-agnostic and can be used by any vertical that works
with git or the filesystem.

Example:
    from victor.framework.config import SafetyEnforcer, SafetyConfig, SafetyLevel
    from victor.framework.safety import create_git_safety_rules, create_file_safety_rules

    enforcer = SafetyEnforcer(config=SafetyConfig(level=SafetyLevel.HIGH))
    create_git_safety_rules(enforcer)
    create_file_safety_rules(enforcer)

    allowed, reason = enforcer.check_operation("git push --force origin main")
    if not allowed:
        print(f"Blocked: {reason}")
"""

from __future__ import annotations


def __getattr__(name: str):
    """Lazy import safety rule factories to avoid circular import.

    The safety rule factories are defined in victor.framework.safety.py
    (the module), while this __init__.py is in the victor.framework.safety package.
    To avoid circular import, we use __getattr__ for lazy loading.
    """
    if name in ("create_git_safety_rules", "create_file_safety_rules"):
        # Import from the safety.py module in the parent framework package
        # We need to load it from the file system to avoid package/module confusion
        import importlib.util
        import importlib.machinery
        import sys
        from pathlib import Path

        # Find the safety.py module
        framework_path = Path(__file__).parent.parent
        safety_py_path = framework_path / "safety.py"

        if not safety_py_path.exists():
            raise ImportError(f"Cannot find victor.framework.safety.py at {safety_py_path}")

        # Load the module
        spec = importlib.util.spec_from_file_location("victor.framework.safety_rules", safety_py_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {safety_py_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules["victor.framework.safety_rules"] = module
        spec.loader.exec_module(module)

        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "create_git_safety_rules",
    "create_file_safety_rules",
]
