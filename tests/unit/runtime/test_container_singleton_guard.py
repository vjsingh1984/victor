"""Guard tests for ServiceContainer singleton elimination (GS-3).

Tracks get_container() call sites and prevents new ones from being added.
Allowed locations: core/container.py (definition), core/bootstrap.py (init),
core/plugins/ (plugin context), framework/service_provider.py (framework init).
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VICTOR_ROOT = REPO_ROOT / "victor"

# These locations are allowed to call get_container() — they are infrastructure
# bootstrap code, not business logic. Everything else should receive the
# container via constructor injection or ExecutionContext.
ALLOWED_GET_CONTAINER_PREFIXES = (
    "victor/core/container.py",  # Definition
    "victor/core/bootstrap.py",  # Bootstrap infrastructure
    "victor/core/plugins/",  # Plugin context (framework init)
    "victor/framework/service_provider.py",  # Service provider (framework init)
)


def _count_get_container_calls(root: Path) -> list:
    """Find executable get_container() calls via AST."""
    calls = []
    for py_file in root.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        rel = str(py_file.relative_to(REPO_ROOT))
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=rel)
        except (SyntaxError, UnicodeDecodeError):
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "get_container":
                    calls.append((rel, node.lineno))
                elif isinstance(func, ast.Attribute) and func.attr == "get_container":
                    calls.append((rel, node.lineno))
    return calls


class TestContainerSingletonGuard:
    """Prevent proliferation of get_container() calls in business logic."""

    def test_get_container_call_count_capped(self):
        """Track get_container() calls outside allowed infrastructure.

        Current known non-infrastructure callers (to be migrated):
        - agent/ coordinators, prompt pipeline, etc.
        - framework/ step handlers, capability helpers, etc.

        This test tracks the count to ensure it doesn't INCREASE.
        As call sites are migrated to DI, update the cap downward.
        """
        calls = _count_get_container_calls(VICTOR_ROOT)
        non_infra_calls = [
            (f, line)
            for f, line in calls
            if not any(f.startswith(p) for p in ALLOWED_GET_CONTAINER_PREFIXES)
        ]
        # Current count: ~23 non-infrastructure calls.
        # This cap prevents new ones from being added.
        # As sites are migrated, lower this number.
        MAX_ALLOWED = 25
        assert len(non_infra_calls) <= MAX_ALLOWED, (
            f"Found {len(non_infra_calls)} get_container() calls outside infrastructure "
            f"(cap is {MAX_ALLOWED}). Use constructor injection or ExecutionContext instead.\n"
            f"New calls:\n" + "\n".join(f"  {f}:{line}" for f, line in sorted(non_infra_calls))
        )

    def test_no_get_container_in_new_runtime_module(self):
        """The new victor/runtime/ module must not use get_container()."""
        runtime_root = VICTOR_ROOT / "runtime"
        if not runtime_root.exists():
            pytest.skip("victor/runtime/ not yet created")

        calls = _count_get_container_calls(runtime_root)
        assert not calls, (
            "victor/runtime/ must not use get_container(). "
            "Use ExecutionContext.container instead:\n"
            + "\n".join(f"  {f}:{line}" for f, line in calls)
        )

    def test_orchestrator_uses_instance_container(self):
        """Orchestrator must use self._container, not get_container()."""
        import inspect

        from victor.agent.orchestrator import AgentOrchestrator

        source = inspect.getsource(AgentOrchestrator)
        # Count get_container() calls in orchestrator source
        import ast as _ast

        tree = _ast.parse(source)
        get_container_calls = 0
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Call):
                func = node.func
                if isinstance(func, _ast.Name) and func.id == "get_container":
                    get_container_calls += 1
                elif isinstance(func, _ast.Attribute) and func.attr == "get_container":
                    get_container_calls += 1

        assert get_container_calls == 0, (
            f"AgentOrchestrator has {get_container_calls} get_container() call(s). "
            f"Use self._container or self._execution_context.container instead."
        )
