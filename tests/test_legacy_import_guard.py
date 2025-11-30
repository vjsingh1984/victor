"""Guard to ensure imports resolve to the active package, not the archived copy."""

import importlib.util
import pathlib


def test_active_victor_package_used() -> None:
    """Ensure `import victor` resolves to the top-level active package."""
    spec = importlib.util.find_spec("victor")
    assert spec is not None, "victor package should be importable"
    origin = pathlib.Path(spec.origin or "").resolve()
    assert "archive" not in origin.parts, f"victor imports from archived path: {origin}"
    assert origin.name == "__init__.py", f"unexpected victor origin: {origin}"
