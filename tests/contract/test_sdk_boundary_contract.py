from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


_MODULES = [
    "victor_research/assistant.py",
    "victor_research/plugin.py",
    "victor_research/protocols.py",
    "victor_research/prompts.py",
    "victor_research/safety.py",
    "victor_research/safety_enhanced.py",
]

_BANNED_IMPORTS = (
    "victor.core.verticals.protocols",
    "victor.core.verticals.registration",
    "victor.core.verticals.base",
)


def test_sdk_boundary_modules_avoid_core_vertical_protocol_imports() -> None:
    for module in _MODULES:
        source = (_REPO_ROOT / module).read_text(encoding="utf-8")
        for banned in _BANNED_IMPORTS:
            assert banned not in source, f"{module} still imports {banned}"


def test_package_init_avoids_eager_vertical_runtime_imports() -> None:
    source = (_REPO_ROOT / "victor_research/__init__.py").read_text(encoding="utf-8")

    assert "from victor_research.assistant import" not in source
    assert "from victor.core.tool_dependency_loader import" not in source.splitlines()[:20]
