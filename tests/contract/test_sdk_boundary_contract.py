from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


_MODULES = [
    "victor_coding/assistant.py",
    "victor_coding/plugin.py",
    "victor_coding/middleware.py",
    "victor_coding/protocols.py",
    "victor_coding/prompts.py",
    "victor_coding/safety.py",
    "victor_coding/safety_enhanced.py",
    "victor_coding/service_provider.py",
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
    source = (_REPO_ROOT / "victor_coding/__init__.py").read_text(encoding="utf-8")

    assert "from victor_coding.assistant import" not in source
    assert "from victor.core.tool_dependency_loader import" not in source.splitlines()[:20]
