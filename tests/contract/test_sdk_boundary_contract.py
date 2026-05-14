from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


_MODULES = [
    "victor_devops/assistant.py",
    "victor_devops/plugin.py",
    "victor_devops/protocols.py",
    "victor_devops/prompts.py",
    "victor_devops/safety.py",
    "victor_devops/safety_enhanced.py",
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


_CONTRACT_MODULES = [
    "victor_devops/assistant.py",
    "victor_devops/plugin.py",
    "victor_devops/protocols.py",
]

_CONTRACT_IMPORTS = (
    "from victor_sdk import",
    "from victor_sdk.verticals import",
    "from victor_sdk.verticals.protocols import",
)


def test_public_contract_modules_use_contract_namespace() -> None:
    for module in _CONTRACT_MODULES:
        source = (_REPO_ROOT / module).read_text(encoding="utf-8")
        for banned in _CONTRACT_IMPORTS:
            assert banned not in source, f"{module} still imports {banned}"


def test_package_init_avoids_eager_vertical_runtime_imports() -> None:
    source = (_REPO_ROOT / "victor_devops/__init__.py").read_text(encoding="utf-8")

    assert "from victor_devops.assistant import" not in source
