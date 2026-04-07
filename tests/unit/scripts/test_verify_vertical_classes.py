from __future__ import annotations

from pathlib import Path

from scripts.ci.verify_vertical_classes import candidate_import_roots, verify_vertical_class

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_DIR = REPO_ROOT / "examples" / "external_vertical"
EXAMPLE_SRC_PACKAGE_DIR = EXAMPLE_DIR / "src" / "victor_security"


def test_candidate_import_roots_include_src_layout_parent() -> None:
    roots = candidate_import_roots(EXAMPLE_SRC_PACKAGE_DIR)

    assert EXAMPLE_DIR / "src" in roots
    assert EXAMPLE_SRC_PACKAGE_DIR in roots


def test_verify_vertical_class_supports_repo_root_vertical_dir() -> None:
    verified = verify_vertical_class(EXAMPLE_DIR)

    assert verified.module_name == "victor_security.assistant"
    assert verified.class_name == "SecurityAssistant"
    assert verified.base_class_name == "VerticalBase"
    assert verified.instance_name == "security"


def test_verify_vertical_class_supports_nested_src_package_dir() -> None:
    verified = verify_vertical_class(EXAMPLE_SRC_PACKAGE_DIR)

    assert verified.module_name == "victor_security.assistant"
    assert verified.class_name == "SecurityAssistant"
    assert verified.base_class_name == "VerticalBase"
    assert verified.instance_name == "security"
