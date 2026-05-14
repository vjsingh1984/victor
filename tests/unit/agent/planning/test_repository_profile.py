from pathlib import Path

from victor.agent.planning.language_manifests import RustManifestHandler, select_language_manifests
from victor.agent.planning.repository_profile import detect_repository_profile


def test_detect_repository_profile_rust_workspace(tmp_path: Path) -> None:
    (tmp_path / "Cargo.toml").write_text("[workspace]\nmembers = ['crates/core']\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.rs").write_text("pub fn run() {}\n")

    profile = detect_repository_profile(tmp_path)

    assert profile.primary_language == "rust"
    assert profile.manifests["rust"] == ["Cargo.toml"]
    assert "Cargo.toml" in profile.to_planning_context()
    assert "workspace members/crates" in profile.inventory_guidance()


def test_detect_repository_profile_mixed_repo_uses_language_boundaries(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'api'\n")
    (tmp_path / "package.json").write_text('{"name": "web"}')
    (tmp_path / "api.py").write_text("print('api')\n")
    (tmp_path / "app.ts").write_text("export const app = 1;\n")

    profile = detect_repository_profile(tmp_path)
    context = profile.to_planning_context()

    assert profile.is_mixed is True
    assert "python" in profile.languages
    assert "javascript" in profile.languages
    assert "typescript" in profile.languages
    assert "group analysis by detected language/package boundary" in context


def test_detect_repository_profile_unknown_falls_back_to_inventory(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# docs\n")

    profile = detect_repository_profile(tmp_path)

    assert profile.primary_language == "unknown"
    assert "unknown language" in profile.to_planning_context()


def test_rust_manifest_handler_discovers_nested_workspaces(tmp_path: Path) -> None:
    (tmp_path / "rust" / "crates" / "core").mkdir(parents=True)
    (tmp_path / "rust" / "Cargo.toml").write_text("[workspace]\n")
    (tmp_path / "rust" / "crates" / "core" / "Cargo.toml").write_text("[package]\n")
    (tmp_path / "rust" / "target").mkdir()
    (tmp_path / "rust" / "target" / "Cargo.toml").write_text("[package]\n")

    manifests = RustManifestHandler().discover(tmp_path)

    assert manifests == [
        "rust/Cargo.toml",
        "rust/crates/core/Cargo.toml",
    ]


def test_rust_manifest_selection_prefers_explicit_plan_path(tmp_path: Path) -> None:
    (tmp_path / "rust").mkdir()
    (tmp_path / "rust" / "Cargo.toml").write_text("[workspace]\n")

    selection = select_language_manifests(
        "rust",
        "Map Rust workspace structure: read rust/Cargo.toml to identify all workspace members",
        root=tmp_path,
    )

    assert selection.explicit is True
    assert selection.paths == ["rust/Cargo.toml"]


def test_rust_manifest_selection_resolves_missing_root_manifest_to_nested_workspace(
    tmp_path: Path,
) -> None:
    (tmp_path / "rust" / "crates" / "core").mkdir(parents=True)
    (tmp_path / "rust" / "Cargo.toml").write_text("[workspace]\n")
    (tmp_path / "rust" / "crates" / "core" / "Cargo.toml").write_text("[package]\n")

    selection = select_language_manifests(
        "rust",
        "Read root Cargo.toml to identify workspace members and crate layout",
        root=tmp_path,
    )

    assert selection.explicit is True
    assert selection.paths == [
        "rust/Cargo.toml",
        "rust/crates/core/Cargo.toml",
    ]
