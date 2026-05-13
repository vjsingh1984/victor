from pathlib import Path

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
