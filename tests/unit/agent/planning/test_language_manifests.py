"""Unit tests for victor.agent.planning.language_manifests."""
from __future__ import annotations

from pathlib import Path

import pytest

from victor.agent.planning.language_manifests import (
    GoManifestHandler,
    JavaManifestHandler,
    JavaScriptManifestHandler,
    KotlinManifestHandler,
    ManifestSelection,
    PhpManifestHandler,
    PythonManifestHandler,
    RubyManifestHandler,
    RustManifestHandler,
    SimpleManifestHandler,
    TypeScriptManifestHandler,
    _HANDLERS,
    discover_language_manifests,
    get_manifest_handler,
    register_manifest_handler,
    select_language_manifests,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _touch(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


# ---------------------------------------------------------------------------
# RustManifestHandler
# ---------------------------------------------------------------------------


class TestRustManifestHandler:
    def test_discover_workspace_root_sorted_first(self, tmp_path: Path) -> None:
        _touch(tmp_path / "Cargo.toml", "[workspace]")
        _touch(tmp_path / "crates" / "core" / "Cargo.toml", "[package]")
        _touch(tmp_path / "crates" / "util" / "Cargo.toml", "[package]")

        handler = RustManifestHandler()
        manifests = handler.discover(tmp_path)

        assert manifests[0] == "Cargo.toml"
        assert "crates/core/Cargo.toml" in manifests
        assert "crates/util/Cargo.toml" in manifests

    def test_discover_excludes_target_dir(self, tmp_path: Path) -> None:
        _touch(tmp_path / "Cargo.toml", "[workspace]")
        _touch(tmp_path / "target" / "release" / "Cargo.toml", "[package]")

        manifests = RustManifestHandler().discover(tmp_path)

        assert all("target" not in m for m in manifests)
        assert "Cargo.toml" in manifests

    def test_discover_nonexistent_root_returns_empty(self, tmp_path: Path) -> None:
        manifests = RustManifestHandler().discover(tmp_path / "no_such_dir")
        assert manifests == []

    def test_select_for_step_explicit_path_in_text(self, tmp_path: Path) -> None:
        _touch(tmp_path / "rust" / "Cargo.toml", "[workspace]")

        selection = RustManifestHandler().select_for_step(
            "Read rust/Cargo.toml to find members", tmp_path
        )

        assert selection.explicit is True
        assert selection.paths == ["rust/Cargo.toml"]

    def test_select_for_step_explicit_read_root(self, tmp_path: Path) -> None:
        _touch(tmp_path / "Cargo.toml", "[workspace]")

        selection = RustManifestHandler().select_for_step(
            "Read the root Cargo.toml to identify all workspace members", tmp_path
        )

        assert selection.explicit is True
        assert "Cargo.toml" in selection.paths

    def test_select_for_step_resolves_missing_root_to_nested(self, tmp_path: Path) -> None:
        _touch(tmp_path / "rust" / "Cargo.toml", "[workspace]")

        selection = RustManifestHandler().select_for_step(
            "Read root Cargo.toml to identify workspace members", tmp_path
        )

        assert selection.explicit is True
        assert "rust/Cargo.toml" in selection.paths

    def test_select_for_step_falls_back_to_discover(self, tmp_path: Path) -> None:
        _touch(tmp_path / "Cargo.toml", "[workspace]")
        _touch(tmp_path / "core" / "Cargo.toml", "[package]")

        selection = RustManifestHandler().select_for_step(
            "Analyze workspace structure and add benchmarks", tmp_path
        )

        assert selection.explicit is False
        assert "Cargo.toml" in selection.paths
        assert "core/Cargo.toml" in selection.paths

    def test_cargo_toml_files_plural_not_extracted(self, tmp_path: Path) -> None:
        _touch(tmp_path / "Cargo.toml", "[workspace]")

        # "Cargo.toml files" is a reference to discovery, not an explicit path
        selection = RustManifestHandler().select_for_step(
            "Read all Cargo.toml files to map the workspace", tmp_path
        )

        assert selection.explicit is False


# ---------------------------------------------------------------------------
# SimpleManifestHandler (via concrete subclasses)
# ---------------------------------------------------------------------------


class TestPythonManifestHandler:
    def test_discovers_pyproject_toml(self, tmp_path: Path) -> None:
        _touch(tmp_path / "pyproject.toml", "[project]")
        _touch(tmp_path / "backend" / "pyproject.toml", "[project]")

        manifests = PythonManifestHandler().discover(tmp_path)

        assert "pyproject.toml" in manifests
        assert "backend/pyproject.toml" in manifests

    def test_discovers_setup_py(self, tmp_path: Path) -> None:
        _touch(tmp_path / "setup.py", "from setuptools import setup")
        manifests = PythonManifestHandler().discover(tmp_path)
        assert "setup.py" in manifests

    def test_discovers_requirements_txt(self, tmp_path: Path) -> None:
        _touch(tmp_path / "requirements.txt", "flask\n")
        manifests = PythonManifestHandler().discover(tmp_path)
        assert "requirements.txt" in manifests

    def test_excludes_node_modules(self, tmp_path: Path) -> None:
        _touch(tmp_path / "pyproject.toml", "[project]")
        _touch(tmp_path / "node_modules" / "some_dep" / "pyproject.toml", "[project]")

        manifests = PythonManifestHandler().discover(tmp_path)
        assert all("node_modules" not in m for m in manifests)

    def test_select_for_step_explicit_path(self, tmp_path: Path) -> None:
        _touch(tmp_path / "pyproject.toml", "[project]")
        selection = PythonManifestHandler().select_for_step(
            "Read pyproject.toml and install dependencies", tmp_path
        )
        assert selection.explicit is True
        assert "pyproject.toml" in selection.paths

    def test_select_for_step_no_explicit_discovers(self, tmp_path: Path) -> None:
        _touch(tmp_path / "pyproject.toml", "[project]")
        selection = PythonManifestHandler().select_for_step(
            "Set up Python environment and run tests", tmp_path
        )
        assert selection.explicit is False
        assert "pyproject.toml" in selection.paths

    def test_empty_root_returns_empty_list(self, tmp_path: Path) -> None:
        manifests = PythonManifestHandler().discover(tmp_path)
        assert manifests == []


class TestJavaScriptManifestHandler:
    def test_discovers_package_json(self, tmp_path: Path) -> None:
        _touch(tmp_path / "package.json", '{"name":"app"}')
        _touch(tmp_path / "packages" / "ui" / "package.json", '{"name":"ui"}')

        manifests = JavaScriptManifestHandler().discover(tmp_path)

        assert "package.json" in manifests
        assert "packages/ui/package.json" in manifests

    def test_excludes_node_modules(self, tmp_path: Path) -> None:
        _touch(tmp_path / "package.json", "{}")
        _touch(tmp_path / "node_modules" / "react" / "package.json", "{}")

        manifests = JavaScriptManifestHandler().discover(tmp_path)
        assert all("node_modules" not in m for m in manifests)

    def test_select_for_step_explicit(self, tmp_path: Path) -> None:
        _touch(tmp_path / "package.json", '{"name":"app"}')
        selection = JavaScriptManifestHandler().select_for_step(
            "Read package.json to find workspace members", tmp_path
        )
        assert selection.explicit is True


class TestTypeScriptManifestHandler:
    def test_discovers_tsconfig(self, tmp_path: Path) -> None:
        _touch(tmp_path / "tsconfig.json", "{}")
        _touch(tmp_path / "package.json", '{"name":"ts-app"}')

        manifests = TypeScriptManifestHandler().discover(tmp_path)
        assert "tsconfig.json" in manifests
        assert "package.json" in manifests


class TestGoManifestHandler:
    def test_discovers_go_mod(self, tmp_path: Path) -> None:
        _touch(tmp_path / "go.mod", "module example.com/app")
        _touch(tmp_path / "services" / "auth" / "go.mod", "module example.com/auth")

        manifests = GoManifestHandler().discover(tmp_path)

        assert "go.mod" in manifests
        assert "services/auth/go.mod" in manifests

    def test_select_for_step_explicit_go_mod(self, tmp_path: Path) -> None:
        _touch(tmp_path / "go.mod", "module example.com/app")
        selection = GoManifestHandler().select_for_step("Read go.mod for module info", tmp_path)
        assert selection.explicit is True
        assert "go.mod" in selection.paths


class TestJavaManifestHandler:
    def test_discovers_pom_xml(self, tmp_path: Path) -> None:
        _touch(tmp_path / "pom.xml", "<project/>")
        manifests = JavaManifestHandler().discover(tmp_path)
        assert "pom.xml" in manifests

    def test_discovers_gradle_files(self, tmp_path: Path) -> None:
        _touch(tmp_path / "build.gradle", "plugins {}")
        _touch(tmp_path / "settings.gradle", "rootProject.name = 'app'")
        manifests = JavaManifestHandler().discover(tmp_path)
        assert "build.gradle" in manifests
        assert "settings.gradle" in manifests


class TestKotlinManifestHandler:
    def test_discovers_gradle_kts(self, tmp_path: Path) -> None:
        _touch(tmp_path / "build.gradle.kts", "plugins {}")
        _touch(tmp_path / "settings.gradle.kts", "rootProject.name = \"app\"")
        manifests = KotlinManifestHandler().discover(tmp_path)
        assert "build.gradle.kts" in manifests
        assert "settings.gradle.kts" in manifests


class TestRubyManifestHandler:
    def test_discovers_gemfile(self, tmp_path: Path) -> None:
        _touch(tmp_path / "Gemfile", "source 'https://rubygems.org'")
        manifests = RubyManifestHandler().discover(tmp_path)
        assert "Gemfile" in manifests

    def test_select_for_step_explicit_gemfile(self, tmp_path: Path) -> None:
        _touch(tmp_path / "Gemfile", "source 'https://rubygems.org'")
        selection = RubyManifestHandler().select_for_step("Read Gemfile to check dependencies", tmp_path)
        assert selection.explicit is True


class TestPhpManifestHandler:
    def test_discovers_composer_json(self, tmp_path: Path) -> None:
        _touch(tmp_path / "composer.json", '{"require":{}}')
        manifests = PhpManifestHandler().discover(tmp_path)
        assert "composer.json" in manifests

    def test_select_for_step_explicit_composer(self, tmp_path: Path) -> None:
        _touch(tmp_path / "composer.json", '{"require":{}}')
        selection = PhpManifestHandler().select_for_step("Read composer.json for dependencies", tmp_path)
        assert selection.explicit is True


# ---------------------------------------------------------------------------
# Registry functions
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_all_default_languages_registered(self) -> None:
        for lang in ("rust", "python", "javascript", "typescript", "go", "java", "kotlin", "ruby", "php"):
            assert get_manifest_handler(lang) is not None, f"{lang} not registered"

    def test_get_manifest_handler_case_insensitive(self) -> None:
        assert get_manifest_handler("Python") is get_manifest_handler("python")
        assert get_manifest_handler("RUST") is get_manifest_handler("rust")

    def test_get_manifest_handler_unknown_returns_none(self) -> None:
        assert get_manifest_handler("cobol") is None

    def test_register_and_retrieve_custom_handler(self) -> None:
        class SwiftHandler(SimpleManifestHandler):
            language = "swift"
            manifest_names = ("Package.swift",)

        original = get_manifest_handler("swift")
        handler = SwiftHandler()
        register_manifest_handler("swift", handler)
        try:
            assert get_manifest_handler("swift") is handler
        finally:
            # Restore original state
            if original is None:
                _HANDLERS.pop("swift", None)
            else:
                register_manifest_handler("swift", original)

    def test_register_replaces_existing_handler(self) -> None:
        class RicherPythonHandler(SimpleManifestHandler):
            language = "python"
            manifest_names = ("pyproject.toml",)

        original = get_manifest_handler("python")
        register_manifest_handler("python", RicherPythonHandler())
        try:
            handler = get_manifest_handler("python")
            assert isinstance(handler, RicherPythonHandler)
        finally:
            register_manifest_handler("python", original)  # type: ignore[arg-type]

    def test_register_handler_case_normalized(self) -> None:
        class ErlangHandler(SimpleManifestHandler):
            language = "erlang"
            manifest_names = ("rebar.config",)

        register_manifest_handler("Erlang", ErlangHandler())
        try:
            assert get_manifest_handler("erlang") is not None
        finally:
            _HANDLERS.pop("erlang", None)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    def test_select_language_manifests_unknown_language_returns_empty(self, tmp_path: Path) -> None:
        result = select_language_manifests("cobol", "some step text", root=tmp_path)
        assert isinstance(result, ManifestSelection)
        assert result.paths == []

    def test_select_language_manifests_uses_cwd_when_root_none(self) -> None:
        result = select_language_manifests("go", "some text")
        assert isinstance(result, ManifestSelection)

    def test_discover_language_manifests_unknown_language(self, tmp_path: Path) -> None:
        paths = discover_language_manifests("cobol", root=tmp_path)
        assert list(paths) == []

    def test_discover_language_manifests_go(self, tmp_path: Path) -> None:
        _touch(tmp_path / "go.mod", "module example.com/app")
        paths = discover_language_manifests("go", root=tmp_path)
        assert "go.mod" in list(paths)

    def test_discover_language_manifests_max_files_limits_scan(self, tmp_path: Path) -> None:
        for i in range(5):
            _touch(tmp_path / f"mod{i}" / "go.mod", "module example.com")
        paths = discover_language_manifests("go", root=tmp_path, max_files=2)
        # Should return at most max_files results
        assert len(list(paths)) <= 2


# ---------------------------------------------------------------------------
# ManifestSelection dataclass
# ---------------------------------------------------------------------------


class TestManifestSelection:
    def test_explicit_defaults_false(self) -> None:
        sel = ManifestSelection("go", ["go.mod"])
        assert sel.explicit is False

    def test_frozen(self) -> None:
        sel = ManifestSelection("go", ["go.mod"], explicit=True)
        with pytest.raises((AttributeError, TypeError)):
            sel.language = "rust"  # type: ignore[misc]

    def test_equality(self) -> None:
        a = ManifestSelection("python", ["pyproject.toml"])
        b = ManifestSelection("python", ["pyproject.toml"])
        assert a == b
