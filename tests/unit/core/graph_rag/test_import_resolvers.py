"""Tests for per-language import resolution strategies.

The Rust strategy is what makes Martin coupling metrics honest for Rust
projects: module-level Ca/Ce come from IMPORTS edges, and those edges only
exist if ``use`` declarations parse and resolve to project files.
"""

from pathlib import Path

from victor.core.graph_rag.import_resolvers import (
    ImportResolverRegistry,
    PythonImportResolver,
    RustImportResolver,
)


def _make_workspace(root: Path) -> None:
    """Two-crate Cargo workspace mirroring the shapes seen in real repos.

    root crate ``mainapp``:
        src/main.rs
        src/services/observer.rs
        src/storage/mod.rs
        src/storage/engine.rs
        src/core.rs            (file module with a child dir)
        src/core/meta.rs
        tests/integration.rs   (own crate root, outside src/)
    workspace member ``proxi-catalog``:
        crates/catalog/src/lib.rs
        crates/catalog/src/schema.rs
    """
    (root / "Cargo.toml").write_text(
        '[package]\nname = "mainapp"\nversion = "0.1.0"\n\n'
        '[workspace]\nmembers = ["crates/catalog"]\n'
    )
    (root / "src" / "services").mkdir(parents=True)
    (root / "src" / "storage").mkdir()
    (root / "src" / "core").mkdir()
    (root / "src" / "main.rs").write_text("fn main() {}\n")
    (root / "src" / "services" / "observer.rs").write_text("")
    (root / "src" / "storage" / "mod.rs").write_text("pub mod engine;\n")
    (root / "src" / "storage" / "engine.rs").write_text("")
    (root / "src" / "core.rs").write_text("pub mod meta;\n")
    (root / "src" / "core" / "meta.rs").write_text("")
    (root / "tests").mkdir()
    (root / "tests" / "integration.rs").write_text("")

    catalog = root / "crates" / "catalog"
    (catalog / "src").mkdir(parents=True)
    (catalog / "Cargo.toml").write_text('[package]\nname = "proxi-catalog"\nversion = "0.1.0"\n')
    (catalog / "src" / "lib.rs").write_text("pub mod schema;\n")
    (catalog / "src" / "schema.rs").write_text("")


class TestRegistry:
    def test_known_languages_have_strategies(self):
        assert isinstance(ImportResolverRegistry.create("python"), PythonImportResolver)
        assert isinstance(ImportResolverRegistry.create("rust"), RustImportResolver)
        assert ImportResolverRegistry.create("typescript") is None

    def test_create_returns_fresh_instances(self):
        # Per-run caches (Rust crate map) must not leak across runs.
        assert ImportResolverRegistry.create("rust") is not ImportResolverRegistry.create("rust")


class TestRustParse:
    def test_crate_path_emits_module_prefixes_not_symbol(self, tmp_path: Path):
        _make_workspace(tmp_path)
        resolver = RustImportResolver()
        src = str(tmp_path / "src" / "services" / "observer.rs")
        # Every module on the path gets a candidate; the trailing type does not.
        assert resolver.parse("use crate::storage::engine::Engine;", src, tmp_path) == [
            "src::storage",
            "src::storage::engine",
        ]

    def test_external_workspace_crate_includes_crate_root(self, tmp_path: Path):
        _make_workspace(tmp_path)
        resolver = RustImportResolver()
        src = str(tmp_path / "src" / "services" / "observer.rs")
        # Depending on any item in a sibling crate is a dependency on the
        # crate root (lib.rs) — this is what makes Ca(lib.rs) match
        # `grep -rl 'use proxi_catalog'`.
        assert resolver.parse("use proxi_catalog::schema::Table;", src, tmp_path) == [
            "crates/catalog/src",
            "crates/catalog/src::schema",
        ]

    def test_builtin_and_unknown_crates_are_skipped(self, tmp_path: Path):
        _make_workspace(tmp_path)
        resolver = RustImportResolver()
        src = str(tmp_path / "src" / "main.rs")
        assert resolver.parse("use std::collections::HashMap;", src, tmp_path) == []
        assert resolver.parse("use serde::Deserialize;", src, tmp_path) == []

    def test_brace_groups_expand_including_self(self, tmp_path: Path):
        _make_workspace(tmp_path)
        resolver = RustImportResolver()
        src = str(tmp_path / "src" / "main.rs")
        got = resolver.parse("use crate::storage::{self, engine::Engine};", src, tmp_path)
        assert got == ["src::storage", "src::storage::engine"]

    def test_nested_groups_glob_and_alias(self, tmp_path: Path):
        _make_workspace(tmp_path)
        resolver = RustImportResolver()
        src = str(tmp_path / "src" / "main.rs")
        got = resolver.parse("use crate::{storage::{engine, *}, core::meta as m};", src, tmp_path)
        assert got == ["src::storage", "src::storage::engine", "src::core", "src::core::meta"]

    def test_pub_use_visibility_qualifiers_are_stripped(self, tmp_path: Path):
        _make_workspace(tmp_path)
        resolver = RustImportResolver()
        src = str(tmp_path / "src" / "main.rs")
        assert resolver.parse("pub use crate::storage::Engine;", src, tmp_path) == ["src::storage"]
        assert resolver.parse("pub(crate) use crate::storage::Engine;", src, tmp_path) == [
            "src::storage"
        ]

    def test_super_resolves_against_parent_module(self, tmp_path: Path):
        _make_workspace(tmp_path)
        resolver = RustImportResolver()
        # engine.rs lives in module storage; super:: is the storage module.
        src = str(tmp_path / "src" / "storage" / "engine.rs")
        assert resolver.parse("use super::io_helpers;", src, tmp_path) == [
            "src/storage::io_helpers"
        ]
        # mod.rs *is* module storage; super:: is the crate root's dir.
        src = str(tmp_path / "src" / "storage" / "mod.rs")
        assert resolver.parse("use super::core::meta;", src, tmp_path) == [
            "src::core",
            "src::core::meta",
        ]

    def test_self_resolves_to_child_modules(self, tmp_path: Path):
        _make_workspace(tmp_path)
        resolver = RustImportResolver()
        # core.rs owns src/core/ — self::meta is src/core/meta.rs.
        src = str(tmp_path / "src" / "core.rs")
        assert resolver.parse("use self::meta::Meta;", src, tmp_path) == ["src/core::meta"]

    def test_crate_outside_src_is_not_bound_to_library(self, tmp_path: Path):
        _make_workspace(tmp_path)
        resolver = RustImportResolver()
        # tests/integration.rs is its own crate root; crate:: there does NOT
        # mean the library crate — binding it to src/ would fabricate edges.
        src = str(tmp_path / "tests" / "integration.rs")
        assert resolver.parse("use crate::common::helpers;", src, tmp_path) == []

    def test_non_use_text_returns_empty(self, tmp_path: Path):
        _make_workspace(tmp_path)
        resolver = RustImportResolver()
        src = str(tmp_path / "src" / "main.rs")
        assert resolver.parse("mod storage;", src, tmp_path) == []
        assert resolver.parse("", src, tmp_path) == []


class TestRustResolve:
    def test_resolves_file_module_and_mod_rs(self, tmp_path: Path):
        _make_workspace(tmp_path)
        resolver = RustImportResolver()
        # Directory module → mod.rs
        assert resolver.resolve("src::storage", tmp_path) == (
            tmp_path / "src" / "storage" / "mod.rs"
        )
        # Plain file module
        assert resolver.resolve("src::storage::engine", tmp_path) == (
            tmp_path / "src" / "storage" / "engine.rs"
        )
        # File module that also owns a child dir prefers the .rs file
        assert resolver.resolve("src::core", tmp_path) == (tmp_path / "src" / "core.rs")

    def test_bare_src_dir_resolves_to_crate_root(self, tmp_path: Path):
        _make_workspace(tmp_path)
        resolver = RustImportResolver()
        assert resolver.resolve("crates/catalog/src", tmp_path) == (
            tmp_path / "crates" / "catalog" / "src" / "lib.rs"
        )
        assert resolver.resolve("src", tmp_path) == (tmp_path / "src" / "main.rs")

    def test_unknown_or_hostile_candidates_return_none(self, tmp_path: Path):
        _make_workspace(tmp_path)
        resolver = RustImportResolver()
        assert resolver.resolve("src::nonexistent", tmp_path) is None
        assert resolver.resolve("", tmp_path) is None
        # A symbol-looking segment (function import) fails cleanly.
        assert resolver.resolve("src::storage::read_all", tmp_path) is None
        # Nothing may escape the project tree.
        assert resolver.resolve("..::etc", tmp_path) is None


class TestPythonResolver:
    """Behavior parity for the strategy moved out of indexing.py."""

    def test_parse_and_resolve_roundtrip(self, tmp_path: Path):
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "__init__.py").write_text("")
        (tmp_path / "pkg" / "util.py").write_text("")
        src = tmp_path / "pkg" / "main.py"
        src.write_text("from pkg import util\n")

        resolver = PythonImportResolver()
        candidates = resolver.parse("from pkg import util", str(src), tmp_path)
        assert candidates == ["pkg", "pkg.util"]
        assert resolver.resolve("pkg.util", tmp_path) == tmp_path / "pkg" / "util.py"
        assert resolver.resolve("os", tmp_path) is None
