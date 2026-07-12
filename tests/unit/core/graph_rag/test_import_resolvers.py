"""Tests for per-language import resolution strategies.

The Rust strategy is what makes Martin coupling metrics honest for Rust
projects: module-level Ca/Ce come from IMPORTS edges, and those edges only
exist if ``use`` declarations parse and resolve to project files.
"""

from pathlib import Path

from victor.core.graph_rag.import_resolvers import (
    ImportResolverRegistry,
    JsTsImportResolver,
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
        for lang in ("javascript", "typescript", "jsx", "tsx"):
            assert isinstance(ImportResolverRegistry.create(lang), JsTsImportResolver)
        assert ImportResolverRegistry.create("go") is None

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


def _make_ts_project(root: Path) -> None:
    """Vite/tsconfig-style UI project.

    tsconfig.json         baseUrl "." + paths {"@/*": ["src/*"]}
    src/main.ts
    src/utils/graph.ts
    src/components/Button.tsx
    src/components/index.ts   (barrel)
    src/legacy.js
    """
    (root / "tsconfig.json").write_text(
        "{\n"
        "  // JSONC: comments and trailing commas are the norm\n"
        '  "compilerOptions": {\n'
        '    "baseUrl": ".",\n'
        '    "paths": {\n'
        '      "@/*": ["src/*"],\n'
        "    },\n"
        "  },\n"
        "}\n"
    )
    (root / "src" / "utils").mkdir(parents=True)
    (root / "src" / "components").mkdir()
    (root / "src" / "main.ts").write_text("")
    (root / "src" / "utils" / "graph.ts").write_text("")
    (root / "src" / "components" / "Button.tsx").write_text("")
    (root / "src" / "components" / "index.ts").write_text("")
    (root / "src" / "legacy.js").write_text("")


class TestJsTsParse:
    def test_relative_import_forms(self, tmp_path: Path):
        _make_ts_project(tmp_path)
        resolver = JsTsImportResolver()
        src = str(tmp_path / "src" / "main.ts")
        assert resolver.parse("import { graph } from './utils/graph'", src, tmp_path) == [
            "src/utils/graph"
        ]
        assert resolver.parse("import './setup'", src, tmp_path) == ["src/setup"]
        assert resolver.parse("export * from './components'", src, tmp_path) == ["src/components"]
        assert resolver.parse("const x = require('./legacy')", src, tmp_path) == ["src/legacy"]
        assert resolver.parse("const m = await import('./utils/graph')", src, tmp_path) == [
            "src/utils/graph"
        ]

    def test_parent_relative_and_escape(self, tmp_path: Path):
        _make_ts_project(tmp_path)
        resolver = JsTsImportResolver()
        src = str(tmp_path / "src" / "components" / "Button.tsx")
        assert resolver.parse("import { graph } from '../utils/graph'", src, tmp_path) == [
            "src/utils/graph"
        ]
        # Escaping the project tree yields nothing rather than a bad path.
        assert resolver.parse("import x from '../../../outside'", src, tmp_path) == []

    def test_tsconfig_alias_and_baseurl(self, tmp_path: Path):
        _make_ts_project(tmp_path)
        resolver = JsTsImportResolver()
        src = str(tmp_path / "src" / "main.ts")
        # "@/*" alias from tsconfig paths.
        assert resolver.parse("import Button from '@/components/Button'", src, tmp_path) == [
            "src/components/Button"
        ]
        # Bare specifier probing under baseUrl.
        assert resolver.parse("import { g } from 'src/utils/graph'", src, tmp_path) == [
            "src/utils/graph"
        ]

    def test_bare_package_names_are_external(self, tmp_path: Path):
        _make_ts_project(tmp_path)
        resolver = JsTsImportResolver()
        src = str(tmp_path / "src" / "main.ts")
        # baseUrl candidates for real packages don't exist on disk, so they
        # fail resolution — parse may emit them, resolve must return None.
        for raw in ("import React from 'react'", "import { z } from '@scope/pkg'"):
            for candidate in resolver.parse(raw, src, tmp_path):
                assert resolver.resolve(candidate, tmp_path) is None

    def test_bundler_query_suffix_stripped(self, tmp_path: Path):
        _make_ts_project(tmp_path)
        resolver = JsTsImportResolver()
        src = str(tmp_path / "src" / "main.ts")
        assert resolver.parse("import g from './utils/graph?raw'", src, tmp_path) == [
            "src/utils/graph"
        ]


class TestJsTsResolve:
    def test_extension_and_index_probing(self, tmp_path: Path):
        _make_ts_project(tmp_path)
        resolver = JsTsImportResolver()
        assert resolver.resolve("src/utils/graph", tmp_path) == (
            tmp_path / "src" / "utils" / "graph.ts"
        )
        assert resolver.resolve("src/components/Button", tmp_path) == (
            tmp_path / "src" / "components" / "Button.tsx"
        )
        # Directory import binds to the barrel index.
        assert resolver.resolve("src/components", tmp_path) == (
            tmp_path / "src" / "components" / "index.ts"
        )
        assert resolver.resolve("src/legacy", tmp_path) == (tmp_path / "src" / "legacy.js")
        assert resolver.resolve("src/nonexistent", tmp_path) is None
        assert resolver.resolve("", tmp_path) is None
        assert resolver.resolve("../etc", tmp_path) is None

    def test_esm_js_specifier_binds_to_ts_source(self, tmp_path: Path):
        _make_ts_project(tmp_path)
        resolver = JsTsImportResolver()
        # NodeNext projects write `import './graph.js'` for a graph.ts source.
        assert resolver.resolve("src/utils/graph.js", tmp_path) == (
            tmp_path / "src" / "utils" / "graph.ts"
        )
        # A .js specifier whose .js file exists binds as-is.
        assert resolver.resolve("src/legacy.js", tmp_path) == (tmp_path / "src" / "legacy.js")


def _make_cpp_project(root: Path) -> None:
    """CMake-style layout mirroring inferflux.

    server/logging/logger.h + logger.cpp
    runtime/engine.h + engine.cpp        (engine.cpp includes "engine.h" and
                                          root-relative "server/logging/logger.h")
    runtime/util/config.h                 \\ two config.h — basename-only
    server/config.h                       // includes are ambiguous
    """
    (root / "server" / "logging").mkdir(parents=True)
    (root / "runtime" / "util").mkdir(parents=True)
    (root / "server" / "logging" / "logger.h").write_text("#pragma once\n")
    (root / "server" / "logging" / "logger.cpp").write_text('#include "logger.h"\n')
    (root / "runtime" / "engine.h").write_text("#pragma once\n")
    (root / "runtime" / "engine.cpp").write_text('#include "engine.h"\n')
    (root / "runtime" / "util" / "config.h").write_text("#pragma once\n")
    (root / "server" / "config.h").write_text("#pragma once\n")


class TestCppResolver:
    def test_quoted_include_resolves_relative_to_source(self, tmp_path: Path):
        from victor.core.graph_rag.import_resolvers import CppImportResolver

        _make_cpp_project(tmp_path)
        resolver = CppImportResolver()
        src = str(tmp_path / "runtime" / "engine.cpp")
        assert resolver.parse('#include "engine.h"', src, tmp_path) == ["runtime/engine.h"]

    def test_root_relative_include_resolves(self, tmp_path: Path):
        from victor.core.graph_rag.import_resolvers import CppImportResolver

        _make_cpp_project(tmp_path)
        resolver = CppImportResolver()
        src = str(tmp_path / "runtime" / "engine.cpp")
        # Both quote styles support the -I <root> convention.
        assert resolver.parse('#include "server/logging/logger.h"', src, tmp_path) == [
            "server/logging/logger.h"
        ]
        assert resolver.parse("#include <server/logging/logger.h>", src, tmp_path) == [
            "server/logging/logger.h"
        ]

    def test_unique_suffix_match_resolves(self, tmp_path: Path):
        from victor.core.graph_rag.import_resolvers import CppImportResolver

        _make_cpp_project(tmp_path)
        resolver = CppImportResolver()
        # <logging/logger.h> — an -I server/ include dir we can't see;
        # unique suffix match against the header index recovers it.
        src = str(tmp_path / "runtime" / "engine.cpp")
        assert resolver.parse("#include <logging/logger.h>", src, tmp_path) == [
            "server/logging/logger.h"
        ]

    def test_ambiguous_and_system_includes_are_skipped(self, tmp_path: Path):
        from victor.core.graph_rag.import_resolvers import CppImportResolver

        _make_cpp_project(tmp_path)
        resolver = CppImportResolver()
        src = str(tmp_path / "runtime" / "engine.cpp")
        # Two files named config.h: a wrong edge is worse than none.
        assert resolver.parse("#include <config.h>", src, tmp_path) == []
        # System headers never match the project index.
        assert resolver.parse("#include <vector>", src, tmp_path) == []
        assert resolver.parse("#include <sys/types.h>", src, tmp_path) == []
        # Absolute and escaping paths are rejected.
        assert resolver.parse('#include "/etc/passwd"', src, tmp_path) == []
        assert resolver.parse('#include "../../../etc/passwd"', src, tmp_path) == []

    def test_resolve_is_existence_check(self, tmp_path: Path):
        from victor.core.graph_rag.import_resolvers import CppImportResolver

        _make_cpp_project(tmp_path)
        resolver = CppImportResolver()
        assert resolver.resolve("server/logging/logger.h", tmp_path) == (
            tmp_path / "server" / "logging" / "logger.h"
        )
        assert resolver.resolve("nope/missing.h", tmp_path) is None
        assert resolver.resolve("..", tmp_path) is None
        assert resolver.resolve("", tmp_path) is None

    def test_registered_for_c_and_cpp(self):
        from victor.core.graph_rag.import_resolvers import CppImportResolver

        for lang in ("c", "cpp"):
            assert isinstance(ImportResolverRegistry.create(lang), CppImportResolver)


def _make_jvm_project(root: Path) -> None:
    """Maven multi-module + sbt mixed layout.

    core/src/main/java/com/acme/core/Engine.java     (nested class Engine.Builder)
    core/src/main/java/com/acme/core/util/Log.java
    api/src/main/java/com/acme/api/Handler.java
    api/src/main/scala/com/acme/api/Routes.scala
    core/src/test/java/com/acme/core/EngineTest.java
    Ambiguity trap: two Config.java in different packages.
    """
    for rel in (
        "core/src/main/java/com/acme/core/Engine.java",
        "core/src/main/java/com/acme/core/util/Log.java",
        "core/src/main/java/com/acme/core/Config.java",
        "core/src/test/java/com/acme/core/EngineTest.java",
        "api/src/main/java/com/acme/api/Handler.java",
        "api/src/main/java/com/acme/api/Config.java",
    ):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("class X {}\n")
    scala = root / "api/src/main/scala/com/acme/api/Routes.scala"
    scala.parent.mkdir(parents=True, exist_ok=True)
    scala.write_text("class Routes\n")


class TestJavaResolver:
    def test_fqn_resolves_across_source_roots_and_modules(self, tmp_path: Path):
        from victor.core.graph_rag.import_resolvers import JavaImportResolver

        _make_jvm_project(tmp_path)
        resolver = JavaImportResolver()
        src = str(tmp_path / "api/src/main/java/com/acme/api/Handler.java")
        # Cross-module import: api → core, source roots discovered by suffix.
        assert resolver.parse("import com.acme.core.Engine;", src, tmp_path) == [
            "core/src/main/java/com/acme/core/Engine.java"
        ]
        assert resolver.parse("import com.acme.core.util.Log;", src, tmp_path) == [
            "core/src/main/java/com/acme/core/util/Log.java"
        ]

    def test_static_and_nested_imports_drop_to_declaring_class(self, tmp_path: Path):
        from victor.core.graph_rag.import_resolvers import JavaImportResolver

        _make_jvm_project(tmp_path)
        resolver = JavaImportResolver()
        src = str(tmp_path / "api/src/main/java/com/acme/api/Handler.java")
        assert resolver.parse("import static com.acme.core.Engine.start;", src, tmp_path) == [
            "core/src/main/java/com/acme/core/Engine.java"
        ]
        assert resolver.parse("import com.acme.core.Engine.Builder;", src, tmp_path) == [
            "core/src/main/java/com/acme/core/Engine.java"
        ]

    def test_wildcards_externals_and_ambiguity_are_skipped(self, tmp_path: Path):
        from victor.core.graph_rag.import_resolvers import JavaImportResolver

        _make_jvm_project(tmp_path)
        resolver = JavaImportResolver()
        src = str(tmp_path / "api/src/main/java/com/acme/api/Handler.java")
        assert resolver.parse("import com.acme.core.*;", src, tmp_path) == []
        assert resolver.parse("import java.util.List;", src, tmp_path) == []
        # Config.java exists in two packages — but the FQN disambiguates.
        assert resolver.parse("import com.acme.api.Config;", src, tmp_path) == [
            "api/src/main/java/com/acme/api/Config.java"
        ]
        # A same-suffix collision (hypothetical duplicate FQN) yields nothing:
        dup = tmp_path / "other/src/main/java/com/acme/api/Config.java"
        dup.parent.mkdir(parents=True)
        dup.write_text("class X {}\n")
        fresh = JavaImportResolver()  # new index
        assert fresh.parse("import com.acme.api.Config;", src, tmp_path) == []


class TestScalaResolver:
    def test_selector_groups_renames_and_multi_imports(self, tmp_path: Path):
        from victor.core.graph_rag.import_resolvers import ScalaImportResolver

        _make_jvm_project(tmp_path)
        resolver = ScalaImportResolver()
        src = str(tmp_path / "api/src/main/scala/com/acme/api/Routes.scala")
        # Selector group with rename; Scala imports Java classes too.
        assert resolver.parse("import com.acme.core.{Engine => E, Config}", src, tmp_path) == [
            "core/src/main/java/com/acme/core/Engine.java",
            "core/src/main/java/com/acme/core/Config.java",
        ]
        # Multi-import clause.
        assert resolver.parse(
            "import com.acme.core.Engine, com.acme.api.Routes", src, tmp_path
        ) == [
            "core/src/main/java/com/acme/core/Engine.java",
            "api/src/main/scala/com/acme/api/Routes.scala",
        ]

    def test_wildcards_hidden_members_and_externals_skipped(self, tmp_path: Path):
        from victor.core.graph_rag.import_resolvers import ScalaImportResolver

        _make_jvm_project(tmp_path)
        resolver = ScalaImportResolver()
        src = str(tmp_path / "api/src/main/scala/com/acme/api/Routes.scala")
        assert resolver.parse("import com.acme.core._", src, tmp_path) == []
        assert resolver.parse("import com.acme.core.*", src, tmp_path) == []
        assert resolver.parse("import scala.collection.mutable.Map", src, tmp_path) == []
        # `Engine => _` hides the member; only Config imports.
        assert resolver.parse("import com.acme.core.{Engine => _, Config}", src, tmp_path) == [
            "core/src/main/java/com/acme/core/Config.java"
        ]

    def test_registered_for_java_and_scala(self):
        from victor.core.graph_rag.import_resolvers import (
            JavaImportResolver,
            ScalaImportResolver,
        )

        assert isinstance(ImportResolverRegistry.create("java"), JavaImportResolver)
        assert isinstance(ImportResolverRegistry.create("scala"), ScalaImportResolver)
