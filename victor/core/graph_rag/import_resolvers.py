# Copyright 2025 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-language import resolution strategies for Graph RAG indexing.

This module is the SEAM between the language-agnostic indexing pipeline
and language-specific import semantics, mirroring the registry pattern in
``language_handlers.py``:

- ``_resolve_imports()`` in ``indexing.py`` buffers raw import strings
  (extracted by the TreeSitterAnalysisProtocol provider) and drives the
  generic resolve loop.
- Each language contributes a :class:`LanguageImportResolver` strategy that
  knows how to (a) parse one raw import statement into module-name
  candidates and (b) resolve a candidate to a project file.
- Strategies are discovered through :class:`ImportResolverRegistry`.
  Resolver instances are created per resolution run, so a strategy may
  cache project-layout state (e.g. the Rust workspace crate map) on
  ``self`` without staleness across runs.

Python and Rust are implemented today. TS/JS/Go need their own strategies
(path-string imports, package names) — register them here rather than
adding language branches to ``indexing.py``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Set

logger = logging.getLogger(__name__)


class LanguageImportResolver(Protocol):
    """Strategy protocol: parse and resolve one language's imports.

    ``parse`` turns a raw import statement into zero or more self-contained
    module-name candidates; ``resolve`` maps one candidate to a project
    file (or None for stdlib/third-party targets). Candidates must be
    self-contained strings — the pipeline caches ``resolve`` results keyed
    by (candidate, language), so a candidate may not depend on which source
    file emitted it.
    """

    def parse(self, raw: str, src_file: str, root_path: Path) -> List[str]:
        """Parse one raw import statement into module-name candidates."""
        ...

    def resolve(self, module: str, root_path: Path) -> Optional[Path]:
        """Resolve a candidate to a project file, or None if external."""
        ...


class ImportResolverRegistry:
    """Registry of per-language import resolver factories.

    ``create`` returns a fresh strategy instance so per-run caches (e.g.
    the Rust workspace crate map) start clean each indexing run.
    """

    _factories: Dict[str, Callable[[], LanguageImportResolver]] = {}

    @classmethod
    def register(cls, language: str, factory: Callable[[], LanguageImportResolver]) -> None:
        cls._factories[language.lower()] = factory
        logger.debug("Registered import resolver for language: %s", language)

    @classmethod
    def create(cls, language: str) -> Optional[LanguageImportResolver]:
        factory = cls._factories.get(language.lower())
        return factory() if factory is not None else None

    @classmethod
    def supported_languages(cls) -> List[str]:
        return sorted(cls._factories)


class PythonImportResolver:
    """Resolve Python ``import``/``from`` statements to project modules."""

    def parse(self, raw: str, src_file: str, root_path: Path) -> List[str]:
        """Extract candidate dotted module names from a Python import.

        Handles: ``import a``, ``import a as x``, ``import a, b``,
        ``import a.b.c``, ``from a.b import x``, ``from a.b import x, y``,
        ``from .rel import x`` (resolved against ``src_file``'s package),
        and dotted relative imports (``from ..pkg import x``).

        ``from X import Y`` is genuinely ambiguous: ``Y`` could be a symbol
        defined in ``X/__init__.py`` *or* a sibling module ``X/Y.py``. We
        emit both candidates (``X`` and ``X.Y``) and let ``resolve`` decide
        which one actually exists. The downstream resolver dedupes edges by
        (src, dst) pair so emitting both is cheap and prevents losing
        legitimate submodule imports.
        """
        text = raw.strip()
        if text.startswith("import "):
            tail = text[len("import ") :].strip()
            modules: List[str] = []
            for piece in tail.split(","):
                name = piece.strip().split(" as ")[0].strip()
                if name:
                    modules.append(name)
            return modules
        if text.startswith("from "):
            tail = text[len("from ") :].strip()
            if " import " not in tail:
                return []
            module_part, names_part = tail.split(" import ", 1)
            module_part = module_part.strip()
            # Compute the base module path (absolute or resolved-relative).
            base: Optional[str]
            if module_part.startswith("."):
                dots = 0
                while dots < len(module_part) and module_part[dots] == ".":
                    dots += 1
                rest = module_part[dots:]
                try:
                    rel = Path(src_file).resolve().relative_to(root_path.resolve())
                except ValueError:
                    return []
                pkg_parts = list(rel.parts[:-1])
                for _ in range(dots - 1):
                    if not pkg_parts:
                        return []
                    pkg_parts.pop()
                if rest:
                    base = ".".join([*pkg_parts, rest])
                else:
                    base = ".".join(pkg_parts) if pkg_parts else None
            else:
                base = module_part

            candidates: List[str] = []
            if base:
                candidates.append(base)
            # Also emit base.<name> for each imported name so submodule
            # imports (``from pkg import submodule``) get an edge to the
            # submodule itself, not just the parent package. Strip aliases
            # and wildcard form.
            for piece in names_part.split(","):
                name = piece.strip().split(" as ")[0].strip().rstrip(")").lstrip("(")
                if not name or name == "*":
                    continue
                if base:
                    candidates.append(f"{base}.{name}")
                else:
                    candidates.append(name)
            return candidates
        return []

    def resolve(self, module: str, root_path: Path) -> Optional[Path]:
        """Resolve a dotted module name to a project file path.

        Prefers ``foo/bar/baz.py``, falls back to
        ``foo/bar/baz/__init__.py``. Returns None for stdlib/third-party
        modules (anything not under ``root_path``).
        """
        if not module:
            return None
        parts = module.split(".")
        file_candidate = root_path.joinpath(*parts).with_suffix(".py")
        if file_candidate.is_file():
            return file_candidate
        init_candidate = root_path.joinpath(*parts) / "__init__.py"
        if init_candidate.is_file():
            return init_candidate
        return None


class RustImportResolver:
    """Resolve Rust ``use`` declarations to project modules.

    Candidate encoding: ``<src-dir-relpath>`` (crate root) or
    ``<src-dir-relpath>::seg1::seg2`` where the prefix is the anchor
    directory relative to the project root — self-contained so the
    pipeline can cache resolutions without re-deriving crate layout.

    Instances cache the workspace crate map and Cargo.toml walk-ups, so
    create one instance per resolution run (``ImportResolverRegistry
    .create`` does exactly that).
    """

    #: First segments of ``use`` paths that can never be project crates.
    _BUILTIN_CRATES = frozenset({"std", "core", "alloc", "proc_macro", "test"})

    _PACKAGE_NAME_RE = re.compile(r'(?ms)^\[package\][^\[]*?^\s*name\s*=\s*"([^"]+)"')

    def __init__(self) -> None:
        # Workspace crate map (underscored crate name → src dir), built on
        # first external-crate lookup.
        self._crate_src_map: Optional[Dict[str, Path]] = None
        # Cargo.toml walk-up memo (source dir → crate src dir).
        self._cargo_src_cache: Dict[str, Optional[Path]] = {}

    def parse(self, raw: str, src_file: str, root_path: Path) -> List[str]:
        """Extract candidate module names from a Rust ``use`` declaration.

        For each ``use`` path we emit a candidate for *every module prefix*
        below the anchor (``use crate::a::b::X`` → ``a`` and ``a::b``):
        Rust name resolution traverses each of those modules, and the
        prefix set is what makes module-level fan-in (Martin Ca) match
        real use-statement counts. For external workspace crates the crate
        root itself is also emitted (``use other_crate::x`` is a dependency
        on ``other_crate/src/lib.rs``); for ``crate::``/``self::``/
        ``super::`` the anchor is the importer's own crate, so the anchor
        root is deliberately *not* emitted.

        Trailing segments that are clearly symbols (leading uppercase, glob
        ``*``) are dropped; a lowercase final segment that turns out to be
        a function simply fails resolution downstream, mirroring how the
        Python strategy over-emits ``base.name`` candidates.
        """
        try:
            src_path = Path(src_file).resolve()
            src_path.relative_to(root_path.resolve())
        except (OSError, ValueError):
            return []

        text = raw.strip().removesuffix(";").strip()
        # Strip visibility qualifier: pub, pub(crate), pub(in some::path).
        if text.startswith("pub"):
            rest = text[3:].lstrip()
            if rest.startswith("("):
                close = rest.find(")")
                if close == -1:
                    return []
                rest = rest[close + 1 :].lstrip()
            text = rest
        if not text.startswith("use"):
            return []
        text = text[3:].strip()

        candidates: List[str] = []
        seen: Set[str] = set()
        for path in self._expand_use_tree(text):
            segments = [seg.strip().removeprefix("r#") for seg in path.split("::")]
            segments = [seg for seg in segments if seg and seg != "*"]
            if not segments:
                continue

            anchor: Optional[Path] = None
            include_anchor_root = False
            first = segments[0]
            if first == "crate":
                anchor = self._crate_src_dir(src_path, root_path)
                segments = segments[1:]
            elif first in ("self", "super"):
                anchor = self._module_dir(src_path)
                while segments and segments[0] == "super":
                    anchor = anchor.parent
                    segments = segments[1:]
                if segments and segments[0] == "self":
                    segments = segments[1:]
            elif first in self._BUILTIN_CRATES:
                continue
            else:
                # External crate: dependency on a sibling workspace crate.
                anchor = self._workspace_crate_srcs(root_path).get(first)
                segments = segments[1:]
                include_anchor_root = True

            if anchor is None:
                continue
            try:
                anchor_rel = anchor.resolve().relative_to(root_path.resolve()).as_posix()
            except (OSError, ValueError):
                continue  # super:: walked out of the project tree

            # Module-shaped prefixes only: stop at the first segment that
            # can't be a module path component (uppercase = type/const,
            # anything non-identifier = macro noise).
            prefixes: List[str] = []
            for seg in segments:
                if not seg.isidentifier() or seg[0].isupper():
                    break
                prefixes.append(seg)

            if include_anchor_root and anchor_rel not in seen:
                seen.add(anchor_rel)
                candidates.append(anchor_rel)
            for depth in range(1, len(prefixes) + 1):
                name = anchor_rel + "::" + "::".join(prefixes[:depth])
                if name not in seen:
                    seen.add(name)
                    candidates.append(name)
        return candidates

    def resolve(self, module: str, root_path: Path) -> Optional[Path]:
        """Resolve a ``<src-dir>::seg…`` candidate to a project file.

        Prefers ``seg….rs``, falls back to ``seg…/mod.rs``; a bare
        ``<src-dir>`` resolves to the crate root
        (``lib.rs``/``main.rs``/``mod.rs``).
        """
        if not module:
            return None
        dir_part, sep, seg_part = module.partition("::")
        base = root_path / dir_part if dir_part not in ("", ".") else root_path
        segments = [seg for seg in seg_part.split("::") if seg] if sep else []
        # Reject anything that could escape the project tree.
        if ".." in Path(dir_part).parts or any(not seg.isidentifier() for seg in segments):
            return None
        if not base.is_dir():
            return None
        if not segments:
            for name in ("lib.rs", "main.rs", "mod.rs"):
                candidate = base / name
                if candidate.is_file():
                    return candidate
            return None
        file_candidate = base.joinpath(*segments[:-1]) / f"{segments[-1]}.rs"
        if file_candidate.is_file():
            return file_candidate
        mod_candidate = base.joinpath(*segments) / "mod.rs"
        if mod_candidate.is_file():
            return mod_candidate
        return None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @classmethod
    def _expand_use_tree(cls, text: str) -> List[str]:
        """Expand a Rust use-tree into plain ``::`` paths.

        Handles brace groups (nested), ``self`` group members (bind to the
        group prefix), aliases (``as x`` stripped), and leading ``::``.
        """
        text = text.strip()
        brace = text.find("{")
        if brace == -1:
            path = text.split(" as ")[0].strip().removeprefix("::").strip()
            return [path] if path else []
        prefix = text[:brace].strip().removesuffix("::").removeprefix("::").strip()
        close = text.rfind("}")
        if close < brace:
            return []
        inner = text[brace + 1 : close]

        # Split group members on top-level commas.
        parts: List[str] = []
        depth = 0
        current: List[str] = []
        for ch in inner:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            elif ch == "," and depth == 0:
                parts.append("".join(current))
                current = []
                continue
            current.append(ch)
        parts.append("".join(current))

        out: List[str] = []
        for part in parts:
            for sub in cls._expand_use_tree(part):
                if sub == "self":
                    if prefix:
                        out.append(prefix)
                elif prefix:
                    out.append(f"{prefix}::{sub}")
                else:
                    out.append(sub)
        return out

    def _crate_src_dir(self, src_path: Path, root_path: Path) -> Optional[Path]:
        """Find the crate src/ dir that ``crate::`` refers to for a file.

        Walks up to the nearest ``Cargo.toml``. Files outside that package's
        ``src/`` (tests/, benches/, examples/) are their own crate roots, so
        ``crate::`` there does *not* point at ``src/`` — return None rather
        than emit false edges into the library crate.
        """
        parent_str = str(src_path.parent)
        if parent_str in self._cargo_src_cache:
            return self._cargo_src_cache[parent_str]
        result: Optional[Path] = None
        d = src_path.parent
        root_resolved = root_path.resolve()
        while True:
            if (d / "Cargo.toml").is_file():
                src_dir = d / "src"
                if src_dir == src_path.parent or src_dir in src_path.parents:
                    result = src_dir
                break
            if d == root_resolved or d == d.parent:
                break
            d = d.parent
        self._cargo_src_cache[parent_str] = result
        return result

    @staticmethod
    def _module_dir(src_path: Path) -> Path:
        """Directory holding a Rust file's child modules (``self::`` anchor).

        ``src/foo/bar.rs`` owns ``src/foo/bar/``; root/section files
        (``lib.rs``, ``main.rs``, ``mod.rs``) own their containing dir.
        """
        if src_path.name in ("lib.rs", "main.rs", "mod.rs"):
            return src_path.parent
        return src_path.parent / src_path.stem

    def _workspace_crate_srcs(self, root_path: Path) -> Dict[str, Path]:
        """Map underscored crate names to their src/ dirs across the workspace.

        Scans for ``Cargo.toml`` files (skipping ``target/`` and hidden
        dirs) and pulls ``[package] name``. Built once per instance, i.e.
        once per resolution run.
        """
        if self._crate_src_map is not None:
            return self._crate_src_map
        crate_map: Dict[str, Path] = {}
        try:
            manifests = list(root_path.rglob("Cargo.toml"))
        except OSError:
            manifests = []
        for manifest in manifests:
            rel_parts = manifest.relative_to(root_path).parts
            if any(part == "target" or part.startswith(".") for part in rel_parts):
                continue
            try:
                match = self._PACKAGE_NAME_RE.search(
                    manifest.read_text(encoding="utf-8", errors="replace")
                )
            except OSError:
                continue
            if not match:
                continue  # workspace-only manifest
            src_dir = manifest.parent / "src"
            if src_dir.is_dir():
                crate_map[match.group(1).replace("-", "_")] = src_dir
        self._crate_src_map = crate_map
        return crate_map


ImportResolverRegistry.register("python", PythonImportResolver)
ImportResolverRegistry.register("rust", RustImportResolver)


__all__ = [
    "ImportResolverRegistry",
    "LanguageImportResolver",
    "PythonImportResolver",
    "RustImportResolver",
]
