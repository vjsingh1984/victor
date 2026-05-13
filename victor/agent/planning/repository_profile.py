"""Repository-type profiling for planning guidance.

The planner uses this lightweight profile to choose an inventory-first
strategy before broad semantic search. It intentionally stays generic and fast:
manifest detection is preferred, with extension counts as a fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


MANIFEST_LANGUAGES: Dict[str, Tuple[str, ...]] = {
    "rust": ("Cargo.toml",),
    "python": ("pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "Pipfile"),
    "javascript": ("package.json", "pnpm-lock.yaml", "yarn.lock", "package-lock.json"),
    "typescript": ("tsconfig.json",),
    "go": ("go.mod",),
    "java": ("pom.xml", "build.gradle", "settings.gradle", "build.gradle.kts"),
    "dotnet": ("*.csproj", "*.sln"),
    "ruby": ("Gemfile", "*.gemspec"),
    "php": ("composer.json",),
    "swift": ("Package.swift",),
    "kotlin": ("settings.gradle.kts", "build.gradle.kts"),
}

EXTENSION_LANGUAGES: Dict[str, str] = {
    ".rs": "rust",
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".cs": "dotnet",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
}

EXCLUDED_DIRS = {
    ".git",
    ".victor",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    ".next",
    ".nuxt",
    ".mypy_cache",
    ".pytest_cache",
}


@dataclass(frozen=True)
class RepositoryProfile:
    """Compact repository profile used to seed planning prompts."""

    root: Path
    languages: List[str] = field(default_factory=list)
    manifests: Dict[str, List[str]] = field(default_factory=dict)
    extension_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def primary_language(self) -> str:
        """Return the leading detected language or ``unknown``."""
        return self.languages[0] if self.languages else "unknown"

    @property
    def is_mixed(self) -> bool:
        """Whether more than one language family appears material."""
        return len(self.languages) > 1

    def to_planning_context(self) -> str:
        """Render concise, language-aware inventory guidance for plan generation."""
        if not self.languages and not self.manifests:
            return (
                "Repository profile: unknown language.\n"
                "Inventory guidance: start by listing top-level files/directories and key "
                "manifests before semantic search. Use file-extension inventory to identify "
                "the dominant language and test/build system."
            )

        manifest_bits = []
        for language, paths in sorted(self.manifests.items()):
            manifest_bits.append(f"{language}: {', '.join(paths[:6])}")
        manifest_text = "; ".join(manifest_bits) if manifest_bits else "none"
        language_text = ", ".join(self.languages) if self.languages else "unknown"

        return (
            f"Repository profile: languages={language_text}; primary={self.primary_language}; "
            f"mixed={self.is_mixed}; manifests={manifest_text}.\n"
            f"Inventory guidance: {self.inventory_guidance()}"
        )

    def inventory_guidance(self) -> str:
        """Return manifest-aware guidance for the first repository mapping step."""
        guidance = [
            "Before broad semantic code_search, map the repository from manifests, "
            "top-level directories, and file inventory."
        ]
        language_guidance = {
            "rust": "For Rust, read Cargo.toml files first to identify workspace members/crates, then inspect src/, crates/, benches/, and tests/.",
            "python": "For Python, read pyproject.toml/setup files first to identify packages, test tooling, and src-layout vs flat-layout.",
            "javascript": "For Node/JavaScript, read package.json and lockfiles first to identify workspaces, scripts, and package manager.",
            "typescript": "For TypeScript, read package.json plus tsconfig files first to identify project references, apps, and packages.",
            "go": "For Go, read go.mod first to identify module path, then map cmd/, internal/, pkg/, and test packages.",
            "java": "For Java, read Maven/Gradle manifests first to identify modules, source sets, and test tasks.",
            "dotnet": "For .NET, read solution/project files first to identify projects, target frameworks, and test projects.",
        }
        for language in self.languages:
            if language in language_guidance:
                guidance.append(language_guidance[language])
        if self.is_mixed:
            guidance.append(
                "For mixed repositories, group analysis by detected language/package boundary "
                "instead of assuming a single workspace model."
            )
        return " ".join(guidance)


def detect_repository_profile(root: Path, *, max_files: int = 2500) -> RepositoryProfile:
    """Detect repository languages and manifests quickly.

    Args:
        root: Repository root.
        max_files: Maximum files to inspect while counting extensions.

    Returns:
        RepositoryProfile for planning prompt enrichment.
    """
    root = root.expanduser().resolve()
    manifest_hits: Dict[str, List[str]] = {}
    extension_counts: Dict[str, int] = {}
    inspected = 0

    for path in _iter_repo_files(root, max_files=max_files):
        inspected += 1
        rel = _safe_relative(path, root)
        name = path.name
        for language, patterns in MANIFEST_LANGUAGES.items():
            if _matches_manifest(name, patterns):
                manifest_hits.setdefault(language, []).append(rel)

        language = EXTENSION_LANGUAGES.get(path.suffix.lower())
        if language:
            extension_counts[language] = extension_counts.get(language, 0) + 1

        if inspected >= max_files:
            break

    languages = _rank_languages(manifest_hits, extension_counts)
    return RepositoryProfile(
        root=root,
        languages=languages,
        manifests={key: sorted(value) for key, value in manifest_hits.items()},
        extension_counts=dict(sorted(extension_counts.items())),
    )


def _iter_repo_files(root: Path, *, max_files: int) -> Iterable[Path]:
    """Yield files under root while skipping common generated directories."""
    yielded = 0
    if not root.exists():
        return
    for path in root.rglob("*"):
        if yielded >= max_files:
            return
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        if path.is_file():
            yielded += 1
            yield path


def _matches_manifest(filename: str, patterns: Sequence[str]) -> bool:
    return any(Path(filename).match(pattern) for pattern in patterns)


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _rank_languages(
    manifest_hits: Dict[str, List[str]],
    extension_counts: Dict[str, int],
) -> List[str]:
    candidates = set(manifest_hits) | set(extension_counts)
    ranked = sorted(
        candidates,
        key=lambda language: (
            len(manifest_hits.get(language, [])) > 0,
            extension_counts.get(language, 0),
            len(manifest_hits.get(language, [])),
            language,
        ),
        reverse=True,
    )
    return ranked[:5]
