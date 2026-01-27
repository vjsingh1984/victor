"""
Unified Language Capability Registry.

Single source of truth for language capabilities, used by:
- Indexing (UnifiedSymbolExtractor)
- Validation (CodeValidationPipeline)
- Language detection
- Feature flag management
"""

import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Set

from .types import (
    ASTAccessMethod,
    ASTCapability,
    LanguageTier,
    LSPCapability,
    TreeSitterCapability,
    UnifiedLanguageCapability,
)
from .feature_flags import FeatureFlagManager

logger = logging.getLogger(__name__)


def _create_default_capabilities() -> List[UnifiedLanguageCapability]:
    """Create default language capability definitions."""
    capabilities = []

    # ==========================================================================
    # Tier 1 Languages (Full Support)
    # ==========================================================================

    # Python - Native AST + Tree-sitter + LSP
    capabilities.append(
        UnifiedLanguageCapability(
            name="python",
            tier=LanguageTier.TIER_1,
            extensions=[".py", ".pyw", ".pyi"],
            filenames=["Pipfile", "pyproject.toml", "setup.py"],
            native_ast=ASTCapability(
                library="ast",
                access_method=ASTAccessMethod.NATIVE,
                has_type_info=True,
                has_error_recovery=False,
                has_semantic_analysis=False,
            ),
            tree_sitter=TreeSitterCapability(
                grammar_package="tree_sitter_python",
                has_highlights=True,
            ),
            lsp=LSPCapability(
                server_name="pylsp",
                language_id="python",
                install_command="pip install python-lsp-server",
                has_type_info=True,
            ),
            indexing_strategy=[
                ASTAccessMethod.NATIVE,
                ASTAccessMethod.TREE_SITTER,
                ASTAccessMethod.LSP,
            ],
            validation_strategy=[
                ASTAccessMethod.NATIVE,
                ASTAccessMethod.TREE_SITTER,
                ASTAccessMethod.LSP,
            ],
        )
    )

    # TypeScript - Subprocess (Node.js) + Tree-sitter + LSP
    capabilities.append(
        UnifiedLanguageCapability(
            name="typescript",
            tier=LanguageTier.TIER_1,
            extensions=[".ts"],
            filenames=["tsconfig.json"],
            native_ast=ASTCapability(
                library="typescript",
                access_method=ASTAccessMethod.SUBPROCESS,
                requires_runtime="node",
                subprocess_command=["npx", "ts-node", "--eval"],
                output_format="json",
                has_type_info=True,
                has_error_recovery=True,
                has_semantic_analysis=True,
            ),
            tree_sitter=TreeSitterCapability(
                grammar_package="tree_sitter_typescript",
                language_function="language_typescript",
            ),
            lsp=LSPCapability(
                server_name="typescript-language-server",
                language_id="typescript",
                install_command="npm install -g typescript-language-server typescript",
                has_type_info=True,
            ),
            indexing_strategy=[
                ASTAccessMethod.TREE_SITTER,
                ASTAccessMethod.SUBPROCESS,
                ASTAccessMethod.LSP,
            ],
            validation_strategy=[
                ASTAccessMethod.TREE_SITTER,
                ASTAccessMethod.LSP,
                ASTAccessMethod.SUBPROCESS,
            ],
        )
    )

    # TSX - TypeScript with JSX
    capabilities.append(
        UnifiedLanguageCapability(
            name="tsx",
            tier=LanguageTier.TIER_1,
            extensions=[".tsx"],
            native_ast=ASTCapability(
                library="typescript",
                access_method=ASTAccessMethod.SUBPROCESS,
                requires_runtime="node",
                has_type_info=True,
                has_error_recovery=True,
            ),
            tree_sitter=TreeSitterCapability(
                grammar_package="tree_sitter_typescript",
                language_function="language_tsx",
            ),
            lsp=LSPCapability(
                server_name="typescript-language-server",
                language_id="typescriptreact",
                has_type_info=True,
            ),
            indexing_strategy=[
                ASTAccessMethod.TREE_SITTER,
                ASTAccessMethod.LSP,
            ],
            validation_strategy=[
                ASTAccessMethod.TREE_SITTER,
                ASTAccessMethod.LSP,
            ],
        )
    )

    # JavaScript - Tree-sitter + LSP (no static types)
    capabilities.append(
        UnifiedLanguageCapability(
            name="javascript",
            tier=LanguageTier.TIER_1,
            extensions=[".js", ".mjs", ".cjs"],
            filenames=["package.json"],
            native_ast=ASTCapability(
                library="typescript",
                access_method=ASTAccessMethod.SUBPROCESS,
                requires_runtime="node",
                has_type_info=False,
                has_error_recovery=True,
            ),
            tree_sitter=TreeSitterCapability(
                grammar_package="tree_sitter_javascript",
            ),
            lsp=LSPCapability(
                server_name="typescript-language-server",
                language_id="javascript",
                has_type_info=False,
            ),
            indexing_strategy=[
                ASTAccessMethod.TREE_SITTER,
                ASTAccessMethod.SUBPROCESS,
                ASTAccessMethod.LSP,
            ],
            validation_strategy=[
                ASTAccessMethod.TREE_SITTER,
                ASTAccessMethod.SUBPROCESS,
                ASTAccessMethod.LSP,
            ],
        )
    )

    # JSX - JavaScript with JSX
    capabilities.append(
        UnifiedLanguageCapability(
            name="jsx",
            tier=LanguageTier.TIER_1,
            extensions=[".jsx"],
            tree_sitter=TreeSitterCapability(
                grammar_package="tree_sitter_javascript",
            ),
            lsp=LSPCapability(
                server_name="typescript-language-server",
                language_id="javascriptreact",
            ),
            indexing_strategy=[
                ASTAccessMethod.TREE_SITTER,
                ASTAccessMethod.LSP,
            ],
            validation_strategy=[
                ASTAccessMethod.TREE_SITTER,
                ASTAccessMethod.LSP,
            ],
        )
    )

    # ==========================================================================
    # Tier 2 Languages (Good Support)
    # ==========================================================================

    # Go - gopygo (pure Python) + Tree-sitter + gopls
    capabilities.append(
        UnifiedLanguageCapability(
            name="go",
            tier=LanguageTier.TIER_2,
            extensions=[".go"],
            filenames=["go.mod", "go.sum"],
            native_ast=ASTCapability(
                library="gopygo",
                access_method=ASTAccessMethod.PYTHON_LIB,
                python_package="gopygo",
                has_type_info=False,
                has_error_recovery=False,
            ),
            tree_sitter=TreeSitterCapability(
                grammar_package="tree_sitter_go",
            ),
            lsp=LSPCapability(
                server_name="gopls",
                language_id="go",
                install_command="go install golang.org/x/tools/gopls@latest",
                has_type_info=True,
            ),
            indexing_strategy=[
                ASTAccessMethod.TREE_SITTER,
                ASTAccessMethod.PYTHON_LIB,
                ASTAccessMethod.LSP,
            ],
            validation_strategy=[
                ASTAccessMethod.TREE_SITTER,
                ASTAccessMethod.LSP,
                ASTAccessMethod.PYTHON_LIB,
            ],
        )
    )

    # Rust - Tree-sitter + rust-analyzer (no good native Python AST)
    capabilities.append(
        UnifiedLanguageCapability(
            name="rust",
            tier=LanguageTier.TIER_2,
            extensions=[".rs"],
            filenames=["Cargo.toml", "Cargo.lock"],
            tree_sitter=TreeSitterCapability(
                grammar_package="tree_sitter_rust",
            ),
            lsp=LSPCapability(
                server_name="rust-analyzer",
                language_id="rust",
                install_command="rustup component add rust-analyzer",
                has_type_info=True,
            ),
            indexing_strategy=[
                ASTAccessMethod.TREE_SITTER,
                ASTAccessMethod.LSP,
            ],
            validation_strategy=[
                ASTAccessMethod.TREE_SITTER,
                ASTAccessMethod.LSP,
            ],
        )
    )

    # Java - javalang (pure Python, Java 8 only) + Tree-sitter + jdtls
    capabilities.append(
        UnifiedLanguageCapability(
            name="java",
            tier=LanguageTier.TIER_2,
            extensions=[".java"],
            filenames=["pom.xml", "build.gradle", "build.gradle.kts"],
            native_ast=ASTCapability(
                library="javalang",
                access_method=ASTAccessMethod.PYTHON_LIB,
                python_package="javalang",
                has_type_info=False,
                has_error_recovery=True,
            ),
            tree_sitter=TreeSitterCapability(
                grammar_package="tree_sitter_java",
            ),
            lsp=LSPCapability(
                server_name="jdtls",
                language_id="java",
                has_type_info=True,
            ),
            indexing_strategy=[
                ASTAccessMethod.TREE_SITTER,
                ASTAccessMethod.PYTHON_LIB,
                ASTAccessMethod.LSP,
            ],
            validation_strategy=[
                ASTAccessMethod.TREE_SITTER,
                ASTAccessMethod.PYTHON_LIB,
                ASTAccessMethod.LSP,
            ],
        )
    )

    # C - libclang (FFI) + Tree-sitter + clangd
    capabilities.append(
        UnifiedLanguageCapability(
            name="c",
            tier=LanguageTier.TIER_2,
            extensions=[".c", ".h"],
            native_ast=ASTCapability(
                library="libclang",
                access_method=ASTAccessMethod.FFI,
                python_package="libclang",
                has_type_info=True,
                has_error_recovery=True,
                has_semantic_analysis=True,
            ),
            tree_sitter=TreeSitterCapability(
                grammar_package="tree_sitter_c",
            ),
            lsp=LSPCapability(
                server_name="clangd",
                language_id="c",
                has_type_info=True,
            ),
            indexing_strategy=[
                ASTAccessMethod.TREE_SITTER,
                ASTAccessMethod.FFI,
                ASTAccessMethod.LSP,
            ],
            validation_strategy=[
                ASTAccessMethod.TREE_SITTER,
                ASTAccessMethod.FFI,
                ASTAccessMethod.LSP,
            ],
        )
    )

    # C++ - libclang (FFI) + Tree-sitter + clangd
    capabilities.append(
        UnifiedLanguageCapability(
            name="cpp",
            tier=LanguageTier.TIER_2,
            extensions=[".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx", ".h++"],
            native_ast=ASTCapability(
                library="libclang",
                access_method=ASTAccessMethod.FFI,
                python_package="libclang",
                has_type_info=True,
                has_error_recovery=True,
                has_semantic_analysis=True,
            ),
            tree_sitter=TreeSitterCapability(
                grammar_package="tree_sitter_cpp",
            ),
            lsp=LSPCapability(
                server_name="clangd",
                language_id="cpp",
                has_type_info=True,
            ),
            indexing_strategy=[
                ASTAccessMethod.TREE_SITTER,
                ASTAccessMethod.FFI,
                ASTAccessMethod.LSP,
            ],
            validation_strategy=[
                ASTAccessMethod.TREE_SITTER,
                ASTAccessMethod.FFI,
                ASTAccessMethod.LSP,
            ],
        )
    )

    # ==========================================================================
    # Tier 3 Languages (Tree-sitter only)
    # ==========================================================================

    tier3_languages = [
        # Language, extensions, filenames, grammar_package, lsp_server, lsp_language_id
        (
            "ruby",
            [".rb", ".rake"],
            ["Gemfile", "Rakefile"],
            "tree_sitter_ruby",
            "solargraph",
            "ruby",
        ),
        ("php", [".php", ".phtml"], ["composer.json"], "tree_sitter_php", "phpactor", "php"),
        ("csharp", [".cs"], [], "tree_sitter_c_sharp", "omnisharp", "csharp"),
        ("scala", [".scala", ".sc"], ["build.sbt"], "tree_sitter_scala", "metals", "scala"),
        ("kotlin", [".kt", ".kts"], [], "tree_sitter_kotlin", "kotlin-language-server", "kotlin"),
        ("swift", [".swift"], ["Package.swift"], "tree_sitter_swift", "sourcekit-lsp", "swift"),
        ("lua", [".lua"], [], "tree_sitter_lua", "lua-language-server", "lua"),
        ("elixir", [".ex", ".exs"], ["mix.exs"], "tree_sitter_elixir", "elixir-ls", "elixir"),
        (
            "haskell",
            [".hs", ".lhs"],
            ["*.cabal"],
            "tree_sitter_haskell",
            "haskell-language-server",
            "haskell",
        ),
        ("r", [".r", ".R"], [".Rprofile"], "tree_sitter_r", "languageserver", "r"),
        ("julia", [".jl"], ["Project.toml"], "tree_sitter_julia", "LanguageServer.jl", "julia"),
        ("perl", [".pl", ".pm"], [], "tree_sitter_perl", "Perl-LanguageServer", "perl"),
        (
            "bash",
            [".sh", ".bash"],
            [".bashrc", ".bash_profile"],
            "tree_sitter_bash",
            "bash-language-server",
            "shellscript",
        ),
        ("sql", [".sql"], [], "tree_sitter_sql", None, "sql"),
        ("yaml", [".yaml", ".yml"], [], "tree_sitter_yaml", "yaml-language-server", "yaml"),
        ("json", [".json"], [], "tree_sitter_json", None, "json"),
        ("toml", [".toml"], [], "tree_sitter_toml", None, "toml"),
        ("html", [".html", ".htm"], [], "tree_sitter_html", "html-languageserver", "html"),
        ("css", [".css"], [], "tree_sitter_css", "css-languageserver", "css"),
        ("markdown", [".md", ".markdown"], [], "tree_sitter_markdown", None, "markdown"),
        ("vue", [".vue"], [], "tree_sitter_vue", "volar", "vue"),
        ("svelte", [".svelte"], [], "tree_sitter_svelte", "svelte-language-server", "svelte"),
        ("zig", [".zig"], [], "tree_sitter_zig", "zls", "zig"),
        ("nim", [".nim"], [], "tree_sitter_nim", "nimlsp", "nim"),
        ("dart", [".dart"], ["pubspec.yaml"], "tree_sitter_dart", "dart", "dart"),
        (
            "clojure",
            [".clj", ".cljs", ".cljc"],
            ["deps.edn"],
            "tree_sitter_clojure",
            "clojure-lsp",
            "clojure",
        ),
        ("erlang", [".erl", ".hrl"], ["rebar.config"], "tree_sitter_erlang", "erlang_ls", "erlang"),
        ("terraform", [".tf", ".tfvars"], [], "tree_sitter_hcl", "terraform-ls", "terraform"),
        (
            "dockerfile",
            [],
            ["Dockerfile"],
            "tree_sitter_dockerfile",
            "dockerfile-language-server-nodejs",
            "dockerfile",
        ),
        ("make", [], ["Makefile", "GNUmakefile"], "tree_sitter_make", None, "makefile"),
        # Additional configuration and documentation formats
        ("xml", [".xml", ".xsd", ".xsl", ".xslt", ".pom"], [], "tree_sitter_xml", "lemminx", "xml"),
        (
            "hocon",
            [".conf", ".hocon"],
            ["application.conf", "reference.conf"],
            "tree_sitter_hocon",
            None,
            "hocon",
        ),
        ("ini", [".ini", ".cfg"], [".editorconfig", ".gitconfig"], "tree_sitter_ini", None, "ini"),
        ("properties", [".properties"], [], "tree_sitter_properties", None, "properties"),
        ("graphql", [".graphql", ".gql"], [], "tree_sitter_graphql", "graphql-lsp", "graphql"),
        ("proto", [".proto"], [], "tree_sitter_proto", "bufls", "proto"),
        (
            "jsonnet",
            [".jsonnet", ".libsonnet"],
            [],
            "tree_sitter_jsonnet",
            "jsonnet-language-server",
            "jsonnet",
        ),
        ("rst", [".rst"], [], "tree_sitter_rst", None, "restructuredtext"),
        ("latex", [".tex", ".latex"], [], "tree_sitter_latex", "texlab", "latex"),
        (
            "gitignore",
            [],
            [".gitignore", ".dockerignore", ".npmignore"],
            "tree_sitter_gitignore",
            None,
            "gitignore",
        ),
        (
            "dotenv",
            [".env"],
            [".env.local", ".env.development", ".env.production"],
            "tree_sitter_dotenv",
            None,
            "dotenv",
        ),
        ("csv", [".csv", ".tsv"], [], "tree_sitter_csv", None, "csv"),
    ]

    # Config languages that have native Python library validators
    # These get NATIVE validation in addition to tree-sitter
    config_languages_with_native = {
        "json": "json",  # Built-in json module
        "yaml": "pyyaml",  # PyYAML library
        "toml": "tomllib",  # Built-in (Python 3.11+) or tomli
        "xml": "xml.etree",  # Built-in xml.etree.ElementTree
        "hocon": "pyhocon",  # pyhocon library
        "markdown": "markdown-it-py",  # markdown-it-py or mistune
    }

    for lang_name, extensions, filenames, grammar, lsp_server, lsp_lang_id in tier3_languages:
        lsp_cap = None
        if lsp_server:
            lsp_cap = LSPCapability(
                server_name=lsp_server,
                language_id=lsp_lang_id,
            )

        # Check if this is a config language with native Python support
        native_ast = None
        validation_strategy = [ASTAccessMethod.TREE_SITTER]
        if lang_name in config_languages_with_native:
            native_ast = ASTCapability(
                library=config_languages_with_native[lang_name],
                access_method=ASTAccessMethod.NATIVE,
                has_type_info=False,
                has_error_recovery=False,
                has_semantic_analysis=False,
            )
            # Native validation comes first for config languages
            validation_strategy = [ASTAccessMethod.NATIVE, ASTAccessMethod.TREE_SITTER]

        if lsp_cap:
            validation_strategy.append(ASTAccessMethod.LSP)

        capabilities.append(
            UnifiedLanguageCapability(
                name=lang_name,
                tier=LanguageTier.TIER_3,
                extensions=extensions,
                filenames=filenames,
                native_ast=native_ast,
                tree_sitter=TreeSitterCapability(
                    grammar_package=grammar,
                ),
                lsp=lsp_cap,
                indexing_strategy=[
                    ASTAccessMethod.TREE_SITTER,
                    ASTAccessMethod.LSP,
                ],
                validation_strategy=validation_strategy,
            )
        )

    return capabilities


class LanguageCapabilityRegistry:
    """
    Unified registry for language capabilities.

    Single source of truth used by:
    - Indexing (UnifiedSymbolExtractor)
    - Validation (CodeValidationPipeline)
    - Language detection
    - Feature flag management

    Thread-safe singleton with lazy initialization.
    """

    _instance: Optional["LanguageCapabilityRegistry"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._capabilities: Dict[str, UnifiedLanguageCapability] = {}
        self._extension_map: Dict[str, str] = {}
        self._filename_map: Dict[str, str] = {}
        self._feature_flags = FeatureFlagManager.instance()
        self._internal_lock = threading.RLock()

    @classmethod
    def instance(cls) -> "LanguageCapabilityRegistry":
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    cls._instance._load_default_capabilities()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None
            FeatureFlagManager.reset_instance()

    def register(self, capability: UnifiedLanguageCapability) -> None:
        """Register a language capability."""
        with self._internal_lock:
            self._capabilities[capability.name] = capability
            for ext in capability.extensions:
                ext_lower = ext.lower()
                if not ext_lower.startswith("."):
                    ext_lower = f".{ext_lower}"
                self._extension_map[ext_lower] = capability.name
            for fname in capability.filenames:
                self._filename_map[fname.lower()] = capability.name

    def unregister(self, language: str) -> Optional[UnifiedLanguageCapability]:
        """Unregister a language capability. Returns the removed capability."""
        with self._internal_lock:
            cap = self._capabilities.pop(language, None)
            if cap:
                # Remove extension mappings
                for ext in cap.extensions:
                    ext_lower = ext.lower()
                    if not ext_lower.startswith("."):
                        ext_lower = f".{ext_lower}"
                    self._extension_map.pop(ext_lower, None)
                # Remove filename mappings
                for fname in cap.filenames:
                    self._filename_map.pop(fname.lower(), None)
            return cap

    def get(self, language: str) -> Optional[UnifiedLanguageCapability]:
        """Get capability by language name."""
        return self._capabilities.get(language.lower())

    def get_for_file(self, file_path: Path) -> Optional[UnifiedLanguageCapability]:
        """Detect language and get capability for a file."""

        # Try filename first (exact matches like Dockerfile, Makefile)
        name = file_path.name.lower()
        if name in self._filename_map:
            return self.get(self._filename_map[name])

        # Try extension
        ext = file_path.suffix.lower()
        if ext in self._extension_map:
            return self.get(self._extension_map[ext])

        return None

    def detect_language(self, file_path: Path) -> Optional[str]:
        """Detect language from file path."""
        cap = self.get_for_file(file_path)
        return cap.name if cap else None

    def get_indexing_method(self, language: str) -> Optional[ASTAccessMethod]:
        """Get best indexing method for a language (respects feature flags)."""
        cap = self.get(language)
        if not cap or not cap.indexing_enabled:
            return None
        if not self._feature_flags.is_indexing_enabled(language):
            return None
        return cap.get_best_indexing_method()

    def get_validation_method(self, language: str) -> Optional[ASTAccessMethod]:
        """Get best validation method for a language (respects feature flags)."""
        cap = self.get(language)
        if not cap or not cap.validation_enabled:
            return None
        if not self._feature_flags.is_validation_enabled(language):
            return None
        return cap.get_best_validation_method()

    def list_supported_languages(self, tier: Optional[LanguageTier] = None) -> List[str]:
        """List all supported languages, optionally filtered by tier."""
        languages = list(self._capabilities.keys())
        if tier:
            languages = [lang for lang in languages if self._capabilities[lang].tier == tier]
        return sorted(languages)

    def list_extensions(self) -> Set[str]:
        """List all supported file extensions."""
        return set(self._extension_map.keys())

    def list_filenames(self) -> Set[str]:
        """List all supported special filenames."""
        return set(self._filename_map.keys())

    def get_languages_by_tier(self) -> Dict[LanguageTier, List[str]]:
        """Get languages grouped by tier."""
        result: Dict[LanguageTier, List[str]] = {
            tier: [] for tier in LanguageTier if tier != LanguageTier.UNSUPPORTED
        }
        for lang, cap in self._capabilities.items():
            if cap.tier in result:
                result[cap.tier].append(lang)
        for tier in result:
            result[tier].sort()
        return result

    def _load_default_capabilities(self) -> None:
        """Load default language capabilities.

        Tries to load from YAML configuration first, falls back to
        hardcoded defaults if YAML is not available.
        """
        # Try loading from YAML first
        loaded_from_yaml = self._try_load_from_yaml()

        if not loaded_from_yaml:
            # Fall back to hardcoded defaults
            for cap in _create_default_capabilities():
                self.register(cap)

        logger.debug(
            f"Loaded {len(self._capabilities)} language capabilities: "
            f"Tier1={len(self.list_supported_languages(LanguageTier.TIER_1))}, "
            f"Tier2={len(self.list_supported_languages(LanguageTier.TIER_2))}, "
            f"Tier3={len(self.list_supported_languages(LanguageTier.TIER_3))}"
        )

    def _try_load_from_yaml(self) -> bool:
        """Try to load capabilities from YAML configuration.

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            from .config_loader import load_capabilities_from_yaml

            capabilities = load_capabilities_from_yaml()
            if capabilities:
                for cap in capabilities:
                    self.register(cap)
                logger.info(f"Loaded {len(capabilities)} languages from YAML config")
                return True
            return False

        except Exception as e:
            logger.debug(f"Failed to load from YAML, using defaults: {e}")
            return False

    def reload_from_yaml(self, config_path: Optional[Path] = None) -> int:
        """Reload capabilities from YAML configuration.

        Args:
            config_path: Optional path to capabilities.yaml

        Returns:
            Number of languages loaded
        """
        from .config_loader import load_capabilities_from_yaml

        with self._internal_lock:
            # Clear existing capabilities
            self._capabilities.clear()
            self._extension_map.clear()
            self._filename_map.clear()

            # Load from YAML
            capabilities = load_capabilities_from_yaml(config_path)
            for cap in capabilities:
                self.register(cap)

            logger.info(f"Reloaded {len(capabilities)} languages from YAML")
            return len(capabilities)

    @property
    def feature_flags(self) -> FeatureFlagManager:
        """Get the feature flag manager."""
        return self._feature_flags
