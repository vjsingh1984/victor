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

"""Smart codebase analyzer for generating comprehensive init.md files.

This package analyzes Python codebases to extract:
- Package structure and layout
- Key classes and their locations (with line numbers)
- Architectural patterns (providers, tools, managers, etc.)
- CLI commands from pyproject.toml
- Configuration files and their purposes

Output location: .victor/init.md (configurable via settings.py)

Decomposed from a single module into focused submodules:
- models: Data structures (ClassInfo, ModuleInfo, CodebaseAnalysis)
- scanner: File discovery, walking, filtering, and parsing
- metrics: Code metrics, complexity, statistics, and analysis
- query: Graph insights, conversation insights, embedding status
- generator: Markdown generation and LLM-powered enhancement
"""

import logging
from pathlib import Path
from typing import List, Optional

# --- Data models ---
from .models import (
    ClassInfo,
    CodebaseAnalysis,
    ModuleInfo,
)

# --- Scanner ---
from .scanner import (
    CONFIG_EXTENSIONS,
    DEFAULT_SKIP_DIRS,
    KEY_CLASS_PATTERNS,
    LANGUAGE_EXTENSIONS,
    CodebaseScanner,
    categorize_class,
    is_hidden_path,
    should_ignore_path,
)

# --- Metrics ---
from .metrics import (
    CodebaseMetrics,
)

# --- Query ---
from .query import (
    _build_analyzer_section,
    _collect_embedding_status,
    extract_conversation_insights,
    extract_graph_insights,
)

# --- Generator ---
from .generator import (
    CONTEXT_FILE_ALIASES,
    VictorMDBuilder,
    _build_quick_start,
    _extract_readme_description,
    _find_config_files,
    _find_docs_files,
    _generate_generic_victor_md,
    _infer_commands,
    _infer_directory_purpose,
    _infer_env_vars,
    _infer_python_requires,
    build_llm_prompt_for_victor_md,
    create_context_symlinks,
    gather_project_context,
    generate_enhanced_init_md,
    generate_smart_victor_md,
    generate_victor_md_from_index,
    generate_victor_md_with_llm,
    remove_context_symlinks,
)

logger = logging.getLogger(__name__)


class CodebaseAnalyzer:
    """Analyzes codebases to extract structure and architecture (language-agnostic).

    Thin facade that delegates to CodebaseScanner (file discovery/parsing)
    and CodebaseMetrics (statistics/analysis).
    """

    # Expose class-level constants for backward compatibility
    KEY_CLASS_PATTERNS = KEY_CLASS_PATTERNS
    LANGUAGE_EXTENSIONS = LANGUAGE_EXTENSIONS
    CONFIG_EXTENSIONS = CONFIG_EXTENSIONS
    SKIP_DIRS = DEFAULT_SKIP_DIRS

    def __init__(
        self,
        root_path: Optional[str] = None,
        include_dirs: Optional[List[str]] = None,
        exclude_dirs: Optional[List[str]] = None,
    ):
        """Initialize analyzer.

        Args:
            root_path: Root directory to analyze. Defaults to current directory.
            include_dirs: List of directories to include in the analysis.
            exclude_dirs: List of directories to exclude from the analysis.
        """
        self.root = Path(root_path).resolve() if root_path else Path.cwd()
        self.analysis = CodebaseAnalysis(project_name=self.root.name, root_path=self.root)
        self.include_dirs = include_dirs

        # Combine default and user-provided exclude dirs
        self.effective_skip_dirs = self.SKIP_DIRS.copy()
        if exclude_dirs:
            self.effective_skip_dirs.update(exclude_dirs)

        # Create delegates
        self._scanner = CodebaseScanner(
            root=self.root,
            effective_skip_dirs=self.effective_skip_dirs,
            include_dirs=self.include_dirs,
        )
        self._metrics = CodebaseMetrics(
            root=self.root,
            effective_skip_dirs=self.effective_skip_dirs,
            include_dirs=self.include_dirs,
        )

    def analyze(self) -> CodebaseAnalysis:
        """Perform full codebase analysis (language-agnostic).

        Returns:
            Complete CodebaseAnalysis object.
        """
        logger.info(f"Analyzing codebase at {self.root}")

        # Step 1: Detect package/source layout (any language)
        self._scanner.detect_package_layout(self.analysis)

        # Step 2: Analyze source files (Python AST or regex for other languages)
        self._scanner.analyze_source_files(self.analysis)

        # Step 3: Identify key components
        self._metrics.identify_key_components(self.analysis)

        # Step 4: Extract entry points from config files
        self._metrics.extract_entry_points(self.analysis)

        # Step 5: Detect architecture patterns
        self._metrics.detect_architecture_patterns(self.analysis)

        # Step 6: Find config files
        self._metrics.find_config_files(self.analysis)

        # Step 7: Extract dependencies from pyproject.toml/package.json
        self._metrics.extract_dependencies(self.analysis)

        # Step 8: Calculate LOC stats
        self._metrics.calculate_loc_stats(self.analysis)

        # Step 9: Extract top imports
        self._metrics.extract_top_imports(self.analysis)

        # Step 10: Try to get test coverage
        self._metrics.extract_test_coverage(self.analysis)

        return self.analysis

    # --- Backward-compatible delegated methods ---

    def _detect_package_layout(self) -> None:
        self._scanner.detect_package_layout(self.analysis)

    def _analyze_source_files(self) -> None:
        self._scanner.analyze_source_files(self.analysis)

    def _identify_key_components(self) -> None:
        self._metrics.identify_key_components(self.analysis)

    def _extract_entry_points(self) -> None:
        self._metrics.extract_entry_points(self.analysis)

    def _detect_architecture_patterns(self) -> None:
        self._metrics.detect_architecture_patterns(self.analysis)

    def _find_config_files(self) -> None:
        self._metrics.find_config_files(self.analysis)

    def _extract_dependencies(self) -> None:
        self._metrics.extract_dependencies(self.analysis)

    def _calculate_loc_stats(self) -> None:
        self._metrics.calculate_loc_stats(self.analysis)

    def _extract_top_imports(self) -> None:
        self._metrics.extract_top_imports(self.analysis)

    def _extract_test_coverage(self) -> None:
        self._metrics.extract_test_coverage(self.analysis)

    def _categorize_class(self, name: str, base_classes: List[str]) -> Optional[str]:
        return categorize_class(name, base_classes)


__all__ = [
    # Facade
    "CodebaseAnalyzer",
    # Data models
    "ClassInfo",
    "ModuleInfo",
    "CodebaseAnalysis",
    # Scanner
    "CodebaseScanner",
    "DEFAULT_SKIP_DIRS",
    "LANGUAGE_EXTENSIONS",
    "CONFIG_EXTENSIONS",
    "KEY_CLASS_PATTERNS",
    "is_hidden_path",
    "should_ignore_path",
    "categorize_class",
    # Metrics
    "CodebaseMetrics",
    # Query
    "extract_conversation_insights",
    "extract_graph_insights",
    "_collect_embedding_status",
    "_build_analyzer_section",
    # Generator
    "VictorMDBuilder",
    "CONTEXT_FILE_ALIASES",
    "generate_smart_victor_md",
    "generate_victor_md_from_index",
    "generate_victor_md_with_llm",
    "generate_enhanced_init_md",
    "gather_project_context",
    "build_llm_prompt_for_victor_md",
    "create_context_symlinks",
    "remove_context_symlinks",
    "_extract_readme_description",
    "_generate_generic_victor_md",
    "_infer_directory_purpose",
    "_infer_python_requires",
    "_infer_commands",
    "_infer_env_vars",
    "_build_quick_start",
    "_find_config_files",
    "_find_docs_files",
]
