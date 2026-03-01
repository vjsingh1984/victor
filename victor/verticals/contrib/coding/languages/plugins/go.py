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

"""Go language plugin."""

from pathlib import Path
from typing import Optional

from victor.verticals.contrib.coding.languages.base import (
    BaseLanguagePlugin,
    BuildSystem,
    CommentStyle,
    DocCommentPattern,
    Formatter,
    LanguageCapabilities,
    LanguageConfig,
    Linter,
    QueryPattern,
    TestRunner,
    TreeSitterQueries,
)


class GoPlugin(BaseLanguagePlugin):
    """Go language plugin.

    Supports:
    - Testing: go test
    - Formatting: gofmt, goimports
    - Linting: golangci-lint, staticcheck
    - Building: go build
    """

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="go",
            display_name="Go",
            aliases=["golang"],
            extensions=[".go"],
            filenames=["go.mod", "go.sum"],
            shebangs=[],
            comment_style=CommentStyle.C_STYLE,
            line_comment="//",
            block_comment_start="/*",
            block_comment_end="*/",
            string_delimiters=['"', "`"],
            indent_size=4,
            use_tabs=True,  # Go convention
            package_managers=["go mod"],
            build_systems=["go build"],
            test_frameworks=["go test"],
            language_server="gopls",
            language_server_name="gopls",
            tree_sitter_language="go",
            doc_comment_pattern=DocCommentPattern(
                line_prefixes=["//"],
            ),
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=True,
            supports_type_checking=True,
            supports_rename=True,
            supports_extract_function=True,
            supports_inline=True,
            supports_organize_imports=True,
            supports_test_discovery=True,
            supports_test_execution=True,
            supports_coverage=True,
            supports_debugging=True,
            supports_breakpoints=True,
            supports_step_debugging=True,
            supports_formatting=True,
            supports_linting=True,
            supports_completion=True,
        )

    def _create_tree_sitter_queries(self) -> TreeSitterQueries:
        """Create tree-sitter queries for Go symbol/call extraction."""
        return TreeSitterQueries(
            symbols=[
                QueryPattern("function", "(function_declaration name: (identifier) @name)"),
                QueryPattern("function", "(method_declaration name: (field_identifier) @name)"),
                QueryPattern(
                    "class", "(type_declaration (type_spec name: (type_identifier) @name))"
                ),
            ],
            calls="""
                (call_expression function: (identifier) @callee)
                (call_expression function: (selector_expression field: (field_identifier) @callee))
                (type_conversion_expression type: (type_identifier) @callee)
            """,
            references="""
                (call_expression function: (identifier) @name)
                (call_expression function: (selector_expression field: (field_identifier) @name))
                (selector_expression field: (field_identifier) @name)
                (identifier) @name
            """,
            composition="""
                (type_declaration
                    (type_spec
                        name: (type_identifier) @owner
                        type: (struct_type
                            (field_declaration
                                type: (type_identifier) @type))))
            """,
            enclosing_scopes=[
                ("function_declaration", "name"),
                ("method_declaration", "name"),
            ],
        )

    def get_test_runner(self, project_root: Path) -> Optional[TestRunner]:
        """Get go test runner."""
        go_mod = project_root / "go.mod"

        if not go_mod.exists():
            return None

        return TestRunner(
            name="go test",
            command=["go", "test"],
            file_pattern="*_test.go",
            discover_args=["-list", "."],
            run_args=["-v", "./..."],
            coverage_args=["-cover", "-coverprofile=coverage.out"],
            parallel_args=["-parallel", "auto"],
            output_format="json",
            json_args=["-json"],
        )

    def get_formatter(self, project_root: Path) -> Optional[Formatter]:
        """Get gofmt formatter."""
        return Formatter(
            name="gofmt",
            command=["gofmt", "-w", "."],
            check_args=["-d"],
        )

    def get_linter(self, project_root: Path) -> Optional[Linter]:
        """Get golangci-lint linter."""
        return Linter(
            name="golangci-lint",
            command=["golangci-lint", "run"],
            fix_args=["--fix"],
            config_file=".golangci.yml",
            output_format="json",
        )

    def get_build_system(self, project_root: Path) -> Optional[BuildSystem]:
        """Get go build system."""
        go_mod = project_root / "go.mod"

        if not go_mod.exists():
            return None

        return BuildSystem(
            name="go",
            build_command=["go", "build", "./..."],
            run_command=["go", "run", "."],
            clean_command=["go", "clean"],
            install_command=["go", "install", "."],
            manifest_file="go.mod",
        )
