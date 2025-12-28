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

"""Rust language plugin."""

from pathlib import Path
from typing import Optional

from victor_coding.languages.base import (
    BaseLanguagePlugin,
    BuildSystem,
    CommentStyle,
    Formatter,
    LanguageCapabilities,
    LanguageConfig,
    Linter,
    QueryPattern,
    TestRunner,
    TreeSitterQueries,
)


class RustPlugin(BaseLanguagePlugin):
    """Rust language plugin.

    Supports:
    - Testing: cargo test
    - Formatting: rustfmt
    - Linting: clippy
    - Building: cargo build
    """

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="rust",
            display_name="Rust",
            aliases=["rs"],
            extensions=[".rs"],
            filenames=["Cargo.toml", "Cargo.lock"],
            shebangs=[],
            comment_style=CommentStyle.C_STYLE,
            line_comment="//",
            block_comment_start="/*",
            block_comment_end="*/",
            string_delimiters=['"'],
            indent_size=4,
            use_tabs=False,
            package_managers=["cargo"],
            build_systems=["cargo"],
            test_frameworks=["cargo test"],
            language_server="rust-analyzer",
            language_server_name="rust-analyzer",
            tree_sitter_language="rust",
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
        """Create tree-sitter queries for Rust symbol/call extraction."""
        return TreeSitterQueries(
            symbols=[
                QueryPattern("class", "(struct_item name: (type_identifier) @name)"),
                QueryPattern("class", "(enum_item name: (type_identifier) @name)"),
                QueryPattern("class", "(trait_item name: (type_identifier) @name)"),
                QueryPattern("function", "(function_item name: (identifier) @name)"),
                QueryPattern("function", "(impl_item (function_item name: (identifier) @name))"),
            ],
            calls="""
                (call_expression function: (identifier) @callee)
                (call_expression function: (field_expression field: (field_identifier) @callee))
                (call_expression function: (scoped_identifier name: (identifier) @callee))
            """,
            references="""
                (call_expression function: (identifier) @name)
                (call_expression function: (field_expression field: (field_identifier) @name))
                (call_expression function: (scoped_identifier name: (identifier) @name))
                (identifier) @name
                (type_identifier) @name
            """,
            implements="""
                (impl_item
                    trait: (type_identifier) @interface
                    type: (type_identifier) @child)
            """,
            composition="""
                (struct_item
                    name: (type_identifier) @owner
                    body: (field_declaration_list
                        (field_declaration
                            type: (type_identifier) @type)))
            """,
            enclosing_scopes=[
                ("function_item", "name"),
                ("impl_item", "type"),
            ],
        )

    def get_test_runner(self, project_root: Path) -> Optional[TestRunner]:
        """Get cargo test runner."""
        cargo_toml = project_root / "Cargo.toml"

        if not cargo_toml.exists():
            return None

        return TestRunner(
            name="cargo test",
            command=["cargo", "test"],
            file_pattern="*_test.rs",
            discover_args=["--no-run"],
            run_args=["--", "--nocapture"],
            coverage_args=["--", "--show-output"],  # Use cargo-tarpaulin for real coverage
            parallel_args=["--", "--test-threads=auto"],
            output_format="text",
        )

    def get_formatter(self, project_root: Path) -> Optional[Formatter]:
        """Get rustfmt formatter."""
        return Formatter(
            name="rustfmt",
            command=["cargo", "fmt"],
            check_args=["--check"],
            config_file="rustfmt.toml",
        )

    def get_linter(self, project_root: Path) -> Optional[Linter]:
        """Get clippy linter."""
        return Linter(
            name="clippy",
            command=["cargo", "clippy"],
            fix_args=["--fix", "--allow-dirty"],
            output_format="text",
        )

    def get_build_system(self, project_root: Path) -> Optional[BuildSystem]:
        """Get cargo build system."""
        cargo_toml = project_root / "Cargo.toml"

        if not cargo_toml.exists():
            return None

        return BuildSystem(
            name="cargo",
            build_command=["cargo", "build"],
            run_command=["cargo", "run"],
            clean_command=["cargo", "clean"],
            install_command=["cargo", "install", "--path", "."],
            debug_args=[],
            release_args=["--release"],
            manifest_file="Cargo.toml",
        )
