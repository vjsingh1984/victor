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

import logging
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from victor_coding.languages.base import (
    BaseLanguagePlugin,
    BuildSystem,
    CallEdge,
    CommentStyle,
    ConfigurableASTTraverser,
    DocCommentPattern,
    EdgeDetectionResult,
    Formatter,
    LanguageCapabilities,
    LanguageConfig,
    Linter,
    QueryPattern,
    TestRunner,
    TraversalConfig,
    TreeSitterQueries,
)

if TYPE_CHECKING:
    from tree_sitter import Node, Tree

logger = logging.getLogger(__name__)


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
            doc_comment_pattern=DocCommentPattern(
                line_prefixes=["///", "//!"],
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

    def detect_calls_edges(
        self,
        tree: "Tree",
        source_code: str,
        file_path: Path,
    ) -> EdgeDetectionResult:
        """Detect CALLS edges in Rust source code.

        Finds function calls, method calls, and macro calls.

        Args:
            tree: Parsed tree-sitter tree
            source_code: Raw source code text
            file_path: Path to source file

        Returns:
            EdgeDetectionResult with detected calls
        """
        calls: List[CallEdge] = []
        call_nodes = self._find_call_nodes(tree.root_node)

        for call_node, caller_name, caller_line in call_nodes:
            callee_name = self._extract_callee_name(call_node)
            if callee_name and caller_name:
                calls.append(CallEdge(
                    caller_name=caller_name,
                    callee_name=callee_name,
                    caller_line=caller_line,
                ))

        logger.debug(f"Detected {len(calls)} CALLS edges in {file_path.name}")

        return EdgeDetectionResult(
            calls=calls,
            metadata={
                "language": "rust",
                "file": str(file_path),
            },
        )

    def _find_call_nodes(
        self,
        root: "Node",
    ) -> List[tuple["Node", str, Optional[int]]]:
        """Find all call nodes with their enclosing function context.

        Uses the shared ConfigurableASTTraverser to eliminate code duplication.
        """
        config = TraversalConfig(
            function_types=["function_item"],
            class_types=["impl_item"],
            call_types=["call_expression", "macro_invocation"],
            name_field="identifier",
        )
        traverser = ConfigurableASTTraverser(config, self._get_node_text)
        return traverser.find_call_nodes(root)

    def _extract_callee_name(self, call_node: "Node") -> Optional[str]:
        """Extract the name of the called function from a call node.

        Handles:
        - Simple calls: foo()
        - Field access calls: obj.method()
        - Scoped calls: std::mem::drop()
        - Macro calls: println!, vec!
        """
        # For macro invocations
        if call_node.type == "macro_invocation":
            for child in call_node.children:
                if child.type == "identifier":
                    return self._get_node_text(child)
                elif child.type == "scoped_identifier":
                    # std::println -> extract "println"
                    return self._extract_scoped_name(child)
            return None

        # For call expressions
        for child in call_node.children:
            if child.type == "field_expression":
                # obj.method() -> extract "method"
                return self._extract_field_name(child)
            elif child.type == "scoped_identifier":
                # std::mem::drop() -> extract "drop"
                return self._extract_scoped_name(child)
            elif child.type == "identifier":
                # foo() -> extract "foo"
                return self._get_node_text(child)

        return None

    def _extract_field_name(self, field_node: "Node") -> Optional[str]:
        """Extract field name from a field_expression node.

        For obj.method, extracts "method".
        For obj.field1.field2, extracts "field2" (the final field).
        """
        # field_expression: value (identifier or field_expression) . field_ref (field_identifier)
        for child in reversed(field_node.children):
            if child.type == "field_identifier":
                return self._get_node_text(child)
            elif child.type == "field_expression":
                result = self._extract_field_name(child)
                if result:
                    return result

        return None

    def _extract_scoped_name(self, scoped_node: "Node") -> Optional[str]:
        """Extract the final identifier from a scoped identifier.

        For std::mem::drop, extracts "drop".
        For std::collections::HashMap, extracts "HashMap".
        """
        # scoped_identifier: [identifier::] identifier
        # We want the last identifier
        for child in reversed(scoped_node.children):
            if child.type == "identifier":
                return self._get_node_text(child)
            elif child.type == "scoped_identifier":
                result = self._extract_scoped_name(child)
                if result:
                    return result

        return None

    def _get_node_text(self, node: "Node") -> Optional[str]:
        """Get text content of a node."""
        if node is None or not hasattr(node, "text"):
            return None
        text = node.text
        if isinstance(text, bytes):
            return text.decode("utf-8", errors="ignore")
        return text