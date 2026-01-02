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

"""TypeScript language plugin."""

from pathlib import Path
from typing import Optional

from victor.coding.languages.base import (
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


class TypeScriptPlugin(BaseLanguagePlugin):
    """TypeScript language plugin.

    Supports:
    - Testing: jest, vitest
    - Formatting: prettier
    - Linting: eslint, typescript-eslint
    - Type checking: tsc
    """

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="typescript",
            display_name="TypeScript",
            aliases=["ts", "tsx"],
            extensions=[".ts", ".tsx", ".mts", ".cts"],
            filenames=["tsconfig.json"],
            shebangs=["ts-node", "tsx"],
            comment_style=CommentStyle.C_STYLE,
            line_comment="//",
            block_comment_start="/*",
            block_comment_end="*/",
            string_delimiters=['"', "'", "`"],
            indent_size=2,
            use_tabs=False,
            package_managers=["npm", "yarn", "pnpm", "bun"],
            build_systems=["tsc", "webpack", "vite", "esbuild"],
            test_frameworks=["jest", "vitest", "mocha"],
            language_server="typescript-language-server",
            language_server_name="TypeScript Language Server",
            tree_sitter_language="typescript",
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
        """Create tree-sitter queries for TypeScript symbol/call extraction.

        The @def capture is used for end_line (function body boundaries).
        The @name capture is used for the symbol name.
        """
        return TreeSitterQueries(
            symbols=[
                QueryPattern("class", "(class_declaration name: (identifier) @name) @def"),
                QueryPattern("function", "(function_declaration name: (identifier) @name) @def"),
                QueryPattern(
                    "function", "(method_signature name: (property_identifier) @name) @def"
                ),
                QueryPattern(
                    "function", "(method_definition name: (property_identifier) @name) @def"
                ),
                QueryPattern(
                    "function",
                    "(lexical_declaration (variable_declarator name: (identifier) @name value: (arrow_function))) @def",
                ),
                QueryPattern(
                    "function",
                    "(lexical_declaration (variable_declarator name: (identifier) @name value: (function_expression))) @def",
                ),
                QueryPattern(
                    "function",
                    "(assignment_expression left: (identifier) @name right: (arrow_function)) @def",
                ),
            ],
            calls="""
                (call_expression function: (identifier) @callee)
                (call_expression function: (member_expression property: (property_identifier) @callee))
                (call_expression function: (subscript_expression index: (property_identifier) @callee))
                (new_expression constructor: (identifier) @callee)
            """,
            references="""
                (call_expression function: (identifier) @name)
                (call_expression function: (member_expression property: (property_identifier) @name))
                (member_expression property: (property_identifier) @name)
                (new_expression constructor: (identifier) @name)
                (identifier) @name
            """,
            inheritance="""
                (class_declaration
                    name: (identifier) @child
                    heritage: (class_heritage (identifier) @base))
            """,
            implements="""
                (class_declaration
                    name: (identifier) @child
                    (heritage_clause (identifier) @interface))
            """,
            composition="""
                (class_declaration
                    name: (identifier) @owner
                    body: (class_body
                        (field_definition
                            type: (type_annotation (type_identifier) @type))
                        (public_field_definition
                            type: (type_annotation (type_identifier) @type))
                        (method_definition
                            body: (statement_block
                                (expression_statement
                                    (assignment_expression
                                        left: (member_expression object: (this) property: (property_identifier))
                                        right: (new_expression constructor: (identifier) @type)))))))
            """,
            enclosing_scopes=[
                ("function_declaration", "name"),
                ("method_definition", "name"),
                ("method_signature", "name"),
                ("class_declaration", "name"),
            ],
        )

    def get_test_runner(self, project_root: Path) -> Optional[TestRunner]:
        """Get jest or vitest runner."""
        package_json = project_root / "package.json"

        if package_json.exists():
            content = package_json.read_text()

            if '"vitest"' in content:
                return TestRunner(
                    name="vitest",
                    command=["npx", "vitest"],
                    file_pattern="*.test.ts",
                    run_args=["run"],
                    coverage_args=["--coverage"],
                    output_format="json",
                    json_args=["--reporter=json"],
                )

            if '"jest"' in content:
                return TestRunner(
                    name="jest",
                    command=["npx", "jest"],
                    file_pattern="*.test.ts",
                    coverage_args=["--coverage"],
                    output_format="json",
                    json_args=["--json"],
                )

        return TestRunner(
            name="jest",
            command=["npx", "jest"],
            file_pattern="*.test.ts",
        )

    def get_formatter(self, project_root: Path) -> Optional[Formatter]:
        """Get prettier formatter."""
        return Formatter(
            name="prettier",
            command=["npx", "prettier", "--write", "."],
            check_args=["--check"],
            config_file=".prettierrc",
        )

    def get_linter(self, project_root: Path) -> Optional[Linter]:
        """Get eslint linter with TypeScript support."""
        return Linter(
            name="eslint",
            command=["npx", "eslint", "--ext", ".ts,.tsx", "."],
            fix_args=["--fix"],
            config_file=".eslintrc",
        )

    def get_build_system(self, project_root: Path) -> Optional[BuildSystem]:
        """Get TypeScript build system."""
        tsconfig = project_root / "tsconfig.json"

        if not tsconfig.exists():
            return None

        return BuildSystem(
            name="tsc",
            build_command=["npx", "tsc"],
            run_command=["npx", "ts-node"],
            clean_command=["rm", "-rf", "dist"],
            manifest_file="tsconfig.json",
        )
