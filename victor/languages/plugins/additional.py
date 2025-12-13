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

"""Additional language plugins for less common but supported languages.

These plugins support embedding-based indexing for:
- C (clang, gcc)
- Kotlin (kotlinc, gradle)
- C# (dotnet, msbuild)
- Ruby (bundler, rspec)
- PHP (composer, phpunit)
- Swift (swift build, xcode)
- Scala (sbt, mill)
- Bash/Shell scripts
- SQL
- HTML/CSS/SCSS
- Lua
- Elixir
- Haskell
- R
- Markdown/reStructuredText
"""

from pathlib import Path
from typing import Optional

from victor.languages.base import (
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


class CPlugin(BaseLanguagePlugin):
    """C language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="c",
            display_name="C",
            aliases=["clang"],
            extensions=[".c", ".h"],
            filenames=["Makefile", "CMakeLists.txt"],
            shebangs=[],
            comment_style=CommentStyle.C_STYLE,
            line_comment="//",
            block_comment_start="/*",
            block_comment_end="*/",
            string_delimiters=['"'],
            indent_size=4,
            use_tabs=False,
            package_managers=[],
            build_systems=["make", "cmake", "meson"],
            test_frameworks=["unity", "cunit"],
            language_server="clangd",
            language_server_name="clangd",
            tree_sitter_language="c",
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
        return TreeSitterQueries(
            symbols=[
                QueryPattern(
                    "function",
                    "(function_definition declarator: (function_declarator declarator: (identifier) @name))",
                ),
                QueryPattern("class", "(struct_specifier name: (type_identifier) @name)"),
                QueryPattern("class", "(enum_specifier name: (type_identifier) @name)"),
            ],
            calls="""
                (call_expression function: (identifier) @callee)
                (call_expression function: (field_expression field: (field_identifier) @callee))
            """,
            references="""
                (call_expression function: (identifier) @name)
                (identifier) @name
            """,
            enclosing_scopes=[
                ("function_definition", "declarator"),
            ],
        )


class KotlinPlugin(BaseLanguagePlugin):
    """Kotlin language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="kotlin",
            display_name="Kotlin",
            aliases=["kt", "kts"],
            extensions=[".kt", ".kts"],
            filenames=["build.gradle.kts", "settings.gradle.kts"],
            shebangs=[],
            comment_style=CommentStyle.C_STYLE,
            line_comment="//",
            block_comment_start="/*",
            block_comment_end="*/",
            string_delimiters=['"', '"""'],
            indent_size=4,
            use_tabs=False,
            package_managers=["gradle", "maven"],
            build_systems=["gradle", "maven"],
            test_frameworks=["junit", "kotest"],
            language_server="kotlin-language-server",
            language_server_name="Kotlin Language Server",
            tree_sitter_language="kotlin",
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
        return TreeSitterQueries(
            symbols=[
                QueryPattern("class", "(class_declaration (type_identifier) @name)"),
                QueryPattern("function", "(function_declaration (simple_identifier) @name)"),
            ],
            calls="""
                (call_expression (simple_identifier) @callee)
                (call_expression (navigation_expression (simple_identifier) @callee))
            """,
            references="""
                (simple_identifier) @name
            """,
            inheritance="""
                (class_declaration
                    (type_identifier) @child
                    (delegation_specifier (user_type (type_identifier) @base)))
            """,
            implements="""
                (class_declaration
                    (type_identifier) @child
                    (delegation_specifier (user_type (type_identifier) @interface)))
            """,
            composition="""
                (class_declaration
                    (type_identifier) @owner
                    (class_body
                        (property_declaration
                            (variable_declaration (type_identifier) @type))))
            """,
            enclosing_scopes=[
                ("function_declaration", "name"),
                ("class_declaration", "name"),
            ],
        )


class CSharpPlugin(BaseLanguagePlugin):
    """C# language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="csharp",
            display_name="C#",
            aliases=["cs", "c_sharp"],
            extensions=[".cs"],
            filenames=["*.csproj", "*.sln"],
            shebangs=[],
            comment_style=CommentStyle.C_STYLE,
            line_comment="//",
            block_comment_start="/*",
            block_comment_end="*/",
            string_delimiters=['"', '@"', '"""'],
            indent_size=4,
            use_tabs=False,
            package_managers=["nuget"],
            build_systems=["dotnet", "msbuild"],
            test_frameworks=["xunit", "nunit", "mstest"],
            language_server="omnisharp",
            language_server_name="OmniSharp",
            tree_sitter_language="c_sharp",
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
        return TreeSitterQueries(
            symbols=[
                QueryPattern("class", "(class_declaration name: (identifier) @name)"),
                QueryPattern("class", "(interface_declaration name: (identifier) @name)"),
                QueryPattern("function", "(method_declaration name: (identifier) @name)"),
            ],
            calls="""
                (invocation_expression (identifier) @callee)
                (invocation_expression (member_access_expression name: (identifier) @callee))
                (object_creation_expression type: (identifier) @callee)
            """,
            references="""
                (identifier) @name
            """,
            inheritance="""
                (class_declaration
                    name: (identifier) @child
                    bases: (base_list (identifier) @base))
            """,
            implements="""
                (class_declaration
                    name: (identifier) @child
                    bases: (base_list (identifier) @interface))
            """,
            composition="""
                (class_declaration
                    name: (identifier) @owner
                    body: (declaration_list
                        (field_declaration
                            type: (identifier) @type)))
            """,
            enclosing_scopes=[
                ("method_declaration", "name"),
                ("class_declaration", "name"),
            ],
        )


class RubyPlugin(BaseLanguagePlugin):
    """Ruby language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="ruby",
            display_name="Ruby",
            aliases=["rb"],
            extensions=[".rb", ".rake", ".gemspec"],
            filenames=["Gemfile", "Rakefile", ".ruby-version"],
            shebangs=["ruby"],
            comment_style=CommentStyle.HASH,
            line_comment="#",
            block_comment_start="=begin",
            block_comment_end="=end",
            string_delimiters=['"', "'", '"""'],
            indent_size=2,
            use_tabs=False,
            package_managers=["bundler", "gem"],
            build_systems=["rake"],
            test_frameworks=["rspec", "minitest"],
            language_server="solargraph",
            language_server_name="Solargraph",
            tree_sitter_language="ruby",
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=True,
            supports_type_checking=False,  # Ruby is dynamic
            supports_rename=True,
            supports_extract_function=True,
            supports_inline=True,
            supports_organize_imports=False,
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
        return TreeSitterQueries(
            symbols=[
                QueryPattern("class", "(class name: (constant) @name)"),
                QueryPattern("class", "(module name: (constant) @name)"),
                QueryPattern("function", "(method name: (identifier) @name)"),
                QueryPattern("function", "(singleton_method name: (identifier) @name)"),
            ],
            calls="""
                (call method: (identifier) @callee)
                (call receiver: (identifier) method: (identifier) @callee)
            """,
            references="""
                (identifier) @name
                (constant) @name
            """,
            inheritance="""
                (class
                    name: (constant) @child
                    superclass: (superclass (constant) @base))
            """,
            implements="""
                (class
                    name: (constant) @child
                    (call method: (identifier) @_include (#eq? @_include "include")
                        arguments: (argument_list (constant) @interface)))
            """,
            enclosing_scopes=[
                ("method", "name"),
                ("class", "name"),
                ("module", "name"),
            ],
        )


class PhpPlugin(BaseLanguagePlugin):
    """PHP language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="php",
            display_name="PHP",
            aliases=[],
            extensions=[".php", ".phtml", ".php5", ".php7"],
            filenames=["composer.json", "composer.lock"],
            shebangs=["php"],
            comment_style=CommentStyle.C_STYLE,
            line_comment="//",
            block_comment_start="/*",
            block_comment_end="*/",
            string_delimiters=['"', "'"],
            indent_size=4,
            use_tabs=False,
            package_managers=["composer"],
            build_systems=[],
            test_frameworks=["phpunit", "pest"],
            language_server="intelephense",
            language_server_name="Intelephense",
            tree_sitter_language="php",
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
        return TreeSitterQueries(
            symbols=[
                QueryPattern("class", "(class_declaration name: (name) @name)"),
                QueryPattern("class", "(interface_declaration name: (name) @name)"),
                QueryPattern("class", "(trait_declaration name: (name) @name)"),
                QueryPattern("function", "(function_definition name: (name) @name)"),
                QueryPattern("function", "(method_declaration name: (name) @name)"),
            ],
            calls="""
                (function_call_expression function: (name) @callee)
                (member_call_expression name: (name) @callee)
                (object_creation_expression (name) @callee)
            """,
            references="""
                (name) @name
            """,
            inheritance="""
                (class_declaration
                    name: (name) @child
                    (base_clause (name) @base))
            """,
            implements="""
                (class_declaration
                    name: (name) @child
                    (class_interface_clause (name) @interface))
            """,
            composition="""
                (class_declaration
                    name: (name) @owner
                    (declaration_list
                        (property_declaration
                            type: (type_list (named_type (name) @type)))))
            """,
            enclosing_scopes=[
                ("function_definition", "name"),
                ("method_declaration", "name"),
                ("class_declaration", "name"),
            ],
        )


class SwiftPlugin(BaseLanguagePlugin):
    """Swift language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="swift",
            display_name="Swift",
            aliases=[],
            extensions=[".swift"],
            filenames=["Package.swift"],
            shebangs=[],
            comment_style=CommentStyle.C_STYLE,
            line_comment="//",
            block_comment_start="/*",
            block_comment_end="*/",
            string_delimiters=['"', '"""'],
            indent_size=4,
            use_tabs=False,
            package_managers=["swift package manager"],
            build_systems=["swift build", "xcodebuild"],
            test_frameworks=["xctest"],
            language_server="sourcekit-lsp",
            language_server_name="SourceKit-LSP",
            tree_sitter_language="swift",
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
        return TreeSitterQueries(
            symbols=[
                QueryPattern("class", "(class_declaration name: (type_identifier) @name)"),
                QueryPattern("class", "(struct_declaration name: (type_identifier) @name)"),
                QueryPattern("class", "(protocol_declaration name: (type_identifier) @name)"),
                QueryPattern("function", "(function_declaration name: (simple_identifier) @name)"),
            ],
            calls="""
                (call_expression (simple_identifier) @callee)
            """,
            references="""
                (simple_identifier) @name
                (type_identifier) @name
            """,
            inheritance="""
                (class_declaration
                    name: (type_identifier) @child
                    (type_inheritance_clause (type_identifier) @base))
            """,
            implements="""
                (class_declaration
                    name: (type_identifier) @child
                    (type_inheritance_clause
                        (type_identifier)
                        (type_identifier) @interface))
            """,
            composition="""
                (class_declaration
                    name: (type_identifier) @owner
                    (class_body
                        (property_declaration
                            (pattern (simple_identifier))
                            (type_annotation (type_identifier) @type))))
            """,
            enclosing_scopes=[
                ("function_declaration", "name"),
                ("class_declaration", "name"),
            ],
        )


class ScalaPlugin(BaseLanguagePlugin):
    """Scala language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="scala",
            display_name="Scala",
            aliases=[],
            extensions=[".scala", ".sc"],
            filenames=["build.sbt"],
            shebangs=[],
            comment_style=CommentStyle.C_STYLE,
            line_comment="//",
            block_comment_start="/*",
            block_comment_end="*/",
            string_delimiters=['"', '"""'],
            indent_size=2,
            use_tabs=False,
            package_managers=["sbt", "mill"],
            build_systems=["sbt", "mill"],
            test_frameworks=["scalatest", "specs2"],
            language_server="metals",
            language_server_name="Metals",
            tree_sitter_language="scala",
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
        return TreeSitterQueries(
            symbols=[
                QueryPattern("class", "(class_definition name: (identifier) @name)"),
                QueryPattern("class", "(object_definition name: (identifier) @name)"),
                QueryPattern("class", "(trait_definition name: (identifier) @name)"),
                QueryPattern("function", "(function_definition name: (identifier) @name)"),
            ],
            calls="""
                (call_expression function: (identifier) @callee)
            """,
            references="""
                (identifier) @name
            """,
            inheritance="""
                (class_definition
                    name: (identifier) @child
                    extends_clause: (extends_clause (type_identifier) @base))
            """,
            implements="""
                (class_definition
                    name: (identifier) @child
                    extends_clause: (extends_clause
                        (type_identifier)
                        (type_identifier) @interface))
            """,
            composition="""
                (class_definition
                    name: (identifier) @owner
                    body: (template_body
                        (val_definition
                            pattern: (identifier)
                            type: (type_identifier) @type)))
            """,
            enclosing_scopes=[
                ("function_definition", "name"),
                ("class_definition", "name"),
            ],
        )


class BashPlugin(BaseLanguagePlugin):
    """Bash/Shell script plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="bash",
            display_name="Bash",
            aliases=["sh", "shell", "zsh"],
            extensions=[".sh", ".bash", ".zsh", ".ksh"],
            filenames=[".bashrc", ".zshrc", ".profile", ".bash_profile"],
            shebangs=["bash", "sh", "zsh"],
            comment_style=CommentStyle.HASH,
            line_comment="#",
            block_comment_start="",
            block_comment_end="",
            string_delimiters=['"', "'"],
            indent_size=2,
            use_tabs=False,
            package_managers=[],
            build_systems=[],
            test_frameworks=["bats"],
            language_server="bash-language-server",
            language_server_name="Bash Language Server",
            tree_sitter_language="bash",
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=False,
            supports_type_checking=False,
            supports_rename=False,
            supports_extract_function=False,
            supports_inline=False,
            supports_organize_imports=False,
            supports_test_discovery=True,
            supports_test_execution=True,
            supports_coverage=False,
            supports_debugging=True,
            supports_breakpoints=True,
            supports_step_debugging=True,
            supports_formatting=True,
            supports_linting=True,
            supports_completion=True,
        )

    def _create_tree_sitter_queries(self) -> TreeSitterQueries:
        return TreeSitterQueries(
            symbols=[
                QueryPattern("function", "(function_definition name: (word) @name)"),
            ],
            calls="""
                (command name: (command_name (word) @callee))
            """,
            references="""
                (word) @name
                (variable_name) @name
            """,
            enclosing_scopes=[
                ("function_definition", "name"),
            ],
        )


class SqlPlugin(BaseLanguagePlugin):
    """SQL language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="sql",
            display_name="SQL",
            aliases=["mysql", "postgresql", "sqlite"],
            extensions=[".sql"],
            filenames=[],
            shebangs=[],
            comment_style=CommentStyle.DOUBLE_DASH,
            line_comment="--",
            block_comment_start="/*",
            block_comment_end="*/",
            string_delimiters=["'", '"'],
            indent_size=2,
            use_tabs=False,
            package_managers=[],
            build_systems=[],
            test_frameworks=[],
            language_server="sql-language-server",
            language_server_name="SQL Language Server",
            tree_sitter_language="sql",
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=False,
            supports_type_checking=False,
            supports_rename=False,
            supports_extract_function=False,
            supports_inline=False,
            supports_organize_imports=False,
            supports_test_discovery=False,
            supports_test_execution=False,
            supports_coverage=False,
            supports_debugging=False,
            supports_breakpoints=False,
            supports_step_debugging=False,
            supports_formatting=True,
            supports_linting=True,
            supports_completion=True,
        )

    def _create_tree_sitter_queries(self) -> TreeSitterQueries:
        return TreeSitterQueries(
            symbols=[
                QueryPattern("function", "(create_function_statement name: (identifier) @name)"),
                QueryPattern("class", "(create_table_statement name: (identifier) @name)"),
            ],
            enclosing_scopes=[],
        )


class HtmlPlugin(BaseLanguagePlugin):
    """HTML language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="html",
            display_name="HTML",
            aliases=["htm"],
            extensions=[".html", ".htm", ".xhtml"],
            filenames=["index.html"],
            shebangs=[],
            comment_style=CommentStyle.HTML,
            line_comment="",
            block_comment_start="<!--",
            block_comment_end="-->",
            string_delimiters=['"', "'"],
            indent_size=2,
            use_tabs=False,
            package_managers=[],
            build_systems=[],
            test_frameworks=[],
            language_server="html-languageserver",
            language_server_name="HTML Language Server",
            tree_sitter_language="html",
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=False,
            supports_type_checking=False,
            supports_rename=False,
            supports_extract_function=False,
            supports_inline=False,
            supports_organize_imports=False,
            supports_test_discovery=False,
            supports_test_execution=False,
            supports_coverage=False,
            supports_debugging=False,
            supports_breakpoints=False,
            supports_step_debugging=False,
            supports_formatting=True,
            supports_linting=True,
            supports_completion=True,
        )

    def _create_tree_sitter_queries(self) -> TreeSitterQueries:
        return TreeSitterQueries()


class CssPlugin(BaseLanguagePlugin):
    """CSS language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="css",
            display_name="CSS",
            aliases=["scss", "less", "sass"],
            extensions=[".css", ".scss", ".less", ".sass"],
            filenames=[],
            shebangs=[],
            comment_style=CommentStyle.C_STYLE,
            line_comment="//",  # Only for SCSS/Less
            block_comment_start="/*",
            block_comment_end="*/",
            string_delimiters=['"', "'"],
            indent_size=2,
            use_tabs=False,
            package_managers=["npm"],
            build_systems=["sass", "less"],
            test_frameworks=[],
            language_server="css-languageserver",
            language_server_name="CSS Language Server",
            tree_sitter_language="css",
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=False,
            supports_type_checking=False,
            supports_rename=False,
            supports_extract_function=False,
            supports_inline=False,
            supports_organize_imports=False,
            supports_test_discovery=False,
            supports_test_execution=False,
            supports_coverage=False,
            supports_debugging=False,
            supports_breakpoints=False,
            supports_step_debugging=False,
            supports_formatting=True,
            supports_linting=True,
            supports_completion=True,
        )

    def _create_tree_sitter_queries(self) -> TreeSitterQueries:
        return TreeSitterQueries()


class LuaPlugin(BaseLanguagePlugin):
    """Lua language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="lua",
            display_name="Lua",
            aliases=[],
            extensions=[".lua"],
            filenames=[],
            shebangs=["lua"],
            comment_style=CommentStyle.DOUBLE_DASH,
            line_comment="--",
            block_comment_start="--[[",
            block_comment_end="]]",
            string_delimiters=['"', "'", "[[", "]]"],
            indent_size=2,
            use_tabs=False,
            package_managers=["luarocks"],
            build_systems=[],
            test_frameworks=["busted"],
            language_server="lua-language-server",
            language_server_name="Lua Language Server",
            tree_sitter_language="lua",
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=True,
            supports_type_checking=False,
            supports_rename=True,
            supports_extract_function=False,
            supports_inline=False,
            supports_organize_imports=False,
            supports_test_discovery=True,
            supports_test_execution=True,
            supports_coverage=False,
            supports_debugging=True,
            supports_breakpoints=True,
            supports_step_debugging=True,
            supports_formatting=True,
            supports_linting=True,
            supports_completion=True,
        )

    def _create_tree_sitter_queries(self) -> TreeSitterQueries:
        return TreeSitterQueries(
            symbols=[
                QueryPattern("function", "(function_declaration name: (identifier) @name)"),
                QueryPattern("function", "(local_function_declaration name: (identifier) @name)"),
            ],
            calls="""
                (function_call name: (identifier) @callee)
            """,
            references="""
                (identifier) @name
            """,
            enclosing_scopes=[
                ("function_declaration", "name"),
                ("local_function_declaration", "name"),
            ],
        )


class ElixirPlugin(BaseLanguagePlugin):
    """Elixir language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="elixir",
            display_name="Elixir",
            aliases=["ex", "exs"],
            extensions=[".ex", ".exs"],
            filenames=["mix.exs"],
            shebangs=["elixir"],
            comment_style=CommentStyle.HASH,
            line_comment="#",
            block_comment_start="",
            block_comment_end="",
            string_delimiters=['"', '"""'],
            indent_size=2,
            use_tabs=False,
            package_managers=["mix", "hex"],
            build_systems=["mix"],
            test_frameworks=["exunit"],
            language_server="elixir-ls",
            language_server_name="ElixirLS",
            tree_sitter_language="elixir",
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=True,
            supports_type_checking=True,  # Via Dialyzer
            supports_rename=True,
            supports_extract_function=True,
            supports_inline=False,
            supports_organize_imports=False,
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
        return TreeSitterQueries(
            symbols=[
                QueryPattern(
                    "class",
                    '(call target: (identifier) @_defmodule (#eq? @_defmodule "defmodule") (arguments (alias) @name))',
                ),
                QueryPattern(
                    "function",
                    '(call target: (identifier) @_def (#match? @_def "^def") (arguments (call target: (identifier) @name)))',
                ),
            ],
            calls="""
                (call target: (identifier) @callee)
            """,
            references="""
                (identifier) @name
                (alias) @name
            """,
            enclosing_scopes=[],
        )


class HaskellPlugin(BaseLanguagePlugin):
    """Haskell language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="haskell",
            display_name="Haskell",
            aliases=["hs"],
            extensions=[".hs", ".lhs"],
            filenames=["stack.yaml", "package.yaml", "*.cabal"],
            shebangs=["runhaskell"],
            comment_style=CommentStyle.DOUBLE_DASH,
            line_comment="--",
            block_comment_start="{-",
            block_comment_end="-}",
            string_delimiters=['"'],
            indent_size=2,
            use_tabs=False,
            package_managers=["cabal", "stack"],
            build_systems=["cabal", "stack"],
            test_frameworks=["hspec", "quickcheck"],
            language_server="haskell-language-server",
            language_server_name="Haskell Language Server",
            tree_sitter_language="haskell",
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
        return TreeSitterQueries(
            symbols=[
                QueryPattern("function", "(function name: (variable) @name)"),
                QueryPattern("class", "(type_alias name: (type) @name)"),
                QueryPattern("class", "(newtype name: (type) @name)"),
                QueryPattern("class", "(data name: (type) @name)"),
            ],
            calls="""
                (exp_apply (exp_name (variable) @callee))
            """,
            references="""
                (variable) @name
                (constructor) @name
            """,
            enclosing_scopes=[
                ("function", "name"),
            ],
        )


class RPlugin(BaseLanguagePlugin):
    """R language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="r",
            display_name="R",
            aliases=["rlang"],
            extensions=[".r", ".R", ".Rmd"],
            filenames=["DESCRIPTION", "NAMESPACE"],
            shebangs=["Rscript"],
            comment_style=CommentStyle.HASH,
            line_comment="#",
            block_comment_start="",
            block_comment_end="",
            string_delimiters=['"', "'"],
            indent_size=2,
            use_tabs=False,
            package_managers=["cran", "devtools"],
            build_systems=[],
            test_frameworks=["testthat"],
            language_server="languageserver",
            language_server_name="R Language Server",
            tree_sitter_language="r",
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=True,
            supports_semantic_analysis=True,
            supports_type_checking=False,
            supports_rename=True,
            supports_extract_function=False,
            supports_inline=False,
            supports_organize_imports=False,
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
        return TreeSitterQueries(
            symbols=[
                QueryPattern(
                    "function",
                    "(binary_operator lhs: (identifier) @name rhs: (function_definition))",
                ),
            ],
            calls="""
                (call function: (identifier) @callee)
            """,
            references="""
                (identifier) @name
            """,
            enclosing_scopes=[],
        )


class MarkdownPlugin(BaseLanguagePlugin):
    """Markdown documentation plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="markdown",
            display_name="Markdown",
            aliases=["md"],
            extensions=[".md", ".markdown", ".mdown"],
            filenames=["README.md", "CHANGELOG.md", "CONTRIBUTING.md"],
            shebangs=[],
            comment_style=CommentStyle.HTML,
            line_comment="",
            block_comment_start="<!--",
            block_comment_end="-->",
            string_delimiters=[],
            indent_size=2,
            use_tabs=False,
            package_managers=[],
            build_systems=[],
            test_frameworks=[],
            language_server="marksman",
            language_server_name="Marksman",
            tree_sitter_language=None,  # No tree-sitter for markdown in our setup
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=False,
            supports_semantic_analysis=False,
            supports_type_checking=False,
            supports_rename=False,
            supports_extract_function=False,
            supports_inline=False,
            supports_organize_imports=False,
            supports_test_discovery=False,
            supports_test_execution=False,
            supports_coverage=False,
            supports_debugging=False,
            supports_breakpoints=False,
            supports_step_debugging=False,
            supports_formatting=True,
            supports_linting=True,
            supports_completion=True,
        )

    def _create_tree_sitter_queries(self) -> TreeSitterQueries:
        return TreeSitterQueries()


class XmlPlugin(BaseLanguagePlugin):
    """XML language plugin."""

    def _create_config(self) -> LanguageConfig:
        return LanguageConfig(
            name="xml",
            display_name="XML",
            aliases=["xsl", "xslt"],
            extensions=[".xml", ".xsl", ".xslt", ".xsd", ".wsdl", ".svg"],
            filenames=["pom.xml", "web.xml", "AndroidManifest.xml"],
            shebangs=[],
            comment_style=CommentStyle.HTML,
            line_comment="",
            block_comment_start="<!--",
            block_comment_end="-->",
            string_delimiters=['"', "'"],
            indent_size=2,
            use_tabs=False,
            package_managers=[],
            build_systems=[],
            test_frameworks=[],
            language_server="lemminx",
            language_server_name="LemMinX",
            tree_sitter_language=None,
        )

    def _create_capabilities(self) -> LanguageCapabilities:
        return LanguageCapabilities(
            supports_syntax_analysis=False,
            supports_semantic_analysis=False,
            supports_type_checking=False,
            supports_rename=False,
            supports_extract_function=False,
            supports_inline=False,
            supports_organize_imports=False,
            supports_test_discovery=False,
            supports_test_execution=False,
            supports_coverage=False,
            supports_debugging=False,
            supports_breakpoints=False,
            supports_step_debugging=False,
            supports_formatting=True,
            supports_linting=True,
            supports_completion=True,
        )

    def _create_tree_sitter_queries(self) -> TreeSitterQueries:
        return TreeSitterQueries()
