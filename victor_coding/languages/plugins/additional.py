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

import logging
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from victor_coding.languages.base import (
    BaseLanguagePlugin,
    BuildSystem,
    CallEdge,
    CommentStyle,
    DocCommentPattern,
    EdgeDetectionResult,
    Formatter,
    LanguageCapabilities,
    LanguageConfig,
    Linter,
    QueryPattern,
    TestRunner,
    TreeSitterQueries,
)

if TYPE_CHECKING:
    from tree_sitter import Node, Tree

logger = logging.getLogger(__name__)


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
        """Create tree-sitter queries for C/C++ symbol extraction.

        The @def capture is used for end_line (function body boundaries).
        """
        return TreeSitterQueries(
            symbols=[
                QueryPattern(
                    "function",
                    "(function_definition declarator: (function_declarator declarator: (identifier) @name)) @def",
                ),
                QueryPattern("class", "(struct_specifier name: (type_identifier) @name) @def"),
                QueryPattern("class", "(enum_specifier name: (type_identifier) @name) @def"),
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

    def detect_calls_edges(
        self,
        tree: "Tree",
        source_code: str,
        file_path: Path,
    ) -> EdgeDetectionResult:
        """Detect CALLS edges in C source code.

        Finds function calls and method calls through pointers.
        Handles C-specific features like function pointers and struct methods.

        Args:
            tree: Parsed tree-sitter tree
            source_code: Raw source code text
            file_path: Path to source file

        Returns:
            EdgeDetectionResult with detected calls
        """
        calls: List[CallEdge] = []
        call_nodes = self._find_call_nodes_c(tree.root_node)

        for call_node, caller_name, caller_line in call_nodes:
            callee_name = self._extract_callee_name_c(call_node)
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
                "language": "c",
                "file": str(file_path),
            },
        )

    def _find_call_nodes_c(
        self,
        root: "Node",
    ) -> List[tuple["Node", str, Optional[int]]]:
        """Find all call nodes with their enclosing function context.

        Args:
            root: Tree-sitter root node

        Returns:
            List of (call_node, caller_name, caller_line) tuples
        """
        results: List[tuple["Node", str, Optional[int]]] = []

        def traverse(node: "Node", enclosing_function: Optional[str] = None) -> None:
            # Check if this is a function definition
            if node.type == "function_definition":
                func_name = self._extract_function_name_c(node)
                if not func_name:
                    func_name = enclosing_function

                # Process children with new enclosing context
                for child in node.children:
                    traverse(child, func_name)
                return

            # Check if this is a struct specifier
            if node.type == "struct_specifier":
                struct_name = None
                for child in node.children:
                    if child.type == "type_identifier":
                        struct_name = self._get_node_text(child)
                        break

                # Process struct body
                for child in node.children:
                    if child.type == "field_declaration_list":
                        for grandchild in child.children:
                            traverse(grandchild, struct_name or enclosing_function)
                    else:
                        traverse(child, enclosing_function)
                return

            # Check if this is a call expression
            if node.type == "call_expression":
                results.append((node, enclosing_function or "", node.start_point[0] + 1))

            # Recurse into children
            for child in node.children:
                traverse(child, enclosing_function)

        traverse(root)
        return results

    def _extract_function_name_c(self, node: "Node") -> Optional[str]:
        """Extract function name from a function definition node.

        Args:
            node: Function definition node

        Returns:
            Function name or None
        """
        # Look for function_declarator
        for child in node.children:
            if child.type == "function_declarator":
                # Get the declarator (identifier)
                for grandchild in child.children:
                    if grandchild.type == "declarator":
                        for ggchild in grandchild.children:
                            if ggchild.type == "identifier":
                                return self._get_node_text(ggchild)
                    elif grandchild.type == "identifier":
                        return self._get_node_text(grandchild)

        # Fallback: look for identifier directly
        for child in node.children:
            if child.type == "identifier":
                return self._get_node_text(child)

        return None

    def _extract_callee_name_c(self, call_node: "Node") -> Optional[str]:
        """Extract the name of the called function from a call node.

        Handles:
        - Simple calls: foo()
        - Field expression calls: ptr->method()

        Args:
            call_node: Call expression node

        Returns:
            Callee name or None
        """
        # Get the function part of the call
        for child in call_node.children:
            if child.type == "field_expression":
                # ptr->method() -> extract "method"
                return self._extract_field_name_c(child)
            elif child.type == "identifier":
                # foo() -> extract "foo"
                return self._get_node_text(child)

        return None

    def _extract_field_name_c(self, field_node: "Node") -> Optional[str]:
        """Extract field name from a field_expression node.

        For ptr->method, extracts "method".

        Args:
            field_node: Field expression node

        Returns:
            Field name or None
        """
        # field_expression: argument . field: (field_identifier)
        # or for pointer access: argument -> field: (field_identifier)
        for child in reversed(field_node.children):
            if child.type == "field_identifier":
                return self._get_node_text(child)

        return None

    def _get_node_text(self, node: "Node") -> Optional[str]:
        """Get text content of a node.

        Args:
            node: Tree-sitter node

        Returns:
            Node text or None
        """
        if node is None or not hasattr(node, "text"):
            return None
        text = node.text
        if isinstance(text, bytes):
            return text.decode("utf-8", errors="ignore")
        return text


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

    def detect_calls_edges(
        self,
        tree: "Tree",
        source_code: str,
        file_path: Path,
    ) -> EdgeDetectionResult:
        """Detect CALLS edges in Kotlin source code.

        Finds function calls, method calls, and property access chains.
        Handles Kotlin-specific features like safe call operators and extension functions.

        Args:
            tree: Parsed tree-sitter tree
            source_code: Raw source code text
            file_path: Path to source file

        Returns:
            EdgeDetectionResult with detected calls
        """
        calls: List[CallEdge] = []
        call_nodes = self._find_call_nodes_kotlin(tree.root_node)

        for call_node, caller_name, caller_line in call_nodes:
            callee_name = self._extract_callee_name_kotlin(call_node)
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
                "language": "kotlin",
                "file": str(file_path),
            },
        )

    def _find_call_nodes_kotlin(
        self,
        root: "Node",
    ) -> List[tuple["Node", str, Optional[int]]]:
        """Find all call nodes with their enclosing function context.

        Args:
            root: Tree-sitter root node

        Returns:
            List of (call_node, caller_name, caller_line) tuples
        """
        results: List[tuple["Node", str, Optional[int]]] = []

        def traverse(node: "Node", enclosing_function: Optional[str] = None) -> None:
            # Check if this is a function declaration
            if node.type == "function_declaration":
                func_name = None
                for child in node.children:
                    if child.type == "identifier":
                        func_name = self._get_node_text(child)
                        break

                # Process children with new enclosing context
                for child in node.children:
                    traverse(child, func_name or enclosing_function)
                return

            # Check for class declarations
            if node.type == "class_declaration":
                class_name = None
                for child in node.children:
                    if child.type == "type_identifier":
                        class_name = self._get_node_text(child)
                        break

                # Process class body
                for child in node.children:
                    if child.type == "class_body":
                        for grandchild in child.children:
                            traverse(grandchild, class_name or enclosing_function)
                    else:
                        traverse(child, enclosing_function)
                return

            # Check for object declarations (Kotlin singletons)
            if node.type == "object_declaration":
                obj_name = None
                for child in node.children:
                    if child.type == "simple_identifier":
                        obj_name = self._get_node_text(child)
                        break

                # Process object body
                for child in node.children:
                    if child.type == "class_body":
                        for grandchild in child.children:
                            traverse(grandchild, obj_name or enclosing_function)
                    else:
                        traverse(child, enclosing_function)
                return

            # Check if this is a call expression
            if node.type == "call_expression":
                results.append((node, enclosing_function or "", node.start_point[0] + 1))

            # Recurse into children
            for child in node.children:
                traverse(child, enclosing_function)

        traverse(root)
        return results

    def _extract_callee_name_kotlin(self, call_node: "Node") -> Optional[str]:
        """Extract the name of the called function from a call node.

        Handles:
        - Simple calls: foo()
        - Navigation expressions: obj.method()
        - Safe navigation: obj?.method()

        Args:
            call_node: Call expression node

        Returns:
            Callee name or None
        """
        for child in call_node.children:
            if child.type == "navigation_expression":
                # obj.method() -> extract "method"
                return self._extract_navigation_name(child)
            elif child.type == "identifier":
                # foo() -> extract "foo"
                return self._get_node_text(child)

        return None

    def _extract_navigation_name(self, navigation_node: "Node") -> Optional[str]:
        """Extract method name from a navigation_expression node.

        For obj.method, extracts "method".
        Handles safe navigation: obj?.method

        Args:
            navigation_node: Navigation expression node

        Returns:
            Method name or None
        """
        # navigation_expression can be nested: obj1.obj2.method
        # We want the last identifier (not simple_identifier!)
        for child in reversed(navigation_node.children):
            if child.type == "identifier":
                return self._get_node_text(child)
            elif child.type == "navigation_expression":
                result = self._extract_navigation_name(child)
                if result:
                    return result

        return None

    def _get_node_text(self, node: "Node") -> Optional[str]:
        """Get text content of a node.

        Args:
            node: Tree-sitter node

        Returns:
            Node text or None
        """
        if node is None or not hasattr(node, "text"):
            return None
        text = node.text
        if isinstance(text, bytes):
            return text.decode("utf-8", errors="ignore")
        return text


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

    def detect_calls_edges(
        self,
        tree: "Tree",
        source_code: str,
        file_path: Path,
    ) -> EdgeDetectionResult:
        """Detect CALLS edges in C# source code.

        Finds method calls, constructor calls, and static method calls.
        Handles C#-specific features like generics, extension methods, and LINQ.

        Args:
            tree: Parsed tree-sitter tree
            source_code: Raw source code text
            file_path: Path to source file

        Returns:
            EdgeDetectionResult with detected calls
        """
        calls: List[CallEdge] = []
        call_nodes = self._find_call_nodes_csharp(tree.root_node)

        for call_node, caller_name, caller_line in call_nodes:
            callee_name = self._extract_callee_name_csharp(call_node)
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
                "language": "csharp",
                "file": str(file_path),
            },
        )

    def _find_call_nodes_csharp(
        self,
        root: "Node",
    ) -> List[tuple["Node", str, Optional[int]]]:
        """Find all call nodes with their enclosing function context.

        Args:
            root: Tree-sitter root node

        Returns:
            List of (call_node, caller_name, caller_line) tuples
        """
        results: List[tuple["Node", str, Optional[int]]] = []

        def traverse(node: "Node", enclosing_function: Optional[str] = None,
                   namespace_context: List[str] = None) -> None:
            """Recursively traverse tree finding calls."""
            if namespace_context is None:
                namespace_context = []

            # Check if this is a namespace declaration
            if node.type == "namespace_declaration":
                # Get namespace name
                ns_name = None
                for child in node.children:
                    if child.type == "identifier":
                        ns_name = self._get_node_text(child)
                        break

                new_namespace = namespace_context.copy()
                if ns_name:
                    new_namespace.append(ns_name)

                # Process children with namespace context
                for child in node.children:
                    traverse(child, enclosing_function, new_namespace)
                return

            # Check if this is a method declaration
            if node.type == "method_declaration":
                func_name = None
                for child in node.children:
                    if child.type == "identifier":
                        func_name = self._get_node_text(child)
                        break

                # Process children with new enclosing context
                for child in node.children:
                    traverse(child, func_name or enclosing_function, namespace_context)
                return

            # Check for class/interface declarations
            if node.type in ("class_declaration", "struct_declaration", "interface_declaration"):
                class_name = None
                for child in node.children:
                    if child.type == "identifier":
                        class_name = self._get_node_text(child)
                        break

                # Process class body
                for child in node.children:
                    if child.type == "declaration_list":
                        for grandchild in child.children:
                            traverse(grandchild, class_name or enclosing_function, namespace_context)
                    else:
                        traverse(child, enclosing_function, namespace_context)
                return

            # Check if this is an invocation expression (method call)
            if node.type == "invocation_expression":
                results.append((node, enclosing_function or "", node.start_point[0] + 1))

            # Check for object creation expressions (constructor calls)
            if node.type == "object_creation_expression":
                results.append((node, enclosing_function or "", node.start_point[0] + 1))

            # Recurse into children
            for child in node.children:
                traverse(child, enclosing_function, namespace_context)

        traverse(root)
        return results

    def _extract_callee_name_csharp(self, call_node: "Node") -> Optional[str]:
        """Extract the name of the called method from a call node.

        Handles:
        - Simple calls: Method()
        - Member access calls: obj.Method()
        - Static calls: ClassName.Method()

        Args:
            call_node: Call expression node

        Returns:
            Callee name or None
        """
        # For invocation_expression, find the function being called
        for child in call_node.children:
            if child.type == "member_access_expression":
                # obj.Method() -> extract "Method"
                return self._extract_member_name_csharp(child)
            elif child.type == "identifier":
                # Method() -> extract "Method"
                return self._get_node_text(child)

        # For object_creation_expression
        if call_node.type == "object_creation_expression":
            for child in call_node.children:
                if child.type == "identifier":
                    return self._get_node_text(child)
                elif child.type == "generic_name":
                    # GenericType<>() -> extract "GenericType"
                    for ggchild in child.children:
                        if ggchild.type == "identifier":
                            return self._get_node_text(ggchild)

        return None

    def _extract_member_name_csharp(self, member_node: "Node") -> Optional[str]:
        """Extract member name from a member_access_expression node.

        For obj.Method, extracts "Method".
        For obj1.obj2.Method, extracts "Method" (the final member).

        Args:
            member_node: Member access expression node

        Returns:
            Member name or None
        """
        # member_access_expression: object . name: (identifier)
        for child in reversed(member_node.children):
            if child.type == "identifier":
                return self._get_node_text(child)
            elif child.type == "generic_name":
                # Generic method: Method<T>()
                for ggchild in child.children:
                    if ggchild.type == "identifier":
                        return self._get_node_text(ggchild)
            elif child.type == "member_access_expression":
                result = self._extract_member_name_csharp(child)
                if result:
                    return result

        return None

    def _get_node_text(self, node: "Node") -> Optional[str]:
        """Get text content of a node.

        Args:
            node: Tree-sitter node

        Returns:
            Node text or None
        """
        if node is None or not hasattr(node, "text"):
            return None
        text = node.text
        if isinstance(text, bytes):
            return text.decode("utf-8", errors="ignore")
        return text


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

    def detect_calls_edges(
        self,
        tree: "Tree",
        source_code: str,
        file_path: Path,
    ) -> EdgeDetectionResult:
        """Detect CALLS edges in Ruby source code.

        Finds method calls, including blocks and dynamic calls.
        Handles Ruby-specific features like safe navigation and singleton methods.

        Args:
            tree: Parsed tree-sitter tree
            source_code: Raw source code text
            file_path: Path to source file

        Returns:
            EdgeDetectionResult with detected calls
        """
        calls: List[CallEdge] = []
        call_nodes = self._find_call_nodes_ruby(tree.root_node)

        for call_node, caller_name, caller_line in call_nodes:
            callee_name = self._extract_callee_name_ruby(call_node)
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
                "language": "ruby",
                "file": str(file_path),
            },
        )

    def _find_call_nodes_ruby(
        self,
        root: "Node",
    ) -> List[tuple["Node", str, Optional[int]]]:
        """Find all call nodes with their enclosing function context.

        Args:
            root: Tree-sitter root node

        Returns:
            List of (call_node, caller_name, caller_line) tuples
        """
        results: List[tuple["Node", str, Optional[int]]] = []

        def traverse(node: "Node", enclosing_function: Optional[str] = None) -> None:
            # Check if this is a method definition
            if node.type == "method":
                func_name = None
                for child in node.children:
                    if child.type == "identifier":
                        func_name = self._get_node_text(child)
                        break

                # Process children with new enclosing context
                for child in node.children:
                    traverse(child, func_name or enclosing_function)
                return

            # Check for singleton method definitions (def self.method)
            if node.type == "singleton_method":
                func_name = None
                for child in node.children:
                    if child.type == "identifier":
                        func_name = self._get_node_text(child)
                        break

                # Process children with new enclosing context
                for child in node.children:
                    traverse(child, func_name or enclosing_function)
                return

            # Check for class declarations
            if node.type == "class":
                class_name = None
                for child in node.children:
                    if child.type == "constant":
                        class_name = self._get_node_text(child)
                        break

                # Process class body
                for child in node.children:
                    if child.type == "body":
                        for grandchild in child.children:
                            traverse(grandchild, class_name or enclosing_function)
                    else:
                        traverse(child, enclosing_function)
                return

            # Check for module declarations
            if node.type == "module":
                module_name = None
                for child in node.children:
                    if child.type == "constant":
                        module_name = self._get_node_text(child)
                        break

                # Process module body
                for child in node.children:
                    if child.type == "body":
                        for grandchild in child.children:
                            traverse(grandchild, module_name or enclosing_function)
                    else:
                        traverse(child, enclosing_function)
                return

            # Check if this is a call expression
            if node.type == "call":
                results.append((node, enclosing_function or "", node.start_point[0] + 1))

            # Check for bare identifiers as method calls (Ruby allows foo without parens)
            # Only capture identifiers that are direct children of body_statement
            # This avoids capturing variable declarations, parameters, etc.
            if node.type == "body_statement" and enclosing_function:
                for child in node.children:
                    if child.type == "identifier":
                        # This is likely a bare method call like: foo
                        # Create a synthetic call node for consistency
                        results.append((child, enclosing_function or "", child.start_point[0] + 1))
                        break  # Only capture the first identifier per statement

            # Recurse into children
            for child in node.children:
                traverse(child, enclosing_function)

        traverse(root)
        return results

    def _extract_callee_name_ruby(self, call_node: "Node") -> Optional[str]:
        """Extract the name of the called method from a call node.

        Handles:
        - Simple calls: method (identifier node itself)
        - Method calls with receiver: obj.method (call node)
        - Safe navigation: obj&.method

        For simple calls, returns the identifier's text.
        For calls with receivers (obj.method), returns the last identifier.

        Args:
            call_node: Call node

        Returns:
            Callee name or None
        """
        # If the node itself is an identifier (bare method call)
        if call_node.type == "identifier":
            return self._get_node_text(call_node)

        # For method calls with receivers (obj.method), the LAST identifier is the method name
        # (first would be the receiver like "obj" in "obj.method")
        identifiers = []
        for child in call_node.children:
            if child.type == "identifier":
                identifiers.append(self._get_node_text(child))

        # Return the last identifier (method name), or first if only one
        if identifiers:
            return identifiers[-1] if len(identifiers) > 1 else identifiers[0]

        return None

    def _get_node_text(self, node: "Node") -> Optional[str]:
        """Get text content of a node.

        Args:
            node: Tree-sitter node

        Returns:
            Node text or None
        """
        if node is None or not hasattr(node, "text"):
            return None
        text = node.text
        if isinstance(text, bytes):
            return text.decode("utf-8", errors="ignore")
        return text


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

    def detect_calls_edges(
        self,
        tree: "Tree",
        source_code: str,
        file_path: Path,
    ) -> EdgeDetectionResult:
        """Detect CALLS edges in PHP source code.

        Finds function calls, method calls, and constructor calls.
        Handles PHP-specific features like static methods and namespaces.

        Args:
            tree: Parsed tree-sitter tree
            source_code: Raw source code text
            file_path: Path to source file

        Returns:
            EdgeDetectionResult with detected calls
        """
        calls: List[CallEdge] = []
        call_nodes = self._find_call_nodes_php(tree.root_node)

        for call_node, caller_name, caller_line in call_nodes:
            callee_name = self._extract_callee_name_php(call_node)
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
                "language": "php",
                "file": str(file_path),
            },
        )

    def _find_call_nodes_php(
        self,
        root: "Node",
    ) -> List[tuple["Node", str, Optional[int]]]:
        """Find all call nodes with their enclosing function context.

        Args:
            root: Tree-sitter root node

        Returns:
            List of (call_node, caller_name, caller_line) tuples
        """
        results: List[tuple["Node", str, Optional[int]]] = []

        def traverse(node: "Node", enclosing_function: Optional[str] = None,
                   namespace_context: List[str] = None) -> None:
            """Recursively traverse tree finding calls."""
            if namespace_context is None:
                namespace_context = []

            # Check if this is a namespace definition
            if node.type == "namespace_definition":
                # Get namespace name
                ns_name = None
                for child in node.children:
                    if child.type == "namespace_name":
                        ns_name = self._get_node_text(child)
                        break

                new_namespace = namespace_context.copy()
                if ns_name:
                    new_namespace.append(ns_name)

                # Process children with namespace context
                for child in node.children:
                    traverse(child, enclosing_function, new_namespace)
                return

            # Check if this is a function definition
            if node.type == "function_definition":
                func_name = None
                for child in node.children:
                    if child.type == "name":
                        func_name = self._get_node_text(child)
                        break

                # Process children with new enclosing context
                for child in node.children:
                    traverse(child, func_name or enclosing_function, namespace_context)
                return

            # Check if this is a method declaration
            if node.type == "method_declaration":
                func_name = None
                for child in node.children:
                    if child.type == "name":
                        func_name = self._get_node_text(child)
                        break

                # Process children with new enclosing context
                for child in node.children:
                    traverse(child, func_name or enclosing_function, namespace_context)
                return

            # Check for class/interface/trait declarations
            if node.type in ("class_declaration", "interface_declaration", "trait_declaration"):
                class_name = None
                for child in node.children:
                    if child.type == "name":
                        class_name = self._get_node_text(child)
                        break

                # Process class body
                for child in node.children:
                    if child.type == "declaration_list":
                        for grandchild in child.children:
                            traverse(grandchild, class_name or enclosing_function, namespace_context)
                    else:
                        traverse(child, enclosing_function, namespace_context)
                return

            # Check if this is a function call expression
            if node.type == "function_call_expression":
                results.append((node, enclosing_function or "", node.start_point[0] + 1))

            # Check if this is a member call expression ($obj->method())
            if node.type == "member_call_expression":
                results.append((node, enclosing_function or "", node.start_point[0] + 1))

            # Check for scoped call expression (ClassName::method())
            if node.type == "scoped_call_expression":
                results.append((node, enclosing_function or "", node.start_point[0] + 1))

            # Check for object creation expression (new Class())
            if node.type == "object_creation_expression":
                results.append((node, enclosing_function or "", node.start_point[0] + 1))

            # Recurse into children
            for child in node.children:
                traverse(child, enclosing_function, namespace_context)

        traverse(root)
        return results

    def _extract_callee_name_php(self, call_node: "Node") -> Optional[str]:
        """Extract the name of the called function from a call node.

        Handles:
        - Simple calls: function()
        - Member calls: $obj->method()
        - Static calls: ClassName::method()
        - Object creation: new ClassName()

        Args:
            call_node: Call expression node

        Returns:
            Callee name or None
        """
        # For function_call_expression
        if call_node.type == "function_call_expression":
            for child in call_node.children:
                if child.type == "name":
                    return self._get_node_text(child)

        # For member_call_expression ($obj->method())
        if call_node.type == "member_call_expression":
            for child in call_node.children:
                if child.type == "name":
                    return self._get_node_text(child)

        # For scoped_call_expression (ClassName::method())
        if call_node.type == "scoped_call_expression":
            # The last name child is the method name
            names = []
            for child in call_node.children:
                if child.type == "name":
                    names.append(self._get_node_text(child))
            if names:
                # Return the last name (method name)
                return names[-1]

        # For object_creation_expression
        if call_node.type == "object_creation_expression":
            for child in call_node.children:
                if child.type == "name":
                    return self._get_node_text(child)

        return None

    def _get_node_text(self, node: "Node") -> Optional[str]:
        """Get text content of a node.

        Args:
            node: Tree-sitter node

        Returns:
            Node text or None
        """
        if node is None or not hasattr(node, "text"):
            return None
        text = node.text
        if isinstance(text, bytes):
            return text.decode("utf-8", errors="ignore")
        return text


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

    def detect_calls_edges(
        self,
        tree: "Tree",
        source_code: str,
        file_path: Path,
    ) -> EdgeDetectionResult:
        """Detect CALLS edges in Swift source code.

        Finds function calls and method calls.
        Handles Swift-specific features like optional chaining and protocol methods.

        Args:
            tree: Parsed tree-sitter tree
            source_code: Raw source code text
            file_path: Path to source file

        Returns:
            EdgeDetectionResult with detected calls
        """
        calls: List[CallEdge] = []
        call_nodes = self._find_call_nodes_swift(tree.root_node)

        for call_node, caller_name, caller_line in call_nodes:
            callee_name = self._extract_callee_name_swift(call_node)
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
                "language": "swift",
                "file": str(file_path),
            },
        )

    def _find_call_nodes_swift(
        self,
        root: "Node",
    ) -> List[tuple["Node", str, Optional[int]]]:
        """Find all call nodes with their enclosing function context.

        Args:
            root: Tree-sitter root node

        Returns:
            List of (call_node, caller_name, caller_line) tuples
        """
        results: List[tuple["Node", str, Optional[int]]] = []

        def traverse(node: "Node", enclosing_function: Optional[str] = None) -> None:
            # Check if this is a function declaration
            if node.type == "function_declaration":
                func_name = None
                for child in node.children:
                    if child.type == "simple_identifier":
                        func_name = self._get_node_text(child)
                        break

                # Process children with new enclosing context
                for child in node.children:
                    traverse(child, func_name or enclosing_function)
                return

            # Check for class declarations
            if node.type == "class_declaration":
                class_name = None
                for child in node.children:
                    if child.type == "type_identifier":
                        class_name = self._get_node_text(child)
                        break

                # Process class body
                for child in node.children:
                    if child.type == "class_body":
                        for grandchild in child.children:
                            traverse(grandchild, class_name or enclosing_function)
                    else:
                        traverse(child, enclosing_function)
                return

            # Check for struct declarations
            if node.type == "struct_declaration":
                struct_name = None
                for child in node.children:
                    if child.type == "type_identifier":
                        struct_name = self._get_node_text(child)
                        break

                # Process struct body
                for child in node.children:
                    if child.type == "class_body":
                        for grandchild in child.children:
                            traverse(grandchild, struct_name or enclosing_function)
                    else:
                        traverse(child, enclosing_function)
                return

            # Check for protocol declarations
            if node.type == "protocol_declaration":
                protocol_name = None
                for child in node.children:
                    if child.type == "type_identifier":
                        protocol_name = self._get_node_text(child)
                        break

                # Process protocol body
                for child in node.children:
                    if child.type == "class_body":
                        for grandchild in child.children:
                            traverse(grandchild, protocol_name or enclosing_function)
                    else:
                        traverse(child, enclosing_function)
                return

            # Check if this is a call expression
            if node.type == "call_expression":
                results.append((node, enclosing_function or "", node.start_point[0] + 1))

            # Recurse into children
            for child in node.children:
                traverse(child, enclosing_function)

        traverse(root)
        return results

    def _extract_callee_name_swift(self, call_node: "Node") -> Optional[str]:
        """Extract the name of the called function from a call node.

        Handles:
        - Simple calls: function()
        - Member calls: obj.method()
        - Optional chaining: obj?.method

        Args:
            call_node: Call expression node

        Returns:
            Callee name or None
        """
        for child in call_node.children:
            if child.type == "navigation_expression":
                # obj.method() -> extract "method" from navigation_suffix
                return self._extract_navigation_method_name(child)
            elif child.type == "simple_identifier":
                # foo() -> extract "foo"
                return self._get_node_text(child)

        return None

    def _extract_navigation_method_name(self, navigation_node: "Node") -> Optional[str]:
        """Extract method name from a navigation_expression node.

        For obj.method, extracts "method" from navigation_suffix.
        Handles nested navigation: obj1.obj2.method

        Args:
            navigation_node: Navigation expression node

        Returns:
            Method name or None
        """
        # Look for navigation_suffix containing the method name
        for child in navigation_node.children:
            if child.type == "navigation_suffix":
                for grandchild in child.children:
                    if grandchild.type == "simple_identifier":
                        return self._get_node_text(grandchild)
            elif child.type == "navigation_expression":
                result = self._extract_navigation_method_name(child)
                if result:
                    return result

        return None

    def _get_node_text(self, node: "Node") -> Optional[str]:
        """Get text content of a node.

        Args:
            node: Tree-sitter node

        Returns:
            Node text or None
        """
        if node is None or not hasattr(node, "text"):
            return None
        text = node.text
        if isinstance(text, bytes):
            return text.decode("utf-8", errors="ignore")
        return text


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
