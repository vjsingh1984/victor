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

"""AST visitor for extracting symbols from Python code."""

import ast
from typing import List, Optional

from victor.verticals.contrib.coding.codebase.utils.ast_helpers import (
    build_signature,
    extract_base_classes,
)
from victor.verticals.contrib.coding.codebase.indexer.models import FileMetadata, Symbol


class SymbolVisitor(ast.NodeVisitor):
    """AST visitor to extract symbols from Python code."""

    def __init__(self, metadata: FileMetadata):
        self.metadata = metadata
        self.current_class: Optional[str] = None
        self.current_function: Optional[str] = None
        self.call_edges: List[tuple[str, str]] = []
        self.composition_edges: List[tuple[str, str]] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition."""
        bases = extract_base_classes(node)
        symbol = Symbol(
            name=node.name,
            type="class",
            file_path=self.metadata.path,
            line_number=node.lineno,
            docstring=ast.get_docstring(node),
            base_classes=bases,
        )
        self.metadata.symbols.append(symbol)

        # Visit class methods
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition."""
        name = node.name
        if self.current_class:
            name = f"{self.current_class}.{name}"

        signature = build_signature(node)

        symbol = Symbol(
            name=name,
            type="function",
            file_path=self.metadata.path,
            line_number=node.lineno,
            docstring=ast.get_docstring(node),
            signature=signature,
        )
        self.metadata.symbols.append(symbol)
        old_function = self.current_function
        self.current_function = name
        self.generic_visit(node)
        self.current_function = old_function

    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statement."""
        for alias in node.names:
            self.metadata.imports.append(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from...import statement."""
        if node.module:
            self.metadata.imports.append(node.module)

    def visit_Call(self, node: ast.Call) -> None:
        """Capture simple call relationships for intra-file graph edges."""
        if self.current_function:
            callee = None
            if isinstance(node.func, ast.Name):
                callee = node.func.id
            elif isinstance(node.func, ast.Attribute):
                callee = node.func.attr

            if callee:
                self.call_edges.append((self.current_function, callee))

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Detect has-a relationships for class attributes."""
        if self.current_class:
            target_type: Optional[str] = None
            if isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Name):
                    target_type = func.id
                elif isinstance(func, ast.Attribute):
                    target_type = func.attr
            elif isinstance(node.value, ast.Name):
                target_type = node.value.id
            if target_type:
                self.composition_edges.append((self.current_class, target_type))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Capture annotated attributes inside classes for composition edges."""
        if self.current_class:
            target_type: Optional[str] = None
            if isinstance(node.annotation, ast.Name):
                target_type = node.annotation.id
            elif isinstance(node.annotation, ast.Attribute):
                target_type = node.annotation.attr
            if target_type:
                self.composition_edges.append((self.current_class, target_type))
        self.generic_visit(node)
