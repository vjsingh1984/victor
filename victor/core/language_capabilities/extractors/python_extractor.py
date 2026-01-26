"""
Python AST extractor using the built-in ast module.

Provides deep symbol extraction for Python code including:
- Type annotations
- Parameter types
- Async function detection
- Decorators
- Docstrings
"""

import ast
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..types import ExtractedSymbol

logger = logging.getLogger(__name__)


class PythonASTExtractor:
    """
    Python AST extractor using the built-in ast module.

    Extracts rich symbol information including type hints,
    decorators, and docstrings.
    """

    def extract(
        self,
        code: str,
        file_path: Path,
    ) -> List[ExtractedSymbol]:
        """
        Extract symbols from Python code.

        Args:
            code: Python source code
            file_path: Path to the source file

        Returns:
            List of extracted symbols
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            logger.warning(f"Failed to parse Python file {file_path}: {e}")
            return []

        symbols: List[ExtractedSymbol] = []
        self._extract_from_node(tree, symbols, str(file_path), None)
        return symbols

    def _extract_from_node(
        self,
        node: ast.AST,
        symbols: List[ExtractedSymbol],
        file_path: str,
        parent: Optional[str],
    ) -> None:
        """Recursively extract symbols from AST nodes."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                symbol = self._extract_class(child, file_path, parent)
                symbols.append(symbol)
                # Extract nested classes and methods
                self._extract_from_node(child, symbols, file_path, child.name)

            elif isinstance(child, ast.FunctionDef):
                symbol = self._extract_function(child, file_path, parent, is_async=False)
                symbols.append(symbol)
                # Extract nested functions
                self._extract_from_node(child, symbols, file_path, child.name)

            elif isinstance(child, ast.AsyncFunctionDef):
                symbol = self._extract_function(child, file_path, parent, is_async=True)
                symbols.append(symbol)
                # Extract nested functions
                self._extract_from_node(child, symbols, file_path, child.name)

            elif isinstance(child, (ast.Assign, ast.AnnAssign)):
                # Module-level or class-level variables
                if parent is None or isinstance(node, ast.ClassDef):
                    variable_symbols = self._extract_variable(child, file_path, parent)
                    symbols.extend(variable_symbols)

    def _extract_class(
        self,
        node: ast.ClassDef,
        file_path: str,
        parent: Optional[str],
    ) -> ExtractedSymbol:
        """Extract a class definition."""
        # Get base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(self._get_attribute_name(base))

        # Get decorators
        decorators = self._extract_decorators(node.decorator_list)

        # Get docstring
        docstring = ast.get_docstring(node)

        # Build signature
        signature = f"class {node.name}"
        if bases:
            signature += f"({', '.join(bases)})"

        return ExtractedSymbol(
            name=node.name,
            symbol_type="class",
            file_path=file_path,
            line_number=node.lineno,
            end_line=node.end_lineno,
            signature=signature,
            docstring=docstring,
            parent_symbol=parent,
            decorators=decorators,
            metadata={"bases": bases},
        )

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        file_path: str,
        parent: Optional[str],
        is_async: bool,
    ) -> ExtractedSymbol:
        """Extract a function or method definition."""
        # Determine if this is a method
        symbol_type = "method" if parent else "function"

        # Get parameters with types
        parameters = self._extract_parameters(node.args)

        # Get return type
        return_type = None
        if node.returns:
            return_type = self._annotation_to_string(node.returns)

        # Get decorators
        decorators = self._extract_decorators(node.decorator_list)

        # Determine visibility
        visibility = "public"
        if node.name.startswith("__") and not node.name.endswith("__"):
            visibility = "private"
        elif node.name.startswith("_"):
            visibility = "protected"

        # Get docstring
        docstring = ast.get_docstring(node)

        # Build signature
        async_prefix = "async " if is_async else ""
        param_str = ", ".join(parameters)
        signature = f"{async_prefix}def {node.name}({param_str})"
        if return_type:
            signature += f" -> {return_type}"

        return ExtractedSymbol(
            name=node.name,
            symbol_type=symbol_type,
            file_path=file_path,
            line_number=node.lineno,
            end_line=node.end_lineno,
            signature=signature,
            docstring=docstring,
            parent_symbol=parent,
            return_type=return_type,
            parameters=parameters,
            visibility=visibility,
            is_async=is_async,
            decorators=decorators,
        )

    def _extract_variable(
        self,
        node: ast.Assign | ast.AnnAssign,
        file_path: str,
        parent: Optional[str],
    ) -> List[ExtractedSymbol]:
        """Extract variable assignments."""
        symbols = []

        if isinstance(node, ast.AnnAssign):
            # Annotated assignment: x: int = 5
            if isinstance(node.target, ast.Name):
                type_annotation = self._annotation_to_string(node.annotation)
                symbols.append(
                    ExtractedSymbol(
                        name=node.target.id,
                        symbol_type="variable",
                        file_path=file_path,
                        line_number=node.lineno,
                        end_line=node.end_lineno,
                        parent_symbol=parent,
                        return_type=type_annotation,
                        metadata={"annotated": True},
                    )
                )
        elif isinstance(node, ast.Assign):
            # Simple assignment: x = 5
            for target in node.targets:
                if isinstance(target, ast.Name):
                    # Skip private variables at module level
                    if parent is None and target.id.startswith("_"):
                        continue
                    symbols.append(
                        ExtractedSymbol(
                            name=target.id,
                            symbol_type="variable",
                            file_path=file_path,
                            line_number=node.lineno,
                            end_line=node.end_lineno,
                            parent_symbol=parent,
                        )
                    )

        return symbols

    def _extract_parameters(self, args: ast.arguments) -> List[str]:
        """Extract parameter names with type annotations."""
        parameters = []

        # Regular args
        for i, arg in enumerate(args.args):
            param = arg.arg
            if arg.annotation:
                param += f": {self._annotation_to_string(arg.annotation)}"

            # Check for defaults
            default_offset = len(args.args) - len(args.defaults)
            if i >= default_offset:
                default = args.defaults[i - default_offset]
                param += f" = {self._value_to_string(default)}"

            parameters.append(param)

        # *args
        if args.vararg:
            param = f"*{args.vararg.arg}"
            if args.vararg.annotation:
                param += f": {self._annotation_to_string(args.vararg.annotation)}"
            parameters.append(param)

        # Keyword-only args
        for i, arg in enumerate(args.kwonlyargs):
            param = arg.arg
            if arg.annotation:
                param += f": {self._annotation_to_string(arg.annotation)}"
            if args.kw_defaults[i]:
                param += f" = {self._value_to_string(args.kw_defaults[i])}"
            parameters.append(param)

        # **kwargs
        if args.kwarg:
            param = f"**{args.kwarg.arg}"
            if args.kwarg.annotation:
                param += f": {self._annotation_to_string(args.kwarg.annotation)}"
            parameters.append(param)

        return parameters

    def _extract_decorators(self, decorator_list: List[ast.expr]) -> List[str]:
        """Extract decorator names."""
        decorators = []
        for dec in decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorators.append(self._get_attribute_name(dec))
            elif isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Name):
                    decorators.append(dec.func.id)
                elif isinstance(dec.func, ast.Attribute):
                    decorators.append(self._get_attribute_name(dec.func))
        return decorators

    def _annotation_to_string(self, node: ast.expr) -> str:
        """Convert annotation AST node to string."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return node.value
            return repr(node.value)
        elif isinstance(node, ast.Subscript):
            value = self._annotation_to_string(node.value)
            slice_str = self._annotation_to_string(node.slice)
            return f"{value}[{slice_str}]"
        elif isinstance(node, ast.Attribute):
            return self._get_attribute_name(node)
        elif isinstance(node, ast.Tuple):
            elts = [self._annotation_to_string(e) for e in node.elts]
            return ", ".join(elts)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            # Union type: X | Y
            left = self._annotation_to_string(node.left)
            right = self._annotation_to_string(node.right)
            return f"{left} | {right}"
        elif isinstance(node, ast.List):
            elts = [self._annotation_to_string(e) for e in node.elts]
            return f"[{', '.join(elts)}]"
        else:
            try:
                return ast.unparse(node)
            except Exception:
                return "..."

    def _value_to_string(self, node: ast.expr) -> str:
        """Convert value AST node to string representation."""
        if isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.List):
            return "[...]"
        elif isinstance(node, ast.Dict):
            return "{...}"
        elif isinstance(node, ast.Tuple):
            return "(...)"
        else:
            return "..."

    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """Get full attribute name (e.g., 'module.Class')."""
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))

    def can_parse(self, code: str) -> bool:
        """Check if code can be parsed without errors."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def get_syntax_errors(self, code: str) -> List[Dict[str, Any]]:
        """Get syntax errors from code."""
        try:
            ast.parse(code)
            return []
        except SyntaxError as e:
            return [{
                "line": e.lineno or 1,
                "column": e.offset or 0,
                "message": str(e.msg),
                "text": e.text,
            }]
