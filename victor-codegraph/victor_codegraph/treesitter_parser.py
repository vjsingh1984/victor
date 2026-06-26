"""Generic tree-sitter parser for non-Python languages.

Includes a *real* JavaScript/TypeScript extractor — the donor ProximaDB ``code.py``
shipped a JS/TS stub that returned no symbols (CLAUDE-mandate "plausible-but-wrong"
failure). Here JS/TS, plus a language-agnostic node-walk for common grammars, produce
functions, classes, methods, and imports.

Requires the ``treesitter`` extra (``tree-sitter`` + ``tree-sitter-language-pack``).
If the grammar is unavailable, :func:`parse_treesitter` raises :class:`GrammarUnavailable`
so the orchestrator can fall back.
"""

from __future__ import annotations

from .languages import TREE_SITTER_GRAMMAR
from .model import (
    CodeRelation,
    CodeRelationType,
    CodeSymbol,
    CodeSymbolType,
    ParsedCode,
    SourceLocation,
    content_hash,
    deterministic_symbol_id,
)


class GrammarUnavailable(RuntimeError):
    """Raised when a tree-sitter grammar can't be loaded for a language."""


# Node types that denote a callable, per common tree-sitter grammars.
_FUNC_NODES = {
    "function_declaration",
    "function_definition",
    "function_item",  # rust
    "method_definition",  # js/ts
    "method_declaration",  # java/go
    "function",
    "arrow_function",
}
_CLASS_NODES = {
    "class_declaration",
    "class_definition",
    "class_specifier",
    "struct_item",  # rust
    "struct_specifier",
    "interface_declaration",
    "impl_item",  # rust
}
_IMPORT_NODES = {
    "import_statement",
    "import_declaration",
    "import_from_statement",
    "use_declaration",  # rust
    "preproc_include",  # c/cpp
}


def _get_parser(grammar: str):
    try:
        from tree_sitter_language_pack import get_parser
    except Exception as e:  # ImportError or native load failure
        raise GrammarUnavailable(f"tree-sitter-language-pack unavailable: {e}") from e
    try:
        return get_parser(grammar)
    except Exception as e:
        raise GrammarUnavailable(f"grammar '{grammar}' unavailable: {e}") from e


def _text(node, src: bytes) -> str:
    return src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _attr(obj, name):
    """Access a tree-sitter attribute that may be a property OR a zero-arg method.

    tree-sitter-language-pack's bundled binding exposes `root_node`/`children` as
    *methods* (callables), whereas the canonical `tree_sitter` exposes them as
    properties. A list/Node is never callable, so this is safe for both shapes.
    """

    v = getattr(obj, name)
    return v() if callable(v) else v


def _children(node):
    """Return a node's children across tree-sitter binding flavors.

    `children` may be a property, a zero-arg method, or absent entirely (the
    bundled binding exposes only `child_count` + `child(i)`, the universal C API).
    """

    children = getattr(node, "children", None)
    if children is not None:
        return children() if callable(children) else children
    count = _attr(node, "child_count")
    return [node.child(i) for i in range(count)]


def _name_of(node, src: bytes) -> str | None:
    """Find the declared name of a function/class node (grammar-agnostic)."""

    field = node.child_by_field_name("name")
    if field is not None:
        return _text(field, src)
    for child in _children(node):
        if child.type in ("identifier", "type_identifier", "field_identifier", "property_identifier"):
            return _text(child, src)
    return None


def _walk_collect(node, src, file_path, language, scope, symbols, relations):
    for child in _children(node):
        t = child.type
        if t in _CLASS_NODES:
            name = _name_of(child, src) or "<anonymous>"
            stype = (
                CodeSymbolType.STRUCT
                if t.startswith("struct")
                else CodeSymbolType.INTERFACE
                if "interface" in t
                else CodeSymbolType.CLASS
            )
            sym = _mk(child, src, file_path, language, name, stype, scope)
            symbols.append(sym)
            # Recurse into the class body with the class name on the scope chain.
            body = child.child_by_field_name("body")
            _walk_collect(
                body if body is not None else child,
                src,
                file_path,
                language,
                [*scope, name],
                symbols,
                relations,
            )
        elif t in _FUNC_NODES:
            name = _name_of(child, src)
            if name is None and t == "arrow_function":
                continue  # anonymous arrow not bound to a name; skip
            name = name or "<anonymous>"
            stype = CodeSymbolType.METHOD if scope else CodeSymbolType.FUNCTION
            if name in ("constructor", "__init__", "new"):
                stype = CodeSymbolType.CONSTRUCTOR
            symbols.append(_mk(child, src, file_path, language, name, stype, scope))
            # Don't recurse into function bodies for nested defs (kept flat, like donors).
        else:
            _walk_collect(child, src, file_path, language, scope, symbols, relations)


def _handle_const_arrow(node, src, file_path, language, symbols):
    """JS/TS: ``const foo = (...) => {...}`` / ``export const foo = () => {}``."""

    for decl in _children(node):
        if decl.type != "variable_declarator":
            continue
        name_node = decl.child_by_field_name("name")
        value = decl.child_by_field_name("value")
        if name_node is not None and value is not None and value.type == "arrow_function":
            name = _text(name_node, src)
            symbols.append(
                _mk(decl, src, file_path, language, name, CodeSymbolType.FUNCTION, [])
            )


def _mk(node, src, file_path, language, name, stype, scope) -> CodeSymbol:
    start_line = node.start_point[0] + 1
    end_line = node.end_point[0] + 1
    fqn = "::".join([file_path.replace("/", "."), *scope, name])
    return CodeSymbol(
        id=deterministic_symbol_id(file_path, name, start_line, node.start_point[1]),
        symbol_type=stype,
        fully_qualified_name=fqn,
        simple_name=name,
        location=SourceLocation(
            file_path=file_path,
            start_line=start_line,
            start_column=node.start_point[1],
            end_line=end_line,
            end_column=node.end_point[1],
            byte_offset=node.start_byte,
            byte_length=node.end_byte - node.start_byte,
        ),
        source_code=_text(node, src),
        language=language,
        scope_chain=list(scope),
        complexity={"lines": end_line - start_line + 1},
    )


def parse_treesitter(content: str, file_path: str, language: str) -> ParsedCode:
    """Parse non-Python source via tree-sitter. Raises GrammarUnavailable on fallback."""

    grammar = TREE_SITTER_GRAMMAR.get(language)
    if grammar is None:
        raise GrammarUnavailable(f"no grammar mapping for language '{language}'")
    parser = _get_parser(grammar)

    src = content.encode("utf-8")
    # tree-sitter's Parser.parse() takes bytes on most builds, but some
    # (notably via tree-sitter-language-pack) require a str and raise
    # TypeError on bytes. Node byte offsets are UTF-8 positions either way,
    # so `src` stays valid for slicing in _text() regardless of which we pass.
    try:
        tree = parser.parse(src)
    except TypeError:
        tree = parser.parse(content)
    root = _attr(tree, "root_node")

    symbols: list[CodeSymbol] = []
    relations: list[CodeRelation] = []
    imports: list[str] = []

    for child in _children(root):
        if child.type in _IMPORT_NODES:
            imports.append(_text(child, src))

    _walk_collect(root, src, file_path, language, [], symbols, relations)

    # JS/TS arrow-function-as-const: a real surface the stub missed.
    if language in ("javascript", "typescript", "tsx"):
        for child in _children(root):
            target = child
            # unwrap `export const ...`
            if child.type in ("export_statement",) and child.child_count:
                for c in _children(child):
                    if c.type in ("lexical_declaration", "variable_declaration"):
                        target = c
                        break
            if target.type in ("lexical_declaration", "variable_declaration"):
                _handle_const_arrow(target, src, file_path, language, symbols)

    # Best-effort CONTAINS edges (class -> its methods) by scope.
    by_scope: dict[str, str] = {
        s.simple_name: s.id for s in symbols if s.symbol_type == CodeSymbolType.CLASS
    }
    for s in symbols:
        if s.scope_chain:
            parent = by_scope.get(s.scope_chain[-1])
            if parent is not None and parent != s.id:
                relations.append(
                    CodeRelation(
                        from_symbol_id=parent,
                        to_symbol_id=s.id,
                        relation_type=CodeRelationType.CONTAINS,
                    )
                )

    return ParsedCode(
        file_path=file_path,
        language=language,
        symbols=symbols,
        relations=relations,
        imports=imports,
        content_hash=content_hash(content),
    )
