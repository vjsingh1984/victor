"""Projection to the ProximaDB substrate-keystone ``ProximaRecord`` shape.

Per ProximaDB ``CODE_GRAPH_CORRELATED_SUBSTRATE_2026_06_22.adoc`` a code symbol is *one*
record addressable as a relational row, a graph node, and a vector at once. This adapter
emits the **shape** as plain dicts — it does not import proximadb, embed, or write. The
consumer (Victor embedded, AnvaiOps service) supplies the embedder and the DB write.
"""

from __future__ import annotations

from typing import Any, Callable

from .model import CodeRelation, CodeSymbol, ParsedCode

Embedder = Callable[[str], list[float]]


def _symbol_oid(repo_graph_id: str, symbol: CodeSymbol) -> str:
    return f"graph/{repo_graph_id}/node/{symbol.id}"


def symbol_to_record(
    symbol: CodeSymbol,
    repo_graph_id: str,
    branch_id: str = "main",
    embedder: Embedder | None = None,
    model_id: str = "bge-small-en-v1.5",
    dim: int = 384,
) -> dict[str, Any]:
    """Project one symbol to a node record (row + graph node + optional vector)."""

    oid = _symbol_oid(repo_graph_id, symbol)
    record: dict[str, Any] = {
        "oid": oid,
        "labels": ["graph_node", "code_symbol"],
        "branch_id": branch_id,
        "props": {
            "name": symbol.simple_name,
            "fully_qualified_name": symbol.fully_qualified_name,
            "file": symbol.location.file_path,
            "line": symbol.location.start_line,
            "end_line": symbol.location.end_line,
            "lang": symbol.language,
            "ast_kind": symbol.symbol_type.name,
            "signature": symbol.signature,
            "visibility": "private" if "private" in symbol.modifiers else "public",
            "module_path": "::".join(symbol.scope_chain),
            "snippet": symbol.source_code,
            "documentation": symbol.documentation,
        },
        "embeddings": [],
    }
    if embedder is not None:
        record["embeddings"].append(
            {
                "model_id": model_id,
                "modality": "code",
                "dim": dim,
                "values": embedder(symbol.source_code),
            }
        )
    return record


def relation_to_record(relation: CodeRelation, repo_graph_id: str, branch_id: str = "main") -> dict[str, Any]:
    """Project one relation to an edge record."""

    return {
        "labels": ["graph_edge"],
        "branch_id": branch_id,
        "edge": {
            "from_oid": f"graph/{repo_graph_id}/node/{relation.from_symbol_id}",
            "to_oid": f"graph/{repo_graph_id}/node/{relation.to_symbol_id}",
            "edge_type": relation.relation_type.name,
        },
        "props": {"confidence": relation.confidence, "context": relation.context},
    }


def to_proxima_records(
    parsed: ParsedCode,
    repo_graph_id: str,
    branch_id: str = "main",
    embedder: Embedder | None = None,
) -> list[dict[str, Any]]:
    """Project an entire parsed file to node + edge records (shapes only)."""

    records = [
        symbol_to_record(s, repo_graph_id, branch_id, embedder) for s in parsed.symbols
    ]
    records.extend(
        relation_to_record(r, repo_graph_id, branch_id) for r in parsed.relations
    )
    return records
