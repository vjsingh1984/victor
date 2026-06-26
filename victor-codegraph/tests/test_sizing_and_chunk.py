"""Size-capping tests — the headline gap fix vs ProximaDB's donor code.py."""

from __future__ import annotations

from victor_codegraph import ChunkConfig, chunk
from victor_codegraph.model import (
    CodeSymbol,
    CodeSymbolType,
    SourceLocation,
)
from victor_codegraph.sizing import chunks_for_symbol


def _big_symbol(n_lines: int) -> CodeSymbol:
    body = "\n".join(f"    x{i} = compute({i})" for i in range(n_lines))
    source = f"def big():\n{body}\n"
    return CodeSymbol(
        id="sym1",
        symbol_type=CodeSymbolType.FUNCTION,
        fully_qualified_name="m::big",
        simple_name="big",
        location=SourceLocation(file_path="m.py", start_line=1, end_line=n_lines + 2),
        source_code=source,
        language="python",
    )


def test_small_symbol_is_single_chunk():
    sym = _big_symbol(3)
    chunks = chunks_for_symbol(sym, ChunkConfig())
    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_total"] == 1


def test_oversized_symbol_is_body_split():
    sym = _big_symbol(400)
    cfg = ChunkConfig(max_chunk_tokens=128, chunk_overlap_tokens=16)
    chunks = chunks_for_symbol(sym, cfg)
    assert len(chunks) > 1, "oversized symbol must split"
    # No chunk exceeds the char budget.
    for c in chunks:
        assert len(c.text) <= cfg.max_chunk_chars
    # All sub-chunks share the parent symbol id and have hierarchical ids.
    assert all(c.symbol_id == "sym1" for c in chunks)
    assert all(c.chunk_id.startswith("sym1#body#") for c in chunks)
    assert all(c.metadata["chunk_total"] == len(chunks) for c in chunks)


def test_chunk_end_to_end_python_respects_budget():
    src = "def f():\n" + "\n".join(f"    a{i} = {i}" for i in range(300)) + "\n"
    cfg = ChunkConfig(max_chunk_tokens=100)
    chunks = chunk(src, file_path="big.py", config=cfg)
    assert chunks
    for c in chunks:
        assert len(c.text) <= cfg.max_chunk_chars


def test_unknown_language_falls_back_to_sliding_window():
    src = "\n".join(f"line {i}" for i in range(200))
    chunks = chunk(src, file_path="notes.unknownext", config=ChunkConfig(max_chunk_tokens=50))
    assert chunks
    assert all(c.metadata.get("strategy") == "sliding_window" for c in chunks)
