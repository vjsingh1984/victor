"""Public entrypoints: ``parse`` (symbols+relations) and ``chunk`` (size-capped).

Fallback chain (Victor's posture): python-ast -> tree-sitter -> sliding-window. A parse
never hard-fails; an unknown or grammar-less language degrades to line-window chunks.
"""

from __future__ import annotations

from .config import ChunkConfig
from .languages import detect_language
from .model import CodeChunk, ParsedCode, content_hash
from .python_parser import parse_python
from .sizing import chunks_for_symbol
from .treesitter_parser import GrammarUnavailable, parse_treesitter


def parse(content: str, language: str | None = None, file_path: str = "<unknown>") -> ParsedCode:
    """Parse source into symbols + relations, falling back gracefully."""

    language = language or detect_language(file_path)

    if language == "python":
        try:
            return parse_python(content, file_path)
        except SyntaxError:
            pass  # fall through to window chunking via empty ParsedCode

    if language is not None and language != "python":
        try:
            return parse_treesitter(content, file_path, language)
        except GrammarUnavailable:
            pass

    # Last resort: no symbols (caller's chunk() will sliding-window the raw text).
    return ParsedCode(
        file_path=file_path,
        language=language or "text",
        symbols=[],
        relations=[],
        imports=[],
        content_hash=content_hash(content),
    )


def _sliding_window(content: str, file_path: str, language: str, config: ChunkConfig) -> list[CodeChunk]:
    """Universal fallback when no symbols were extracted."""

    if not content:
        return []
    lines = content.splitlines(keepends=True)
    out: list[CodeChunk] = []
    cur: list[str] = []
    cur_len = 0
    start_line = 1
    idx = 0
    for n, ln in enumerate(lines, start=1):
        if cur_len + len(ln) > config.max_chunk_chars and cur:
            text = "".join(cur)
            out.append(
                CodeChunk(
                    chunk_id=f"{file_path}#window#{idx}",
                    text=text,
                    symbol_id=f"{file_path}#window#{idx}",
                    start_pos=0,
                    end_pos=0,
                    metadata={
                        "file_path": file_path,
                        "language": language,
                        "chunk_index": idx,
                        "start_line": start_line,
                        "end_line": n - 1,
                        "strategy": "sliding_window",
                    },
                )
            )
            idx += 1
            cur, cur_len, start_line = [], 0, n
        cur.append(ln)
        cur_len += len(ln)
    if cur:
        out.append(
            CodeChunk(
                chunk_id=f"{file_path}#window#{idx}",
                text="".join(cur),
                symbol_id=f"{file_path}#window#{idx}",
                start_pos=0,
                end_pos=0,
                metadata={
                    "file_path": file_path,
                    "language": language,
                    "chunk_index": idx,
                    "start_line": start_line,
                    "end_line": len(lines),
                    "strategy": "sliding_window",
                },
            )
        )
    return out


def chunk(
    content: str,
    language: str | None = None,
    file_path: str = "<unknown>",
    config: ChunkConfig | None = None,
) -> list[CodeChunk]:
    """Parse + project into size-capped, embeddable chunks."""

    config = config or ChunkConfig()
    parsed = parse(content, language, file_path)

    if not parsed.symbols:
        return _sliding_window(content, file_path, parsed.language, config)

    out: list[CodeChunk] = []
    for sym in parsed.symbols:
        if not config.include_private and "private" in sym.modifiers:
            continue
        out.extend(chunks_for_symbol(sym, config))
    return out
