"""Shared quality helpers for code-oriented search results.

These helpers keep result enrichment and lightweight utility reranking out of
tool implementations so multiple code-retrieval surfaces can reuse them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_OPAQUE_CONTENT_PREFIXES = ("symbol:", "node:", "file:")
_QUERY_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]*")
_WHITESPACE_RE = re.compile(r"\s+")
_QUERY_STOPWORDS = {
    "about",
    "after",
    "before",
    "change",
    "entry",
    "find",
    "for",
    "from",
    "help",
    "how",
    "into",
    "main",
    "point",
    "show",
    "that",
    "the",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
    "why",
}
_TEST_PATH_MARKERS = (
    "/test/",
    "/tests/",
    "test_",
    "_test.",
    ".spec.",
    "spec_",
    "/examples/",
    "examples/",
)
_NON_IMPLEMENTATION_PREFIXES = (
    "docs/",
    "site/",
    "examples/",
    "benchmarks/",
)


@dataclass(frozen=True)
class CodeSearchQualityConfig:
    """Configuration for snippet enrichment and bounded reranking."""

    context_before_lines: int = 1
    max_snippet_lines: int = 8
    max_snippet_chars: int = 400
    unique_file_bonus: float = 0.04
    repeated_file_penalty: float = 0.03
    duplicate_snippet_penalty: float = 0.05
    implementation_bonus: float = 0.03
    test_path_penalty: float = 0.02
    identifier_bonus: float = 0.03


def enrich_code_search_results(
    results: Sequence[Dict[str, Any]],
    *,
    root_path: Path,
    config: Optional[CodeSearchQualityConfig] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Resolve compact preview snippets and opaque semantic payloads.

    Search providers sometimes return compact identifiers instead of actual
    content. This helper resolves short file-window snippets from the reported
    file and line span so the user and model see grounded code rather than an
    opaque ID.
    """

    quality_config = config or CodeSearchQualityConfig()
    file_cache: Dict[Path, Optional[List[str]]] = {}
    enriched_results: List[Dict[str, Any]] = []
    snippet_enriched_hits = 0

    for raw_result in results:
        result = dict(raw_result)
        metadata = result.get("metadata")
        metadata_dict = dict(metadata) if isinstance(metadata, dict) else {}

        file_path = _as_non_empty_str(result.get("file_path", result.get("path")))
        if not file_path:
            file_path = _as_non_empty_str(metadata_dict.get("file_path"))
            if file_path:
                result["file_path"] = file_path

        line_number = _coerce_int(result.get("line_number", result.get("line")))
        if line_number is None:
            line_number = _coerce_int(metadata_dict.get("line_number"))
            if line_number is not None:
                result["line_number"] = line_number

        end_line = _coerce_int(result.get("end_line"))
        if end_line is None:
            end_line = _coerce_int(metadata_dict.get("end_line"))
            if end_line is not None:
                result["end_line"] = end_line

        resolved_snippet = _resolve_file_window(
            root_path=root_path,
            file_path=file_path,
            line_number=line_number,
            end_line=end_line,
            cache=file_cache,
            config=quality_config,
        )
        snippet = _as_non_empty_str(result.get("snippet"))
        if not snippet and resolved_snippet:
            result["snippet"] = resolved_snippet
            snippet_enriched_hits += 1
            metadata_dict.setdefault("snippet_source", "line_window")
        elif not snippet:
            fallback_snippet = _snippet_from_text(result.get("content"), quality_config.max_snippet_chars)
            if fallback_snippet:
                result["snippet"] = fallback_snippet

        if resolved_snippet and _should_replace_content(result.get("content"), metadata_dict):
            result["content"] = resolved_snippet
            metadata_dict.setdefault("content_source", "line_window")

        if metadata_dict:
            result["metadata"] = metadata_dict

        enriched_results.append(result)

    enrichment_metadata: Dict[str, Any] = {
        "chunking_strategy": infer_code_search_chunking_strategy(enriched_results)
    }
    if snippet_enriched_hits:
        enrichment_metadata["snippet_strategy"] = "line_window"
        enrichment_metadata["snippet_enriched_hits"] = snippet_enriched_hits

    return enriched_results, enrichment_metadata


def rerank_code_search_results(
    results: Sequence[Dict[str, Any]],
    *,
    query: str,
    config: Optional[CodeSearchQualityConfig] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Apply bounded utility-aware reranking to code search hits."""

    quality_config = config or CodeSearchQualityConfig()
    query_signals = _extract_query_signals(query)
    seen_files: set[str] = set()
    seen_snippets: set[str] = set()
    rescored_results: List[Tuple[float, float, int, Dict[str, Any]]] = []
    repeated_file_hits = 0
    duplicate_snippet_hits = 0
    implementation_hits = 0
    identifier_hits = 0
    total_relevance = 0.0

    for position, raw_result in enumerate(results):
        result = dict(raw_result)
        base_score = _coerce_float(
            result.get("combined_score", result.get("score", result.get("similarity", 0.0)))
        )
        total_relevance += base_score
        utility_score = base_score

        file_key = _normalize_space(_as_non_empty_str(result.get("file_path", result.get("path"))) or "")
        snippet_key = _normalize_snippet_key(result)

        if file_key:
            if file_key in seen_files:
                repeated_file_hits += 1
                utility_score -= quality_config.repeated_file_penalty
            else:
                seen_files.add(file_key)
                utility_score += quality_config.unique_file_bonus

        if snippet_key:
            if snippet_key in seen_snippets:
                duplicate_snippet_hits += 1
                utility_score -= quality_config.duplicate_snippet_penalty
            else:
                seen_snippets.add(snippet_key)

        if _is_implementation_path(file_key):
            implementation_hits += 1
            utility_score += quality_config.implementation_bonus
        elif _is_test_path(file_key):
            utility_score -= quality_config.test_path_penalty

        if query_signals and _matches_query_signal(result, query_signals):
            identifier_hits += 1
            utility_score += quality_config.identifier_bonus

        result["utility_rank_score"] = round(max(0.0, utility_score), 6)
        rescored_results.append((result["utility_rank_score"], base_score, -position, result))

    rescored_results.sort(reverse=True)
    reranked_results = [result for _, _, _, result in rescored_results]
    average_relevance = total_relevance / len(results) if results else 0.0
    average_rank_score = (
        sum(score for score, _, _, _ in rescored_results) / len(results) if results else 0.0
    )

    reranking_metadata = {
        "strategy": "bounded_code_utility",
        "candidate_count": len(results),
        "file_diversity": len(seen_files),
        "repeated_file_hits": repeated_file_hits,
        "duplicate_snippet_hits": duplicate_snippet_hits,
        "implementation_hits": implementation_hits,
        "identifier_hits": identifier_hits,
        "average_relevance": round(average_relevance, 4),
        "average_rank_score": round(average_rank_score, 4),
    }
    return reranked_results, reranking_metadata


def infer_code_search_chunking_strategy(results: Sequence[Dict[str, Any]]) -> str:
    """Infer the chunking strategy reported by the active backend."""

    for result in results:
        value = _extract_chunking_strategy(result)
        if value:
            return value
    return "symbol_only"


def _extract_chunking_strategy(result: Dict[str, Any]) -> str:
    candidates = [result.get("chunking_strategy")]
    metadata = result.get("metadata")
    if isinstance(metadata, dict):
        candidates.append(metadata.get("chunking_strategy"))
    for candidate in candidates:
        text = _as_non_empty_str(candidate)
        if text:
            return text
    return ""


def _resolve_file_window(
    *,
    root_path: Path,
    file_path: Optional[str],
    line_number: Optional[int],
    end_line: Optional[int],
    cache: Dict[Path, Optional[List[str]]],
    config: CodeSearchQualityConfig,
) -> Optional[str]:
    if not file_path or line_number is None or line_number <= 0:
        return None

    candidate = Path(file_path)
    root_resolved = root_path.resolve()
    if not candidate.is_absolute():
        candidate = (root_path / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if candidate != root_resolved and root_resolved not in candidate.parents:
        return None

    lines = cache.get(candidate)
    if lines is None and candidate not in cache:
        try:
            text = candidate.read_text(encoding="utf-8", errors="ignore")
            lines = text.splitlines()
        except OSError:
            lines = None
        cache[candidate] = lines

    if not lines or line_number > len(lines):
        return None

    start_line = max(1, line_number - config.context_before_lines)
    max_end_line = min(len(lines), start_line + config.max_snippet_lines - 1)
    if end_line is not None and end_line >= line_number:
        resolved_end = min(max_end_line, end_line)
    else:
        resolved_end = max_end_line

    snippet = "\n".join(lines[start_line - 1 : resolved_end]).strip()
    if not snippet:
        return None
    if len(snippet) > config.max_snippet_chars:
        snippet = snippet[: config.max_snippet_chars - 24].rstrip() + "... [snippet truncated]"
    return snippet


def _should_replace_content(content: Any, metadata: Dict[str, Any]) -> bool:
    text = _as_non_empty_str(content)
    if not text:
        return True

    normalized = text.strip()
    unified_id = _as_non_empty_str(metadata.get("unified_id"))
    if unified_id and normalized == unified_id:
        return True

    if (
        normalized.startswith(_OPAQUE_CONTENT_PREFIXES)
        and "\n" not in normalized
        and " " not in normalized
    ):
        return True

    return bool(
        ":" in normalized
        and "\n" not in normalized
        and " " not in normalized
        and len(normalized) > 24
        and "/" in normalized
    )


def _snippet_from_text(content: Any, max_chars: int) -> str:
    text = _as_non_empty_str(content)
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    snippet = " ".join(lines[:2]).strip()
    if len(snippet) > max_chars:
        return snippet[: max_chars - 3].rstrip() + "..."
    return snippet


def _normalize_snippet_key(result: Dict[str, Any]) -> str:
    text = _as_non_empty_str(result.get("snippet")) or _snippet_from_text(result.get("content"), 160)
    if not text:
        return ""
    return _normalize_space(text)[:160]


def _matches_query_signal(result: Dict[str, Any], query_signals: Sequence[str]) -> bool:
    metadata = result.get("metadata")
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    haystack_parts = [
        _as_non_empty_str(result.get("file_path", result.get("path"))),
        _as_non_empty_str(result.get("symbol_name", result.get("name"))),
        _as_non_empty_str(metadata_dict.get("symbol_name")),
    ]
    haystack = " ".join(part.lower() for part in haystack_parts if part)
    return any(signal in haystack for signal in query_signals)


def _extract_query_signals(query: str) -> List[str]:
    signals: List[str] = []
    seen: set[str] = set()
    for token in _QUERY_TOKEN_RE.findall(query):
        normalized = token.strip().lower().strip(".,:;()[]{}<>\"'")
        if len(normalized) < 3 or normalized in _QUERY_STOPWORDS:
            continue
        is_high_signal = (
            any(ch in normalized for ch in "._/-:")
            or "_" in normalized
            or token[:1].isupper()
        )
        if not is_high_signal or normalized in seen:
            continue
        seen.add(normalized)
        signals.append(normalized)
    return signals[:8]


def _is_test_path(file_path: str) -> bool:
    lowered = file_path.lower()
    return any(marker in lowered for marker in _TEST_PATH_MARKERS)


def _is_implementation_path(file_path: str) -> bool:
    lowered = file_path.lower()
    if not lowered or _is_test_path(lowered):
        return False
    return not any(lowered.startswith(prefix) for prefix in _NON_IMPLEMENTATION_PREFIXES)


def _as_non_empty_str(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    stripped = value.strip()
    return stripped if stripped else ""


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _normalize_space(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.strip().lower())


__all__ = [
    "CodeSearchQualityConfig",
    "enrich_code_search_results",
    "infer_code_search_chunking_strategy",
    "rerank_code_search_results",
]
