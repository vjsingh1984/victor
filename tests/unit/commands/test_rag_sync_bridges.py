from __future__ import annotations

from unittest.mock import Mock, call, patch

import victor.verticals.contrib.rag.commands.rag as rag_cmd


def _call_demo_sec(**overrides: object) -> None:
    rag_cmd.demo_sec(
        company=overrides.pop("company", None),
        preset=overrides.pop("preset", None),
        filing_type=overrides.pop("filing_type", "10-K"),
        count=overrides.pop("count", 1),
        query=overrides.pop("query", None),
        synthesize=overrides.pop("synthesize", False),
        provider=overrides.pop("provider", "ollama"),
        model=overrides.pop("model", None),
        sector=overrides.pop("sector", None),
        stats=overrides.pop("stats", False),
        list_companies=overrides.pop("list_companies", False),
        clear=overrides.pop("clear", False),
        max_concurrent=overrides.pop("max_concurrent", 5),
    )
    assert not overrides


def _demo_sec_loader(
    *,
    company_presets: dict[str, list[str]] | None = None,
    sp500_companies: dict[str, dict[str, str]] | None = None,
    clear_sec_filings: object | None = None,
    ingest_sec_filings: object | None = None,
    list_companies: object | None = None,
    query_filings: object | None = None,
    show_stats: object | None = None,
):
    attrs = {
        ("victor.rag.demo_sec_filings", "COMPANY_PRESETS"): company_presets or {"faang": ["AAPL"]},
        ("victor.rag.demo_sec_filings", "SP500_COMPANIES"): sp500_companies
        or {"AAPL": {"name": "Apple Inc."}},
        ("victor.rag.demo_sec_filings", "clear_sec_filings"): clear_sec_filings or Mock(),
        ("victor.rag.demo_sec_filings", "ingest_sec_filings"): ingest_sec_filings or Mock(),
        ("victor.rag.demo_sec_filings", "list_companies"): list_companies or Mock(),
        ("victor.rag.demo_sec_filings", "query_filings"): query_filings or Mock(),
        ("victor.rag.demo_sec_filings", "show_stats"): show_stats or Mock(),
    }

    def _loader(module_path: str, attr_name: str):
        return attrs[(module_path, attr_name)]

    return _loader


class TestRagSyncBridge:
    def test_ingest_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(rag_cmd, "_ingest_async", mock_async),
            patch.object(rag_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            rag_cmd.ingest(
                source="./docs",
                recursive=True,
                pattern="*.md",
                doc_type="markdown",
                doc_id="doc-1",
            )

        mock_async.assert_called_once_with("./docs", True, "*.md", "markdown", "doc-1")
        mock_run_sync.assert_called_once_with(coro)

    def test_search_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(rag_cmd, "_search_async", mock_async),
            patch.object(rag_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            rag_cmd.search(query="auth flow", top_k=7)

        mock_async.assert_called_once_with("auth flow", 7)
        mock_run_sync.assert_called_once_with(coro)

    def test_query_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(rag_cmd, "_query_async", mock_async),
            patch.object(rag_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            rag_cmd.query(
                question="How does auth work?",
                synthesize=True,
                provider="openai",
                model="gpt-5.4",
                top_k=4,
                show_enrichment=False,
            )

        mock_async.assert_called_once_with(
            "How does auth work?",
            True,
            "openai",
            "gpt-5.4",
            4,
            False,
        )
        mock_run_sync.assert_called_once_with(coro)

    def test_list_stats_and_delete_use_shared_sync_bridge(self) -> None:
        list_coro = object()
        stats_coro = object()
        delete_coro = object()

        with (
            patch.object(rag_cmd, "_list_async", Mock(return_value=list_coro)) as mock_list,
            patch.object(rag_cmd, "_stats_async", Mock(return_value=stats_coro)) as mock_stats,
            patch.object(rag_cmd, "_delete_async", Mock(return_value=delete_coro)) as mock_delete,
            patch.object(rag_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            rag_cmd.list_docs()
            rag_cmd.stats()
            rag_cmd.delete(doc_id="doc-123", force=True)

        mock_list.assert_called_once_with()
        mock_stats.assert_called_once_with()
        mock_delete.assert_called_once_with("doc-123")
        assert mock_run_sync.call_args_list == [call(list_coro), call(stats_coro), call(delete_coro)]

    def test_demo_docs_uses_shared_sync_bridge(self) -> None:
        coro = object()
        mock_async = Mock(return_value=coro)

        with (
            patch.object(rag_cmd, "_demo_docs", mock_async),
            patch.object(rag_cmd, "run_sync", return_value=None) as mock_run_sync,
            patch.object(rag_cmd.console, "print"),
        ):
            rag_cmd.demo(demo_type="docs")

        mock_async.assert_called_once_with()
        mock_run_sync.assert_called_once_with(coro)

    def test_demo_sec_stats_uses_shared_sync_bridge(self) -> None:
        coro = object()
        show_stats = Mock(return_value=coro)

        with (
            patch.object(
                rag_cmd,
                "_load_rag_attr",
                side_effect=_demo_sec_loader(show_stats=show_stats),
            ),
            patch.object(rag_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            _call_demo_sec(stats=True)

        show_stats.assert_called_once_with()
        mock_run_sync.assert_called_once_with(coro)

    def test_demo_sec_clear_uses_shared_sync_bridge(self) -> None:
        coro = object()
        clear_sec_filings = Mock(return_value=coro)

        with (
            patch.object(
                rag_cmd,
                "_load_rag_attr",
                side_effect=_demo_sec_loader(clear_sec_filings=clear_sec_filings),
            ),
            patch.object(rag_cmd, "run_sync", return_value=5) as mock_run_sync,
            patch.object(rag_cmd.console, "print"),
        ):
            _call_demo_sec(clear=True)

        clear_sec_filings.assert_called_once_with()
        mock_run_sync.assert_called_once_with(coro)

    def test_demo_sec_query_uses_shared_sync_bridge(self) -> None:
        coro = object()
        query_filings = Mock(return_value=coro)

        with (
            patch.object(
                rag_cmd,
                "_load_rag_attr",
                side_effect=_demo_sec_loader(query_filings=query_filings),
            ),
            patch.object(rag_cmd, "run_sync", return_value=None) as mock_run_sync,
        ):
            _call_demo_sec(
                company=["AAPL"],
                query="revenue growth",
                synthesize=True,
                provider="anthropic",
                model="claude",
                sector="technology",
            )

        query_filings.assert_called_once_with(
            "revenue growth",
            synthesize=True,
            provider="anthropic",
            model="claude",
            filter_sector="technology",
            filter_symbol="AAPL",
        )
        mock_run_sync.assert_called_once_with(coro)

    def test_demo_sec_ingest_uses_shared_sync_bridge(self) -> None:
        coro = object()
        ingest_sec_filings = Mock(return_value=coro)

        with (
            patch.object(
                rag_cmd,
                "_load_rag_attr",
                side_effect=_demo_sec_loader(ingest_sec_filings=ingest_sec_filings),
            ),
            patch.object(rag_cmd, "run_sync", return_value={"AAPL": 3}) as mock_run_sync,
            patch.object(rag_cmd.console, "print"),
        ):
            _call_demo_sec(
                preset="faang",
                filing_type="10-Q",
                count=2,
                max_concurrent=3,
            )

        ingest_sec_filings.assert_called_once_with(
            companies=["AAPL"],
            filing_type="10-Q",
            count=2,
            max_concurrent=3,
        )
        mock_run_sync.assert_called_once_with(coro)
