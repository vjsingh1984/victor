from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import victor.agent.coordinators.protocol_based_injection_example as protocol_example
import victor.verticals.contrib.rag.demo_docs as demo_docs
import victor.verticals.contrib.rag.demo_sec_filings as demo_sec


class TestDemoDocsSyncBridge:
    def test_main_uses_shared_sync_bridge_for_stats(self) -> None:
        coro = object()
        args = SimpleNamespace(path=None, pattern=None, query=None, stats=True, victor=False)

        with (
            patch.object(demo_docs.argparse.ArgumentParser, "parse_args", return_value=args),
            patch.object(demo_docs, "show_stats", Mock(return_value=coro)) as mock_show_stats,
            patch.object(demo_docs, "run_sync", return_value=None) as mock_run_sync,
        ):
            demo_docs.main()

        mock_show_stats.assert_called_once_with()
        mock_run_sync.assert_called_once_with(coro)

    def test_main_uses_shared_sync_bridge_for_query(self) -> None:
        coro = object()
        args = SimpleNamespace(
            path=None, pattern=None, query="auth flow", stats=False, victor=False
        )

        with (
            patch.object(demo_docs.argparse.ArgumentParser, "parse_args", return_value=args),
            patch.object(demo_docs, "query_docs", Mock(return_value=coro)) as mock_query_docs,
            patch.object(demo_docs, "run_sync", return_value=None) as mock_run_sync,
        ):
            demo_docs.main()

        mock_query_docs.assert_called_once_with("auth flow")
        mock_run_sync.assert_called_once_with(coro)

    def test_main_uses_shared_sync_bridge_for_custom_ingest(self) -> None:
        coro = object()
        args = SimpleNamespace(
            path="/tmp/project",
            pattern=["*.rst"],
            query=None,
            stats=False,
            victor=False,
        )

        with (
            patch.object(demo_docs.argparse.ArgumentParser, "parse_args", return_value=args),
            patch.object(
                demo_docs, "ingest_project_docs", Mock(return_value=coro)
            ) as mock_ingest_docs,
            patch.object(demo_docs, "run_sync", return_value={"*.rst": 4}) as mock_run_sync,
            patch("builtins.print"),
        ):
            demo_docs.main()

        mock_ingest_docs.assert_called_once_with(
            project_path=Path("/tmp/project"),
            patterns=["*.rst"],
        )
        mock_run_sync.assert_called_once_with(coro)


class TestDemoSecSyncBridge:
    def test_main_uses_shared_sync_bridge_for_stats(self) -> None:
        coro = object()
        args = SimpleNamespace(
            company=None,
            preset=None,
            filing_type="10-K",
            count=1,
            query=None,
            synthesize=False,
            provider="ollama",
            model=None,
            stats=True,
            list=False,
            sector=None,
            clear=False,
            max_concurrent=5,
        )

        with (
            patch.object(demo_sec.argparse.ArgumentParser, "parse_args", return_value=args),
            patch.object(demo_sec, "show_stats", Mock(return_value=coro)) as mock_show_stats,
            patch.object(demo_sec, "run_sync", return_value=None) as mock_run_sync,
        ):
            demo_sec.main()

        mock_show_stats.assert_called_once_with()
        mock_run_sync.assert_called_once_with(coro)

    def test_main_uses_shared_sync_bridge_for_query(self) -> None:
        coro = object()
        args = SimpleNamespace(
            company=["AAPL"],
            preset=None,
            filing_type="10-K",
            count=1,
            query="risk factors",
            synthesize=True,
            provider="anthropic",
            model="claude",
            stats=False,
            list=False,
            sector="Technology",
            clear=False,
            max_concurrent=5,
        )

        with (
            patch.object(demo_sec.argparse.ArgumentParser, "parse_args", return_value=args),
            patch.object(demo_sec, "query_filings", Mock(return_value=coro)) as mock_query_filings,
            patch.object(demo_sec, "run_sync", return_value=None) as mock_run_sync,
        ):
            demo_sec.main()

        mock_query_filings.assert_called_once_with(
            "risk factors",
            synthesize=True,
            provider="anthropic",
            model="claude",
            filter_sector="Technology",
            filter_symbol="AAPL",
        )
        mock_run_sync.assert_called_once_with(coro)

    def test_main_uses_shared_sync_bridge_for_ingest(self) -> None:
        coro = object()
        args = SimpleNamespace(
            company=None,
            preset="faang",
            filing_type="10-Q",
            count=2,
            query=None,
            synthesize=False,
            provider="ollama",
            model=None,
            stats=False,
            list=False,
            sector=None,
            clear=False,
            max_concurrent=3,
        )

        with (
            patch.object(demo_sec.argparse.ArgumentParser, "parse_args", return_value=args),
            patch.object(demo_sec, "run_sync", return_value={"AAPL": 2}) as mock_run_sync,
            patch.object(demo_sec, "ingest_sec_filings", Mock(return_value=coro)) as mock_ingest,
            patch("builtins.print"),
        ):
            demo_sec.main()

        mock_ingest.assert_called_once_with(
            companies=demo_sec.COMPANY_PRESETS["faang"],
            filing_type="10-Q",
            count=2,
            max_concurrent=3,
        )
        mock_run_sync.assert_called_once_with(coro)


class TestProtocolInjectionExampleSyncBridge:
    def test_main_uses_shared_sync_bridge(self) -> None:
        coro = object()

        with (
            patch.object(protocol_example, "print_benefits"),
            patch.object(
                protocol_example,
                "example_chat_with_mock_orchestrator",
                Mock(return_value=coro),
            ) as mock_example,
            patch.object(protocol_example, "run_sync", return_value=None) as mock_run_sync,
            patch("builtins.print"),
        ):
            protocol_example.main()

        mock_example.assert_called_once_with()
        mock_run_sync.assert_called_once_with(coro)
