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

"""RAG CLI commands for Victor.

Provides CLI commands for RAG (Retrieval-Augmented Generation) operations:
- Ingest files, URLs, or directories
- Search the knowledge base
- Query with LLM synthesis
- Manage documents

Usage:
    victor rag ingest ./docs --pattern "*.md" --recursive
    victor rag ingest https://example.com/docs
    victor rag search "authentication"
    victor rag query "What is the auth flow?" --synthesize
    victor rag list
    victor rag stats
    victor rag --log-level DEBUG query "question" --synthesize
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from victor.ui.commands.utils import setup_logging

rag_app = typer.Typer(
    name="rag",
    help="RAG (Retrieval-Augmented Generation) operations - ingest, search, query documents.",
)
console = Console()


def _configure_log_level(log_level: Optional[str]) -> None:
    """Configure logging for RAG commands using centralized config.

    Uses the centralized logging config system with proper priority chain:
    1. CLI argument (log_level)
    2. Environment variable (VICTOR_LOG_LEVEL)
    3. User config (~/.victor/config.yaml)
    4. Command-specific override from package config
    5. Package defaults (WARNING console, INFO file)
    """
    # Validate log level if provided
    if log_level is not None:
        log_level = log_level.upper()
        valid_levels = {"DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL"}

        if log_level not in valid_levels:
            console.print(
                f"[bold red]Error:[/] Invalid log level '{log_level}'. "
                f"Valid options: {', '.join(sorted(valid_levels))}"
            )
            raise typer.Exit(1)

        if log_level == "WARN":
            log_level = "WARNING"

    # Use centralized logging config
    setup_logging(command="rag", cli_log_level=log_level, stream=sys.stderr)

    # Also configure debug logger levels if explicitly set
    if log_level:
        try:
            from victor.agent.debug_logger import configure_logging_levels

            configure_logging_levels(log_level)
        except ImportError:
            pass


@rag_app.callback()
def rag_callback(
    log_level: Optional[str] = typer.Option(
        None,
        "--log-level",
        "-l",
        help="Set logging level (DEBUG, INFO, WARN, ERROR). Defaults to WARNING or VICTOR_LOG_LEVEL env var.",
    ),
) -> None:
    """RAG (Retrieval-Augmented Generation) operations."""
    _configure_log_level(log_level)


@rag_app.command("ingest")
def ingest(
    source: str = typer.Argument(
        ...,
        help="File path, directory path, or URL to ingest",
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive",
        "-r",
        help="Recursively ingest directory contents",
    ),
    pattern: str = typer.Option(
        "*",
        "--pattern",
        "-p",
        help="Glob pattern for directory ingestion (e.g., '*.md')",
    ),
    doc_type: str = typer.Option(
        "auto",
        "--type",
        "-t",
        help="Document type: auto, text, markdown, code, pdf, html",
    ),
    doc_id: Optional[str] = typer.Option(
        None,
        "--id",
        help="Custom document ID",
    ),
) -> None:
    """Ingest documents into the RAG knowledge base.

    Examples:
        victor rag ingest ./README.md
        victor rag ingest ./docs --recursive --pattern "*.md"
        victor rag ingest https://example.com/api-docs
    """
    asyncio.run(_ingest_async(source, recursive, pattern, doc_type, doc_id))


async def _ingest_async(
    source: str,
    recursive: bool,
    pattern: str,
    doc_type: str,
    doc_id: Optional[str],
) -> None:
    """Async implementation of ingest command."""
    from victor.rag.tools.ingest import RAGIngestTool

    tool = RAGIngestTool()

    # Determine if source is URL or path
    is_url = source.startswith(("http://", "https://"))

    with console.status(f"[bold blue]Ingesting {source}...[/]"):
        if is_url:
            result = await tool.execute(
                url=source,
                doc_type=doc_type if doc_type != "auto" else "text",
                doc_id=doc_id,
            )
        else:
            path = Path(source).resolve()
            result = await tool.execute(
                path=str(path),
                recursive=recursive,
                pattern=pattern,
                doc_type=doc_type if doc_type != "auto" else "text",
                doc_id=doc_id,
            )

    if result.success:
        console.print(f"[green]{result.output}[/]")
    else:
        console.print(f"[red]Error: {result.output}[/]")
        raise typer.Exit(1)


@rag_app.command("search")
def search(
    query: str = typer.Argument(
        ...,
        help="Search query",
    ),
    top_k: int = typer.Option(
        5,
        "--top-k",
        "-k",
        help="Number of results to return",
    ),
) -> None:
    """Search the RAG knowledge base for relevant documents.

    Examples:
        victor rag search "authentication"
        victor rag search "error handling" --top-k 10
    """
    asyncio.run(_search_async(query, top_k))


async def _search_async(query: str, top_k: int) -> None:
    """Async implementation of search command."""
    from victor.rag.tools.search import RAGSearchTool

    tool = RAGSearchTool()

    with console.status(f"[bold blue]Searching for: {query}...[/]"):
        result = await tool.execute(query=query, k=top_k)

    if result.success:
        console.print(result.output)
    else:
        console.print(f"[red]Error: {result.output}[/]")
        raise typer.Exit(1)


@rag_app.command("query")
def query(
    question: str = typer.Argument(
        ...,
        help="Question to answer using the knowledge base",
    ),
    synthesize: bool = typer.Option(
        False,
        "--synthesize",
        "-S",
        help="Use LLM to synthesize answer",
    ),
    provider: str = typer.Option(
        "ollama",
        "--provider",
        "-p",
        help="LLM provider for synthesis (e.g., 'ollama', 'anthropic', 'openai')",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model name for synthesis",
    ),
    top_k: int = typer.Option(
        5,
        "--top-k",
        "-k",
        help="Number of context chunks to retrieve",
    ),
    show_enrichment: bool = typer.Option(
        True,
        "--show-enrichment/--no-enrichment",
        "-e/-E",
        help="Show query enrichment details (entities, enhanced query)",
    ),
) -> None:
    """Query the knowledge base and optionally synthesize an answer.

    Examples:
        victor rag query "What is the authentication flow?"
        victor rag query "How do I add a new provider?" --synthesize
        victor rag query "API endpoints" -S -p anthropic -m claude-sonnet-4-20250514
        victor rag query "Compare Apple and Microsoft" -S --no-enrichment
    """
    asyncio.run(_query_async(question, synthesize, provider, model, top_k, show_enrichment))


async def _query_async(
    question: str,
    synthesize: bool,
    provider: str,
    model: Optional[str],
    top_k: int,
    show_enrichment: bool = True,
) -> None:
    """Async implementation of query command."""
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.text import Text

    from victor.rag.tools.query import RAGQueryTool

    # Show enrichment details if enabled
    if show_enrichment:
        await _show_enrichment_details(question)

    tool = RAGQueryTool()

    status_msg = f"Querying: {question}"
    if synthesize:
        status_msg += f" (synthesizing with {provider})"

    with console.status(f"[bold blue]{status_msg}...[/]"):
        result = await tool.execute(
            question=question,
            k=top_k,
            synthesize=synthesize,
            provider=provider if synthesize else None,
            model=model if synthesize else None,
        )

    if result.success:
        if synthesize:
            # Render markdown for synthesized answers
            console.print(Markdown(result.output))
        else:
            # Plain text for context-only mode
            console.print(result.output)
    else:
        console.print(f"[red]Error: {result.output}[/]")
        raise typer.Exit(1)


async def _show_enrichment_details(question: str) -> None:
    """Display query enrichment details for educational purposes.

    Args:
        question: The original query
    """
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    from victor.rag.document_store import DocumentStore
    from victor.rag.enrichment import get_rag_enrichment_strategy, reset_rag_enrichment_strategy

    # Reset to ensure fresh initialization
    reset_rag_enrichment_strategy()

    try:
        # Initialize document store and enrichment strategy
        store = DocumentStore()
        await store.initialize()

        enrichment_strategy = get_rag_enrichment_strategy(document_store=store)
        await enrichment_strategy.initialize(document_store=store)

        # Analyze query
        analysis = await enrichment_strategy.analyze_query_async(question)

        # Build enrichment panel content
        content_lines = []

        # Original query
        content_lines.append(f"[bold]Original Query:[/] {question}")
        content_lines.append("")

        # Entities detected
        entities = analysis.get("entities", [])
        if entities:
            content_lines.append(f"[bold]Entities Detected:[/] {len(entities)}")
            for e in entities:
                if hasattr(e, "name"):
                    parts = [f"  • [cyan]{e.name}[/]"]
                    if hasattr(e, "ticker") and e.ticker:
                        parts.append(f"([yellow]{e.ticker}[/])")
                    if hasattr(e, "sector") and e.sector:
                        parts.append(f"[dim][{e.sector}][/]")
                    content_lines.append(" ".join(parts))
                else:
                    content_lines.append(f"  • {str(e)}")
            content_lines.append("")

        # Query type
        is_comparison = analysis.get("is_comparison", False)
        if is_comparison:
            content_lines.append("[bold]Query Type:[/] [magenta]Comparison Query[/]")
        else:
            content_lines.append("[bold]Query Type:[/] Standard Query")

        # Recommended k
        recommended_k = analysis.get("recommended_k", 5)
        content_lines.append(f"[bold]Recommended k:[/] {recommended_k}")
        content_lines.append("")

        # Enhanced query
        enhanced_query = analysis.get("enhanced_query", question)
        if enhanced_query != question:
            content_lines.append("[bold]Enhanced Query:[/]")
            content_lines.append(f"  [green]{enhanced_query}[/]")
        else:
            content_lines.append("[bold]Enhanced Query:[/] [dim](no enhancement needed)[/]")

        # Create panel
        panel = Panel(
            "\n".join(content_lines),
            title="[bold blue]Query Enrichment[/]",
            subtitle="[dim]Dynamic entity resolution from document metadata[/]",
            border_style="blue",
            padding=(1, 2),
        )
        console.print(panel)
        console.print()  # Add spacing

    except Exception as e:
        # Don't fail the query if enrichment display fails
        console.print(f"[dim yellow]Note: Could not display enrichment details: {e}[/]")
        console.print()


@rag_app.command("list")
def list_docs() -> None:
    """List all documents in the RAG knowledge base.

    Example:
        victor rag list
    """
    asyncio.run(_list_async())


async def _list_async() -> None:
    """Async implementation of list command."""
    from victor.rag.tools.management import RAGListTool

    tool = RAGListTool()
    result = await tool.execute()

    if result.success:
        console.print(result.output)
    else:
        console.print(f"[red]Error: {result.output}[/]")
        raise typer.Exit(1)


@rag_app.command("stats")
def stats() -> None:
    """Show RAG knowledge base statistics.

    Example:
        victor rag stats
    """
    asyncio.run(_stats_async())


async def _stats_async() -> None:
    """Async implementation of stats command."""
    from victor.rag.tools.management import RAGStatsTool

    tool = RAGStatsTool()
    result = await tool.execute()

    if result.success:
        console.print(result.output)
    else:
        console.print(f"[red]Error: {result.output}[/]")
        raise typer.Exit(1)


@rag_app.command("delete")
def delete(
    doc_id: str = typer.Argument(
        ...,
        help="Document ID to delete",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Delete without confirmation",
    ),
) -> None:
    """Delete a document from the RAG knowledge base.

    Example:
        victor rag delete doc_abc123
        victor rag delete doc_abc123 --force
    """
    if not force:
        confirm = typer.confirm(f"Delete document '{doc_id}'?")
        if not confirm:
            raise typer.Abort()

    asyncio.run(_delete_async(doc_id))


async def _delete_async(doc_id: str) -> None:
    """Async implementation of delete command."""
    from victor.rag.tools.management import RAGDeleteTool

    tool = RAGDeleteTool()
    result = await tool.execute(doc_id=doc_id)

    if result.success:
        console.print(f"[green]{result.output}[/]")
    else:
        console.print(f"[red]Error: {result.output}[/]")
        raise typer.Exit(1)


@rag_app.command("demo")
def demo(
    demo_type: str = typer.Argument(
        "docs",
        help="Demo type: 'docs' for project docs, 'sec' for SEC filings",
    ),
) -> None:
    """Run RAG demo to ingest sample documents.

    Examples:
        victor rag demo docs     # Ingest project documentation
        victor rag demo sec      # Ingest SEC filings (requires network)
    """
    if demo_type == "docs":
        console.print("[bold blue]Running project documentation demo...[/]")
        console.print("This will ingest documentation from the current project.\n")
        asyncio.run(_demo_docs())
    elif demo_type == "sec":
        console.print("[bold blue]Running SEC filing demo...[/]")
        console.print("Use 'victor rag demo-sec' for full SEC filing functionality.\n")
        console.print("Quick start: victor rag demo-sec --preset faang")
    else:
        console.print(f"[red]Unknown demo type: {demo_type}[/]")
        console.print("Available: docs, sec")
        raise typer.Exit(1)


@rag_app.command("demo-sec")
def demo_sec(
    company: Optional[list[str]] = typer.Option(
        None,
        "--company",
        "-c",
        help="Company symbol(s) to process (can be specified multiple times)",
    ),
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        "-P",
        help="Use a preset: faang, mag7, top10, top25, top50, top100, tech, healthcare, financials, energy",
    ),
    filing_type: str = typer.Option(
        "10-K",
        "--filing-type",
        "-t",
        help="Filing type to ingest (10-K or 10-Q)",
    ),
    count: int = typer.Option(
        1,
        "--count",
        "-n",
        help="Number of filings per company",
    ),
    query: Optional[str] = typer.Option(
        None,
        "--query",
        "-q",
        help="Query the ingested filings instead of ingesting",
    ),
    synthesize: bool = typer.Option(
        False,
        "--synthesize",
        "-S",
        help="Use LLM to synthesize answer",
    ),
    provider: str = typer.Option(
        "ollama",
        "--provider",
        "-p",
        help="LLM provider for synthesis",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model name for synthesis",
    ),
    sector: Optional[str] = typer.Option(
        None,
        "--sector",
        help="Filter by sector",
    ),
    stats: bool = typer.Option(
        False,
        "--stats",
        "-s",
        help="Show SEC filings statistics",
    ),
    list_companies: bool = typer.Option(
        False,
        "--list",
        "-l",
        help="List available companies",
    ),
    clear: bool = typer.Option(
        False,
        "--clear",
        help="Clear all SEC filings from RAG store",
    ),
    max_concurrent: int = typer.Option(
        5,
        "--max-concurrent",
        help="Maximum concurrent downloads",
    ),
) -> None:
    """Ingest and query SEC 10-K/10-Q filings for S&P 500 stocks.

    Examples:
        victor rag demo-sec --preset faang           # FAANG stocks
        victor rag demo-sec --preset top50           # Top 50 S&P 500
        victor rag demo-sec --company AAPL           # Single company
        victor rag demo-sec --list                   # List available companies
        victor rag demo-sec --list --preset tech     # List tech companies
        victor rag demo-sec --stats                  # Show SEC filing stats
        victor rag demo-sec --clear                  # Clear all SEC filings
    """
    from victor.rag.demo_sec_filings import (
        COMPANY_PRESETS,
        SP500_COMPANIES,
        clear_sec_filings,
        ingest_sec_filings,
        list_companies as _list_companies,
        query_filings,
        show_stats,
    )

    if list_companies:
        _list_companies(preset=preset, sector=sector)
        return

    if stats:
        asyncio.run(show_stats())
        return

    if clear:
        console.print("[bold yellow]Clearing SEC filings from RAG store...[/]")
        count_deleted = asyncio.run(clear_sec_filings())
        console.print(f"[green]Removed {count_deleted} SEC filing documents[/]")
        return

    if query:
        asyncio.run(
            query_filings(
                query,
                synthesize=synthesize,
                provider=provider,
                model=model,
                filter_sector=sector,
                filter_symbol=company[0] if company else None,
            )
        )
        return

    # Determine companies to process
    if preset:
        if preset not in COMPANY_PRESETS:
            console.print(f"[red]Unknown preset: {preset}[/]")
            console.print(f"Available: {', '.join(COMPANY_PRESETS.keys())}")
            raise typer.Exit(1)
        companies = COMPANY_PRESETS[preset]
    elif company:
        companies = [c.upper() for c in company]
    else:
        # Default to FAANG for quick demo
        companies = COMPANY_PRESETS["faang"]
        preset = "faang"

    console.print("\n[bold blue]SEC Filing RAG Demo - S&P 500[/]")
    console.print(f"[dim]Filing Type: {filing_type}[/]")
    console.print(f"[dim]Companies: {len(companies)} ({preset or 'custom'})[/]")
    console.print(f"[dim]Count per company: {count}[/]")
    console.print(f"[dim]Max concurrent: {max_concurrent}[/]\n")

    with console.status("[bold blue]Ingesting SEC filings...[/]"):
        results = asyncio.run(
            ingest_sec_filings(
                companies=companies,
                filing_type=filing_type,
                count=count,
                max_concurrent=max_concurrent,
            )
        )

    # Display results
    table = Table(title="Ingestion Results")
    table.add_column("Symbol", style="cyan")
    table.add_column("Company", style="white")
    table.add_column("Chunks", justify="right", style="green")

    total_chunks = 0
    for symbol, chunks in sorted(results.items()):
        company_name = SP500_COMPANIES.get(symbol, {}).get("name", symbol)
        table.add_row(symbol, company_name, str(chunks))
        total_chunks += chunks

    table.add_row("", "[bold]TOTAL[/]", f"[bold]{total_chunks}[/]", style="bold")
    console.print(table)

    console.print("\n[bold green]Ingestion complete![/]")
    console.print("\nQuery the filings:")
    console.print('  [dim]victor rag query "What is Apple\'s revenue?"[/]')
    console.print('  [dim]victor rag query "Compare revenue growth" --synthesize[/]')


async def _demo_docs() -> None:
    """Run the project docs demo."""
    from victor.rag.demo_docs import ingest_victor_docs

    try:
        results = await ingest_victor_docs()
        console.print("\n[green]Demo complete![/]")
        total = sum(results.values())
        console.print(f"Ingested {total} chunks from project documentation.")
        console.print("\nTry querying:")
        console.print('  victor rag query "How do I add a new provider?"')
        console.print('  victor rag query "What tools are available?" --synthesize')
    except Exception as e:
        console.print(f"[red]Demo failed: {e}[/]")
        raise typer.Exit(1)
