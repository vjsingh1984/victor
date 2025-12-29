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
"""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

rag_app = typer.Typer(
    name="rag",
    help="RAG (Retrieval-Augmented Generation) operations - ingest, search, query documents.",
)
console = Console()


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
    from victor.verticals.rag.tools.ingest import RAGIngestTool

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
    from victor.verticals.rag.tools.search import RAGSearchTool

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
) -> None:
    """Query the knowledge base and optionally synthesize an answer.

    Examples:
        victor rag query "What is the authentication flow?"
        victor rag query "How do I add a new provider?" --synthesize
        victor rag query "API endpoints" -S -p anthropic -m claude-sonnet-4-20250514
    """
    asyncio.run(_query_async(question, synthesize, provider, model, top_k))


async def _query_async(
    question: str,
    synthesize: bool,
    provider: str,
    model: Optional[str],
    top_k: int,
) -> None:
    """Async implementation of query command."""
    from victor.verticals.rag.tools.query import RAGQueryTool

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
        console.print(result.output)
    else:
        console.print(f"[red]Error: {result.output}[/]")
        raise typer.Exit(1)


@rag_app.command("list")
def list_docs() -> None:
    """List all documents in the RAG knowledge base.

    Example:
        victor rag list
    """
    asyncio.run(_list_async())


async def _list_async() -> None:
    """Async implementation of list command."""
    from victor.verticals.rag.tools.management import RAGListTool

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
    from victor.verticals.rag.tools.management import RAGStatsTool

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
    from victor.verticals.rag.tools.management import RAGDeleteTool

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
        console.print("This will download and ingest SEC 10-K filings for FAANG stocks.\n")
        console.print("Run with: python -m victor.verticals.rag.demo_sec_filings --company AAPL")
    else:
        console.print(f"[red]Unknown demo type: {demo_type}[/]")
        console.print("Available: docs, sec")
        raise typer.Exit(1)


async def _demo_docs() -> None:
    """Run the project docs demo."""
    from victor.verticals.rag.demo_docs import ingest_victor_docs

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
