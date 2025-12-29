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

"""SEC Filing RAG Demo - Ingest 10-K/10-Q filings for FAANG stocks.

This demo shows how to use the RAG vertical to ingest and query
SEC filings for major tech companies.

Usage:
    python -m victor.verticals.rag.demo_sec_filings [--company SYMBOL] [--filing-type 10-K|10-Q]

Example:
    # Ingest latest 10-K filings for all FAANG companies
    python -m victor.verticals.rag.demo_sec_filings

    # Ingest only Apple 10-Q filings
    python -m victor.verticals.rag.demo_sec_filings --company AAPL --filing-type 10-Q

    # Query the ingested filings
    python -m victor.verticals.rag.demo_sec_filings --query "What is Apple's revenue?"
"""

import argparse
import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# FAANG company SEC CIK numbers
FAANG_COMPANIES: Dict[str, Dict[str, str]] = {
    "META": {
        "name": "Meta Platforms (Facebook)",
        "cik": "0001326801",
    },
    "AMZN": {
        "name": "Amazon.com Inc",
        "cik": "0001018724",
    },
    "AAPL": {
        "name": "Apple Inc",
        "cik": "0000320193",
    },
    "NFLX": {
        "name": "Netflix Inc",
        "cik": "0001065280",
    },
    "GOOGL": {
        "name": "Alphabet Inc (Google)",
        "cik": "0001652044",
    },
}


@dataclass
class Filing:
    """SEC Filing metadata."""

    company: str
    cik: str
    filing_type: str
    filing_date: str
    accession_number: str
    primary_doc_url: str


class SECFilingFetcher:
    """Fetch SEC filings from EDGAR."""

    BASE_URL = "https://www.sec.gov"
    EDGAR_API = "https://data.sec.gov"

    def __init__(self):
        self.headers = {
            "User-Agent": "VictorRAG/1.0 (research@example.com)",
            "Accept": "application/json",
        }
        # Create SSL context that doesn't verify certificates (for demo purposes)
        import ssl

        self._ssl_context = ssl.create_default_context()
        self._ssl_context.check_hostname = False
        self._ssl_context.verify_mode = ssl.CERT_NONE

    async def get_company_filings(
        self,
        cik: str,
        filing_type: str = "10-K",
        count: int = 1,
    ) -> List[Filing]:
        """Get recent filings for a company.

        Args:
            cik: SEC CIK number (with leading zeros)
            filing_type: Filing type (10-K, 10-Q)
            count: Number of filings to retrieve

        Returns:
            List of Filing objects
        """
        # Remove leading zeros for API call
        cik_no_zeros = cik.lstrip("0")

        # Fetch company submissions
        url = f"{self.EDGAR_API}/submissions/CIK{cik}.json"

        connector = aiohttp.TCPConnector(ssl=self._ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(url, headers=self.headers) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch filings for CIK {cik}: {response.status}")
                    return []

                data = await response.json()

        # Parse recent filings
        filings = []
        recent = data.get("filings", {}).get("recent", {})

        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        for i, form in enumerate(forms):
            if form == filing_type and len(filings) < count:
                accession = accessions[i].replace("-", "")
                filings.append(
                    Filing(
                        company=data.get("name", "Unknown"),
                        cik=cik,
                        filing_type=form,
                        filing_date=dates[i],
                        accession_number=accessions[i],
                        primary_doc_url=f"{self.BASE_URL}/Archives/edgar/data/{cik_no_zeros}/{accession}/{primary_docs[i]}",
                    )
                )

        return filings

    async def fetch_filing_content(self, filing: Filing) -> str:
        """Fetch the full text content of a filing.

        Args:
            filing: Filing metadata

        Returns:
            Text content of the filing
        """
        connector = aiohttp.TCPConnector(ssl=self._ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(
                filing.primary_doc_url,
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as response:
                if response.status != 200:
                    raise Exception(f"Failed to fetch filing: {response.status}")

                content = await response.text()

                # Extract text from HTML
                return self._extract_text_from_html(content)

    def _extract_text_from_html(self, html: str) -> str:
        """Extract readable text from HTML filing."""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style"]):
                element.decompose()

            # Get text
            text = soup.get_text(separator="\n", strip=True)

            # Clean up excessive whitespace
            lines = []
            for line in text.splitlines():
                line = line.strip()
                if line:
                    lines.append(line)

            return "\n".join(lines)

        except ImportError:
            # Fallback: basic regex-based extraction
            text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text)
            return text.strip()


async def ingest_sec_filings(
    companies: Optional[List[str]] = None,
    filing_type: str = "10-K",
    count: int = 1,
) -> Dict[str, int]:
    """Ingest SEC filings into RAG store.

    Args:
        companies: List of company symbols (default: all FAANG)
        filing_type: Filing type to ingest
        count: Number of filings per company

    Returns:
        Dict mapping company symbols to chunk counts
    """
    from victor.verticals.rag.document_store import Document, DocumentStore

    if companies is None:
        companies = list(FAANG_COMPANIES.keys())

    fetcher = SECFilingFetcher()
    store = DocumentStore()
    await store.initialize()

    results = {}

    for symbol in companies:
        if symbol not in FAANG_COMPANIES:
            logger.warning(f"Unknown company symbol: {symbol}")
            continue

        company_info = FAANG_COMPANIES[symbol]
        logger.info(f"Fetching {filing_type} filings for {company_info['name']}...")

        try:
            # Get filing metadata
            filings = await fetcher.get_company_filings(
                cik=company_info["cik"],
                filing_type=filing_type,
                count=count,
            )

            if not filings:
                logger.warning(f"No {filing_type} filings found for {symbol}")
                continue

            total_chunks = 0

            for filing in filings:
                logger.info(f"  Downloading {filing.filing_type} from {filing.filing_date}...")

                # Fetch content
                content = await fetcher.fetch_filing_content(filing)

                if not content:
                    logger.warning(f"  Empty content for {filing.accession_number}")
                    continue

                # Create document
                doc_id = f"sec_{symbol}_{filing.filing_type}_{filing.filing_date}"
                doc = Document(
                    id=doc_id,
                    content=content,
                    source=filing.primary_doc_url,
                    doc_type="text",
                    metadata={
                        "company": company_info["name"],
                        "symbol": symbol,
                        "cik": company_info["cik"],
                        "filing_type": filing.filing_type,
                        "filing_date": filing.filing_date,
                        "accession_number": filing.accession_number,
                    },
                )

                # Ingest
                chunks = await store.add_document(doc)
                total_chunks += len(chunks)
                logger.info(f"  Ingested {len(chunks)} chunks ({len(content):,} chars)")

            results[symbol] = total_chunks

        except Exception as e:
            logger.error(f"Failed to process {symbol}: {e}")
            results[symbol] = 0

    return results


async def query_filings(
    query: str,
    top_k: int = 5,
    synthesize: bool = False,
    provider: str = "ollama",
    model: str = None,
) -> None:
    """Query ingested SEC filings.

    Args:
        query: Search query
        top_k: Number of results to return
        synthesize: Use LLM to synthesize answer
        provider: LLM provider for synthesis
        model: Model name for synthesis
    """
    from victor.verticals.rag.tools.query import RAGQueryTool

    tool = RAGQueryTool()

    logger.info(f"\nQuerying: {query}\n")
    if synthesize:
        logger.info(f"Using {provider} for answer synthesis...")

    result = await tool.execute(
        question=query,
        k=top_k,
        synthesize=synthesize,
        provider=provider if synthesize else None,
        model=model if synthesize else None,
    )

    if not result.success:
        logger.error(f"Query failed: {result.output}")
        return

    print("=" * 80)
    print(result.output)
    print("=" * 80)


async def show_stats() -> None:
    """Show RAG store statistics."""
    from victor.verticals.rag.document_store import DocumentStore

    store = DocumentStore()
    await store.initialize()

    stats = await store.get_stats()

    print("\n" + "=" * 50)
    print("RAG Store Statistics")
    print("=" * 50)
    print(f"Total documents: {stats.get('total_documents', 0)}")
    print(f"Total chunks: {stats.get('total_chunks', 0)}")
    print(f"Store location: {stats.get('store_path', 'N/A')}")

    docs = await store.list_documents()
    if docs:
        print("\nDocuments by company:")
        by_company: Dict[str, int] = {}
        for doc in docs:
            symbol = doc.metadata.get("symbol", "Other")
            by_company[symbol] = by_company.get(symbol, 0) + 1
        for symbol, count in sorted(by_company.items()):
            print(f"  {symbol}: {count} documents")

    print("=" * 50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SEC Filing RAG Demo - Ingest and query 10-K/10-Q filings"
    )
    parser.add_argument(
        "--company",
        "-c",
        type=str,
        help="Company symbol (META, AMZN, AAPL, NFLX, GOOGL). Default: all",
    )
    parser.add_argument(
        "--filing-type",
        "-t",
        type=str,
        default="10-K",
        choices=["10-K", "10-Q"],
        help="Filing type to ingest (default: 10-K)",
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=1,
        help="Number of filings per company (default: 1)",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="Query the ingested filings instead of ingesting",
    )
    parser.add_argument(
        "--synthesize",
        "-S",
        action="store_true",
        help="Use LLM to synthesize answer (requires provider)",
    )
    parser.add_argument(
        "--provider",
        "-p",
        type=str,
        default="ollama",
        help="LLM provider for synthesis (default: ollama)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Model name for synthesis (provider-specific)",
    )
    parser.add_argument(
        "--stats",
        "-s",
        action="store_true",
        help="Show RAG store statistics",
    )
    parser.add_argument(
        "--list-companies",
        "-l",
        action="store_true",
        help="List available companies",
    )

    args = parser.parse_args()

    if args.list_companies:
        print("\nAvailable FAANG Companies:")
        print("-" * 50)
        for symbol, info in FAANG_COMPANIES.items():
            print(f"  {symbol}: {info['name']} (CIK: {info['cik']})")
        print()
        return

    if args.stats:
        asyncio.run(show_stats())
        return

    if args.query:
        asyncio.run(
            query_filings(
                args.query,
                synthesize=args.synthesize,
                provider=args.provider,
                model=args.model,
            )
        )
        return

    # Ingest filings
    companies = [args.company.upper()] if args.company else None

    print(f"\n{'=' * 60}")
    print("SEC Filing RAG Demo")
    print(f"{'=' * 60}")
    print(f"Filing Type: {args.filing_type}")
    print(f"Companies: {companies or 'All FAANG'}")
    print(f"Count per company: {args.count}")
    print(f"{'=' * 60}\n")

    results = asyncio.run(
        ingest_sec_filings(
            companies=companies,
            filing_type=args.filing_type,
            count=args.count,
        )
    )

    print(f"\n{'=' * 60}")
    print("Ingestion Complete!")
    print(f"{'=' * 60}")
    for symbol, chunks in results.items():
        company_name = FAANG_COMPANIES.get(symbol, {}).get("name", symbol)
        print(f"  {company_name}: {chunks} chunks")
    print(f"{'=' * 60}")
    print("\nYou can now query the filings using victor CLI:")
    print("  # Context only (default):")
    print('  victor rag query "What is Apple\'s revenue?"')
    print("")
    print("  # With LLM synthesis (using default provider):")
    print('  victor rag query "What is Apple\'s revenue?" --synthesize')
    print("")
    print("  # With specific provider/model:")
    print('  victor rag query "Risk factors" -S -p anthropic -m claude-sonnet-4-20250514')
    print("")
    print("  # Or use the demo script:")
    print('  python -m victor.verticals.rag.demo_sec_filings --query "Revenue" --synthesize')
    print()


if __name__ == "__main__":
    main()
