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

"""Unified symbol extraction facade with tier-aware strategy selection.

Provides a single interface for symbol extraction that automatically
selects the optimal parsing strategy based on language tier:

- Tier 1 (Python, TS/JS): Native AST + Tree-sitter + LSP enrichment
- Tier 2 (Go, Rust, Java, C/C++): Tree-sitter + LSP enrichment
- Tier 3 (All others): Tree-sitter only

Usage:
    from victor.coding.codebase.unified_extractor import UnifiedSymbolExtractor

    extractor = UnifiedSymbolExtractor(
        tree_sitter=ts_extractor,
        lsp_service=lsp_pool,
    )
    symbols = await extractor.extract_symbols(Path("main.py"))

    for symbol in symbols:
        print(f"{symbol.name}: {symbol.symbol_type} @ line {symbol.line_number}")
        if symbol.return_type:
            print(f"  Returns: {symbol.return_type}")
        if symbol.parameters:
            print(f"  Params: {', '.join(symbol.parameters)}")

SOLID Principles Applied:
    - SRP: Facade delegates to specialized extractors
    - OCP: New tiers/languages add configs, not code changes
    - LSP: All extractors produce compatible EnrichedSymbol
    - ISP: LSPServiceProtocol is minimal interface
    - DIP: Depends on protocols, not concrete implementations

Integration with LanguageCapabilityRegistry:
    This module now uses the unified LanguageCapabilityRegistry from
    victor.core.language_capabilities for language detection and tier
    configuration. This provides a single source of truth for both
    indexing and validation across the codebase.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from victor.core.errors import FileError, ValidationError

logger = logging.getLogger(__name__)

# Import from unified language capabilities registry (new system)
from victor.core.language_capabilities import (
    LanguageCapabilityRegistry,
    LanguageTier as UnifiedLanguageTier,
    UnifiedLanguageCapability,
)

# Also import legacy tier system for backward compatibility
from victor.coding.languages.tiers import LanguageTier, get_tier

if TYPE_CHECKING:
    from victor.coding.codebase.tree_sitter_extractor import (
        ExtractedSymbol,
        TreeSitterExtractor,
    )
    from victor.framework.lsp_protocols import LSPServiceProtocol

logger = logging.getLogger(__name__)


@dataclass
class EnrichedSymbol:
    """Symbol with tier-appropriate enrichment.

    Base fields (all tiers):
        - name, symbol_type, file_path, line_number, end_line
        - signature, docstring, parent_symbol

    Tier 1/2 enrichments (from LSP or native AST):
        - return_type, parameters, visibility
        - is_async, decorators

    Metadata:
        - source_tier: Which tier was used for extraction
    """

    name: str
    symbol_type: str  # class, function, method, variable, etc.
    file_path: str
    line_number: int
    end_line: Optional[int] = None
    signature: Optional[str] = None
    docstring: Optional[str] = None
    parent_symbol: Optional[str] = None
    # Tier 1/2 enrichments (from LSP or native AST)
    return_type: Optional[str] = None
    parameters: list[str] = field(default_factory=list)
    visibility: Optional[str] = None  # public, private, protected
    is_async: bool = False
    decorators: list[str] = field(default_factory=list)
    # Source tier for debugging/analytics
    source_tier: Optional[LanguageTier] = None


class UnifiedSymbolExtractor:
    """Unified facade for symbol extraction across all language tiers.

    Automatically selects the optimal extraction strategy based on
    language tier configuration. This centralizes extraction logic
    for use by both the indexer and LSP tools.

    Attributes:
        tree_sitter: TreeSitterExtractor for syntax analysis
        lsp_service: Optional LSP service for type enrichment
        enable_lsp_enrichment: Whether to use LSP (can be disabled)
        use_unified_registry: Whether to use the new LanguageCapabilityRegistry
    """

    def __init__(
        self,
        tree_sitter: Optional["TreeSitterExtractor"] = None,
        lsp_service: Optional["LSPServiceProtocol"] = None,
        enable_lsp_enrichment: bool = True,
        use_unified_registry: bool = True,
    ):
        """Initialize the unified extractor.

        Args:
            tree_sitter: Tree-sitter extractor instance
            lsp_service: LSP service for type enrichment (optional)
            enable_lsp_enrichment: Whether to use LSP for enrichment
            use_unified_registry: Whether to use the new LanguageCapabilityRegistry
                                  (default: True for forward compatibility)
        """
        self._ts = tree_sitter
        self._lsp = lsp_service if enable_lsp_enrichment else None
        self._enable_lsp = enable_lsp_enrichment
        self._use_unified_registry = use_unified_registry
        self._registry = LanguageCapabilityRegistry.instance() if use_unified_registry else None

    def set_tree_sitter(self, tree_sitter: "TreeSitterExtractor") -> None:
        """Set or update the tree-sitter extractor.

        Args:
            tree_sitter: TreeSitterExtractor instance
        """
        self._ts = tree_sitter

    def _convert_unified_to_legacy(self, unified_cap: Any) -> Any:
        """Convert unified capability to legacy tier config interface.

        This provides backward compatibility by creating an object that
        matches the legacy get_tier() interface while using the new
        unified capability system.

        Args:
            unified_cap: UnifiedLanguageCapability from the new registry

        Returns:
            Object with tier, has_native_ast, and has_lsp attributes
        """
        # Map unified tier to legacy tier
        tier_mapping = {
            UnifiedLanguageTier.TIER_1: LanguageTier.TIER_1,
            UnifiedLanguageTier.TIER_2: LanguageTier.TIER_2,
            UnifiedLanguageTier.TIER_3: LanguageTier.TIER_3,
            UnifiedLanguageTier.UNSUPPORTED: LanguageTier.TIER_3,  # Fallback
        }

        class LegacyTierConfig:
            """Adapter for legacy tier config interface."""

            def __init__(self, cap: UnifiedLanguageCapability) -> None:
                self.tier = tier_mapping.get(cap.tier, LanguageTier.TIER_3)
                self.has_native_ast = cap.native_ast is not None
                self.has_lsp = cap.lsp is not None
                self.has_tree_sitter = cap.tree_sitter is not None
                self.name = cap.name
                self.extensions = cap.extensions

        return LegacyTierConfig(unified_cap)

    def set_lsp_service(self, lsp_service: Optional["LSPServiceProtocol"]) -> None:
        """Set or update the LSP service.

        Args:
            lsp_service: LSP service instance or None to disable
        """
        if self._enable_lsp:
            self._lsp = lsp_service

    async def extract_symbols(
        self,
        file_path: Path,
        language: Optional[str] = None,
        content: Optional[str] = None,
    ) -> list[EnrichedSymbol]:
        """Extract symbols using optimal strategy for language tier.

        This is the main entry point. It:
        1. Detects language if not provided
        2. Gets tier configuration (from unified registry or legacy system)
        3. Extracts symbols via tree-sitter (all tiers)
        4. Enriches with native AST for Tier 1 Python
        5. Enriches with LSP for Tier 1/2 if available

        Args:
            file_path: Path to the source file
            language: Language identifier (auto-detected if not provided)
            content: File content (read from disk if not provided)

        Returns:
            List of enriched symbols with tier-appropriate information
        """
        if not self._ts:
            logger.warning("TreeSitterExtractor not available, returning empty symbols")
            return []

        # Detect language if not provided
        lang = language or self._ts.detect_language(file_path)
        if not lang:
            logger.debug(f"Could not detect language for {file_path}")
            return []

        # Get tier configuration from unified registry or legacy system
        tier_config = None
        unified_cap = None
        if self._use_unified_registry and self._registry:
            unified_cap = self._registry.get(lang)
            if not unified_cap:
                # Fallback: try to get by file path
                unified_cap = self._registry.get_for_file(file_path)

        if unified_cap:
            # Use unified registry - convert to legacy tier_config interface
            tier_config = self._convert_unified_to_legacy(unified_cap)
            logger.debug(
                f"Extracting symbols from {file_path.name} "
                f"(language={lang}, tier={unified_cap.tier.name}, source=unified_registry)"
            )
        else:
            # Fallback to legacy tier system
            tier_config = get_tier(lang)
            logger.debug(
                f"Extracting symbols from {file_path.name} "
                f"(language={lang}, tier={tier_config.tier.name}, source=legacy)"
            )

        # Read content if not provided
        if content is None:
            try:
                content = file_path.read_text()
            except (FileError, OSError, UnicodeDecodeError) as e:
                # Known error types - file read errors
                logger.warning(f"Failed to read {file_path}: {e}")
                return []
            except Exception as e:
                # Catch-all for truly unexpected errors
                logger.warning(f"Failed to read {file_path}: {e}")
                return []

        # Base extraction via tree-sitter (all tiers)
        try:
            ts_symbols = self._ts.extract_symbols(file_path, lang)
            enriched = [self._convert_ts_symbol(s, file_path, tier_config.tier) for s in ts_symbols]
        except (ValidationError, FileError) as e:
            # Known error types - validation or file errors
            logger.warning(f"Tree-sitter extraction failed for {file_path}: {e}")
            enriched = []
        except Exception as e:
            # Catch-all for truly unexpected errors
            logger.warning(f"Tree-sitter extraction failed for {file_path}: {e}")
            enriched = []

        if not enriched:
            return enriched

        # Tier 1: Add native AST enrichment for Python
        if tier_config.tier == LanguageTier.TIER_1 and tier_config.has_native_ast:
            enriched = self._enrich_with_native_ast(content, enriched, lang)

        # Tier 1 & 2: Add LSP enrichment if available
        if (
            tier_config.tier in (LanguageTier.TIER_1, LanguageTier.TIER_2)
            and self._lsp
            and tier_config.has_lsp
        ):
            enriched = await self._enrich_with_lsp(file_path, content, enriched, lang)

        return enriched

    def extract_symbols_sync(
        self,
        file_path: Path,
        language: Optional[str] = None,
        content: Optional[str] = None,
    ) -> list[EnrichedSymbol]:
        """Synchronous symbol extraction (tree-sitter + native AST only).

        Use this for non-async contexts. LSP enrichment is skipped.

        Args:
            file_path: Path to the source file
            language: Language identifier (auto-detected if not provided)
            content: File content (read from disk if not provided)

        Returns:
            List of enriched symbols (without LSP enrichment)
        """
        if not self._ts:
            return []

        lang = language or self._ts.detect_language(file_path)
        if not lang:
            return []

        # Get tier configuration from unified registry or legacy system
        tier_config = None
        unified_cap = None
        if self._use_unified_registry and self._registry:
            unified_cap = self._registry.get(lang)
            if not unified_cap:
                unified_cap = self._registry.get_for_file(file_path)

        if unified_cap:
            tier_config = self._convert_unified_to_legacy(unified_cap)
        else:
            tier_config = get_tier(lang)

        if content is None:
            try:
                content = file_path.read_text()
            except (FileError, OSError, UnicodeDecodeError):
                # Known error types - file read errors
                return []
            except Exception:
                # Catch-all for truly unexpected errors
                return []

        try:
            ts_symbols = self._ts.extract_symbols(file_path, lang)
            enriched = [self._convert_ts_symbol(s, file_path, tier_config.tier) for s in ts_symbols]
        except (ValidationError, FileError):
            # Known error types - validation or file errors
            return []
        except Exception:
            # Catch-all for truly unexpected errors
            return []

        # Tier 1: Add native AST enrichment for Python
        if tier_config.tier == LanguageTier.TIER_1 and tier_config.has_native_ast:
            enriched = self._enrich_with_native_ast(content, enriched, lang)

        return enriched

    def _convert_ts_symbol(
        self,
        ts_symbol: "ExtractedSymbol",
        file_path: Path,
        tier: LanguageTier,
    ) -> EnrichedSymbol:
        """Convert tree-sitter symbol to EnrichedSymbol."""
        return EnrichedSymbol(
            name=ts_symbol.name,
            symbol_type=ts_symbol.type,
            file_path=str(file_path),
            line_number=ts_symbol.line_number,
            end_line=ts_symbol.end_line,
            parent_symbol=ts_symbol.parent_symbol,
            source_tier=tier,
        )

    def _enrich_with_native_ast(
        self,
        content: str,
        symbols: list[EnrichedSymbol],
        language: str,
    ) -> list[EnrichedSymbol]:
        """Enrich symbols with native AST info (Python only currently).

        Python's ast module provides:
        - Return type annotations
        - Parameter names and types
        - Async function detection
        - Decorator list
        """
        if language != "python":
            return symbols

        try:
            tree = ast.parse(content)
            ast_info = self._extract_python_ast_info(tree)

            for sym in symbols:
                key = (sym.name, sym.line_number)
                if key in ast_info:
                    info = ast_info[key]
                    sym.return_type = info.get("return_type")
                    sym.parameters = info.get("parameters", [])
                    sym.is_async = info.get("is_async", False)
                    sym.decorators = info.get("decorators", [])
                    sym.docstring = info.get("docstring")
        except SyntaxError as e:
            logger.debug(f"Python AST parse failed: {e}")
        except Exception as e:
            logger.warning(f"Python AST enrichment failed: {e}")

        return symbols

    def _extract_python_ast_info(self, tree: ast.AST) -> dict[tuple[str, int], dict[str, Any]]:
        """Extract detailed info from Python AST.

        Returns:
            Dict mapping (name, line) to info dict containing:
            - is_async: bool
            - parameters: List[str]
            - return_type: Optional[str]
            - decorators: List[str]
            - docstring: Optional[str]
        """
        info: dict[tuple[str, int], dict[str, Any]] = {}

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                key = (node.name, node.lineno)
                docstring = ast.get_docstring(node)
                info[key] = {
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                    "parameters": self._extract_parameters(node),
                    "return_type": self._get_annotation_str(node.returns),
                    "decorators": [self._get_decorator_name(d) for d in node.decorator_list],
                    "docstring": docstring,
                }
            elif isinstance(node, ast.ClassDef):
                key = (node.name, node.lineno)
                docstring = ast.get_docstring(node)
                info[key] = {
                    "is_async": False,
                    "decorators": [self._get_decorator_name(d) for d in node.decorator_list],
                    "docstring": docstring,
                    "parameters": [],
                }

        return info

    def _extract_parameters(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
        """Extract parameter names with optional type annotations."""
        params = []
        for arg in node.args.args:
            if arg.annotation:
                params.append(f"{arg.arg}: {self._get_annotation_str(arg.annotation)}")
            else:
                params.append(arg.arg)
        return params

    def _get_annotation_str(self, node: Optional[ast.expr]) -> Optional[str]:
        """Convert AST annotation to string."""
        if node is None:
            return None
        try:
            return ast.unparse(node)
        except Exception:
            return None

    def _get_decorator_name(self, node: ast.expr) -> str:
        """Get decorator name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            parts = []
            current: ast.expr = node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))
        elif isinstance(node, ast.Call):
            return self._get_decorator_name(node.func)
        try:
            return ast.unparse(node)
        except Exception:
            return "<decorator>"

    async def _enrich_with_lsp(
        self,
        file_path: Path,
        content: str,
        symbols: list[EnrichedSymbol],
        language: str,
    ) -> list[EnrichedSymbol]:
        """Enrich symbols with LSP hover info for types.

        LSP provides:
        - Type signatures for functions/methods
        - Return types for languages without native AST
        - Documentation strings
        """
        if not self._lsp:
            return symbols

        try:
            # Open document in LSP
            await self._lsp.open_document(str(file_path), content)

            for sym in symbols:
                try:
                    # Get hover info at symbol definition
                    hover = await self._lsp.get_hover(
                        str(file_path),
                        sym.line_number - 1,  # 0-indexed
                        len(sym.name) // 2,  # Middle of symbol name
                    )
                    if hover and hover.contents:
                        # Parse type/signature from hover
                        parsed = self._parse_hover_contents(hover.contents)
                        if parsed.get("signature"):
                            sym.signature = parsed["signature"]
                        if parsed.get("return_type") and not sym.return_type:
                            sym.return_type = parsed["return_type"]
                        if parsed.get("docstring") and not sym.docstring:
                            sym.docstring = parsed["docstring"]
                except Exception as e:
                    logger.debug(f"LSP hover failed for {sym.name}: {e}")

            self._lsp.close_document(str(file_path))
        except Exception as e:
            logger.debug(f"LSP enrichment failed for {file_path}: {e}")

        return symbols

    def _parse_hover_contents(self, contents: str) -> dict[str, Optional[str]]:
        """Parse signature and type info from LSP hover contents.

        LSP hover format varies by language server:
        - Python (pyright): "def foo(x: int) -> str"
        - TypeScript (tsserver): "(method) Class.foo(x: number): string"
        - Go (gopls): "func (r *Receiver) Method(x int) string"
        """
        result: dict[str, Optional[str]] = {
            "signature": None,
            "return_type": None,
            "docstring": None,
        }

        lines = contents.strip().split("\n")
        in_docstring = False
        docstring_lines = []

        for line in lines:
            line = line.strip()

            # Skip markdown code fences
            if line.startswith("```"):
                continue
            if line.startswith("---"):
                in_docstring = True
                continue

            if in_docstring:
                docstring_lines.append(line)
                continue

            # Look for function signatures
            if "(" in line and ")" in line and not result["signature"]:
                result["signature"] = line
                # Extract return type if present
                if " -> " in line:
                    result["return_type"] = line.split(" -> ")[-1].strip()
                elif "): " in line:
                    # TypeScript style: ): ReturnType
                    result["return_type"] = line.split("): ")[-1].strip()
                continue

            # Look for type annotations (variable types)
            if ":" in line and not line.startswith("#") and not result["signature"]:
                result["signature"] = line

        if docstring_lines:
            result["docstring"] = "\n".join(docstring_lines)

        return result
