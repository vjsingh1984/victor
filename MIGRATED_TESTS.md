# Migrated Tests - victor to victor-coding

This document tracks tests that have been migrated from the victor (framework) 
repository to the victor-coding (vertical) repository.

## Migration Date

2026-03-04

## Tests Moved to victor-coding

### Protocol Tests (moved to `victor-coding/tests/protocols/`)

| Original Path | New Path | Reason |
|----------------|----------|--------|
| `tests/unit/protocols/test_coverage_protocol.py` | `victor-coding/tests/protocols/test_coverage_protocol_impl.py` | Tests victor-coding's coverage protocol implementation |
| `tests/unit/protocols/test_docgen_protocol.py` | `victor-coding/tests/protocols/test_docgen_protocol_impl.py` | Tests victor-coding's docgen protocol implementation |
| `tests/unit/protocols/test_refactor_protocol.py` | `victor-coding/tests/protocols/test_refactor_protocol_impl.py` | Tests victor-coding's refactor protocol implementation |
| `tests/unit/protocols/test_review_protocol.py` | `victor-coding/tests/protocols/test_review_protocol_impl.py` | Tests victor-coding's review protocol implementation |
| `tests/unit/protocols/test_testgen_protocol.py` | `victor-coding/tests/protocols/test_testgen_protocol_impl.py` | Tests victor-coding's testgen protocol implementation |

### Tool Tests (moved to `victor-coding/tests/integration/`)

| Original Path | New Path | Reason |
|----------------|----------|--------|
| `tests/unit/tools/test_documentation_tool_unit.py` | `victor-coding/tests/integration/test_documentation_tool.py` | Tests documentation tool with victor-coding features |
| `tests/unit/tools/test_lsp_tool.py` | `victor-coding/tests/integration/test_lsp_tool.py` | Tests LSP tool with victor-coding features |
| `tests/unit/tools/test_lsp.py` | `victor-coding/tests/integration/test_lsp.py` | Tests LSP integration with victor-coding |
| `tests/unit/tools/test_query_expander.py` | `victor-coding/tests/integration/test_query_expander.py` | Tests query expander with victor-coding features |
| `tests/unit/tools/test_code_search_tool.py` | `victor-coding/tests/integration/test_code_search_tool.py` | Tests code search with victor-coding features |

### Indexer Tests (moved to `victor-coding/tests/indexers/`)

All files in `tests/unit/indexers/` directory moved to `victor-coding/tests/indexers/`:

| Original Path | New Path | Reason |
|----------------|----------|--------|
| `tests/unit/indexers/test_indexer.py` | `victor-coding/tests/indexers/test_indexer.py` | Tests tree-sitter indexing (victor-coding feature) |
| `tests/unit/indexers/test_indexer_cpp.py` | `victor-coding/tests/indexers/test_indexer_cpp.py` | Tests C++ indexing (victor-coding feature) |
| `tests/unit/indexers/test_indexer_edge_extraction.py` | `victor-coding/tests/indexers/test_indexer_edge_extraction.py` | Tests edge extraction (victor-coding feature) |
| `tests/unit/indexers/test_indexer_graph_integration.py` | `victor-coding/tests/indexers/test_indexer_graph_integration.py` | Tests graph integration (victor-coding feature) |
| `tests/unit/indexers/test_indexer_multilang_tree_sitter.py` | `victor-coding/tests/indexers/test_indexer_multilang_tree_sitter.py` | Tests multi-language support (victor-coding feature) |
| `tests/unit/indexers/test_tree_sitter_extractor.py` | `victor-coding/tests/indexers/test_tree_sitter_extractor.py` | Tests tree-sitter extractor (victor-coding feature) |
| `tests/unit/indexers/test_tree_sitter_manager.py` | `victor-coding/tests/indexers/test_tree_sitter_manager.py` | Tests tree-sitter manager (victor-coding feature) |
| `tests/unit/indexers/test_codebase_analyzer.py` | `victor-coding/tests/indexers/test_codebase_analyzer.py` | Tests codebase analyzer (victor-coding feature) |
| `tests/unit/indexers/test_airgapped_codebase_search.py` | `victor-coding/tests/indexers/test_airgapped_codebase_search.py` | Tests airgapped search (victor-coding feature) |

## Tests Remaining in victor

The following tests remain in victor because they test framework functionality:

- `tests/unit/protocols/test_vertical_protocols.py` - Tests framework protocol definitions
- `tests/unit/framework/test_vertical_integration.py` - Tests vertical integration with framework
- `tests/unit/contrib/` - All contrib package tests (no victor_coding dependencies)
- `tests/unit/core/` - Core framework tests

## Migration Rationale

These tests were migrated because they:

1. **Test victor-coding specific features** - Tree-sitter indexing, multi-language support, etc.
2. **Create circular dependency** - Framework tests depend on external vertical package
3. **Cause test skips** - Tests are skipped when victor_coding is not installed
4. **Belong in vertical** - These tests validate victor-coding's implementation of protocols

## Verification

After migration:
- `pytest tests/unit/` in victor runs completely without victor_coding
- `pytest tests/` in victor-coding validates victor-coding features
- No circular dependencies between repos

