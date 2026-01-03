# Architecture Overview

A concise view of Victor’s architecture and where to look for details.

## What It Is
Victor is a provider‑agnostic coding assistant with a CLI/TUI front end, a core orchestrator, and modular tools/verticals.

## Main Pieces
- **CLI/TUI**: entry point for chat and workflows
- **Orchestrator**: coordinates providers, tools, and workflows
- **Providers**: local or cloud model backends
- **Tools**: file ops, git, testing, search, etc.
- **Verticals**: domain presets (coding, research, devops, data, rag)

## Data Flow (Simplified)
1. User input → orchestrator
2. Orchestrator selects tools/providers
3. Tools run → results returned
4. Provider completes → response streamed back

## Where to Dig Deeper
- Full deep‑dive appendix: `docs/ARCHITECTURE_DEEP_DIVE_APPENDIX.md`
