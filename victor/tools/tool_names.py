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

"""Centralized Tool Name Registry.

Design Pattern: Registry Pattern with Alias Support
====================================================
This module provides a single source of truth for all tool names, enabling:
- Token-efficient short names for LLM context
- Backward compatibility via aliases
- Platform-agnostic naming conventions
- Easy future renaming without breaking changes

Usage:
    from victor.tools.tool_names import ToolNames, get_canonical_name

    # Get the canonical (short) name
    name = ToolNames.SHELL  # "shell"

    # Resolve legacy names to canonical
    canonical = get_canonical_name("execute_bash")  # Returns "shell"

Architecture:
    ┌─────────────────────────────────────────────────────┐
    │                  Tool Name Registry                  │
    ├─────────────────────────────────────────────────────┤
    │  Canonical Names (short, token-efficient)           │
    │  ├── shell, read, write, edit, grep, search, ...   │
    ├─────────────────────────────────────────────────────┤
    │  Aliases (backward compatibility)                   │
    │  ├── execute_bash → shell                          │
    │  ├── read_file → read                              │
    │  ├── code_search → grep                            │
    └─────────────────────────────────────────────────────┘

ALIAS USAGE POLICY
==================
Aliases in the @tool decorator should be used SPARINGLY and only for:

1. **External Backward Compatibility**: When external integrations, user scripts,
   or LLM prompts may reference old tool names. This ensures existing workflows
   don't break during migration.

2. **Hotfix/Interim Support**: Temporary backward compatibility during migration
   periods. Should be reviewed and potentially removed in future versions.

3. **LLM Compatibility**: LLMs may have been trained on old tool names and may
   continue to use them. Aliases ensure these calls still work.

Aliases should NOT be used as:
- A long-term integration strategy for internal code
- A way to avoid updating internal callers to use canonical names

INTERNAL CODE POLICY:
- Internal Victor code MUST use canonical (short) names for readability
- Internal tool references in dictionaries/mappings should use canonical names
- Module-level export aliases (e.g., `old_name = new_name`) are acceptable for
  import compatibility but internal code should import the canonical name
"""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Optional, Set


@dataclass(frozen=True)
class ToolNameEntry:
    """Immutable entry for a tool name with its aliases.

    Attributes:
        canonical: The primary short name (token-efficient)
        aliases: Legacy names that resolve to canonical
        deprecated: Names that will be removed in future versions
    """

    canonical: str
    aliases: FrozenSet[str] = field(default_factory=frozenset)
    deprecated: FrozenSet[str] = field(default_factory=frozenset)

    def all_names(self) -> Set[str]:
        """Return all valid names (canonical + aliases)."""
        return {self.canonical} | set(self.aliases)


class ToolNames:
    """Registry of canonical tool names.

    Naming Conventions:
    - Platform-agnostic (no bash, cmd, powershell references)
    - Token-efficient (short but descriptive)
    - Action-oriented (verbs where applicable)
    - Unix-inspired for familiarity (ls, grep, diff)
    """

    # ==========================================================================
    # CORE FILESYSTEM TOOLS
    # ==========================================================================
    READ = "read"  # read_file → read
    WRITE = "write"  # write_file → write
    EDIT = "edit"  # edit_files → edit
    LS = "ls"  # list_directory → ls
    PLAN = "plan"  # plan_files → plan

    # ==========================================================================
    # SHELL / COMMAND EXECUTION (Platform-agnostic)
    # ==========================================================================
    SHELL = "shell"  # execute_bash → shell (works on all platforms)
    SHELL_READONLY = "shell_readonly"  # read-only shell for safe exploration
    SANDBOX = "sandbox"  # execute_python_in_sandbox → sandbox
    SANDBOX_UPLOAD = "sandbox_upload"  # upload_files_to_sandbox → sandbox_upload

    # ==========================================================================
    # SEARCH TOOLS
    # ==========================================================================
    GREP = "grep"  # code_search → grep (keyword search)
    CODE_SEARCH = "code_search"  # semantic_code_search → code_search (semantic/AI)
    WEB_SEARCH = "web_search"  # web_search (internet search)
    WEB_FETCH = "web_fetch"  # web_fetch (fetch URL content)
    SUMMARIZE = "summarize"  # web_summarize → summarize

    # ==========================================================================
    # CODE INTELLIGENCE
    # ==========================================================================
    SYMBOL = "symbol"  # find_symbol → symbol
    REFS = "refs"  # find_references → refs
    LSP = "lsp"  # lsp → lsp (already optimal)

    # ==========================================================================
    # REFACTORING TOOLS
    # ==========================================================================
    RENAME = "rename"  # refactor_rename_symbol → rename
    EXTRACT = "extract"  # refactor_extract_function → extract
    INLINE = "inline"  # refactor_inline_variable → inline
    IMPORTS = "imports"  # refactor_organize_imports → imports

    # ==========================================================================
    # GIT TOOLS
    # ==========================================================================
    GIT = "git"  # git → git (already optimal)
    PR = "pr"  # git_create_pr → pr
    COMMIT_MSG = "commit_msg"  # git_suggest_commit → commit_msg
    CONFLICTS = "conflicts"  # git_analyze_conflicts → conflicts

    # ==========================================================================
    # PATCH / DIFF TOOLS
    # ==========================================================================
    PATCH = "patch"  # apply_patch → patch
    DIFF = "diff"  # create_patch → diff

    # ==========================================================================
    # DOCUMENTATION TOOLS
    # ==========================================================================
    DOCS = "docs"  # generate_docs → docs
    DOCS_COVERAGE = "docs_coverage"  # analyze_docs → docs_coverage

    # ==========================================================================
    # CODE QUALITY TOOLS
    # ==========================================================================
    REVIEW = "review"  # code_review → review
    METRICS = "metrics"  # analyze_metrics → metrics
    SCAN = "scan"  # security_scan → scan
    TEST = "test"  # run_tests → test

    # ==========================================================================
    # INFRASTRUCTURE TOOLS
    # ==========================================================================
    DB = "db"  # database → db
    DOCKER = "docker"  # docker → docker (already optimal)
    CICD = "cicd"  # cicd → cicd (already optimal)
    DEPS = "deps"  # dependency → deps
    HTTP = "http"  # http_request → http
    API_TEST = "api_test"  # http_test → api_test

    # ==========================================================================
    # UTILITY TOOLS
    # ==========================================================================
    BATCH = "batch"  # batch → batch (already optimal)
    CACHE = "cache"  # cache → cache (already optimal)
    MCP = "mcp"  # mcp_call → mcp
    WORKFLOW = "workflow"  # run_workflow → workflow
    SCAFFOLD = "scaffold"  # scaffold → scaffold (already optimal)
    OVERVIEW = "overview"  # get_project_overview → overview

    # ==========================================================================
    # ANALYSIS TOOLS (Class-based BaseTool implementations)
    # ==========================================================================
    MERGE = "merge"  # merge_conflicts → merge (conflict resolution)
    PIPELINE = "pipeline"  # pipeline_analyzer → pipeline (CI/CD analysis)
    AUDIT = "audit"  # audit → audit (codebase auditing)
    IAC = "iac"  # iac_scanner → iac (infrastructure as code)
    GRAPH = "graph"  # graph → graph (code graph analysis: PageRank, dependencies)
    ARCH_SUMMARY = (
        "arch_summary"  # architecture_summary → arch_summary (architectural hubs/coupling)
    )


# =============================================================================
# ALIAS REGISTRY: Maps legacy names to canonical names
# =============================================================================
TOOL_ALIASES: Dict[str, str] = {
    # Filesystem
    "read_file": ToolNames.READ,
    "write_file": ToolNames.WRITE,
    "edit_files": ToolNames.EDIT,
    "list_directory": ToolNames.LS,
    "plan_files": ToolNames.PLAN,
    "get_project_overview": ToolNames.OVERVIEW,
    # Shell / Execution
    "execute_bash": ToolNames.SHELL,
    "run": ToolNames.SHELL,  # LLMs often hallucinate "run" as a tool name
    "bash": ToolNames.SHELL,  # Common alias
    "execute": ToolNames.SHELL,  # Common alias
    "cmd": ToolNames.SHELL,  # Windows-style alias
    "execute_python_in_sandbox": ToolNames.SANDBOX,
    "upload_files_to_sandbox": ToolNames.SANDBOX_UPLOAD,
    # Search - keyword search
    "code_search": ToolNames.GREP,
    # Search - semantic (AI-powered) code search
    "semantic_code_search": ToolNames.CODE_SEARCH,
    # Web tools
    "web_summarize": ToolNames.SUMMARIZE,
    # Code Intelligence
    "find_symbol": ToolNames.SYMBOL,
    "find_references": ToolNames.REFS,
    "rename_symbol": ToolNames.RENAME,  # Duplicate → maps to same as refactor_rename_symbol
    "architecture_summary": ToolNames.ARCH_SUMMARY,
    # Refactoring
    "refactor_rename_symbol": ToolNames.RENAME,
    "refactor_extract_function": ToolNames.EXTRACT,
    "refactor_inline_variable": ToolNames.INLINE,
    "refactor_organize_imports": ToolNames.IMPORTS,
    # Git - operation-specific aliases that resolve to unified "git" tool
    # LLMs may call "git_status" expecting a tool, but we use "git" with operation param
    "git_status": ToolNames.GIT,
    "git_diff": ToolNames.GIT,
    "git_log": ToolNames.GIT,
    "git_commit": ToolNames.GIT,
    "git_branch": ToolNames.GIT,
    "git_stage": ToolNames.GIT,
    # PR and commit message tools
    "git_create_pr": ToolNames.PR,
    "git_suggest_commit": ToolNames.COMMIT_MSG,
    "git_analyze_conflicts": ToolNames.CONFLICTS,
    # Patch
    "apply_patch": ToolNames.PATCH,
    "create_patch": ToolNames.DIFF,
    # Documentation
    "generate_docs": ToolNames.DOCS,
    "analyze_docs": ToolNames.DOCS_COVERAGE,
    # Code Quality
    "code_review": ToolNames.REVIEW,
    "analyze_metrics": ToolNames.METRICS,
    "security_scan": ToolNames.SCAN,
    "run_tests": ToolNames.TEST,
    # Infrastructure
    "database": ToolNames.DB,
    "dependency": ToolNames.DEPS,
    "http_request": ToolNames.HTTP,
    "mcp_call": ToolNames.MCP,
    "run_workflow": ToolNames.WORKFLOW,
    # Analysis Tools (Class-based BaseTool implementations)
    "merge_conflicts": ToolNames.MERGE,
    "pipeline_analyzer": ToolNames.PIPELINE,
    "iac_scanner": ToolNames.IAC,
    # "audit" is already canonical (no alias needed)
}

# Build reverse lookup: canonical → set of aliases
CANONICAL_TO_ALIASES: Dict[str, Set[str]] = {}
for alias, canonical in TOOL_ALIASES.items():
    if canonical not in CANONICAL_TO_ALIASES:
        CANONICAL_TO_ALIASES[canonical] = set()
    CANONICAL_TO_ALIASES[canonical].add(alias)


def get_canonical_name(name: str) -> str:
    """Resolve a tool name to its canonical form.

    Args:
        name: Tool name (can be canonical or alias)

    Returns:
        Canonical (short) tool name

    Example:
        >>> get_canonical_name("execute_bash")
        "shell"
        >>> get_canonical_name("shell")
        "shell"
    """
    return TOOL_ALIASES.get(name, name)


def get_aliases(canonical_name: str) -> Set[str]:
    """Get all aliases for a canonical tool name.

    Args:
        canonical_name: The canonical tool name

    Returns:
        Set of alias names (empty if no aliases)
    """
    return CANONICAL_TO_ALIASES.get(canonical_name, set())


def is_valid_tool_name(name: str) -> bool:
    """Check if a name is a valid tool name (canonical or alias).

    Args:
        name: Name to check

    Returns:
        True if valid, False otherwise
    """
    # Check if it's an alias
    if name in TOOL_ALIASES:
        return True
    # Check if it's a canonical name
    return name in CANONICAL_TO_ALIASES or hasattr(ToolNames, name.upper())


def get_all_canonical_names() -> Set[str]:
    """Get all canonical tool names.

    Returns:
        Set of all canonical names
    """
    return {
        getattr(ToolNames, attr)
        for attr in dir(ToolNames)
        if not attr.startswith("_") and isinstance(getattr(ToolNames, attr), str)
    }


def get_name_mapping() -> Dict[str, str]:
    """Get full mapping of all names (canonical + aliases) to canonical.

    Returns:
        Dict mapping every valid name to its canonical form
    """
    mapping = {}
    # Add canonical names (map to themselves)
    for canonical in get_all_canonical_names():
        mapping[canonical] = canonical
    # Add aliases
    mapping.update(TOOL_ALIASES)
    return mapping
