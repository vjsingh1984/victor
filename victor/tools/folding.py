# Copyright 2026 Vijaykumar Singh <singhvijay@users.noreply.github.com>
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

"""Tool advertisement folding policy.

Folded tools remain registered and executable, but are not advertised as
separate enabled schemas by default. Their common use cases are represented by
the target tool's description to reduce tool count and schema/token overhead.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional


@dataclass(frozen=True)
class ToolFold:
    """A tool that should be represented by another advertised tool."""

    target: str
    hint: str


FOLDED_TOOLS: dict[str, ToolFold] = {
    # Shell folds
    "database": ToolFold(
        target="shell",
        hint=(
            "Use shell with sqlite3/psql/mysql for quick database inspection " "and SQL queries."
        ),
    ),
    "dependency": ToolFold(
        target="shell",
        hint="Use shell with pip, pipdeptree, uv, poetry, npm, cargo, or equivalent package CLIs.",
    ),
    "docker": ToolFold(
        target="shell",
        hint="Use shell with docker/docker compose for container, image, log, build, and run commands.",
    ),
    "test": ToolFold(
        target="shell",
        hint="Use shell with pytest, npm test, cargo test, go test, make test, or project test CLIs.",
    ),
    "rag_ingest": ToolFold(
        target="shell",
        hint="Use shell to execute scripts or CLI tools for knowledge base ingestion, similar to SQL/db tools.",
    ),
    "rag_query": ToolFold(
        target="shell", hint="Use shell to execute scripts for querying the knowledge base."
    ),
    "rag_search": ToolFold(
        target="shell", hint="Use shell to execute scripts for searching the knowledge base."
    ),
    "rag_list": ToolFold(
        target="shell", hint="Use shell to execute scripts for listing the knowledge base."
    ),
    "rag_delete": ToolFold(
        target="shell", hint="Use shell to execute scripts for modifying the knowledge base."
    ),
    "rag_stats": ToolFold(
        target="shell", hint="Use shell to execute scripts for getting stats of the knowledge base."
    ),
    "rename": ToolFold(
        target="shell",
        hint="Use shell with standard tools (sed, fastmod) or fs edit/patch for renaming symbols.",
    ),
    "inline": ToolFold(
        target="shell", hint="Use shell with standard tools (sed) or fs edit/patch for inlining."
    ),
    "extract": ToolFold(
        target="shell", hint="Use shell with standard tools or fs edit/patch for extraction."
    ),
    "organize_imports": ToolFold(
        target="shell", hint="Use shell with standard tools (isort, ruff) for organizing imports."
    ),
    "scaffold": ToolFold(
        target="shell",
        hint="Use shell to scaffold projects via npx, cookiecutter, cargo new, or equivalent CLI tools.",
    ),
    "sandbox": ToolFold(
        target="shell",
        hint="Use shell with the --sandbox flag to execute isolated containerized commands.",
    ),
    # Integration folds
    "jira": ToolFold(
        target="integrations",
        hint="Use integrations tool to interact with Jira for ticket management.",
    ),
    "slack": ToolFold(target="integrations", hint="Use integrations tool to message Slack."),
    "teams": ToolFold(
        target="integrations", hint="Use integrations tool to message Microsoft Teams."
    ),
    # MCP folds
    "mcp": ToolFold(
        target="mcp_bridge", hint="Use mcp_bridge to access Model Context Protocol specific tools."
    ),
}


def tool_folding_enabled() -> bool:
    """Return whether default advertisement folding is enabled."""
    raw = os.getenv("VICTOR_TOOL_FOLDING", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def get_fold(tool_name: str) -> Optional[ToolFold]:
    """Return fold metadata for a tool name, if it is folded."""
    if not tool_folding_enabled():
        return None
    return FOLDED_TOOLS.get(str(tool_name or "").strip())


def is_folded_tool(tool_name: str) -> bool:
    """Return whether ``tool_name`` is folded out of default advertisements."""
    return get_fold(tool_name) is not None


def should_advertise_tool(tool_name: str, *, include_folded: bool = False) -> bool:
    """Return whether a tool should be listed in default enabled tool surfaces."""
    return include_folded or not is_folded_tool(tool_name)


def folded_tool_names_for_target(target: str) -> list[str]:
    """Return folded tool names represented by ``target``."""
    if not tool_folding_enabled():
        return []
    return sorted(name for name, fold in FOLDED_TOOLS.items() if fold.target == target)


def folded_tool_hints_for_target(target: str) -> list[str]:
    """Return human-readable fold hints for a target tool description."""
    if not tool_folding_enabled():
        return []
    return [
        f"{name}: {fold.hint}"
        for name, fold in sorted(FOLDED_TOOLS.items())
        if fold.target == target
    ]
