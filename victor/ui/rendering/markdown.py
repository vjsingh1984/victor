"""Rich markdown helpers with diagram/image hooks."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, List, Tuple

from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.markup import escape as escape_markup
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.tree import Tree

_DIAGRAM_LANGS = {"mermaid"}
_TOKEN_PATTERN = re.compile(
    r"```(?P<lang>[a-zA-Z0-9_-]+)\s*\n(?P<code>.*?)```|!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)",
    re.DOTALL,
)

# Pattern to detect Rich markup tags that could cause parsing errors
# Matches: [tag], [/tag], [tag=value], [tag=value1,value2]
_RICH_MARKUP_PATTERN = re.compile(r"\[/?[\w]+(?:=[^\]]+)?(?:,\s*[^\]]+)*\]")


def _escape_rich_markup_from_text(text: str) -> str:
    """Escape Rich markup tags that could cause parsing errors.

    This function escapes Rich-specific markup tags like [bold], [/], [red], etc.
    while preserving markdown syntax. It's designed to handle cases where LLM
    output contains strings that look like Rich markup but aren't intended to be.

    Note: This is a best-effort escape. It escapes square brackets that look like
    Rich markup tags, which is safer than trying to parse and validate them.

    Args:
        text: The text to escape

    Returns:
        Text with Rich markup tags escaped by replacing [ with \\[
    """
    # Escape Rich markup tags by replacing [ with \\[
    # This is a simple approach that works for most cases
    # We escape patterns that look like Rich tags: [word], [/word], [tag=value]
    def _escape_match(match: re.Match) -> str:
        return match.group(0).replace("[", "\\[")

    return _RICH_MARKUP_PATTERN.sub(_escape_match, text)


def _markdown_block(text: str) -> Markdown:
    """Consistently style markdown output with a lighter base.

    Content is escaped to prevent Rich markup parsing errors when LLM
    output contains strings that look like Rich tags (e.g., file paths).
    """
    # Escape Rich markup to prevent parsing errors
    safe_text = _escape_rich_markup_from_text(text)
    return Markdown(safe_text, style="markdown.text", justify="left")


def render_markdown_with_hooks(content: str) -> RenderableType:
    """Render markdown content with diagram/image hooks.

    Splits markdown into chunks so diagram code blocks and inline images can:
    - Render Mermaid snippets as ASCII trees for quick preview
    - Show textual placeholders for image links (since Rich can't display images)

    Falls back to plain text if Rich markup parsing fails, preventing rendering
    errors from breaking the agentic loop.
    """
    if not content.strip():
        return _markdown_block("")

    try:
        parts: List[RenderableType] = []
        cursor = 0
        for match in _TOKEN_PATTERN.finditer(content):
            start, end = match.span()
            if start > cursor:
                chunk = content[cursor:start]
                if chunk.strip():
                    parts.append(_markdown_block(chunk))

            lang = match.group("lang")
            if lang:
                lang = lang.lower()
                code = (match.group("code") or "").strip()
                if lang in _DIAGRAM_LANGS:
                    parts.append(_render_diagram(lang, code))
                else:
                    parts.append(_markdown_block(match.group(0)))
            else:
                alt = match.group("alt") or "image"
                src = match.group("src") or ""
                parts.append(_render_image_placeholder(alt, src))

            cursor = end

        remaining = content[cursor:]
        if remaining.strip():
            parts.append(_markdown_block(remaining))

        if not parts:
            return _markdown_block(content)
        if len(parts) == 1:
            return parts[0]
        return Group(*parts)

    except Exception as e:
        # If Rich rendering fails, fall back to plain text to prevent
        # rendering errors from breaking the agentic loop
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            "Rich rendering failed, falling back to plain text: %s",
            str(e)[:200],
        )
        # Return as plain text (Rich Text object, not Markdown)
        from rich.text import Text
        return Text(content, style="dim")


def _render_diagram(lang: str, code: str) -> RenderableType:
    if lang == "mermaid":
        return _render_mermaid(code)
    return Panel(
        Syntax(code, lang or "text", theme="monokai", word_wrap=True),
        title=f"{lang.title()} diagram",
        border_style="cyan",
    )


def _render_image_placeholder(alt_text: str, source: str) -> RenderableType:
    description = Text.assemble(
        ("Image: ", "bold cyan"),
        alt_text.strip() or "diagram",
        ("\nSource: ", "bold"),
        source,
    )
    return Panel(description, border_style="magenta")


def _render_mermaid(code: str) -> RenderableType:
    direction = _detect_direction(code)
    edges = _parse_mermaid_edges(code)
    if not edges:
        return Panel(
            Syntax(code or "", "mermaid", theme="monokai", word_wrap=True),
            title="Mermaid diagram (raw)",
            border_style="cyan",
        )

    adjacency = defaultdict(list)
    indegree = defaultdict(int)
    nodes = set()

    for src, dst, label in edges:
        adjacency[src].append((dst, label))
        indegree[dst] += 1
        nodes.add(src)
        nodes.add(dst)
        indegree.setdefault(src, 0)

    roots = [node for node in nodes if indegree[node] == 0] or sorted(nodes)

    forest = []
    for root in roots:
        tree = Tree(Text(f"{root}", style="bold"))
        _build_mermaid_tree(tree, root, adjacency, visited=set())
        forest.append(tree)

    title = f"Mermaid ({direction})"
    body: Iterable[RenderableType]
    if len(forest) == 1:
        body = [forest[0]]
    else:
        body = [Text("Multiple roots", style="bold dim")] + forest
    return Panel(Group(*body), title=title, border_style="cyan")


def _build_mermaid_tree(
    tree: Tree,
    node: str,
    adjacency: dict[str, List[Tuple[str, str | None]]],
    visited: set[str],
) -> None:
    if node in visited:
        tree.add(Text("(cycle)", style="dim red"))
        return
    visited.add(node)
    for dest, label in adjacency.get(node, []):
        edge_text = f"{label}: {dest}" if label else dest
        child = tree.add(edge_text)
        _build_mermaid_tree(child, dest, adjacency, visited.copy())


def _parse_mermaid_edges(code: str) -> List[Tuple[str, str, str | None]]:
    edges: List[Tuple[str, str, str | None]] = []
    for line in code.splitlines():
        text = line.strip()
        if not text or text.startswith("%") or "graph " in text.lower():
            continue
        if "-->" not in text:
            continue
        label = None

        if "-->|" in text:
            src_part, tail = text.split("-->|", 1)
            if "|" in tail:
                label, dest_part = tail.split("|", 1)
            else:
                dest_part = tail
        else:
            src_part, dest_part = text.split("-->", 1)

        src = _normalize_mermaid_node(src_part)
        dest = _normalize_mermaid_node(dest_part)
        if not src or not dest:
            continue
        edges.append((src, dest, (label or "").strip() or None))
    return edges


def _normalize_mermaid_node(token: str) -> str:
    token = token.strip()
    if not token:
        return ""
    # Remove trailing arrow modifiers or text (e.g., -.->)
    token = token.rstrip(".-")
    # Extract label inside brackets/braces/parentheses if present
    for opener, closer in ("[]", "()", "{}", "<>", "##"):
        o, c = opener[0], closer[0]
        if o in token and c in token:
            start = token.find(o) + 1
            end = token.find(c, start)
            if end > start:
                label = token[start:end].strip()
                if label:
                    return label
    # Strip identifier prefix like A or node1
    ident = re.match(r"[A-Za-z0-9_]+", token)
    if ident:
        remainder = token[ident.end() :].strip()
        if remainder.startswith("[") and "]" in remainder:
            inner = remainder[1 : remainder.find("]")]
            if inner.strip():
                return inner.strip()
        return ident.group(0)
    return token


def _detect_direction(code: str) -> str:
    for line in code.splitlines():
        text = line.strip().lower()
        if text.startswith("graph "):
            return text.split()[1].upper()
    return "TD"
