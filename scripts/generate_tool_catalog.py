# Generate a markdown catalog of currently registered tools.
# Keeps documentation honest by pulling live tool metadata from the agent.

import asyncio
from pathlib import Path
from typing import AsyncIterator

from victor.agent.orchestrator import AgentOrchestrator
from victor.config.settings import load_settings
from victor.providers.base import (
    BaseProvider,
    CompletionResponse,
    Message,
    StreamChunk,
    ToolDefinition,
)


OUTPUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "TOOL_CATALOG.md"


class _DummyProvider(BaseProvider):
    """Minimal provider to satisfy tool registration without network calls."""

    @property
    def name(self) -> str:
        return "dummy"

    def supports_tools(self) -> bool:  # type: ignore[override]
        return True

    def supports_streaming(self) -> bool:  # type: ignore[override]
        return False

    async def chat(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[ToolDefinition] | None = None,
        **kwargs,
    ) -> CompletionResponse:
        return CompletionResponse(content="", role="assistant", model=model)

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[ToolDefinition] | None = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        if False:
            yield StreamChunk()  # pragma: no cover

    async def close(self) -> None:
        return None


async def main() -> None:
    settings = load_settings()
    # Avoid network-dependent setup for catalog generation
    settings.airgapped_mode = True
    settings.use_semantic_tool_selection = False
    settings.analytics_enabled = False
    settings.tool_cache_enabled = False

    dummy_provider = _DummyProvider()
    agent = AgentOrchestrator(settings=settings, provider=dummy_provider, model="dummy")

    lines = [
        "# Tool Catalog (auto-generated)",
        "",
        "Descriptions are truncated for brevity. Use `victor tools list` for full details.",
        "",
        "| Tool | Description |",
        "| --- | --- |",
    ]
    max_len = 140
    for tool in sorted(agent.tools.list_tools(), key=lambda t: t.name):
        # Collapse whitespace to keep table tidy
        desc = " ".join(tool.description.split())
        if len(desc) > max_len:
            cutoff = desc.rfind(" ", 0, max_len - 3)
            if cutoff == -1:
                cutoff = max_len - 3
            desc = f"{desc[:cutoff].rstrip()}..."
        lines.append(f"| `{tool.name}` | {desc} |")

    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")

    await agent.shutdown()
    await agent.provider.close()


if __name__ == "__main__":
    asyncio.run(main())
