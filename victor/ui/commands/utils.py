import asyncio
import logging
import os
import signal
import sys
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from pathlib import Path
import time

from victor.agent.orchestrator import AgentOrchestrator
from victor.agent.safety import (
    ConfirmationRequest,
    RiskLevel,
    set_confirmation_callback,
)
from victor.codebase.indexer import CodebaseIndex
from victor.tools.code_search_tool import _get_or_build_index, _INDEX_CACHE

logger = logging.getLogger(__name__)
console = Console()

# Global reference for signal handler cleanup
_current_agent: Optional[AgentOrchestrator] = None

def configure_logging(log_level: str, stream: Optional[Any] = None) -> None:
    """Configure logging to stderr (separate from Rich console output)."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.WARNING),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=stream or sys.stderr,
        force=True,
    )

def flush_logging() -> None:
    """Flush all logging handlers before shutdown messages."""
    for handler in logging.root.handlers:
        handler.flush()

async def graceful_shutdown(agent: Optional[AgentOrchestrator]) -> None:
    """Perform graceful shutdown of the agent."""
    if agent is None:
        return
    try:
        from victor.agent.rl_model_selector import SessionReward, get_model_selector
        selector = get_model_selector()
        if selector and agent.provider and hasattr(agent, "message_count"):
            msg_count = agent.message_count
            if msg_count > 0:
                import uuid
                metrics = {}
                if hasattr(agent, "get_session_metrics"):
                    metrics = agent.get_session_metrics() or {}
                reward = SessionReward(
                    session_id=str(uuid.uuid4())[:8],
                    provider=agent.provider.name,
                    model=getattr(agent.provider, "model", "unknown"),
                    success=True,
                    latency_seconds=metrics.get("total_latency", 0),
                    token_count=metrics.get("total_tokens", 0),
                    tool_calls_made=metrics.get("tool_calls", 0),
                )
                new_q = selector.update_q_value(reward)
                logger.info(
                    f"RL session feedback: {agent.provider.name} "
                    f"({msg_count} messages, {metrics.get('tool_calls', 0)} tools) → Q={new_q:.3f}"
                )
    except Exception as e:
        logger.debug(f"RL feedback recording skipped: {e}")
    try:
        shutdown_results = await agent.graceful_shutdown()
        logger.debug(f"Graceful shutdown results: {shutdown_results}")
        await agent.shutdown()
    except Exception as e:
        logger.warning(f"Error during graceful shutdown: {e}")
        try:
            if agent.provider:
                await agent.provider.close()
        except Exception as close_error:
            logger.debug(f"Error closing provider: {close_error}")
    finally:
        flush_logging()

def setup_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(sig: int, frame: Any) -> None:
        global _current_agent
        sig_name = signal.Signals(sig).name
        logger.info(f"Received {sig_name}, initiating graceful shutdown...")
        if _current_agent is not None:
            try:
                asyncio.create_task(graceful_shutdown(_current_agent))
            except RuntimeError:
                logger.debug("Event loop not running, attempting direct cleanup")
                try:
                    if _current_agent.provider:
                        pass
                except Exception:
                    pass
        if sig == signal.SIGINT:
            raise KeyboardInterrupt()
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, signal_handler)

async def check_codebase_index(cwd: str, console_obj: Console, silent: bool = False) -> None:
    """Check codebase index status at startup and reindex if needed."""
    try:
        index = CodebaseIndex(root_path=cwd, use_embeddings=False, enable_watcher=False)
        is_stale, modified, deleted = index.check_staleness_by_mtime()
        if not is_stale:
            if not silent:
                logger.debug("Codebase index is up to date")
            return
        total_changes = len(modified) + len(deleted)
        if not silent:
            console_obj.print(f"[dim]Index stale: {len(modified)} modified, {len(deleted)} deleted files[/]")
        if total_changes <= 10:
            await index.incremental_reindex()
            if not silent:
                console_obj.print(f"[green]Incrementally reindexed {total_changes} files[/]")
        else:
            await index.reindex()
            if not silent:
                console_obj.print(f"[green]Full reindex completed ({len(index.files)} files)[/]")
    except ImportError:
        logger.debug("Codebase indexer not available")
    except Exception as e:
        logger.debug(f"Codebase index check failed: {e}")

async def preload_semantic_index(cwd: str, settings: Any, console_obj: Console, force: bool = False) -> bool:
    """Preload semantic codebase index with embeddings and graph upfront."""
    try:
        root_path = Path(cwd).resolve()
        cache_entry = _INDEX_CACHE.get(str(root_path))
        if cache_entry and not force:
            console_obj.print("[dim]✓ Semantic index already loaded[/]")
            return True
        console_obj.print("[dim]⏳ Building semantic code index (one-time)...[/]")
        start_time = time.time()
        index, rebuilt = await _get_or_build_index(root_path, settings, force_reindex=force)
        elapsed = time.time() - start_time

        # Get graph stats if available
        graph_stats = ""
        if index.graph_store:
            try:
                stats = await index.graph_store.stats()
                node_count = stats.get("nodes", 0)
                edge_count = stats.get("edges", 0)
                if node_count or edge_count:
                    graph_stats = f" | Graph: {node_count} symbols, {edge_count} edges"
            except Exception:
                pass

        if rebuilt:
            console_obj.print(f"[green]✓ Semantic index built in {elapsed:.1f}s{graph_stats}[/]")
        else:
            console_obj.print(f"[green]✓ Semantic index loaded in {elapsed:.1f}s{graph_stats}[/]")
        return True
    except ImportError as e:
        logger.warning(f"Semantic indexing dependencies not available: {e}")
        return False
    except Exception as e:
        logger.warning(f"Failed to preload semantic index: {e}")
        console_obj.print(f"[yellow]⚠ Semantic index preload failed: {e}[/]")
        return False

async def cli_confirmation_callback(request: ConfirmationRequest) -> bool:
    """Prompt user for confirmation of dangerous operations."""
    risk_colors = {
        RiskLevel.SAFE: "green",
        RiskLevel.LOW: "blue",
        RiskLevel.MEDIUM: "yellow",
        RiskLevel.HIGH: "red",
        RiskLevel.CRITICAL: "bold red",
    }
    color = risk_colors.get(request.risk_level, "white")
    console.print()
    console.print(
        Panel(
            f"[{color}]{request.risk_level.value.upper()} RISK[/{color}]\n\n"
            f"Tool: [bold]{request.tool_name}[/bold]\n"
            f"Action: {request.description}\n\n"
            + (
                "Details:\n" + "\n".join(f"  • {d}" for d in request.details)
                if request.details
                else ""
            ),
            title="⚠️  Confirmation Required",
            border_style=color,
        )
    )
    try:
        return Confirm.ask("Proceed with this operation?", default=False)
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Operation cancelled[/dim]")
        return False

def setup_safety_confirmation() -> None:
    """Set up the CLI confirmation callback for dangerous operations."""
    set_confirmation_callback(cli_confirmation_callback)

def get_rl_profile_suggestion(current_provider: str, profiles: dict) -> Optional[tuple[str, str, float]]:
    """Get RL-based profile suggestion if different from current."""
    try:
        from victor.agent.rl_model_selector import get_model_selector
        selector = get_model_selector()
        if not selector:
            return None
        rec = selector.recommend()
        if not rec or rec.confidence < 0.3:
            return None
        if rec.provider.lower() == current_provider.lower():
            return None
        matching_profiles = [
            name for name, cfg in profiles.items() if cfg.provider.lower() == rec.provider.lower()
        ]
        if matching_profiles:
            return (matching_profiles[0], rec.provider, rec.q_value)
        return None
    except Exception:
        return None
