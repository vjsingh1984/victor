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
from victor_coding.codebase.indexer import CodebaseIndex
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
        from victor.agent.rl.base import RLOutcome
        from victor.agent.rl.coordinator import get_rl_coordinator

        coordinator = get_rl_coordinator()
        learner = coordinator.get_learner("model_selector")
        if learner and agent.provider and hasattr(agent, "message_count"):
            msg_count = agent.message_count
            if msg_count > 0:
                import uuid

                metrics = {}
                if hasattr(agent, "get_session_metrics"):
                    metrics = agent.get_session_metrics() or {}

                # Compute quality score based on session metrics
                latency = metrics.get("total_latency", 0)
                tool_calls = metrics.get("tool_calls", 0)
                quality_score = 1.0
                if latency > 30:
                    quality_score -= min(0.1 * (latency - 30) / 30, 0.5)
                if tool_calls > 0:
                    quality_score += min(0.05 * tool_calls, 0.2)

                outcome = RLOutcome(
                    provider=agent.provider.name,
                    model=getattr(agent.provider, "model", "unknown"),
                    task_type="chat",
                    success=True,
                    quality_score=max(0.0, min(1.0, quality_score)),
                    metadata={
                        "session_id": str(uuid.uuid4())[:8],
                        "latency_seconds": latency,
                        "token_count": metrics.get("total_tokens", 0),
                        "tool_calls_made": tool_calls,
                        "message_count": msg_count,
                    },
                    vertical="coding",
                )
                coordinator.record_outcome("model_selector", outcome, "coding")

                # Get updated Q-value for logging
                rankings = learner.get_provider_rankings()
                provider_ranking = next(
                    (r for r in rankings if r["provider"] == agent.provider.name), None
                )
                new_q = provider_ranking["q_value"] if provider_ranking else 0.0

                logger.info(
                    f"RL session feedback: {agent.provider.name} "
                    f"({msg_count} messages, {tool_calls} tools) → Q={new_q:.3f}"
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
            console_obj.print(
                f"[dim]Index stale: {len(modified)} modified, {len(deleted)} deleted files[/]"
            )
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


async def preload_semantic_index(
    cwd: str, settings: Any, console_obj: Console, force: bool = False
) -> bool:
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


def get_rl_profile_suggestion(
    current_provider: str, profiles: dict
) -> Optional[tuple[str, str, float]]:
    """Get RL-based profile suggestion if different from current."""
    try:
        from victor.agent.rl.coordinator import get_rl_coordinator
        import json

        coordinator = get_rl_coordinator()
        learner = coordinator.get_learner("model_selector")
        if not learner:
            return None

        # Get available providers from profiles
        available = list(set(cfg.provider for cfg in profiles.values()))

        # Get recommendation
        rec = coordinator.get_recommendation(
            "model_selector",
            json.dumps(available),
            "",
            "chat",
        )
        if not rec or rec.confidence < 0.3:
            return None
        if rec.value.lower() == current_provider.lower():
            return None

        # Find matching profile for recommended provider
        matching_profiles = [
            name for name, cfg in profiles.items() if cfg.provider.lower() == rec.value.lower()
        ]
        if matching_profiles:
            return (matching_profiles[0], rec.value, rec.confidence)
        return None
    except Exception:
        return None
