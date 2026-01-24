#!/usr/bin/env python3
"""Run the unified Victor API server.

This script starts the consolidated Victor API server that includes
all endpoints from the main API, HITL, and workflow editor servers.

Usage:
    # Run with defaults (port 8000)
    python -m victor.integrations.api.run_server

    # Run on custom port
    python -m victor.integrations.api.run_server --port 8080

    # Run without HITL endpoints
    python -m victor.integrations.api.run_server --no-hitl

    # Run with specific workspace
    python -m victor.integrations.api.run_server --workspace /path/to/project
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the unified Victor API server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Run on port 8000
  %(prog)s --port 8080                 # Run on port 8080
  %(prog)s --no-hitl                   # Disable HITL endpoints
  %(prog)s --workspace /path/to/project # Set workspace directory
        """,
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )

    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Workspace root directory (default: current directory)",
    )

    parser.add_argument(
        "--no-hitl",
        action="store_true",
        help="Disable HITL (Human-in-the-Loop) endpoints",
    )

    parser.add_argument(
        "--hitl-in-memory",
        action="store_true",
        help="Use in-memory storage for HITL (default: SQLite)",
    )

    parser.add_argument(
        "--hitl-auth-token",
        type=str,
        default=None,
        help="Optional auth token for HITL endpoints",
    )

    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Logging level (default: info)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Validate workspace
    workspace = args.workspace
    if workspace is None:
        workspace = str(Path.cwd())
    else:
        workspace_path = Path(workspace)
        if not workspace_path.exists():
            logger.error(f"Workspace directory does not exist: {workspace}")
            sys.exit(1)
        workspace = str(workspace_path.absolute())

    # Log configuration
    log_level = getattr(logging, args.log_level.upper())
    logging.getLogger().setLevel(log_level)

    logger.info("=" * 60)
    logger.info("Victor Unified API Server")
    logger.info("=" * 60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Workspace: {workspace}")
    logger.info(f"HITL: {'Disabled' if args.no_hitl else 'Enabled'}")
    if not args.no_hitl:
        hitl_storage = "In-Memory" if args.hitl_in_memory else "SQLite"
        logger.info(f"HITL Storage: {hitl_storage}")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Available UIs:")
    logger.info(f"  - Landing Page:    http://{args.host}:{args.port}/ui")
    logger.info(f"  - Workflow Editor: http://{args.host}:{args.port}/ui/workflow-editor")
    if not args.no_hitl:
        logger.info(f"  - HITL Approvals:  http://{args.host}:{args.port}/ui/hitl")
    logger.info(f"  - API Docs:        http://{args.host}:{args.port}/docs")
    logger.info("")
    logger.info("=" * 60)

    # Import and run server
    try:
        from victor.integrations.api.unified_server import run_unified_server

        run_unified_server(
            host=args.host,
            port=args.port,
            workspace_root=workspace,
            enable_hitl=not args.no_hitl,
            hitl_persistent=not args.hitl_in_memory,
            hitl_auth_token=args.hitl_auth_token,
            log_level=args.log_level,
        )

    except KeyboardInterrupt:
        logger.info("\nShutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
