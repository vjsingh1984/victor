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

"""Shared ProximaDB runtime helpers for the Code Context Graph (CCG) backend.

This module is the single place that knows how to:

- detect whether the ``proximadb_sdk`` package is importable (optional dep),
- start an **embedded** ProximaDB (local single-repo) instance, and
- build the canonical ``oid`` that correlates a code symbol's relational row,
  its ORION graph node, and its HNSW vector into **one** entity.

See ``docs/architecture/proximadb-codegraph-backend.md`` (TD-11/12/13) for the
design. The embedded path uses ProximaDB's in-RAM vector engine — the
equivalent of the Rust ``EmbeddingMode::Memory`` — so semantic seed→expand
scores neighbors inline. The multi-tenant **service** path (``EmbeddingMode::Cold``
/ SQ8) is gated on ProximaDB TD-127 (secondary indexes) + TD-130/131 (graph
bulk-load + REST v2 hybrid) and is treated as WIP until those land.

ProximaDB is an *optional* runtime dependency: every helper degrades gracefully
(returns ``False`` / raises a clear, actionable error) when the package or the
server binary is absent, so SQLite remains the default with no proximadb install.
"""

from __future__ import annotations

import enum
import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional-dependency detection
# ---------------------------------------------------------------------------
def is_proxima_available() -> bool:
    """Return True if the ``proximadb_sdk`` package can be imported.

    This does **not** guarantee the embedded server binary is present — use
    :func:`start_embedded_db` (which raises on a missing binary) for that.
    """
    import importlib.util

    return importlib.util.find_spec("proximadb_sdk") is not None


class ProximaUnavailableError(RuntimeError):
    """Raised when a ProximaDB-backed store is requested but cannot run.

    Carries an actionable message (install the package or build the embedded
    server binary). Callers that want SQLite-style graceful degradation should
    catch this and fall back rather than propagate.
    """


# ---------------------------------------------------------------------------
# Embedding mode (maps to the Rust EmbeddingMode enum)
# ---------------------------------------------------------------------------
class ProximaEmbeddingMode(str, enum.Enum):
    """How per-symbol vectors are held for semantic scoring.

    The Python SDK has no ``EmbeddingMode`` enum; this is Victor's encoding of
    the engine-side concept so callers/settings can express intent:

    - ``MEMORY`` — full-precision (fp32) vectors held in RAM. Used by the
      embedded/local single-repo case so semantic BFS scores neighbors inline.
      This is the default and the only mode that runs against the embedded
      PyO3/subprocess path today.
    - ``COLD`` — quantized (SQ8) vectors to bound RAM in the multi-tenant
      service deployment. Gated on TD-127/130/131; treated as WIP.
    """

    MEMORY = "memory"
    COLD = "cold"

    @classmethod
    def coerce(cls, value: "ProximaEmbeddingMode | str | None") -> "ProximaEmbeddingMode":
        if value is None:
            return cls.MEMORY
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError:
            logger.warning("Unknown ProximaDB embedding mode %r; defaulting to memory", value)
            return cls.MEMORY


# ---------------------------------------------------------------------------
# oid correlation — one identity for relational row + graph node + vector
# ---------------------------------------------------------------------------
_REPO_SANITIZE = re.compile(r"[^A-Za-z0-9_]+")


def repo_id_from_path(project_path: Optional[Path | str]) -> str:
    """Derive a stable, filesystem-safe repo id from a project path.

    Uses the resolved directory name (lowercased, non-alphanumerics collapsed to
    ``_``). Falls back to ``repo`` for an empty/anonymous path.
    """
    if project_path is None:
        return "repo"
    name = Path(project_path).expanduser().resolve(strict=False).name or "repo"
    sanitized = _REPO_SANITIZE.sub("_", name).strip("_").lower()
    return sanitized or "repo"


def symbol_oid(file: str, name: str, symbol_type: Optional[str] = None) -> str:
    """Compute the stable per-symbol id used inside an :func:`node_oid`.

    Correlation is by ``(symbol_type, file, name)`` — the same implicit key the
    SQLite/Lance pair correlate on today (``symbol:{file}:{name}``), made
    explicit and collision-resistant via a short blake2b digest.
    """
    raw = f"{symbol_type or ''}:{file}:{name}"
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=12).hexdigest()


def node_oid(repo: str, symbol: str) -> str:
    """Build the canonical correlated id: ``graph/{repo}/node/{symbol_oid}``.

    The same string is used as the ProximaDB graph node id **and** the vector
    record id, so a vector hit maps to its graph node by identity (no join) and
    the always-empty ``graph_node.embedding_ref`` bridge is retired.
    """
    return f"graph/{repo}/node/{symbol}"


def graph_id_for_repo(repo: str) -> str:
    """ORION graph id for a repo's Code Context Graph."""
    return f"{repo}_codegraph"


# ---------------------------------------------------------------------------
# Embedded bootstrap
# ---------------------------------------------------------------------------
def _pick_free_ports(count: int) -> list[int]:
    """Reserve ``count`` distinct loopback ports and return them.

    The embedded ProximaDB is a managed subprocess reached over loopback TCP, so
    it needs free REST/gRPC ports. The SDK defaults (15678/15679) frequently
    collide (Docker, a second repo, a leftover instance); picking ports the OS
    reports free avoids ``Address already in use`` on startup. There is a small
    TOCTOU window between release and the server's bind, acceptable for a
    local single-instance launch.
    """
    import socket

    held: list[socket.socket] = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            held.append(sock)
        return [sock.getsockname()[1] for sock in held]
    finally:
        for sock in held:
            sock.close()


async def start_embedded_db(
    data_dir: Path | str,
    *,
    log_level: str = "warn",
    vector_engine: str = "SST",
    graph_engine: str = "ORION",
    binary_path: Optional[str] = None,
    timeout: float = 30.0,
    rest_port: Optional[int] = None,
    grpc_port: Optional[int] = None,
) -> Any:
    """Start and return an embedded ``EmbeddedProximaDB`` for a single repo.

    The SDK's "embedded" mode is a self-managed ``proximadb-server`` subprocess
    reached over loopback REST/gRPC (it is not in-process), so it needs ports.
    Unless explicit ports are given, free ones are selected to avoid colliding
    with the SDK defaults (15678/15679), Docker, or a leftover instance.

    Args:
        data_dir: Persistent data directory for this repo's collection/graph.
        log_level: ProximaDB log level.
        vector_engine: Vector storage engine (SST is write-optimized for code).
        graph_engine: Graph engine (ORION = WAL + in-memory).
        binary_path: Explicit path to the ``proximadb-server`` binary (optional).
        timeout: Seconds to wait for the server to become healthy.
        rest_port: Explicit REST port (default: an OS-selected free port).
        grpc_port: Explicit gRPC port (default: an OS-selected free port).

    Returns:
        A started ``EmbeddedProximaDB`` instance.

    Raises:
        ProximaUnavailableError: if the SDK is missing, the server binary cannot
            be located, or the server fails to become healthy. The message is
            actionable so callers can surface it or fall back to SQLite.
    """
    if not is_proxima_available():
        raise ProximaUnavailableError(
            "proximadb_sdk is not installed. Install it (pip install proximadb) "
            "or keep the default SQLite graph backend."
        )

    from proximadb_sdk.embedded import EmbeddedConfig, EmbeddedProximaDB

    data_path = Path(data_dir).expanduser()
    data_path.mkdir(parents=True, exist_ok=True)

    if rest_port is None or grpc_port is None:
        free_rest, free_grpc = _pick_free_ports(2)
        rest_port = rest_port if rest_port is not None else free_rest
        grpc_port = grpc_port if grpc_port is not None else free_grpc

    config = EmbeddedConfig(
        data_dir=str(data_path),
        log_level=log_level,
        vector_engine=vector_engine,
        graph_engine=graph_engine,
        rest_port=rest_port,
        grpc_port=grpc_port,
    )
    db = EmbeddedProximaDB(config=config, binary_path=binary_path)
    try:
        await db.start(timeout=timeout)
    except (FileNotFoundError, TimeoutError, OSError) as exc:
        raise ProximaUnavailableError(
            "Could not start embedded ProximaDB: "
            f"{exc}. Build the proximadb-server binary "
            "(cd proximaDB && cargo build --release) or pass server_url for a "
            "running instance. SQLite remains the default backend."
        ) from exc
    return db
