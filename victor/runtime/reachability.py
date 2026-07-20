"""Runtime reachability recorder — FEP-0022 Phase 1.

Lightweight, ``contextvars``-scoped recorder for "observed use" witnesses. The
recorder is **armed only inside the evaluation/trajectory substrate** (an explicit
:class:`ReachabilityRecorder` / :func:`activate` context, or the
``VICTOR_REACHABILITY_RECORD`` env that the eval harness reads). When disarmed —
the normal production case — a witness call is a single contextvar read plus a
``None`` check: a near-no-op on the hot path.

Witnesses flush to a run-local sidecar (``reachability-<run_id>.jsonl``); never to
the global or project database. The sink idiom mirrors ``NativeMetrics``
(``victor/native/observability.py``); the contextvars scoping mirrors
``TraceContext`` (``victor/runtime/trace_context.py``).

Phase 2 (FEP-0022) merges sidecars across a diverse run corpus into the
ever-observed baseline that the reachability oracle diffs against
``ServiceContainer.get_registered_types()``.

Usage::

    from victor.runtime.reachability import activate

    with activate(out_path=Path("reachability-run.jsonl")) as rec:
        ...  # agent / trajectory run; container.get() witnesses fire automatically
    # sidecar written on exit
"""

from __future__ import annotations

import contextvars
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, List, Optional

__all__ = [
    "ReachabilityRecorder",
    "activate",
    "current_recorder",
    "is_env_armed",
    "record",
    "record_service_resolution",
]

# Contextvar: the recorder active for this async task / thread, or None (disarmed).
# A ContextVar lookup with a default is a C-level operation — the disarmed fast path.
_CURRENT: "contextvars.ContextVar[Optional[ReachabilityRecorder]]" = contextvars.ContextVar(
    "victor_reachability_recorder", default=None
)

# Env the eval/trajectory harness reads to decide whether to wrap a run in
# `activate(...)`. Nothing arms globally at import — the harness owns the scope.
_ENV_ARM = "VICTOR_REACHABILITY_RECORD"
_ENV_OUTPUT_DIR = "VICTOR_REACHABILITY_OUT"


def is_env_armed() -> bool:
    """True if the process asked the harness to record reachability witnesses."""
    return os.environ.get(_ENV_ARM, "").lower() in ("1", "true", "yes")


def current_recorder() -> Optional[ReachabilityRecorder]:
    """Return the active recorder for this context, or ``None`` (disarmed)."""
    return _CURRENT.get()


def record(kind: str, key: str) -> None:
    """Record a reachability witness. Near-no-op when no recorder is active.

    Args:
        kind: artifact class, e.g. ``"di"``.
        key: stable identifier, e.g. ``"module:QualName"``.
    """
    rec = _CURRENT.get()
    if rec is not None:
        rec.add(kind, key)


def record_service_resolution(service_type: type) -> None:
    """Witness that a registered service type was resolved (FEP-0022 Probe A).

    Called by :meth:`ServiceContainer.get`. The qualified-name key is built only
    when a recorder is armed, so the disarmed hot path does no string work.
    """
    rec = _CURRENT.get()
    if rec is not None:
        rec.add("di", f"{service_type.__module__}:{service_type.__qualname__}")


class ReachabilityRecorder:
    """Accumulates observed-use witnesses for one run; flushes a sidecar on close.

    Witnesses are de-duplicated per ``(kind, key)`` within a run; the sidecar is
    the union. Downstream accumulation across runs (FEP-0022 Phase 2) merges
    sidecars into the ever-observed baseline.
    """

    __slots__ = ("_run_id", "_out_path", "_seen", "_lock", "_closed")

    def __init__(
        self,
        run_id: Optional[str] = None,
        out_path: Optional[Path] = None,
    ) -> None:
        self._run_id = run_id or uuid.uuid4().hex[:12]
        self._out_path = out_path
        self._seen: set[str] = set()
        self._lock = threading.Lock()
        self._closed = False

    @property
    def run_id(self) -> str:
        return self._run_id

    def add(self, kind: str, key: str) -> None:
        """Record one witness (de-duplicated). Thread-safe."""
        with self._lock:
            self._seen.add(f"{kind}\t{key}")

    def observed(self) -> List[Dict[str, str]]:
        """Witnesses as ``[{"kind": ..., "key": ...}, ...]``, sorted (deterministic)."""
        out: List[Dict[str, str]] = []
        with self._lock:
            for token in sorted(self._seen):
                kind, key = token.split("\t", 1)
                out.append({"kind": kind, "key": key})
        return out

    def flush(self) -> Optional[Path]:
        """Write the sidecar JSONL. Returns its path, or ``None`` if nothing to write."""
        if self._closed or self._out_path is None:
            return None
        records = self.observed()
        self._out_path.parent.mkdir(parents=True, exist_ok=True)
        with self._out_path.open("w", encoding="utf-8") as fh:
            fh.write(
                json.dumps({"run_id": self._run_id, "ts": time.time(), "count": len(records)})
                + "\n"
            )
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        return self._out_path

    def close(self) -> Optional[Path]:
        """Flush (if configured) and mark closed. Returns the sidecar path if written."""
        path = self.flush()
        self._closed = True
        return path


@contextmanager
def activate(
    run_id: Optional[str] = None,
    out_path: Optional[Path] = None,
    flush_on_exit: bool = True,
) -> Generator[ReachabilityRecorder, None, None]:
    """Arm the recorder for the current context; detach (and optionally flush) on exit.

    If ``out_path`` is None and ``VICTOR_REACHABILITY_OUT`` is set, a default
    sidecar path under it is used. Nesting replaces the active recorder for the
    inner context (the outer recorder is restored on exit).
    """
    if out_path is None:
        env_dir = os.environ.get(_ENV_OUTPUT_DIR)
        if env_dir:
            out_path = Path(env_dir) / f"reachability-{run_id or 'run'}.jsonl"
    rec = ReachabilityRecorder(run_id=run_id, out_path=out_path)
    token = _CURRENT.set(rec)
    try:
        yield rec
    finally:
        _CURRENT.reset(token)
        if flush_on_exit:
            rec.close()
