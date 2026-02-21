#!/usr/bin/env python3
"""Benchmark warm-cache vertical integration behavior.

Measures:
- `apply()` latency for warm full-replay vs warm no-op/delta path
- `apply_async()` latency for warm full-replay vs warm no-op/delta path
- Side-effect handler skip ratio on warm cache hits

Usage:
    python scripts/benchmark_vertical_integration_cache.py
    python scripts/benchmark_vertical_integration_cache.py --iterations 200 --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

# Avoid eager provider/orchestrator imports when running benchmark harness.
os.environ.setdefault("VICTOR_LIGHT_IMPORT", "1")

from victor.agent.vertical_context import VerticalContext, create_vertical_context
from victor.framework.vertical_integration import VerticalIntegrationPipeline


class _BenchmarkVertical:
    name = "benchmark_vertical"

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        return {"name": cls.name}


class _BenchmarkOrchestrator:
    def __init__(self) -> None:
        self._vertical_context = create_vertical_context()
        self._enabled_tools = set()

    @property
    def vertical_context(self) -> VerticalContext:
        return self._vertical_context

    def set_vertical_context(self, context: VerticalContext) -> None:
        self._vertical_context = context

    def get_vertical_context(self) -> VerticalContext:
        return self._vertical_context

    def set_enabled_tools(self, tools) -> None:
        self._enabled_tools = set(tools)

    def get_enabled_tools(self):
        return set(self._enabled_tools)


class _BenchmarkHandler:
    def __init__(
        self,
        *,
        name: str,
        order: int,
        side_effects: bool,
        sync_delay_seconds: float,
        async_delay_seconds: float,
    ) -> None:
        self.name = name
        self.order = order
        self.side_effects = side_effects
        self.parallel_safe = not side_effects
        self.depends_on = ()
        self._sync_delay_seconds = sync_delay_seconds
        self._async_delay_seconds = async_delay_seconds
        self.calls = 0

    def apply(self, orchestrator, vertical, context, result, strict_mode=False) -> None:
        del vertical, strict_mode
        set_context = getattr(orchestrator, "set_vertical_context", None)
        if callable(set_context):
            set_context(context)
        time.sleep(self._sync_delay_seconds)
        self.calls += 1
        result.record_step_status(
            self.name,
            "success",
            details={"call_index": self.calls},
        )

    async def apply_async(self, orchestrator, vertical, context, result, strict_mode=False) -> None:
        del vertical, strict_mode
        set_context = getattr(orchestrator, "set_vertical_context", None)
        if callable(set_context):
            set_context(context)
        await asyncio.sleep(self._async_delay_seconds)
        self.calls += 1
        result.record_step_status(
            self.name,
            "success",
            details={"call_index": self.calls},
        )


class _BenchmarkRegistry:
    def __init__(self, handlers: Sequence[_BenchmarkHandler]) -> None:
        self._handlers = list(handlers)

    def get_ordered_handlers(self):
        return self._handlers


@dataclass
class _BenchResult:
    cold_ms: float
    warm_full_replay_ms: List[float]
    warm_delta_ms: List[float]
    skipped_side_effect_handlers: int
    total_side_effect_handler_slots: int


def _ms(duration_seconds: float) -> float:
    return duration_seconds * 1000.0


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    idx = round((len(ordered) - 1) * q)
    return float(ordered[idx])


def _summarize(result: _BenchResult) -> Dict[str, float]:
    warm_full_mean = statistics.fmean(result.warm_full_replay_ms)
    warm_delta_mean = statistics.fmean(result.warm_delta_ms)
    gain_pct = 0.0
    if warm_full_mean > 0:
        gain_pct = ((warm_full_mean - warm_delta_mean) / warm_full_mean) * 100.0

    skip_ratio = 0.0
    if result.total_side_effect_handler_slots > 0:
        skip_ratio = (
            result.skipped_side_effect_handlers / result.total_side_effect_handler_slots
        ) * 100.0

    return {
        "cold_ms": result.cold_ms,
        "warm_full_replay_mean_ms": warm_full_mean,
        "warm_full_replay_p95_ms": _percentile(result.warm_full_replay_ms, 0.95),
        "warm_delta_mean_ms": warm_delta_mean,
        "warm_delta_p95_ms": _percentile(result.warm_delta_ms, 0.95),
        "warm_latency_gain_pct": gain_pct,
        "side_effect_noop_skip_ratio_pct": skip_ratio,
    }


def _build_pipeline(
    *,
    side_effect_delay_seconds: float,
    pure_delay_seconds: float,
) -> tuple[VerticalIntegrationPipeline, List[str]]:
    handlers = [
        _BenchmarkHandler(
            name="stateful_a",
            order=10,
            side_effects=True,
            sync_delay_seconds=side_effect_delay_seconds,
            async_delay_seconds=side_effect_delay_seconds,
        ),
        _BenchmarkHandler(
            name="stateful_b",
            order=20,
            side_effects=True,
            sync_delay_seconds=side_effect_delay_seconds,
            async_delay_seconds=side_effect_delay_seconds,
        ),
        _BenchmarkHandler(
            name="pure_a",
            order=30,
            side_effects=False,
            sync_delay_seconds=pure_delay_seconds,
            async_delay_seconds=pure_delay_seconds,
        ),
        _BenchmarkHandler(
            name="pure_b",
            order=40,
            side_effects=False,
            sync_delay_seconds=pure_delay_seconds,
            async_delay_seconds=pure_delay_seconds,
        ),
    ]
    pipeline = VerticalIntegrationPipeline(
        enable_cache=True,
        step_registry=_BenchmarkRegistry(handlers),
    )
    # Benchmark measures integration-plan replay behavior only; exclude event bus overhead.
    pipeline._emit_vertical_applied_event = lambda result, cache_hit: None

    async def _noop_emit_async(result, *, cache_hit):
        return None

    pipeline._emit_vertical_applied_event_async = _noop_emit_async
    side_effect_handler_names = [handler.name for handler in handlers if handler.side_effects]
    return pipeline, side_effect_handler_names


def _run_sync(iterations: int, side_effect_delay_seconds: float, pure_delay_seconds: float) -> _BenchResult:
    # Warm full-replay benchmark: cache hit path on fresh orchestrators (no no-op plan state).
    full_pipeline, _ = _build_pipeline(
        side_effect_delay_seconds=side_effect_delay_seconds,
        pure_delay_seconds=pure_delay_seconds,
    )
    full_pipeline.apply(_BenchmarkOrchestrator(), _BenchmarkVertical)  # prime cache
    warm_full_replay_ms: List[float] = []
    for _ in range(iterations):
        orchestrator = _BenchmarkOrchestrator()
        t0 = time.perf_counter()
        full_pipeline.apply(orchestrator, _BenchmarkVertical)
        warm_full_replay_ms.append(_ms(time.perf_counter() - t0))

    # Warm delta benchmark: repeated applies on same orchestrator should skip side-effects.
    delta_pipeline, side_effect_handler_names = _build_pipeline(
        side_effect_delay_seconds=side_effect_delay_seconds,
        pure_delay_seconds=pure_delay_seconds,
    )
    shared_orchestrator = _BenchmarkOrchestrator()
    t0_cold = time.perf_counter()
    delta_pipeline.apply(shared_orchestrator, _BenchmarkVertical)
    cold_ms = _ms(time.perf_counter() - t0_cold)

    warm_delta_ms: List[float] = []
    skipped = 0
    total_slots = 0
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = delta_pipeline.apply(shared_orchestrator, _BenchmarkVertical)
        warm_delta_ms.append(_ms(time.perf_counter() - t0))
        for name in side_effect_handler_names:
            total_slots += 1
            status = result.get_step_status(name) or {}
            if status.get("status") == "skipped":
                skipped += 1

    return _BenchResult(
        cold_ms=cold_ms,
        warm_full_replay_ms=warm_full_replay_ms,
        warm_delta_ms=warm_delta_ms,
        skipped_side_effect_handlers=skipped,
        total_side_effect_handler_slots=total_slots,
    )


async def _run_async(
    iterations: int,
    side_effect_delay_seconds: float,
    pure_delay_seconds: float,
) -> _BenchResult:
    # Warm full-replay benchmark: cache hit path on fresh orchestrators (no no-op plan state).
    full_pipeline, _ = _build_pipeline(
        side_effect_delay_seconds=side_effect_delay_seconds,
        pure_delay_seconds=pure_delay_seconds,
    )
    await full_pipeline.apply_async(_BenchmarkOrchestrator(), _BenchmarkVertical)  # prime cache
    warm_full_replay_ms: List[float] = []
    for _ in range(iterations):
        orchestrator = _BenchmarkOrchestrator()
        t0 = time.perf_counter()
        await full_pipeline.apply_async(orchestrator, _BenchmarkVertical)
        warm_full_replay_ms.append(_ms(time.perf_counter() - t0))

    # Warm delta benchmark: repeated applies on same orchestrator should skip side-effects.
    delta_pipeline, side_effect_handler_names = _build_pipeline(
        side_effect_delay_seconds=side_effect_delay_seconds,
        pure_delay_seconds=pure_delay_seconds,
    )
    shared_orchestrator = _BenchmarkOrchestrator()
    t0_cold = time.perf_counter()
    await delta_pipeline.apply_async(shared_orchestrator, _BenchmarkVertical)
    cold_ms = _ms(time.perf_counter() - t0_cold)

    warm_delta_ms: List[float] = []
    skipped = 0
    total_slots = 0
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = await delta_pipeline.apply_async(shared_orchestrator, _BenchmarkVertical)
        warm_delta_ms.append(_ms(time.perf_counter() - t0))
        for name in side_effect_handler_names:
            total_slots += 1
            status = result.get_step_status(name) or {}
            if status.get("status") == "skipped":
                skipped += 1

    return _BenchResult(
        cold_ms=cold_ms,
        warm_full_replay_ms=warm_full_replay_ms,
        warm_delta_ms=warm_delta_ms,
        skipped_side_effect_handlers=skipped,
        total_side_effect_handler_slots=total_slots,
    )


async def _main_async(args) -> Dict[str, Any]:
    side_effect_delay_seconds = args.side_effect_delay_ms / 1000.0
    pure_delay_seconds = args.pure_delay_ms / 1000.0

    sync_result = _run_sync(
        iterations=args.iterations,
        side_effect_delay_seconds=side_effect_delay_seconds,
        pure_delay_seconds=pure_delay_seconds,
    )
    async_result = await _run_async(
        iterations=args.iterations,
        side_effect_delay_seconds=side_effect_delay_seconds,
        pure_delay_seconds=pure_delay_seconds,
    )

    return {
        "iterations": args.iterations,
        "side_effect_delay_ms": args.side_effect_delay_ms,
        "pure_delay_ms": args.pure_delay_ms,
        "sync": _summarize(sync_result),
        "async": _summarize(async_result),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark vertical integration warm-cache paths.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of warm iterations for each benchmark path.",
    )
    parser.add_argument(
        "--side-effect-delay-ms",
        type=float,
        default=0.35,
        help="Synthetic per-side-effect handler delay in milliseconds.",
    )
    parser.add_argument(
        "--pure-delay-ms",
        type=float,
        default=0.05,
        help="Synthetic per-pure handler delay in milliseconds.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON only (no human-readable summary).",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be > 0")

    data = asyncio.run(_main_async(args))
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
        return 0

    print("Vertical Integration Warm-Cache Benchmark")
    print(json.dumps(data, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
