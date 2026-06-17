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

"""Single-flight background index rebuilds + filename-search timeout override.

Regression for the cascade where every stale ``code_search`` call spawned a new
fire-and-forget rebuild, so concurrent rebuilds thrashed the embeddings store
and re-triggered probe timeouts.
"""

import asyncio

from victor.tools.code_search_tool import (
    _filename_search_timeout,
    _INFLIGHT_INDEX_REBUILDS,
    _spawn_index_rebuild_once,
)


async def test_rebuild_is_deduplicated_while_in_flight() -> None:
    _INFLIGHT_INDEX_REBUILDS.clear()
    started = 0
    gate = asyncio.Event()

    async def _rebuild() -> None:
        nonlocal started
        started += 1
        await gate.wait()

    t1 = _spawn_index_rebuild_once("root-a", _rebuild)
    t2 = _spawn_index_rebuild_once("root-a", _rebuild)

    # Both calls share one in-flight task; the factory ran only once.
    assert t1 is t2
    await asyncio.sleep(0)  # let the task start
    assert started == 1
    assert "root-a" in _INFLIGHT_INDEX_REBUILDS

    gate.set()
    await t1
    await asyncio.sleep(0)  # allow done-callback to clear the registry
    assert "root-a" not in _INFLIGHT_INDEX_REBUILDS


async def test_distinct_roots_run_independently() -> None:
    _INFLIGHT_INDEX_REBUILDS.clear()
    gate = asyncio.Event()

    async def _rebuild() -> None:
        await gate.wait()

    t_a = _spawn_index_rebuild_once("root-a", _rebuild)
    t_b = _spawn_index_rebuild_once("root-b", _rebuild)
    assert t_a is not t_b

    gate.set()
    await asyncio.gather(t_a, t_b)


async def test_completed_rebuild_allows_respawn() -> None:
    _INFLIGHT_INDEX_REBUILDS.clear()
    runs = 0

    async def _rebuild() -> None:
        nonlocal runs
        runs += 1

    await _spawn_index_rebuild_once("root-a", _rebuild)
    await asyncio.sleep(0)
    # Prior task finished and cleared; a new call spawns a fresh rebuild.
    await _spawn_index_rebuild_once("root-a", _rebuild)
    assert runs == 2


def test_filename_search_timeout_env_override(monkeypatch) -> None:
    monkeypatch.delenv("VICTOR_TIMEOUT_FILENAME_SEARCH", raising=False)
    monkeypatch.delenv("VICTOR_FILENAME_SEARCH_TIMEOUT", raising=False)
    assert _filename_search_timeout() == 45.0

    monkeypatch.setenv("VICTOR_TIMEOUT_FILENAME_SEARCH", "90")
    assert _filename_search_timeout() == 90.0
