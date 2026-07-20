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

"""ADR-0047 D10a step 2 (safe scope) — populate the prompt-cache split from sandhi's
single-sourced parser without changing the load-bearing ``prompt_tokens`` count."""

from __future__ import annotations

import pytest

from victor.providers.httpx_openai_compat import _augment_cache_split, _sg


def test_populates_cache_split_keeping_prompt_full() -> None:
    if _sg is None:
        pytest.skip("sandhi-gateway not installed (victor[sandhi])")
    usage = {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
    usage_data = {
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "prompt_tokens_details": {"cached_tokens": 60},
    }
    _augment_cache_split(usage, usage_data)
    # prompt_tokens stays the FULL count — window/budget logic depends on it (safe scope).
    assert usage["prompt_tokens"] == 100
    # The cache read is now populated (the OpenAI-compat dict dropped it).
    assert usage["cache_read_input_tokens"] == 60
    # OpenAI has no separate cache-creation billing → not added.
    assert "cache_creation_input_tokens" not in usage


def test_noop_when_no_cache() -> None:
    if _sg is None:
        pytest.skip("sandhi-gateway not installed")
    usage = {"prompt_tokens": 50, "completion_tokens": 5}
    _augment_cache_split(usage, {"prompt_tokens": 50, "completion_tokens": 5})
    assert "cache_read_input_tokens" not in usage
    assert usage["prompt_tokens"] == 50


def test_noop_without_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    import victor.providers.httpx_openai_compat as m

    monkeypatch.setattr(m, "_sg", None)
    usage = {"prompt_tokens": 5}
    m._augment_cache_split(
        usage, {"prompt_tokens": 5, "prompt_tokens_details": {"cached_tokens": 3}}
    )
    assert usage == {"prompt_tokens": 5}
