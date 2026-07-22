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

"""Host-side policy tests for GoogleProvider.

Transport (chat/stream/SDK calls) is owned by the Sandhi typed variant
(``SandhiGoogleProvider``); resolver routing and typed execution are covered by
``test_sandhi_transport.py``. This file covers the policy shell that remains:
capability flags, cache/context policy, lifecycle, and API-key resolution.
"""

import asyncio

import pytest

from victor.providers.google_provider import GoogleProvider


@pytest.fixture
def google_provider():
    """GoogleProvider policy shell (API-key mode, no transport)."""
    return GoogleProvider(api_key="test-key")


def test_provider_name(google_provider):
    assert google_provider.name == "google"


def test_supports_tools(google_provider):
    assert google_provider.supports_tools() is True


def test_supports_streaming(google_provider):
    assert google_provider.supports_streaming() is True


def test_supports_prompt_caching(google_provider):
    assert google_provider.supports_prompt_caching() is True


def test_supports_kv_prefix_caching(google_provider):
    assert google_provider.supports_kv_prefix_caching() is True


def test_cache_cost_model(google_provider):
    model = google_provider.cache_cost_model()
    assert model.supported is True
    assert model.min_prefix_tokens == 32768


def test_context_window_lookup(google_provider):
    # Unknown model falls back to the Gemini default (positive int).
    assert google_provider.context_window("gemini-unknown") > 0


def test_initialization_resolves_api_key():
    provider = GoogleProvider(api_key="test-key")
    assert provider._api_key == "test-key"


@pytest.mark.asyncio
async def test_close_is_noop(google_provider):
    # Gemini client needs no explicit closing; close() must not raise.
    await google_provider.close()


def test_raw_chat_stream_are_guard_stubs(google_provider):
    """The policy shell is concrete but transport is delegated to the Sandhi variant."""
    with pytest.raises(NotImplementedError):
        asyncio.run(google_provider.chat(messages=[], model="gemini"))
