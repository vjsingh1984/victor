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

"""Tests for TokenTracker thread-safe token usage accumulation."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, wait

from victor.agent.token_tracker import TokenTracker


class TestTokenTracker:
    """Test TokenTracker basic operations."""

    def test_initial_state_is_zero(self) -> None:
        tracker = TokenTracker()
        usage = tracker.get_usage()
        assert usage == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }

    def test_accumulate_single_response(self) -> None:
        tracker = TokenTracker()
        tracker.accumulate(
            {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            }
        )
        usage = tracker.get_usage()
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 150
        assert usage["cache_read_tokens"] == 0
        assert usage["cache_write_tokens"] == 0

    def test_accumulate_multiple_responses(self) -> None:
        tracker = TokenTracker()
        tracker.accumulate(
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        )
        tracker.accumulate(
            {"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280}
        )
        usage = tracker.get_usage()
        assert usage["prompt_tokens"] == 300
        assert usage["completion_tokens"] == 130
        assert usage["total_tokens"] == 430

    def test_accumulate_with_cache_tokens(self) -> None:
        tracker = TokenTracker()
        tracker.accumulate(
            {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "cache_read_tokens": 30,
                "cache_write_tokens": 20,
            }
        )
        usage = tracker.get_usage()
        assert usage["cache_read_tokens"] == 30
        assert usage["cache_write_tokens"] == 20

    def test_accumulate_ignores_unknown_keys(self) -> None:
        tracker = TokenTracker()
        tracker.accumulate({"prompt_tokens": 10, "unknown_key": 999})
        usage = tracker.get_usage()
        assert usage["prompt_tokens"] == 10
        assert "unknown_key" not in usage

    def test_accumulate_handles_missing_keys(self) -> None:
        tracker = TokenTracker()
        tracker.accumulate({"prompt_tokens": 42})
        usage = tracker.get_usage()
        assert usage["prompt_tokens"] == 42
        assert usage["completion_tokens"] == 0

    def test_accumulate_empty_dict(self) -> None:
        tracker = TokenTracker()
        tracker.accumulate({})
        assert all(v == 0 for v in tracker.get_usage().values())


class TestTokenTrackerProperties:
    """Test convenience properties."""

    def test_total_tokens_property(self) -> None:
        tracker = TokenTracker()
        tracker.accumulate({"total_tokens": 250})
        assert tracker.total_tokens == 250

    def test_prompt_tokens_property(self) -> None:
        tracker = TokenTracker()
        tracker.accumulate({"prompt_tokens": 120})
        assert tracker.prompt_tokens == 120

    def test_completion_tokens_property(self) -> None:
        tracker = TokenTracker()
        tracker.accumulate({"completion_tokens": 80})
        assert tracker.completion_tokens == 80


class TestTokenTrackerReset:
    """Test reset functionality."""

    def test_reset_clears_all_counters(self) -> None:
        tracker = TokenTracker()
        tracker.accumulate(
            {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "cache_read_tokens": 30,
                "cache_write_tokens": 20,
            }
        )
        tracker.reset()
        assert all(v == 0 for v in tracker.get_usage().values())

    def test_accumulate_after_reset(self) -> None:
        tracker = TokenTracker()
        tracker.accumulate({"prompt_tokens": 100})
        tracker.reset()
        tracker.accumulate({"prompt_tokens": 42})
        assert tracker.prompt_tokens == 42


class TestTokenTrackerGetUsageReturnsCopy:
    """Test that get_usage returns a copy, not a reference."""

    def test_mutating_returned_dict_does_not_affect_tracker(self) -> None:
        tracker = TokenTracker()
        tracker.accumulate({"prompt_tokens": 100})
        usage = tracker.get_usage()
        usage["prompt_tokens"] = 999
        assert tracker.prompt_tokens == 100

    def test_successive_calls_return_independent_dicts(self) -> None:
        tracker = TokenTracker()
        tracker.accumulate({"prompt_tokens": 50})
        usage1 = tracker.get_usage()
        tracker.accumulate({"prompt_tokens": 50})
        usage2 = tracker.get_usage()
        assert usage1["prompt_tokens"] == 50
        assert usage2["prompt_tokens"] == 100


class TestTokenTrackerThreadSafety:
    """Test concurrent access from multiple threads."""

    def test_concurrent_accumulation(self) -> None:
        tracker = TokenTracker()
        num_threads = 10
        iterations_per_thread = 1000

        def accumulate_many() -> None:
            for _ in range(iterations_per_thread):
                tracker.accumulate(
                    {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    }
                )

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(accumulate_many) for _ in range(num_threads)]
            wait(futures)
            for f in futures:
                f.result()

        expected = num_threads * iterations_per_thread
        assert tracker.prompt_tokens == expected
        assert tracker.completion_tokens == expected
        assert tracker.total_tokens == expected * 2

    def test_concurrent_accumulate_and_read(self) -> None:
        tracker = TokenTracker()
        stop = threading.Event()
        errors: list[str] = []

        def writer() -> None:
            for _ in range(500):
                tracker.accumulate({"prompt_tokens": 1, "total_tokens": 1})

        def reader() -> None:
            while not stop.is_set():
                usage = tracker.get_usage()
                if usage["prompt_tokens"] < 0:
                    errors.append(f"Negative prompt_tokens: {usage['prompt_tokens']}")
                if usage["total_tokens"] < 0:
                    errors.append(f"Negative total_tokens: {usage['total_tokens']}")

        reader_thread = threading.Thread(target=reader)
        reader_thread.start()

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(writer) for _ in range(4)]
            wait(futures)
            for f in futures:
                f.result()

        stop.set()
        reader_thread.join(timeout=5)
        assert not errors, f"Thread safety errors: {errors}"
        assert tracker.prompt_tokens == 2000

    def test_concurrent_accumulate_and_reset(self) -> None:
        """Reset during concurrent accumulation should not corrupt state."""
        tracker = TokenTracker()

        def accumulate_batch() -> None:
            for _ in range(100):
                tracker.accumulate({"prompt_tokens": 1})

        def reset_once() -> None:
            tracker.reset()

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(accumulate_batch) for _ in range(4)]
            futures.append(executor.submit(reset_once))
            wait(futures)
            for f in futures:
                f.result()

        # After reset + accumulations, value should be non-negative
        assert tracker.prompt_tokens >= 0
