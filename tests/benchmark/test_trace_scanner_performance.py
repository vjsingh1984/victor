"""Benchmark: Rust vs Python JSONL trace scanning."""

import json
import tempfile
import time
import random
import uuid
from pathlib import Path

import pytest
from victor.processing.native._base import _NATIVE_AVAILABLE


def _generate_jsonl(n_events, seed=42):
    """Generate a temporary JSONL file with n events."""
    rng = random.Random(seed)
    path = Path(tempfile.mktemp(suffix=".jsonl"))
    sessions = [str(uuid.UUID(int=rng.getrandbits(128))) for _ in range(max(1, n_events // 10))]

    with open(path, "w") as f:
        for _ in range(n_events):
            sid = rng.choice(sessions)
            etype = rng.choice(["tool_call", "tool_result", "session_start", "user_prompt"])
            event = {
                "session_id": sid,
                "event_type": etype,
                "timestamp": "2026-04-11T00:00:00Z",
                "data": {},
            }
            if etype == "tool_result":
                event["data"]["success"] = rng.random() > 0.2
            f.write(json.dumps(event) + "\n")
    return str(path)


class TestTraceScannerCorrectness:

    def test_scan_empty_file(self):
        from victor.processing.native.trace_scanner import scan_usage_file

        path = Path(tempfile.mktemp(suffix=".jsonl"))
        path.write_text("")
        result = scan_usage_file(str(path))
        assert result == []
        path.unlink()

    def test_scan_nonexistent(self):
        from victor.processing.native.trace_scanner import scan_usage_file

        result = scan_usage_file("/nonexistent/file.jsonl")
        assert result == []

    def test_scan_correctness_vs_python(self):
        from victor.processing.native.trace_scanner import scan_usage_file, _scan_usage_file_python

        path = _generate_jsonl(500)
        rust_result = scan_usage_file(path)
        python_result = _scan_usage_file_python(path)
        # Same number of sessions
        assert len(rust_result) == len(python_result)
        # Same session IDs
        rust_ids = {s.session_id for s in rust_result}
        python_ids = {s.session_id for s in python_result}
        assert rust_ids == python_ids
        Path(path).unlink()

    def test_scan_malformed_lines_skipped(self):
        from victor.processing.native.trace_scanner import scan_usage_file

        path = Path(tempfile.mktemp(suffix=".jsonl"))
        path.write_text(
            '{"session_id":"s1","event_type":"tool_call","data":{}}\n'
            "NOT VALID JSON\n"
            '{"session_id":"s1","event_type":"tool_call","data":{}}\n'
            '{"session_id":"s1","event_type":"tool_call","data":{}}\n'
        )
        result = scan_usage_file(str(path))
        assert len(result) == 1
        assert result[0].tool_calls == 3
        path.unlink()

    def test_scan_filters_low_tool_calls(self):
        from victor.processing.native.trace_scanner import scan_usage_file

        path = Path(tempfile.mktemp(suffix=".jsonl"))
        path.write_text('{"session_id":"s1","event_type":"tool_call","data":{}}\n')
        result = scan_usage_file(str(path))
        assert result == []  # Only 1 tool_call, filtered
        path.unlink()


def _time_fn(func, *args, n=5):
    times = []
    for _ in range(n):
        s = time.perf_counter()
        func(*args)
        times.append((time.perf_counter() - s) * 1000)
    return sum(times) / len(times)


class TestTraceScannerPerformance:

    @pytest.mark.skipif(not _NATIVE_AVAILABLE, reason="No Rust")
    def test_benchmark_5k_events(self):
        from victor.processing.native.trace_scanner import scan_usage_file, _scan_usage_file_python

        path = _generate_jsonl(5000)
        py_ms = _time_fn(_scan_usage_file_python, path)
        rs_ms = _time_fn(scan_usage_file, path)
        sp = py_ms / rs_ms if rs_ms > 0 else float("inf")
        print(f"\n5K events: Py={py_ms:.1f}ms Rust={rs_ms:.1f}ms {sp:.1f}x")
        Path(path).unlink()
        assert sp >= 1.5

    @pytest.mark.skipif(not _NATIVE_AVAILABLE, reason="No Rust")
    def test_benchmark_real_usage_file(self):
        """Benchmark against the actual usage.jsonl if it exists."""
        from victor.processing.native.trace_scanner import scan_usage_file, _scan_usage_file_python

        real_path = str(Path.home() / ".victor/logs/usage.jsonl")
        if not Path(real_path).exists():
            pytest.skip("No real usage.jsonl")
        py_ms = _time_fn(_scan_usage_file_python, real_path, n=3)
        rs_ms = _time_fn(scan_usage_file, real_path, n=3)
        sp = py_ms / rs_ms if rs_ms > 0 else float("inf")
        print(f"\nReal usage.jsonl: Py={py_ms:.1f}ms Rust={rs_ms:.1f}ms {sp:.1f}x")
        assert sp >= 1.0  # At minimum not slower
