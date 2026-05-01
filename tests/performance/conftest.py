"""Configuration for performance regression tests.

This module provides pytest configuration and fixtures for performance testing,
including benchmark setup, CI gates, and regression detection.
"""

import pytest
from pathlib import Path


def pytest_configure(config) -> None:
    """Configure pytest with performance test markers.

    This ensures that performance tests are properly categorized and can
    be run selectively.
    """
    config.addinivalue_line(
        "markers",
        "performance: marks tests as performance regression tests "
        "(use 'pytest -m performance' to run only these)",
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmark tests " "(requires pytest-benchmark plugin)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow-running (may be skipped in quick runs)"
    )


@pytest.fixture(scope="session")
def benchmark_data_dir() -> Path:
    """Get directory for storing benchmark data.

    This directory stores baseline performance data for regression detection.
    """
    data_dir = Path(__file__).parent / ".benchmarks"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture(scope="session")
def baseline_thresholds() -> dict:
    """Define baseline performance thresholds.

    These thresholds are used to detect performance regressions.
    Values are in milliseconds (ms).

    Format: {test_name: max_acceptable_time_ms}
    """
    return {
        "test_register_10_items": 0.5,
        "test_register_100_items": 5.0,
        "test_register_1000_items": 50.0,
        "test_batch_register_100_items": 2.0,  # Batch should be faster
        "test_batch_register_1000_items": 20.0,  # Batch should be faster
        "test_batch_registration_api_100": 2.0,
        "test_batch_registration_api_1000": 20.0,
    }


@pytest.fixture(scope="session")
def regression_threshold() -> float:
    """Define threshold for detecting performance regression.

    A regression is detected when performance degrades by more than
    this percentage compared to baseline.

    Example: 0.20 means a 20% slowdown triggers a regression warning.
    """
    return 0.20  # 20% regression threshold


@pytest.fixture(scope="session")
def improvement_threshold() -> float:
    """Define threshold for detecting performance improvement.

    An improvement is detected when performance improves by more than
    this percentage compared to baseline.

    Example: 0.10 means a 10% speedup triggers an improvement notice.
    """
    return 0.10  # 10% improvement threshold


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    """Print performance regression summary after test run.

    This hook runs after all tests complete and provides a summary
    of any performance regressions or improvements detected.
    """
    if not hasattr(config, "_benchmarks"):
        return

    benchmarks = config._benchmarks
    if not benchmarks:
        return

    terminalreporter.section("Performance Regression Summary")

    # Check for regressions and improvements
    regressions = []
    improvements = []

    for name, stats in benchmarks.items():
        baseline = stats.get("baseline")
        current = stats.get("current")

        if baseline and current:
            change_pct = ((current - baseline) / baseline) * 100

            if change_pct > 20:  # 20% regression
                regressions.append((name, change_pct))
            elif change_pct < -10:  # 10% improvement
                improvements.append((name, change_pct))

    # Report regressions
    if regressions:
        terminalreporter.write_sep("=", red=True)
        terminalreporter.write_line("⚠️  PERFORMANCE REGRESSIONS DETECTED", red=True)
        for name, change_pct in regressions:
            terminalreporter.write_line(f"  {name}: +{change_pct:.1f}% slower", red=True)
        terminalreporter.write_sep("=", red=True)

    # Report improvements
    if improvements:
        terminalreporter.write_sep("=", green=True)
        terminalreporter.write_line("✅ PERFORMANCE IMPROVEMENTS", green=True)
        for name, change_pct in improvements:
            terminalreporter.write_line(f"  {name}: {change_pct:.1f}% faster", green=True)
        terminalreporter.write_sep("=", green=True)

    # Summary
    total = len(benchmarks)
    terminalreporter.write_line(f"\nTotal benchmarks: {total}")
    terminalreporter.write_line(f"Regressions: {len(regressions)}")
    terminalreporter.write_line(f"Improvements: {len(improvements)}")
