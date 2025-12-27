#!/usr/bin/env python3
"""Test Victor with different providers in plan mode to observe native extension speedup.

This script tests DeepSeek, xAI/Grok, Google/Gemini, and OpenAI providers.
"""

import subprocess
import time
import json
import sys
from pathlib import Path

# Test prompt that requires planning with tool usage
TEST_PROMPT = """Create a plan to add a new caching layer to a Python web API.
The cache should:
1. Support Redis and in-memory backends
2. Have configurable TTL
3. Support cache invalidation
4. Include metrics collection

What files need to be created/modified? Outline the implementation steps."""

PROVIDERS = [
    {"name": "DeepSeek", "profile": "deepseek"},
    {"name": "xAI/Grok", "profile": "grok"},
    {"name": "Google/Gemini", "profile": "gemini"},
    {"name": "OpenAI/GPT-4.1", "profile": "gpt-4.1"},
]

def test_provider(profile: str, name: str) -> dict:
    """Test a provider and return timing results."""
    print(f"\n{'='*60}")
    print(f"Testing {name} ({profile})")
    print(f"{'='*60}")

    cmd = [
        "victor", "chat",
        "--no-tui",
        "--profile", profile,
        "--mode", "plan",
        "--plain",
        "--quiet",
        TEST_PROMPT
    ]

    start_time = time.perf_counter()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
            cwd=Path(__file__).parent.parent
        )

        end_time = time.perf_counter()
        elapsed = end_time - start_time

        success = result.returncode == 0
        output = result.stdout if success else result.stderr

        # Truncate output for display
        display_output = output[:500] + "..." if len(output) > 500 else output

        print(f"\nStatus: {'✓ Success' if success else '✗ Failed'}")
        print(f"Time: {elapsed:.2f}s")
        print(f"\nResponse preview:")
        print("-" * 40)
        print(display_output)
        print("-" * 40)

        return {
            "provider": name,
            "profile": profile,
            "success": success,
            "time_seconds": elapsed,
            "output_length": len(output),
            "error": result.stderr if not success else None
        }

    except subprocess.TimeoutExpired:
        print(f"✗ Timeout after 120s")
        return {
            "provider": name,
            "profile": profile,
            "success": False,
            "time_seconds": 120,
            "error": "Timeout"
        }
    except Exception as e:
        print(f"✗ Error: {e}")
        return {
            "provider": name,
            "profile": profile,
            "success": False,
            "error": str(e)
        }

def main():
    """Run tests for all providers."""
    print("Victor Provider Plan Mode Test")
    print("=" * 60)
    print(f"Test prompt: {TEST_PROMPT[:80]}...")
    print()

    # Check if native extensions are available
    try:
        from victor_native import is_native_available
        native = is_native_available()
        print(f"Native extensions: {'✓ Available (Rust/SIMD)' if native else '✗ Not available (Python fallback)'}")
    except ImportError:
        print("Native extensions: ✗ Not installed")

    results = []

    for provider in PROVIDERS:
        result = test_provider(provider["profile"], provider["name"])
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Provider':<20} {'Status':<10} {'Time':<10} {'Output':<10}")
    print("-" * 60)

    for r in results:
        status = "✓" if r.get("success") else "✗"
        time_str = f"{r.get('time_seconds', 0):.2f}s"
        output_str = f"{r.get('output_length', 0)} chars" if r.get("success") else r.get("error", "N/A")[:20]
        print(f"{r['provider']:<20} {status:<10} {time_str:<10} {output_str:<10}")

    # Save results
    results_file = Path(__file__).parent / "provider_test_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    main()
