#!/usr/bin/env python3
"""Test script for AgenticBenchmarkRunner with sample tasks.

This script demonstrates and tests the agentic benchmark harness by:
1. Creating sample benchmark tasks (simple Python bug fixes)
2. Running a mock agent that simulates file edits
3. Validating the results through the harness pipeline

Usage:
    python scripts/test_agentic_harness.py
"""

import asyncio
import logging
import tempfile
import time
from pathlib import Path

from victor.evaluation.agentic_harness import (
    AgenticBenchmarkRunner,
    AgenticExecutionTrace,
    FileEdit,
    ToolCall,
    AgenticValidationType,
)
from victor.evaluation.protocol import (
    BenchmarkTask,
    BenchmarkType,
    EvaluationConfig,
    TaskStatus,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def create_sample_tasks() -> list[BenchmarkTask]:
    """Create sample benchmark tasks for testing."""
    tasks = [
        BenchmarkTask(
            task_id="sample-001",
            benchmark=BenchmarkType.SWE_BENCH,
            description="Fix off-by-one error in sum_range",
            prompt="Fix the off-by-one error in the sum_range function",
            test_code="""
def test_sum_range():
    assert sum_range(1, 5) == 15  # 1+2+3+4+5
    assert sum_range(0, 3) == 6   # 0+1+2+3
""",
            repo="test/sample-repo",
            base_commit="abc123",
            issue_text="The sum_range function has an off-by-one error",
        ),
        BenchmarkTask(
            task_id="sample-002",
            benchmark=BenchmarkType.SWE_BENCH,
            description="Add input validation to divide",
            prompt="Add input validation to the divide function",
            test_code="""
def test_divide():
    assert divide(10, 2) == 5
    assert divide(0, 5) == 0
    try:
        divide(5, 0)
        assert False, "Should raise ZeroDivisionError"
    except ZeroDivisionError:
        pass
""",
            repo="test/sample-repo",
            base_commit="def456",
            issue_text="Add zero division check",
        ),
        BenchmarkTask(
            task_id="sample-003",
            benchmark=BenchmarkType.SWE_BENCH,
            description="Fix recursive fibonacci function",
            prompt="Fix the recursive fibonacci function",
            test_code="""
def test_fibonacci():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(10) == 55
""",
            repo="test/sample-repo",
            base_commit="ghi789",
            issue_text="Fibonacci returns wrong values",
        ),
    ]
    return tasks


async def mock_agent_callback(
    task: BenchmarkTask,
    workspace_dir: Path,
) -> AgenticExecutionTrace:
    """Mock agent that simulates fixing simple Python bugs.

    This simulates what Victor's orchestrator would do when solving tasks.
    """
    trace = AgenticExecutionTrace(
        task_id=task.task_id,
        start_time=time.time(),
    )

    # Simulate tool usage
    trace.tool_calls.append(
        ToolCall(
            name="read_file",
            arguments={"path": "main.py"},
            result="def sum_range(start, end):\n    return sum(range(start, end))\n",
            success=True,
            timestamp=time.time(),
        )
    )
    trace.turns += 1

    # Simulate thinking and file edit
    await asyncio.sleep(0.1)  # Simulate some processing time

    # Create the workspace with a "buggy" file
    main_py = workspace_dir / "main.py"

    if task.task_id == "sample-001":
        # Off-by-one bug
        buggy_code = "def sum_range(start, end):\n    return sum(range(start, end))\n"
        fixed_code = "def sum_range(start, end):\n    return sum(range(start, end + 1))\n"
        main_py.write_text(buggy_code)
    elif task.task_id == "sample-002":
        # Missing validation
        buggy_code = "def divide(a, b):\n    return a / b\n"
        fixed_code = "def divide(a, b):\n    if b == 0:\n        raise ZeroDivisionError('Cannot divide by zero')\n    return a / b\n"
        main_py.write_text(buggy_code)
    elif task.task_id == "sample-003":
        # Bad fibonacci
        buggy_code = "def fibonacci(n):\n    if n <= 1:\n        return 1\n    return fibonacci(n-1) + fibonacci(n-2)\n"
        fixed_code = "def fibonacci(n):\n    if n <= 0:\n        return 0\n    if n == 1:\n        return 1\n    return fibonacci(n-1) + fibonacci(n-2)\n"
        main_py.write_text(buggy_code)
    else:
        buggy_code = ""
        fixed_code = ""

    # Apply the fix
    main_py.write_text(fixed_code)

    trace.tool_calls.append(
        ToolCall(
            name="file_edit",
            arguments={"path": "main.py", "content": fixed_code},
            result="File edited successfully",
            success=True,
            timestamp=time.time(),
        )
    )
    trace.turns += 1

    # Record file edit
    trace.file_edits.append(
        FileEdit(
            path="main.py",
            action="modify",
            before_content=buggy_code,
            after_content=fixed_code,
            diff=f"- {buggy_code}\n+ {fixed_code}",
        )
    )

    # Generate a patch
    trace.generated_patch = f"""--- a/main.py
+++ b/main.py
@@ -1,2 +1,2 @@
-{buggy_code.strip()}
+{fixed_code.strip()}
"""

    trace.end_time = time.time()
    trace.messages = [
        {"role": "user", "content": task.prompt},
        {"role": "assistant", "content": "I'll fix the issue in main.py"},
    ]

    return trace


async def run_test():
    """Run the agentic harness test."""
    print("=" * 60)
    print("AgenticBenchmarkRunner Test Drive")
    print("=" * 60)

    # Create sample tasks
    tasks = create_sample_tasks()
    print(f"\nCreated {len(tasks)} sample tasks:")
    for task in tasks:
        print(f"  - {task.task_id}: {task.prompt[:50]}...")

    # Create runner with default validators
    workspace_base = Path(tempfile.mkdtemp(prefix="agentic_test_"))
    runner = AgenticBenchmarkRunner(
        timeout=60,
        workspace_base=workspace_base,
    )
    print(f"\nWorkspace: {workspace_base}")

    # Create evaluation config
    config = EvaluationConfig(
        benchmark=BenchmarkType.SWE_BENCH,
        model="test-model",
        max_tasks=len(tasks),
        timeout_per_task=60,
    )

    # Progress callback
    def progress_callback(current: int, total: int, result):
        status_icon = {
            TaskStatus.PASSED: "[PASS]",
            TaskStatus.FAILED: "[FAIL]",
            TaskStatus.ERROR: "[ERR]",
            TaskStatus.TIMEOUT: "[TIME]",
        }.get(result.status, "[?]")
        print(
            f"  {status_icon} Task {current}/{total}: {result.task_id} "
            f"(score: {result.overall_score:.3f})"
        )

    print("\nRunning benchmark...")
    print("-" * 40)

    # Run the benchmark
    metrics = await runner.run_benchmark(
        tasks=tasks,
        agent_callback=mock_agent_callback,
        config=config,
        progress_callback=progress_callback,
    )

    print("-" * 40)
    print("\nResults Summary:")
    print(f"  Total tasks: {metrics.total_tasks}")
    print(f"  Passed: {metrics.passed}")
    print(f"  Failed: {metrics.failed}")
    print(f"  Errors: {metrics.errors}")
    print(f"  Timeouts: {metrics.timeouts}")
    print(f"  Pass rate: {metrics.pass_rate:.1%}")
    print(f"  Avg score: {metrics.avg_overall_score:.3f}")
    print(f"  Total time: {metrics.total_time_seconds:.2f}s")
    print(f"  Total turns: {metrics.total_turns}")
    print(f"  Total tool calls: {metrics.total_tool_calls}")

    print("\nPer-task details:")
    for result in metrics.task_results:
        print(f"\n  {result.task_id}:")
        print(f"    Status: {result.status.value}")
        print(f"    Patch score: {result.patch_score:.3f}")
        print(f"    Test score: {result.test_score:.3f}")
        print(f"    Edit accuracy: {result.edit_accuracy:.3f}")
        print(f"    Tool efficiency: {result.tool_efficiency:.3f}")
        print(f"    Overall score: {result.overall_score:.3f}")
        if result.error_message:
            print(f"    Error: {result.error_message}")

    # Serialize to dict (test JSON export)
    print("\nJSON export test:")
    report = metrics.to_dict()
    print(f"  Keys: {list(report.keys())}")
    print(f"  Task results: {len(report.get('task_results', []))} entries")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    asyncio.run(run_test())
