#!/usr/bin/env python3
"""Competitive Benchmark Execution Script for Agentic AI Frameworks.

This script executes the competitive benchmark suite defined in
docs/benchmarking/competitive-benchmark-rubric.md.

Usage:
    # Run all tasks for Victor
    python docs/benchmarking/run_benchmark.py --framework victor

    # Run specific task
    python docs/benchmarking/run_benchmark.py --framework victor --task C1

    # Run with verbose output
    python docs/benchmarking/run_benchmark.py --framework victor --verbose

    # Dry run (show what would be executed)
    python docs/benchmarking/run_benchmark.py --framework victor --dry-run
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Task Definitions
# ============================================================================

TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "C1": {
        "name": "Single-file generation",
        "category": "Code Generation",
        "prompt": """Generate a Python class named `DataProcessor` with the following:
- Constructor taking `data_path: str` and `batch_size: int = 100`
- Method `process()` that loads CSV, applies transformations, returns DataFrame
- Method `save()` that saves processed data to CSV
- Type hints and docstrings
- Error handling for missing files

Requirements:
1. Code compiles without syntax errors
2. Class has all required methods
3. Type hints present
4. Error handling included
5. Follows PEP 8 style""",
        "complexity": "Simple",
        "timeout_seconds": 60,
        "max_tokens": 2000,
    },
    "C2": {
        "name": "Multi-file refactoring",
        "category": "Code Generation",
        "prompt": """Refactor the following codebase to use dependency injection:

1. Create an interface for the data service
2. Modify UserService to accept the interface via constructor
3. Update the factory to inject the correct implementation
4. Ensure all tests still pass

Files to modify:
- src/services/user_service.py
- src/services/data_service.py
- src/factories.py

The refactoring should maintain backward compatibility.""",
        "complexity": "Medium",
        "timeout_seconds": 120,
        "max_tokens": 4000,
    },
    "C3": {
        "name": "Bug fix with context",
        "category": "Code Generation",
        "prompt": """Fix the bug in the authentication module.

Context:
- The auth_token becomes invalid after 5 minutes
- Users are being logged out unexpectedly
- The token refresh logic exists but isn't being called

Fix the bug in src/auth/session.py ensuring:
1. Tokens are refreshed automatically before expiry
2. No changes to the public API
3. Thread safety is maintained
4. Tests pass""",
        "complexity": "Medium",
        "timeout_seconds": 120,
        "max_tokens": 3000,
    },
    "C4": {
        "name": "Code review",
        "category": "Code Generation",
        "prompt": """Review the following Python code and provide structured feedback:

```python
def process_data(data: List[Dict]) -> List[Dict]:
    result = []
    for item in data:
        if item.get('active'):
            new_item = {}
            new_item['id'] = item['id']
            new_item['value'] = item.get('value', 0) * 2
            if new_item['value'] > 100:
                new_item['flag'] = True
            result.append(new_item)
    return result
```

Provide feedback on:
1. Code style and PEP 8 compliance
2. Performance considerations
3. Error handling
4. Type safety
5. Suggested improvements""",
        "complexity": "Medium",
        "timeout_seconds": 90,
        "max_tokens": 2000,
    },
    "C5": {
        "name": "Documentation generation",
        "category": "Code Generation",
        "prompt": """Generate comprehensive documentation for this API class:

```python
class APIClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.session = None

    def connect(self) -> bool:
        """Establish connection to API."""
        pass

    def get_resource(self, resource_id: str) -> Optional[Dict]:
        """Fetch a resource by ID."""
        pass

    def create_resource(self, data: Dict) -> Optional[str]:
        """Create a new resource."""
        pass
```

Generate:
1. Module-level docstring
2. Class docstring with usage example
3. Method docstrings with parameters, returns, raises
4. Example usage in a docstring
5. Type hints for all parameters""",
        "complexity": "Simple",
        "timeout_seconds": 60,
        "max_tokens": 2000,
    },
    "R1": {
        "name": "Research synthesis",
        "category": "Multi-Step Reasoning",
        "prompt": """Synthesize findings from the following research sources on microservices architecture:

Source 1: Discusses benefits including scalability, fault isolation, independent deployment
Source 2: Covers challenges: network latency, distributed transactions, debugging complexity
Source 3: Presents best practices: API gateways, service mesh, observability

Provide a comprehensive synthesis covering:
1. Key benefits with real-world examples
2. Major challenges and mitigation strategies
3. When to use microservices vs monolith
4. Recommended technology stack components
5. Common pitfalls to avoid""",
        "complexity": "Complex",
        "timeout_seconds": 180,
        "max_tokens": 4000,
    },
    "R2": {
        "name": "Architecture design",
        "category": "Multi-Step Reasoning",
        "prompt": """Design a system architecture for a real-time collaborative document editing platform with the following requirements:

Functional Requirements:
- Multiple users edit documents simultaneously
- Real-time sync with conflict resolution
- Document history and version control
- Rich text formatting with images
- User comments and suggestions

Non-Functional Requirements:
- Support 10,000 concurrent users
- <100ms latency for edits to propagate
- 99.9% availability
- Data durability guarantees

Provide:
1. System architecture diagram (described in text)
2. Technology choices with justification
3. Data model and persistence strategy
4. Real-time sync mechanism
5. Conflict resolution strategy
6. Scalability approach
7. Security considerations""",
        "complexity": "Complex",
        "timeout_seconds": 300,
        "max_tokens": 5000,
    },
    "T1": {
        "name": "File operations",
        "category": "Tool Usage",
        "prompt": """Perform the following file operations:

1. Create a directory structure: output/{docs,images,data}
2. Read the file at README.md
3. Extract all code blocks from the README
4. Write each code block to a separate file in output/docs/
5. Create a summary file listing all extracted files

Use file system tools to complete these operations.""",
        "complexity": "Simple",
        "timeout_seconds": 90,
        "max_tokens": 2000,
    },
    "W1": {
        "name": "Sequential workflow",
        "category": "Workflow & Coordination",
        "prompt": """Execute a sequential data processing workflow:

Step 1: Load data from data.csv
Step 2: Validate the data (check for missing values, invalid types)
Step 3: If validation fails, clean the data (fill/remove)
Step 4: Transform the data (normalize numeric fields, encode categoricals)
Step 5: Save the processed data to processed.csv

Each step should only execute if the previous step succeeded.
If any step fails, provide an error message and stop.""",
        "complexity": "Medium",
        "timeout_seconds": 180,
        "max_tokens": 3000,
    },
}


# ============================================================================
# Framework Adapters
# ============================================================================

class FrameworkAdapter:
    """Base class for framework benchmark adapters."""

    def __init__(self, timeout: int = 300):
        self.timeout = timeout

    async def execute_task(
        self,
        task_id: str,
        task_def: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a benchmark task.

        Args:
            task_id: Task identifier (e.g., "C1")
            task_def: Task definition from TASK_REGISTRY

        Returns:
            Result dictionary with success, duration, output, error, etc.
        """
        raise NotImplementedError


class VictorAdapter(FrameworkAdapter):
    """Victor framework adapter for benchmark execution."""

    def __init__(self, timeout: int = 300):
        super().__init__(timeout)
        self.orchestrator = None

    async def execute_task(
        self,
        task_id: str,
        task_def: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute task using Victor."""
        try:
            import psutil
            from victor import Agent
            from victor.config import Settings

            # Start resource monitoring
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            start_time = time.time()

            # Create Victor agent with minimal configuration
            settings = Settings(
                model="gpt-4o-mini",  # Use faster model for benchmarks
                temperature=0.7,
                max_tokens=task_def.get("max_tokens", 2000),
                timeout=self.timeout,
            )

            agent = Agent(settings=settings)

            # Execute the task
            response = await agent.arun(
                task_def["prompt"],
                tools=["file_read", "file_write", "file_search"],
            )

            # Calculate metrics
            duration_ms = (time.time() - start_time) * 1000
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_mb = end_memory - start_memory
            cpu_percent = process.cpu_percent()

            return {
                "task_id": task_id,
                "framework": "victor",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "success": True,
                "duration_ms": round(duration_ms, 2),
                "output": response.content or "",
                "output_quality": None,  # To be human-rated
                "memory_mb": round(memory_mb, 2),
                "cpu_percent": cpu_percent,
                "error": None,
                "notes": "Completed successfully",
            }

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return {
                "task_id": task_id,
                "framework": "victor",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "success": False,
                "duration_ms": round(duration_ms, 2),
                "output": "",
                "output_quality": None,
                "memory_mb": 0,
                "cpu_percent": 0,
                "error": str(e),
                "notes": f"Failed: {type(e).__name__}",
            }


class MockAdapter(FrameworkAdapter):
    """Mock adapter for testing and dry-run mode."""

    async def execute_task(
        self,
        task_id: str,
        task_def: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Simulate task execution without actually running."""
        await asyncio.sleep(0.1)  # Simulate minimal work

        return {
            "task_id": task_id,
            "framework": "mock",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "success": True,
            "duration_ms": 1000.0,
            "output": f"Mock output for {task_id}",
            "output_quality": 3,
            "memory_mb": 100.0,
            "cpu_percent": 25.0,
            "error": None,
            "notes": "Mock execution",
        }


class LangGraphAdapter(FrameworkAdapter):
    """LangGraph framework adapter for benchmark execution.

    Note: This is a stub for demonstration. Full implementation requires:
    1. LangGraph and LangChain dependencies
    2. State graph definition for each task
    3. Tool configuration matching Victor's tools
    """

    async def execute_task(
        self,
        task_id: str,
        task_def: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute task using LangGraph.

        Stub implementation for demonstration.
        """
        # TODO: Implement LangGraph execution
        # 1. Create StateGraph with nodes for task execution
        # 2. Configure tools matching the task requirements
        # 3. Execute graph and capture results
        return {
            "task_id": task_id,
            "framework": "langgraph",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "success": False,
            "duration_ms": 0,
            "output": "",
            "output_quality": None,
            "memory_mb": 0,
            "cpu_percent": 0,
            "error": "Not yet implemented",
            "notes": "LangGraph adapter is a stub",
        }


class CrewAIAdapter(FrameworkAdapter):
    """CrewAI framework adapter for benchmark execution.

    Note: This is a stub for demonstration. Full implementation requires:
    1. CrewAI dependencies
    2. Agent definitions for each task type
    3. Tool configuration matching Victor's tools
    """

    async def execute_task(
        self,
        task_id: str,
        task_def: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute task using CrewAI.

        Stub implementation for demonstration.
        """
        # TODO: Implement CrewAI execution
        # 1. Create Agent with appropriate role
        # 2. Configure tools matching the task requirements
        # 3. Execute crew and capture results
        return {
            "task_id": task_id,
            "framework": "crewai",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "success": False,
            "duration_ms": 0,
            "output": "",
            "output_quality": None,
            "memory_mb": 0,
            "cpu_percent": 0,
            "error": "Not yet implemented",
            "notes": "CrewAI adapter is a stub",
        }


# ============================================================================
# Benchmark Runner
# ============================================================================

async def run_benchmark(
    framework: str,
    task_ids: Optional[List[str]] = None,
    timeout: int = 300,
    verbose: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run benchmark suite.

    Args:
        framework: Framework name (victor, langgraph, crewai, etc.)
        task_ids: List of task IDs to run (None = all tasks)
        timeout: Per-task timeout in seconds
        verbose: Enable verbose output
        dry_run: Show what would be executed without running

    Returns:
        Summary dictionary with all results
    """
    # Select tasks to run
    if task_ids is None:
        tasks_to_run = list(TASK_REGISTRY.keys())
    else:
        tasks_to_run = [t for t in task_ids if t in TASK_REGISTRY]

    if not tasks_to_run:
        return {"error": "No valid tasks to run"}

    # Select adapter
    if dry_run:
        adapter = MockAdapter()
    elif framework == "victor":
        adapter = VictorAdapter(timeout=timeout)
    elif framework == "langgraph":
        adapter = LangGraphAdapter(timeout=timeout)
    elif framework == "crewai":
        adapter = CrewAIAdapter(timeout=timeout)
    else:
        return {"error": f"Framework '{framework}' not yet implemented"}

    results = []
    successful = 0
    failed = 0

    print(f"\n{'='*60}")
    print(f"Benchmark: {framework.upper()}")
    print(f"Tasks: {len(tasks_to_run)}")
    print(f"Timeout: {timeout}s per task")
    print(f"{'='*60}\n")

    for task_id in tasks_to_run:
        task_def = TASK_REGISTRY[task_id]

        if dry_run:
            print(f"[DRY RUN] {task_id}: {task_def['name']}")
            print(f"  Category: {task_def['category']}")
            print(f"  Complexity: {task_def['complexity']}")
            print(f"  Timeout: {task_def['timeout_seconds']}s")
            print()
            continue

        print(f"[{task_id}] {task_def['name']} ({task_def['complexity']})...")
        start_time = time.time()

        result = await adapter.execute_task(task_id, task_def)
        results.append(result)

        duration = time.time() - start_time

        if result["success"]:
            successful += 1
            print(f"  ✅ SUCCESS in {duration:.2f}s")
            if verbose:
                print(f"  Output length: {len(result['output'])} chars")
                print(f"  Memory: {result['memory_mb']:.1f} MB")
                print(f"  CPU: {result['cpu_percent']:.1f}%")
        else:
            failed += 1
            print(f"  ❌ FAILED: {result['error'][:100]}")

    # Calculate summary
    if dry_run:
        return {
            "framework": framework,
            "dry_run": True,
            "tasks_count": len(tasks_to_run),
            "tasks": tasks_to_run,
        }

    total_duration = sum(r["duration_ms"] for r in results) / 1000
    success_rate = successful / len(tasks_to_run) if tasks_to_run else 0

    summary = {
        "framework": framework,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "tasks_total": len(tasks_to_run),
        "tasks_successful": successful,
        "tasks_failed": failed,
        "success_rate": round(success_rate * 100, 2),
        "total_duration_seconds": round(total_duration, 2),
        "avg_duration_seconds": round(total_duration / len(tasks_to_run), 2)
        if tasks_to_run else 0,
        "results": results,
    }

    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Total tasks: {len(tasks_to_run)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {summary['success_rate']}%")
    print(f"Total duration: {summary['total_duration_seconds']:.2f}s")
    print(f"{'='*60}\n")

    return summary


def save_results(results: Dict[str, Any], framework: str) -> str:
    """Save benchmark results to file.

    Args:
        results: Results dictionary from run_benchmark
        framework: Framework name

    Returns:
        Path to saved results file
    """
    results_dir = PROJECT_ROOT / "docs" / "benchmarking" / "results" / framework
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"benchmark_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_file}")
    return str(results_file)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run competitive benchmarks for agentic AI frameworks"
    )
    parser.add_argument(
        "--framework",
        type=str,
        default="victor",
        choices=["victor", "langgraph", "crewai", "mock", "dry-run"],
        help="Framework to benchmark",
    )
    parser.add_argument(
        "--task",
        type=str,
        action="append",
        help="Task ID(s) to run (default: all)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-task timeout in seconds",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running",
    )
    parser.add_argument(
        "--output", "-o",
        action="store_true",
        help="Save results to file",
    )

    args = parser.parse_args()

    # Run benchmark
    results = asyncio.run(run_benchmark(
        framework=args.framework,
        task_ids=args.task,
        timeout=args.timeout,
        verbose=args.verbose,
        dry_run=args.dry_run,
    ))

    # Save results if requested
    if args.output and not args.dry_run and "error" not in results:
        save_results(results, args.framework)

    # Exit with error code if any tasks failed
    if results.get("tasks_failed", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
