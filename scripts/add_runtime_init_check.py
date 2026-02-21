#!/usr/bin/env python3
"""Add runtime initialization checking to benchmark_startup_kpi.py.

This script adds:
1. Runtime initialization status checking in the activation probe
2. New CLI flags for coordination/interaction runtime lazy init expectations
3. Validation logic for runtime lazy init
4. Updated text output to show runtime initialization status
"""

import re

def add_runtime_check_function():
    """Add the _check_runtime_initialization function after _registry_totals."""
    with open("scripts/benchmark_startup_kpi.py", "r") as f:
        content = f.read()

    # Find the _registry_totals function and add our new function after it
    registry_totals_end_pattern = r'(            return attempted, applied\n\n)'
    runtime_check_function = r'''\1        def _check_runtime_initialization(orchestrator):
            """Check if coordination/interaction runtime components are lazily initialized."""
            runtime_status = {{}}

            # Check coordination runtime
            coordination_runtime = getattr(orchestrator, "_coordination_runtime", None)
            if coordination_runtime is not None:
                runtime_status["coordination_runtime"] = {{
                    "recovery_coordinator_initialized": bool(
                        getattr(coordination_runtime, "recovery_coordinator", None)
                        and getattr(coordination_runtime.recovery_coordinator, "initialized", False)
                    ),
                    "chunk_generator_initialized": bool(
                        getattr(coordination_runtime, "chunk_generator", None)
                        and getattr(coordination_runtime.chunk_generator, "initialized", False)
                    ),
                    "tool_planner_initialized": bool(
                        getattr(coordination_runtime, "tool_planner", None)
                        and getattr(coordination_runtime.tool_planner, "initialized", False)
                    ),
                    "task_coordinator_initialized": bool(
                        getattr(coordination_runtime, "task_coordinator", None)
                        and getattr(coordination_runtime.task_coordinator, "initialized", False)
                    ),
                }}
            else:
                runtime_status["coordination_runtime"] = None

            # Check interaction runtime
            interaction_runtime = getattr(orchestrator, "_interaction_runtime", None)
            if interaction_runtime is not None:
                runtime_status["interaction_runtime"] = {{
                    "chat_coordinator_initialized": bool(
                        getattr(interaction_runtime, "chat_coordinator", None)
                        and getattr(interaction_runtime.chat_coordinator, "initialized", False)
                    ),
                    "tool_coordinator_initialized": bool(
                        getattr(interaction_runtime, "tool_coordinator", None)
                        and getattr(interaction_runtime.tool_coordinator, "initialized", False)
                    ),
                    "session_coordinator_initialized": bool(
                        getattr(interaction_runtime, "session_coordinator", None)
                        and getattr(interaction_runtime.session_coordinator, "initialized", False)
                    ),
                }}
            else:
                runtime_status["interaction_runtime"] = None

            return runtime_status

'''

    content = re.sub(registry_totals_end_pattern, runtime_check_function, content, count=1)

    with open("scripts/benchmark_startup_kpi.py", "w") as f:
        f.write(content)

    print("Added _check_runtime_initialization function")


def update_measure_once_call():
    """Update the _measure_once function to call _check_runtime_initialization and return the status."""
    with open("scripts/benchmark_startup_kpi.py", "r") as f:
        content = f.read()

    # Find the line where attempted_total, applied_total are returned
    # and add runtime_init_status
    old_return = r'            attempted_total, applied_total = _registry_totals\(registry_metrics\)\n            await agent\.close\(\)\n            return elapsed_ms, runtime_flags, registry_metrics, attempted_total, applied_total'

    new_return = r'''            attempted_total, applied_total = _registry_totals(registry_metrics)

            # Check runtime initialization status
            runtime_init_status = _check_runtime_initialization(orchestrator)

            await agent.close()
            return elapsed_ms, runtime_flags, registry_metrics, attempted_total, applied_total, runtime_init_status'''

    content = re.sub(old_return, new_return, content, count=1)

    with open("scripts/benchmark_startup_kpi.py", "w") as f:
        f.write(content)

    print("Updated _measure_once return to include runtime_init_status")


def update_cold_unpacking():
    """Update the cold_ms unpacking to include runtime_init_status."""
    with open("scripts/benchmark_startup_kpi.py", "r") as f:
        content = f.read()

    # Update the cold_ms unpacking
    old_cold = r'            cold_ms, runtime_flags, registry_metrics, attempted_total, applied_total = \(\n                await _measure_once\(vertical_cls\)\n            \)'
    new_cold = r'''            cold_ms, runtime_flags, registry_metrics, attempted_total, applied_total, cold_runtime_init = (
                await _measure_once(vertical_cls)
            )'''

    content = re.sub(old_cold, new_cold, content, count=1)

    with open("scripts/benchmark_startup_kpi.py", "w") as f:
        f.write(content)

    print("Updated cold_ms unpacking to include cold_runtime_init")


def update_warm_loop():
    """Update the warm loop to handle the extra return value."""
    with open("scripts/benchmark_startup_kpi.py", "r") as f:
        content = f.read()

    # Update warm loop to discard runtime_init_status
    old_warm = r'            for _ in range\(warm_iters\):\n                elapsed_ms, _, _, _, _ = await _measure_once\(vertical_cls\)'
    new_warm = r'            for _ in range(warm_iters):\n                elapsed_ms, _, _, _, _, _ = await _measure_once(vertical_cls)'

    content = re.sub(old_warm, new_warm, content, count=1)

    with open("scripts/benchmark_startup_kpi.py", "w") as f:
        f.write(content)

    print("Updated warm loop to handle extra return value")


def update_json_output():
    """Update the JSON output to include runtime_initialization_status."""
    with open("scripts/benchmark_startup_kpi.py", "r") as f:
        content = f.read()

    # Add runtime_initialization_status to the JSON output
    old_json = r'''                        "framework_registry_applied_total": applied_total,
                    \}\s*\)\s*\)\s*'''
    new_json = r'''                        "framework_registry_applied_total": applied_total,
                        "runtime_initialization_status": cold_runtime_init,
                    }}
                )
            )'''

    content = re.sub(old_json, new_json, content, count=1)

    with open("scripts/benchmark_startup_kpi.py", "w") as f:
        f.write(content)

    print("Updated JSON output to include runtime_initialization_status")


def update_return_dict():
    """Update the return dict to include runtime_initialization_status."""
    with open("scripts/benchmark_startup_kpi.py", "r") as f:
        content = f.read()

    # Find the line with "framework_registry_applied_total" and add runtime_initialization_status after it
    old_line = r'''        "framework_registry_applied_total": int\(payload\.get\("framework_registry_applied_total", 0\)\),\n    \}'''
    new_line = r'''        "framework_registry_applied_total": int(payload.get("framework_registry_applied_total", 0)),
        "runtime_initialization_status": dict(payload.get("runtime_initialization_status", {})),
    }'''

    content = re.sub(old_line, new_line, content, count=1)

    with open("scripts/benchmark_startup_kpi.py", "w") as f:
        f.write(content)

    print("Updated return dictionary to include runtime_initialization_status")


def main():
    """Apply all updates to add runtime initialization checking."""
    print("Adding runtime initialization checking to benchmark_startup_kpi.py...")

    try:
        add_runtime_check_function()
        update_measure_once_call()
        update_cold_unpacking()
        update_warm_loop()
        update_json_output()
        update_return_dict()

        print("\nSuccessfully added runtime initialization checking!")
        print("You still need to manually add:")
        print("1. CLI arguments: --require-coordination-runtime-lazy, --require-interaction-runtime-lazy")
        print("2. _collect_runtime_lazy_expectations function")
        print("3. _evaluate_runtime_lazy_expectation_failures function")
        print("4. Wire up the validation in main()")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
