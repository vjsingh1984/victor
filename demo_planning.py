#!/usr/bin/env python
"""Demo of planning mode detection and logic.

This demonstrates how the planning system would work for complex tasks.
"""

from victor.agent.coordinators.planning_coordinator import (
    PlanningCoordinator,
    PlanningConfig,
    PlanningMode,
)
from victor.agent.planning.readable_schema import TaskComplexity


def demo_planning_detection():
    """Demonstrate planning detection logic."""
    print("=" * 70)
    print("Planning Mode Detection Demo")
    print("=" * 70)

    # Mock orchestrator
    class MockOrchestrator:
        provider = None
        model = "test-model"
        max_tokens = 4096

    coordinator = PlanningCoordinator(MockOrchestrator())

    # Test different task types
    test_cases = [
        ("Simple question", "What is the weather today?", False),
        ("Code question", "How do I use argparse?", False),
        ("Architecture analysis", "Analyze the codebase architecture", True),
        ("Multi-step with keywords", "Analyze architecture and design improvements", True),
        ("Step indicators", "First analyze, then design, finally implement", True),
        ("SOLID evaluation", "Evaluate SOLID principles and identify violations", True),
        ("Complex design", "Design and implement user auth system with analysis", True),
    ]

    print(f"\n{'Task Type':<25} {'Message':<40} {'Use Planning'}")
    print("-" * 70)

    for task_type, message, expected in test_cases:
        should_plan = coordinator._should_use_planning(message)
        status = "✓" if should_plan == expected else "✗"
        print(f"{task_type:<25} {message:<40} {should_plan!s:<5} {status}")

    print("\n" + "=" * 70)


def demo_planning_modes():
    """Demonstrate different planning modes."""
    print("\n" + "=" * 70)
    print("Planning Modes Demo")
    print("=" * 70)

    class MockOrchestrator:
        provider = None
        model = "test-model"
        max_tokens = 4096

    coordinator = PlanningCoordinator(MockOrchestrator())
    complex_message = "Analyze architecture and design improvements"

    print(f"\nTest message: '{complex_message}'")
    print()

    modes = [
        (PlanningMode.AUTO, "Auto-detect based on complexity"),
        (PlanningMode.ALWAYS, "Always use planning"),
        (PlanningMode.NEVER, "Never use planning (direct chat only)"),
    ]

    for mode, description in modes:
        coordinator.set_planning_mode(mode)
        should_plan = coordinator._should_use_planning(complex_message)
        print(f"  {mode.value:<8} - {description:<40} -> {should_plan}")

    print("\n" + "=" * 70)


def demo_config_options():
    """Demonstrate configuration options."""
    print("\n" + "=" * 70)
    print("Planning Configuration Demo")
    print("=" * 70)

    configs = [
        ("Conservative", TaskComplexity.COMPLEX, 5, True),
        ("Default", TaskComplexity.MODERATE, 3, True),
        ("Aggressive", TaskComplexity.SIMPLE, 2, False),
    ]

    print(f"\n{'Profile':<15} {'Min Complexity':<15} {'Min Keywords':<15} {'Show Plan'}")
    print("-" * 70)

    for profile, complexity, keywords, show_plan in configs:
        config = PlanningConfig(
            min_planning_complexity=complexity,
            min_steps_threshold=keywords,
            show_plan_before_execution=show_plan,
        )
        print(
            f"{profile:<15} {complexity.value:<15} {keywords:<15} {show_plan!s:<5}"
        )

    print("\n" + "=" * 70)


def demo_workflow():
    """Demonstrate the planning workflow."""
    print("\n" + "=" * 70)
    print("Planning Workflow Demo")
    print("=" * 70)

    print("""
When planning is enabled, complex tasks follow this workflow:

1. User Message
   ↓
2. Analyze Task Complexity
   ├─ Simple → Direct Chat (fast)
   └─ Complex → Generate Plan → Continue below
   ↓
3. Generate Structured Plan (using ReadableTaskPlan)
   ├─ Step 1: [research] Analyze current state
   ├─ Step 2: [analyze] Evaluate options
   ├─ Step 3: [feature] Implement solution
   └─ Step 4: [test] Verify functionality
   ↓
4. Execute Plan Step-by-Step
   ├─ Each step uses context-aware tools
   ├─ Progressive tool disclosure
   └─ Progress tracking
   ↓
5. Generate Comprehensive Summary
   └─ Combines all step results into final response

Benefits:
- ✓ 50-80% token savings vs direct chat
- ✓ Structured, reliable execution
- ✓ Better completion of complex tasks
- ✓ Progressive tool disclosure
- ✓ Context-aware tool selection per step
    """)

    print("=" * 70)


if __name__ == "__main__":
    demo_planning_detection()
    demo_planning_modes()
    demo_config_options()
    demo_workflow()

    print("\n✅ Planning integration demo complete!")
    print("\n📝 To enable planning in your application:")
    print("   export ENABLE_PLANNING=true")
    print("   # Or set in code:")
    print("   agent.settings.enable_planning = True")
