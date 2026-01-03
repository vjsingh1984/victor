#!/usr/bin/env python3
"""Demo: Enterprise Workflow Automation

This demonstrates a complete enterprise workflow using Victor's 31 tools:
1. Code review with security scanning
2. Test generation
3. Documentation generation
4. Git commit automation
5. CI/CD pipeline validation

Features:
- Multi-tool orchestration
- Air-gapped capable (works offline)
- Production-ready enterprise tools
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from victor.tools.code_review_tool import code_review
from victor.tools.security_scanner_tool import scan
from victor.tools.testing_tool import test
from victor.tools.documentation_tool import docs
from victor.tools.cicd_tool import cicd


async def main():
    print("=" * 70)
    print("🏢 Victor Enterprise Workflow Demo")
    print("=" * 70)
    print()

    # Sample Python code to process
    sample_code = '''
def process_user_input(user_input: str, db_conn):
    """Process user input and save to database."""
    # Security issue: SQL injection vulnerability
    query = f"INSERT INTO users (name) VALUES ('{user_input}')"
    db_conn.execute(query)
    return {"status": "success"}

def calculate_total(items: list) -> float:
    """Calculate total price of items."""
    total = 0
    for item in items:
        total += item.get("price", 0) * item.get("quantity", 1)
    return total
'''

    # Create temporary file for demo
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        temp_file = Path(f.name)
        f.write(sample_code)

    print(f"📝 Sample Code File: {temp_file}")
    print()

    # Step 1: Security Scan
    print("=" * 70)
    print("🔒 Step 1: Security Scanning")
    print("=" * 70)
    print()

    print("Running security scan...")
    security_result = await scan(path=str(temp_file), scan_types=["secrets", "config"])

    if security_result.get("success"):
        print("✅ Security scan complete!")
        print(f"Findings: {security_result.get('total_issues', 0)}")
    else:
        print(f"❌ Security scan failed: {security_result.get('error')}")
    print()

    # Step 2: Code Review
    print("=" * 70)
    print("📊 Step 2: Code Quality Review")
    print("=" * 70)
    print()

    print("Running code review...")
    review_result = await code_review(path=str(temp_file), aspects=["all"], include_metrics=True)

    if review_result.get("success"):
        print("✅ Code review complete!")
        print(f"Issues found: {review_result.get('total_issues', 0)}")
        metrics = review_result.get("results", {}).get("complexity", {})
        if metrics:
            print(f"Complexity: {metrics.get('summary', {}).get('avg_complexity', 'N/A')}")
    else:
        print(f"❌ Code review failed: {review_result.get('error')}")
    print()

    # Step 3: Run Unit Tests
    print("=" * 70)
    print("🧪 Step 3: Run Unit Tests")
    print("=" * 70)
    print()

    test_module = temp_file.stem
    test_file = temp_file.with_name(f"test_{test_module}.py")
    test_file.write_text(
        f"""import {test_module}

def test_calculate_total():
    items = [{{"price": 10, "quantity": 2}}, {{"price": 5, "quantity": 1}}]
    assert {test_module}.calculate_total(items) == 25
"""
    )

    print("Running pytest...")
    test_result = await test(path=str(temp_file.parent))

    if test_result.get("summary"):
        print("✅ Test run complete!")
        summary = test_result["summary"]
        print(f"  Total: {summary.get('total_tests', 0)}")
        print(f"  Passed: {summary.get('passed', 0)}")
        print(f"  Failed: {summary.get('failed', 0)}")
    else:
        print(f"❌ Test run failed: {test_result.get('error')}")
    print()

    # Step 4: Documentation Generation
    print("=" * 70)
    print("📚 Step 4: Generate Documentation")
    print("=" * 70)
    print()

    print("Generating documentation...")
    doc_result = await docs(path=str(temp_file), doc_types=["api"], format="markdown")

    if doc_result.get("success"):
        print("✅ Documentation generated!")
        print(doc_result.get("formatted_report", ""))
    else:
        print(f"❌ Documentation generation failed: {doc_result.get('error')}")
    print()

    # Step 5: CI/CD Pipeline Validation
    print("=" * 70)
    print("🚀 Step 5: CI/CD Pipeline Validation")
    print("=" * 70)
    print()

    print("Validating GitHub Actions workflow...")

    # Sample workflow
    sample_workflow = """
name: CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: pytest
"""

    workflow_file = temp_file.with_suffix(".yml")
    workflow_file.write_text(sample_workflow.strip() + "\n")

    cicd_result = await cicd(operation="validate", platform="github", file=str(workflow_file))

    if cicd_result.get("success"):
        print("✅ CI/CD workflow validated!")
        print(cicd_result.get("formatted_report", ""))
    else:
        print(f"❌ Workflow validation failed: {cicd_result.get('error')}")
    print()

    # Summary
    print("=" * 70)
    print("📊 Workflow Summary")
    print("=" * 70)
    print()

    steps_completed = sum(
        [
            bool(security_result.get("success")),
            bool(review_result.get("success")),
            bool(test_result.get("summary")),
            bool(doc_result.get("success")),
            bool(cicd_result.get("success")),
        ]
    )

    print(f"Steps completed: {steps_completed}/5")
    print()
    print("Workflow stages:")
    print(f"  {'✅' if security_result.get('success') else '❌'} Security scanning")
    print(f"  {'✅' if review_result.get('success') else '❌'} Code review")
    print(f"  {'✅' if test_result.get('summary') else '❌'} Tests")
    print(f"  {'✅' if doc_result.get('success') else '❌'} Documentation")
    print(f"  {'✅' if cicd_result.get('success') else '❌'} CI/CD validation")
    print()

    print("=" * 70)
    print("✅ Enterprise Workflow Demo Complete!")
    print("=" * 70)
    print()
    print("Victor's enterprise tools enable:")
    print("  ✅ Automated code quality assurance")
    print("  ✅ Security vulnerability detection")
    print("  ✅ Test coverage automation")
    print("  ✅ Documentation generation")
    print("  ✅ CI/CD pipeline management")
    print("  ✅ 100% offline capability (air-gapped)")
    print()

    # Cleanup
    temp_file.unlink()


if __name__ == "__main__":
    asyncio.run(main())
