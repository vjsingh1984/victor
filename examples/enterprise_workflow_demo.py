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

from victor.tools.code_review_tool import CodeReviewTool
from victor.tools.security_scanner_tool import SecurityScannerTool
from victor.tools.testing_tool import TestingTool
from victor.tools.documentation_tool import DocumentationTool
from victor.tools.git_tool import GitTool
from victor.tools.cicd_tool import CICDTool


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

    security_tool = SecurityScannerTool()
    print("Running security scan...")
    security_result = await security_tool.execute(path=str(temp_file), scan_type="secrets")

    if security_result.success:
        print("✅ Security scan complete!")
        print(f"Findings: {security_result.metadata.get('findings_count', 0)}")
    else:
        print(f"❌ Security scan failed: {security_result.error}")
    print()

    # Step 2: Code Review
    print("=" * 70)
    print("📊 Step 2: Code Quality Review")
    print("=" * 70)
    print()

    review_tool = CodeReviewTool()
    print("Running code review...")
    review_result = await review_tool.execute(file_path=str(temp_file), review_type="quality")

    if review_result.success:
        print("✅ Code review complete!")
        print(f"Issues found: {review_result.metadata.get('issues_count', 0)}")
        print(f"Complexity: {review_result.metadata.get('complexity', 'N/A')}")
    else:
        print(f"❌ Code review failed: {review_result.error}")
    print()

    # Step 3: Test Generation
    print("=" * 70)
    print("🧪 Step 3: Generate Unit Tests")
    print("=" * 70)
    print()

    testing_tool = TestingTool()
    print("Generating unit tests...")
    test_result = await testing_tool.execute(action="generate", file_path=str(temp_file))

    if test_result.success:
        print("✅ Test generation complete!")
        print(f"Tests created: {test_result.metadata.get('tests_generated', 0)}")
    else:
        print(f"❌ Test generation failed: {test_result.error}")
    print()

    # Step 4: Documentation Generation
    print("=" * 70)
    print("📚 Step 4: Generate Documentation")
    print("=" * 70)
    print()

    doc_tool = DocumentationTool()
    print("Generating documentation...")
    doc_result = await doc_tool.execute(
        action="generate", source_path=str(temp_file), doc_type="api"
    )

    if doc_result.success:
        print("✅ Documentation generated!")
        print(f"Format: {doc_result.metadata.get('format', 'markdown')}")
    else:
        print(f"❌ Documentation generation failed: {doc_result.error}")
    print()

    # Step 5: CI/CD Pipeline Validation
    print("=" * 70)
    print("🚀 Step 5: CI/CD Pipeline Validation")
    print("=" * 70)
    print()

    cicd_tool = CICDTool()
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

    cicd_result = await cicd_tool.execute(
        action="validate", platform="github", workflow_content=sample_workflow
    )

    if cicd_result.success:
        print("✅ CI/CD workflow validated!")
        print(f"Platform: {cicd_result.metadata.get('platform', 'N/A')}")
    else:
        print(f"❌ Workflow validation failed: {cicd_result.error}")
    print()

    # Summary
    print("=" * 70)
    print("📊 Workflow Summary")
    print("=" * 70)
    print()

    steps_completed = sum(
        [
            security_result.success,
            review_result.success,
            test_result.success,
            doc_result.success,
            cicd_result.success,
        ]
    )

    print(f"Steps completed: {steps_completed}/5")
    print()
    print("Workflow stages:")
    print(f"  {'✅' if security_result.success else '❌'} Security scanning")
    print(f"  {'✅' if review_result.success else '❌'} Code review")
    print(f"  {'✅' if test_result.success else '❌'} Test generation")
    print(f"  {'✅' if doc_result.success else '❌'} Documentation")
    print(f"  {'✅' if cicd_result.success else '❌'} CI/CD validation")
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
