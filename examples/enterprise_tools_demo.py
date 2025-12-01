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

"""Demo of Victor's Enterprise Tools.

This demonstrates the enterprise-grade capabilities:
- Code Review Tool for quality analysis
- Project Scaffolding Tool for project generation
- Security Scanner Tool for vulnerability detection

Usage:
    python examples/enterprise_tools_demo.py
"""

import asyncio
import tempfile
from pathlib import Path

from victor.tools.code_review_tool import CodeReviewTool
from victor.tools.scaffold_tool import ScaffoldTool
from victor.tools.security_scanner_tool import SecurityScannerTool


async def demo_code_review():
    """Demo code review tool."""
    print("\n📋 Code Review Tool Demo")
    print("=" * 70)

    tool = CodeReviewTool(max_complexity=10)

    # Create sample code with issues
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(
            """
import os
import random

# Hardcoded credentials (security issue)
API_KEY = "sk-1234567890abcdef1234567890abcdef"
PASSWORD = "admin123"

def process_data(data):
    # Print statement (code smell)
    print("Processing:", data)

    # Insecure random (security issue)
    value = random.randint(1, 100)

    # SQL injection vulnerability
    query = f"SELECT * FROM users WHERE id = {data}"

    # Bare except (code smell)
    try:
        result = execute(query)
    except:
        pass

    return result

def complex_function(a, b, c, d, e):
    # High complexity function
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:
                        return a + b + c + d + e
                    else:
                        return a + b + c + d
                else:
                    return a + b + c
            else:
                return a + b
        else:
            return a
    else:
        return 0

class MyClass:
    # Missing docstring (documentation issue)
    def method_without_docs(self):
        return "No documentation"
"""
        )
        test_file = f.name

    print("\n1️⃣ Review single file for all issues...")
    result = await tool.execute(operation="review_file", path=test_file, include_metrics=True)
    print(result.output)

    print("\n2️⃣ Security-focused scan...")
    result = await tool.execute(operation="review_file", path=test_file)
    if not result.success:
        print("⚠ Security issues detected!")

    print("\n3️⃣ Complexity analysis...")
    result = await tool.execute(operation="complexity", path=test_file)
    print(result.output)

    # Cleanup
    Path(test_file).unlink()

    print("\n✅ Code Review Tool Features:")
    print("  ✓ Security vulnerability detection")
    print("  ✓ Code quality analysis")
    print("  ✓ Complexity metrics")
    print("  ✓ Documentation coverage")
    print("  ✓ Best practices checking")


async def demo_project_scaffolding():
    """Demo project scaffolding tool."""
    print("\n\n🏗️  Project Scaffolding Tool Demo")
    print("=" * 70)

    tool = ScaffoldTool()

    # List available templates
    print("\n1️⃣ List available project templates...")
    result = await tool.execute(operation="list")
    print(result.output)

    # Create FastAPI project
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir) / "demo-api"

        print("\n2️⃣ Create FastAPI project...")
        result = await tool.execute(operation="create", template="fastapi", name=str(project_path))
        print(result.output)

        # Verify files created
        print("\n3️⃣ Verify project structure...")
        for file_path in project_path.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(project_path)
                print(f"  ✓ {rel_path}")

        print("\n4️⃣ Add custom file...")
        custom_file = project_path / "app" / "utils" / "helpers.py"
        result = await tool.execute(
            operation="add_file", path=str(custom_file), content="def helper():\n    pass\n"
        )
        print(result.output)

    print("\n✅ Project Scaffolding Tool Features:")
    print("  ✓ Multiple project templates (FastAPI, Flask, React, CLI)")
    print("  ✓ Best practices structure")
    print("  ✓ Development tooling setup")
    print("  ✓ Configuration files")
    print("  ✓ Testing setup")


async def demo_security_scanner():
    """Demo security scanner tool."""
    print("\n\n🔒 Security Scanner Tool Demo")
    print("=" * 70)

    tool = SecurityScannerTool()

    # Create files with security issues
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create file with secrets
        secrets_file = tmpdir_path / "config.py"
        secrets_file.write_text(
            """
# Configuration with exposed secrets
AWS_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"
AWS_SECRET_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
GITHUB_TOKEN = "ghp_abcd1234efgh5678ijkl90mnopqr123456"
DATABASE_URL = "postgresql://user:password123@localhost/db"

# Google API key
GOOGLE_API_KEY = "XXXXX"
"""
        )

        # Create requirements.txt with vulnerable packages
        req_file = tmpdir_path / "requirements.txt"
        req_file.write_text(
            """
requests==2.25.0
flask==2.0.0
django==3.2.0
pillow==8.0.0
pyyaml==5.4
"""
        )

        # Create .env file (dangerous file)
        env_file = tmpdir_path / ".env"
        env_file.write_text("SECRET_KEY=production_key_12345")

        print("\n1️⃣ Scan for secrets...")
        result = await tool.execute(
            operation="scan_secrets", path=str(tmpdir_path), severity="high"
        )
        print(result.output)

        print("\n2️⃣ Check dependencies for vulnerabilities...")
        result = await tool.execute(operation="scan_dependencies", path=str(tmpdir_path))
        print(result.output)

        print("\n3️⃣ Scan configuration files...")
        result = await tool.execute(operation="scan_config", path=str(tmpdir_path))
        print(result.output)

        print("\n4️⃣ Comprehensive security scan...")
        result = await tool.execute(operation="scan_all", path=str(tmpdir_path), severity="medium")
        if not result.success:
            print("⚠ Security issues detected!")
            print("\nRecommendations:")
            print("  1. Remove all hardcoded secrets")
            print("  2. Use environment variables")
            print("  3. Update vulnerable dependencies")
            print("  4. Add .env to .gitignore")

    print("\n✅ Security Scanner Tool Features:")
    print("  ✓ Secret detection (API keys, tokens, passwords)")
    print("  ✓ Multiple secret pattern detection")
    print("  ✓ Dependency vulnerability scanning")
    print("  ✓ Configuration security analysis")
    print("  ✓ Severity-based filtering")


async def main():
    """Run all enterprise tool demos."""
    print("🎯 Victor Enterprise Tools Demo")
    print("=" * 70)
    print("\nDemonstrating enterprise-grade capabilities:\n")

    # Code Review
    await demo_code_review()

    # Project Scaffolding
    await demo_project_scaffolding()

    # Security Scanner
    await demo_security_scanner()

    print("\n\n✨ Demo Complete!")
    print("\nVictor's Enterprise Tools provide:")
    print("  • Automated code quality analysis")
    print("  • Project scaffolding and templates")
    print("  • Comprehensive security scanning")
    print("  • Production-ready best practices")
    print("\nAll tools are agent-integrated and ready for enterprise use!")


if __name__ == "__main__":
    asyncio.run(main())
