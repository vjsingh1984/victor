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

"""Demo of Victor's CI/CD Integration Tool.

Demonstrates CI/CD pipeline management:
- Generate GitHub Actions workflows
- Validate pipeline configurations
- Create common workflows (test, build, deploy)
- List available templates

Usage:
    python examples/cicd_demo.py
"""

import asyncio
import tempfile
from pathlib import Path

from victor.tools.cicd_tool import cicd


async def demo_list_templates():
    """Demo listing available templates."""
    print("\n\n📋 List Templates Demo")
    print("=" * 70)

    print("\n1️⃣ List all available CI/CD templates...")
    result = await cicd(operation="list")

    if result["success"]:
        print(result.get("formatted_report", ""))
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")


async def demo_generate_test_workflow():
    """Demo generating a test workflow."""
    print("\n\n🧪 Generate Test Workflow Demo")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        print("\n1️⃣ Generate GitHub Actions test workflow...")
        result = await cicd(
            operation="generate",
            platform="github",
            workflow="python-test",
            output=str(temp_path / ".github/workflows/test.yml"),
        )

        if result["success"]:
            print(result.get("formatted_report", ""))
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")


async def demo_generate_publish_workflow():
    """Demo generating a publish workflow."""
    print("\n\n📦 Generate Publish Workflow Demo")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        print("\n1️⃣ Generate GitHub Actions publish workflow...")
        result = await cicd(
            operation="generate",
            platform="github",
            workflow="python-publish",
            output=str(temp_path / ".github/workflows/publish.yml"),
        )

        if result["success"]:
            print(result.get("formatted_report", ""))
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")


async def demo_generate_docker_workflow():
    """Demo generating a Docker workflow."""
    print("\n\n🐳 Generate Docker Workflow Demo")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        print("\n1️⃣ Generate GitHub Actions Docker workflow...")
        result = await cicd(
            operation="generate",
            platform="github",
            workflow="docker-build",
            output=str(temp_path / ".github/workflows/docker.yml"),
        )

        if result["success"]:
            print(result.get("formatted_report", ""))
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")


async def demo_generate_by_type():
    """Demo generating workflows by type shorthand."""
    print("\n\n🔨 Generate Workflow by Type Demo")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        print("\n1️⃣ Create test workflow using type='test'...")
        result = await cicd(
            operation="generate",
            type="test",
            platform="github",
            output=str(temp_path / ".github/workflows/test.yml"),
        )

        if result["success"]:
            print("✓ Test workflow created")
            config = result.get("config", "")
            print(config[:500] + "..." if len(config) > 500 else config)
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")

        print("\n\n2️⃣ Create release workflow using type='release'...")
        result = await cicd(
            operation="generate",
            type="release",
            platform="github",
            output=str(temp_path / ".github/workflows/release.yml"),
        )

        if result["success"]:
            print("✓ Release workflow created")
            config = result.get("config", "")
            print(config[:500] + "..." if len(config) > 500 else config)
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")

        print("\n\n3️⃣ Create build workflow using type='build'...")
        result = await cicd(
            operation="generate",
            type="build",
            platform="github",
            output=str(temp_path / ".github/workflows/build.yml"),
        )

        if result["success"]:
            print("✓ Build workflow created")
            config = result.get("config", "")
            print(config[:500] + "..." if len(config) > 500 else config)
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")


async def demo_validate_config():
    """Demo validating configuration."""
    print("\n\n✅ Validate Configuration Demo")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a valid config
        print("\n1️⃣ Generate a valid configuration first...")
        workflow_path = temp_path / ".github/workflows/test.yml"
        await cicd(
            operation="generate",
            platform="github",
            workflow="python-test",
            output=str(workflow_path),
        )

        print("\n2️⃣ Validate the generated configuration...")
        result = await cicd(
            operation="validate",
            file=str(workflow_path),
        )

        if result["success"]:
            print(result.get("formatted_report", ""))
        else:
            print(f"Validation result: {result.get('formatted_report', result.get('error', ''))}")

        # Create an invalid config
        print("\n\n3️⃣ Test validation with invalid configuration...")
        invalid_config = """
name: Invalid Workflow
# Missing 'on' field
jobs:
  test:
    # Missing 'runs-on' field
    steps:
      - run: echo "test"
"""
        invalid_path = temp_path / ".github/workflows/invalid.yml"
        invalid_path.parent.mkdir(parents=True, exist_ok=True)
        invalid_path.write_text(invalid_config.strip())

        result = await cicd(
            operation="validate",
            file=str(invalid_path),
        )

        print(result.get("formatted_report", ""))


async def demo_real_world_setup():
    """Demo a real-world CI/CD setup."""
    print("\n\n🎯 Real-World CI/CD Setup Demo")
    print("=" * 70)
    print("\nScenario: Setting up CI/CD for a Python project")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        print("\n1️⃣ STEP 1: List available templates...")
        result = await cicd(operation="list")
        if result["success"]:
            print("✓ Templates available")

        print("\n2️⃣ STEP 2: Create test workflow...")
        test_path = temp_path / ".github/workflows/test.yml"
        result = await cicd(
            operation="generate",
            type="test",
            output=str(test_path),
        )
        if result["success"]:
            print("✓ Test workflow created")
        else:
            print(f"⚠ {result.get('error', 'Error creating workflow')}")

        print("\n3️⃣ STEP 3: Create release workflow...")
        release_path = temp_path / ".github/workflows/release.yml"
        result = await cicd(
            operation="generate",
            type="release",
            output=str(release_path),
        )
        if result["success"]:
            print("✓ Release workflow created")
        else:
            print(f"⚠ {result.get('error', 'Error creating workflow')}")

        print("\n4️⃣ STEP 4: Validate all configurations...")
        for workflow_file in [test_path, release_path]:
            if workflow_file.exists():
                result = await cicd(
                    operation="validate",
                    file=str(workflow_file),
                )
                if result["success"] and not result.get("issues"):
                    print(f"✓ {workflow_file.name} is valid")
                else:
                    issues = result.get("issues", [])
                    print(f"⚠ {workflow_file.name}: {len(issues)} issue(s)")

        print("\n\n📊 Project CI/CD Setup Complete!")
        print("\nCreated workflows:")
        print("  • .github/workflows/test.yml")
        print("    - Runs on push and PR")
        print("    - Tests on Python 3.10, 3.11, 3.12")
        print("    - Linting, formatting, type checking")
        print("    - Code coverage reporting")
        print("")
        print("  • .github/workflows/release.yml")
        print("    - Runs on release creation")
        print("    - Builds Python package")
        print("    - Publishes to PyPI")
        print("")

        print("\nNext steps:")
        print("  1. Review generated workflows")
        print("  2. Add PYPI_TOKEN secret to GitHub")
        print("  3. Commit workflows to repository")
        print("  4. Push to trigger first run")
        print("")

        print("\nGenerated test workflow preview:")
        print("-" * 70)
        if test_path.exists():
            content = test_path.read_text()
            print(content[:600] + "...")


async def demo_workflow_features():
    """Demo workflow features and best practices."""
    print("\n\n⭐ Workflow Features Demo")
    print("=" * 70)

    print("\n✨ Features of generated workflows:")
    print("")

    print("1️⃣ Test Workflow Features:")
    print("  • Matrix testing (Python 3.10, 3.11, 3.12)")
    print("  • Linting with Ruff")
    print("  • Format checking with Black")
    print("  • Type checking with mypy")
    print("  • Test coverage with pytest")
    print("  • Coverage upload to Codecov")
    print("  • Runs on push and PR")
    print("")

    print("2️⃣ Publish Workflow Features:")
    print("  • Triggered on release creation")
    print("  • Builds Python package")
    print("  • Publishes to PyPI with token auth")
    print("  • Uses latest Python version")
    print("")

    print("3️⃣ Docker Workflow Features:")
    print("  • Docker Buildx for multi-platform")
    print("  • DockerHub authentication")
    print("  • Automatic tagging")
    print("  • Push on main branch")
    print("")

    print("4️⃣ Best Practices Enforced:")
    print("  • Checkout action for code access")
    print("  • Pinned action versions (v4, v5, etc.)")
    print("  • Secrets for sensitive data")
    print("  • Clear step names")
    print("  • Proper job dependencies")
    print("")

    print("5️⃣ Customization:")
    print("  • Easy to modify triggers")
    print("  • Add/remove Python versions")
    print("  • Configure coverage thresholds")
    print("  • Add deployment steps")
    print("")


async def main():
    """Run all CI/CD demos."""
    print("🎯 Victor CI/CD Integration Tool Demo")
    print("=" * 70)
    print("\nDemonstrating CI/CD pipeline management\n")

    # Run demos
    await demo_list_templates()
    await demo_generate_test_workflow()
    await demo_generate_publish_workflow()
    await demo_generate_docker_workflow()
    await demo_generate_by_type()
    await demo_validate_config()
    await demo_real_world_setup()
    await demo_workflow_features()

    print("\n\n✨ Demo Complete!")
    print("\nVictor's CI/CD Integration Tool provides:")
    print("  • Pre-configured workflow templates")
    print("  • GitHub Actions support")
    print("  • Configuration validation")
    print("  • Best practices enforcement")
    print("  • Easy customization")
    print("")
    print("Supported workflows:")
    print("  • Python testing (multi-version matrix)")
    print("  • Package publishing (PyPI)")
    print("  • Docker build and push")
    print("")
    print("Perfect for:")
    print("  • New project setup")
    print("  • Standardizing CI/CD across projects")
    print("  • Migrating to GitHub Actions")
    print("  • Ensuring best practices")
    print("  • Quick CI/CD configuration")
    print("")
    print("Ready to automate your deployments!")


if __name__ == "__main__":
    asyncio.run(main())
