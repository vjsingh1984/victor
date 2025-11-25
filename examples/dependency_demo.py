"""Demo of Victor's Dependency Management Tool.

Demonstrates dependency management:
- List installed packages
- Check for outdated dependencies
- Security vulnerability scanning
- Generate requirements files
- Dependency analysis

Usage:
    python examples/dependency_demo.py
"""

import asyncio
from victor.tools.dependency_tool import DependencyTool


async def demo_list_packages():
    """Demo listing installed packages."""
    print("\n\n📦 List Installed Packages Demo")
    print("=" * 70)

    tool = DependencyTool()

    print("\n1️⃣ List all installed packages...")
    result = await tool.execute(
        operation="list",
    )

    if result.success:
        print(result.output)


async def demo_check_outdated():
    """Demo checking for outdated packages."""
    print("\n\n🔄 Check Outdated Packages Demo")
    print("=" * 70)

    tool = DependencyTool()

    print("\n1️⃣ Check for outdated packages...")
    result = await tool.execute(
        operation="outdated",
    )

    if result.success:
        print(result.output)
    else:
        print(f"Note: {result.error}")


async def demo_security_audit():
    """Demo security vulnerability check."""
    print("\n\n🔒 Security Audit Demo")
    print("=" * 70)

    tool = DependencyTool()

    print("\n1️⃣ Run security audit...")
    result = await tool.execute(
        operation="security",
    )

    if result.success:
        print(result.output)


async def demo_generate_requirements():
    """Demo generating requirements file."""
    print("\n\n📝 Generate Requirements Demo")
    print("=" * 70)

    tool = DependencyTool()

    print("\n1️⃣ Generate requirements.txt (freeze format)...")
    result = await tool.execute(
        operation="generate",
        output="requirements.txt",
        format="freeze",
    )

    if result.success:
        print(result.output)


async def demo_check_requirements():
    """Demo checking requirements file."""
    print("\n\n✅ Check Requirements Demo")
    print("=" * 70)

    tool = DependencyTool()

    print("\n1️⃣ Check if requirements match installed...")
    result = await tool.execute(
        operation="check",
    )

    if result.success:
        print(result.output)
    else:
        print(f"Note: {result.error}")


async def demo_best_practices():
    """Demo dependency management best practices."""
    print("\n\n⭐ Dependency Management Best Practices")
    print("=" * 70)

    print("\n✨ Best Practices:")
    print("")

    print("1️⃣ Version Pinning:")
    print("  • Use exact versions in requirements.txt (==)")
    print("  • Pin all dependencies, not just direct ones")
    print("  • Use requirements-dev.txt for development dependencies")
    print("")

    print("2️⃣ Security:")
    print("  • Regularly audit for vulnerabilities")
    print("  • Use tools like pip-audit, safety, or Dependabot")
    print("  • Update security patches immediately")
    print("  • Review changelogs before updating")
    print("")

    print("3️⃣ Updates:")
    print("  • Patch updates (1.2.3 → 1.2.4): Apply quickly")
    print("  • Minor updates (1.2.0 → 1.3.0): Test thoroughly")
    print("  • Major updates (1.0.0 → 2.0.0): Plan carefully")
    print("  • Use virtual environments for testing")
    print("")

    print("4️⃣ Organization:")
    print("  • requirements.txt: Production dependencies")
    print("  • requirements-dev.txt: Development dependencies")
    print("  • requirements-test.txt: Testing dependencies")
    print("  • Use pip-tools for compilation")
    print("")

    print("5️⃣ CI/CD:")
    print("  • Automate dependency checks in CI")
    print("  • Fail builds on known vulnerabilities")
    print("  • Use Dependabot for automatic PRs")
    print("  • Test with multiple Python versions")
    print("")


async def main():
    """Run all dependency demos."""
    print("🎯 Victor Dependency Management Tool Demo")
    print("=" * 70)
    print("\nDemonstrating dependency management\n")

    # Run demos
    await demo_list_packages()
    await demo_check_outdated()
    await demo_security_audit()
    await demo_generate_requirements()
    await demo_check_requirements()
    await demo_best_practices()

    print("\n\n✨ Demo Complete!")
    print("\nVictor's Dependency Tool provides:")
    print("  • Package listing and inspection")
    print("  • Outdated package detection")
    print("  • Security vulnerability scanning")
    print("  • Requirements file generation")
    print("  • Dependency verification")
    print("")
    print("Perfect for:")
    print("  • Maintaining up-to-date dependencies")
    print("  • Security compliance")
    print("  • Dependency auditing")
    print("  • Requirements management")
    print("  • CI/CD integration")
    print("")
    print("Ready to manage your dependencies!")


if __name__ == "__main__":
    asyncio.run(main())
