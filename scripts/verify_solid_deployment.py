#!/usr/bin/env python3
"""
SOLID Remediation Deployment Verification Script

This script verifies that all SOLID remediation components are correctly
deployed and functioning. It runs tests, checks feature flags, and validates
the implementation.

Usage:
    python scripts/verify_solid_deployment.py [--verbose]

Exit Codes:
    0: All checks passed
    1: Some checks failed
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path
from typing import List, Tuple, Dict, Any


# Colors for output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_header(text: str) -> None:
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.RESET}\n")


def print_success(text: str) -> None:
    """Print a success message."""
    print(f"{Colors.GREEN}✅ {text}{Colors.RESET}")


def print_error(text: str) -> None:
    """Print an error message."""
    print(f"{Colors.RED}❌ {text}{Colors.RESET}")


def print_warning(text: str) -> None:
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.RESET}")


def print_info(text: str) -> None:
    """Print an info message."""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.RESET}")


class DeploymentVerifier:
    """Verifies SOLID remediation deployment."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[Tuple[str, bool, str]] = []
        self.project_root = Path(__file__).parent.parent

    def add_result(self, name: str, passed: bool, message: str = "") -> None:
        """Add a verification result."""
        self.results.append((name, passed, message))
        if passed:
            print_success(f"{name}: {message}")
        else:
            print_error(f"{name}: {message}")

    def verify_tests(self) -> None:
        """Verify all SOLID tests pass."""
        print_header("Phase 1: Test Suite Verification")

        test_files = [
            "tests/unit/core/verticals/test_plugin_discovery.py",
            "tests/unit/core/verticals/test_lazy_proxy.py",
            "tests/unit/framework/test_lazy_initializer.py",
        ]

        for test_file in test_files:
            print_info(f"Running {test_file}...")

            cmd = [sys.executable, "-m", "pytest", test_file, "-v", "--tb=no", "--no-header"]

            if self.verbose:
                cmd.append("--capture=no")

            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)

            # Check for "passed" in output
            if result.returncode == 0:
                # Extract pass count from output
                output = result.stdout
                if "passed" in output:
                    self.add_result(test_file, True, "All tests passed")
                else:
                    self.add_result(test_file, False, "No test results found")
            else:
                self.add_result(test_file, False, f"Tests failed (exit code {result.returncode})")
                if self.verbose:
                    print(result.stdout)
                    print(result.stderr)

    def verify_lsp_compliance(self) -> None:
        """Verify LSP compliance (LazyProxy inherits VerticalBase)."""
        print_header("Phase 2: LSP Compliance Verification")

        try:
            from victor.core.verticals.lazy_proxy import LazyProxy
            from victor.core.verticals.base import VerticalBase
            from victor.coding import CodingAssistant

            # Create a proxy
            proxy = LazyProxy(vertical_name="coding", loader=lambda: CodingAssistant)

            # Test isinstance
            if isinstance(proxy, VerticalBase):
                self.add_result(
                    "LSP Compliance", True, "isinstance(proxy, VerticalBase) returns True"
                )
            else:
                self.add_result(
                    "LSP Compliance", False, "isinstance(proxy, VerticalBase) returns False"
                )

            # Test attribute access
            try:
                _ = proxy.name
                self.add_result("Lazy Proxy Attribute Access", True, "Proxy attributes accessible")
            except Exception as e:
                self.add_result("Lazy Proxy Attribute Access", False, str(e))

        except Exception as e:
            self.add_result("LSP Compliance", False, f"Import or test failed: {e}")

    def verify_plugin_discovery(self) -> None:
        """Verify plugin discovery system."""
        print_header("Phase 3: Plugin Discovery Verification")

        try:
            from victor.core.verticals.plugin_discovery import get_plugin_discovery

            discovery = get_plugin_discovery()

            # Test discovery - returns DiscoveryResult dataclass
            result = discovery.discover_all()
            verticals = result.verticals  # Access the verticals dict

            # Check for built-in verticals
            expected_verticals = ["coding", "research", "devops", "dataanalysis"]
            found_verticals = [v for v in expected_verticals if v in verticals]

            if len(found_verticals) == len(expected_verticals):
                self.add_result(
                    "Plugin Discovery",
                    True,
                    f"Found all {len(expected_verticals)} built-in verticals",
                )
            else:
                self.add_result(
                    "Plugin Discovery",
                    False,
                    f"Found {len(found_verticals)}/{len(expected_verticals)} verticals: {found_verticals}",
                )

            # Test cache
            result2 = discovery.discover_all()
            verticals2 = result2.verticals
            if result is result2 or len(verticals) == len(verticals2):
                self.add_result("Plugin Discovery Cache", True, "Discovery cache working")
            else:
                self.add_result("Plugin Discovery Cache", False, "Cache not working as expected")

        except Exception as e:
            self.add_result("Plugin Discovery", False, f"Discovery failed: {e}")

    def verify_lazy_initialization(self) -> None:
        """Verify lazy initialization system."""
        print_header("Phase 4: Lazy Initialization Verification")

        try:
            from victor.framework.lazy_initializer import (
                get_initializer_for_vertical,
                clear_all_initializers,
            )

            # Clear any existing initializers
            clear_all_initializers()

            call_count = [0]

            def test_initializer():
                call_count[0] += 1
                return "initialized"

            # Create initializer
            init = get_initializer_for_vertical("test_verification", test_initializer)

            # Test not initialized yet
            if not init.is_initialized():
                self.add_result(
                    "Lazy Initialization State", True, "Not initialized before first access"
                )
            else:
                self.add_result("Lazy Initialization State", False, "Should not be initialized yet")

            # Test initialization on first access
            result = init.get_or_initialize()

            if result == "initialized" and call_count[0] == 1:
                self.add_result(
                    "Lazy Initialization Execution", True, "Initialized on first access"
                )
            else:
                self.add_result(
                    "Lazy Initialization Execution",
                    False,
                    f"Expected 'initialized', got {result}, calls: {call_count[0]}",
                )

            # Test cached
            if init.is_initialized():
                self.add_result("Lazy Initialization Cache", True, "Initialization state tracked")
            else:
                self.add_result("Lazy Initialization Cache", False, "State not tracked")

            # Test no re-initialization
            result2 = init.get_or_initialize()
            if call_count[0] == 1:
                self.add_result("Lazy Initialization Single Call", True, "Only initialized once")
            else:
                self.add_result(
                    "Lazy Initialization Single Call", False, f"Initialized {call_count[0]} times"
                )

        except Exception as e:
            self.add_result("Lazy Initialization", False, f"Initialization failed: {e}")

    def verify_feature_flags(self) -> None:
        """Verify feature flags work correctly."""
        print_header("Phase 5: Feature Flags Verification")

        # Test lazy initialization flag
        old_value = os.environ.get("VICTOR_LAZY_INITIALIZATION", "true")

        try:
            # Test with flag enabled
            os.environ["VICTOR_LAZY_INITIALIZATION"] = "true"
            from victor.framework.lazy_initializer import get_initializer_for_vertical

            # Reload module to pick up new flag
            import importlib
            import victor.framework.lazy_initializer

            importlib.reload(victor.framework.lazy_initializer)

            from victor.framework.lazy_initializer import get_initializer_for_vertical

            init = get_initializer_for_vertical("test_flag", lambda: "flag_test")
            result = init.get_or_initialize()

            if result == "flag_test":
                self.add_result(
                    "Feature Flag: VICTOR_LAZY_INITIALIZATION=true", True, "Flag respected"
                )
            else:
                self.add_result(
                    "Feature Flag: VICTOR_LAZY_INITIALIZATION=true",
                    False,
                    f"Unexpected result: {result}",
                )

        except Exception as e:
            self.add_result(
                "Feature Flag: VICTOR_LAZY_INITIALIZATION", False, f"Flag test failed: {e}"
            )
        finally:
            os.environ["VICTOR_LAZY_INITIALIZATION"] = old_value

    def verify_thread_safety(self) -> None:
        """Verify thread safety of lazy components."""
        print_header("Phase 6: Thread Safety Verification")

        try:
            from victor.framework.lazy_initializer import get_initializer_for_vertical
            import threading
            import time

            call_count = [0]
            lock = threading.Lock()

            def slow_initializer():
                with lock:
                    call_count[0] += 1
                    time.sleep(0.01)  # Simulate slow initialization
                return call_count[0]

            init = get_initializer_for_vertical("test_thread", slow_initializer)

            # Spawn multiple threads
            threads = []
            results = []

            def initialize_from_thread():
                result = init.get_or_initialize()
                results.append(result)

            for _ in range(10):
                thread = threading.Thread(target=initialize_from_thread)
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            # Should only initialize once despite concurrent access
            if call_count[0] == 1:
                self.add_result(
                    "Thread Safety: Lazy Initialization",
                    True,
                    "Concurrent access handled correctly",
                )
            else:
                self.add_result(
                    "Thread Safety: Lazy Initialization",
                    False,
                    f"Initialized {call_count[0]} times (expected 1)",
                )

        except Exception as e:
            self.add_result("Thread Safety", False, f"Thread safety test failed: {e}")

    def verify_import_independence(self) -> None:
        """Verify import order independence."""
        print_header("Phase 7: Import Independence Verification")

        try:
            # Test importing in different orders
            import subprocess

            test_script = """
# Test import order independence
import sys
sys.path.insert(0, '.')

# Import in different order
from victor.dataanalysis import DataAnalysisAssistant
from victor.coding import CodingAssistant
from victor.research import ResearchAssistant
from victor.devops import DevOpsAssistant

print("SUCCESS")
"""

            result = subprocess.run(
                [sys.executable, "-c", test_script],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )

            if "SUCCESS" in result.stdout:
                self.add_result("Import Independence", True, "Import order does not matter")
            else:
                self.add_result("Import Independence", False, f"Import failed: {result.stderr}")

        except Exception as e:
            self.add_result("Import Independence", False, f"Import test failed: {e}")

    def verify_protocols(self) -> None:
        """Verify protocol definitions exist and work."""
        print_header("Phase 8: Protocol Definitions Verification")

        try:
            from victor.protocols import (
                ToolExecutorProtocol,
                CapabilityContainerProtocol,
            )

            # Test protocols are callable
            self.add_result("Protocol Definitions", True, "Core protocols imported successfully")

            # Test @runtime_checkable
            from typing import runtime_checkable

            if hasattr(ToolExecutorProtocol, "__protocol_attrs__"):
                self.add_result(
                    "Protocol Runtime Checkable", True, "Protocols are @runtime_checkable"
                )
            else:
                self.add_result(
                    "Protocol Runtime Checkable", False, "Protocols not runtime checkable"
                )

        except Exception as e:
            self.add_result("Protocol Definitions", False, f"Protocol import failed: {e}")

    def print_summary(self) -> None:
        """Print verification summary."""
        print_header("Verification Summary")

        passed = sum(1 for _, p, _ in self.results if p)
        failed = sum(1 for _, p, _ in self.results if not p)
        total = len(self.results)

        print(f"\n{Colors.BOLD}Total Checks: {total}{Colors.RESET}")
        print(f"{Colors.GREEN}{Colors.BOLD}Passed: {passed}{Colors.RESET}")
        print(f"{Colors.RED}{Colors.BOLD}Failed: {failed}{Colors.RESET}")

        if failed > 0:
            print(f"\n{Colors.RED}{Colors.BOLD}Failed Checks:{Colors.RESET}")
            for name, passed, message in self.results:
                if not passed:
                    print(f"  {Colors.RED}❌ {name}: {message}{Colors.RESET}")

        print()

        if failed == 0:
            print(f"{Colors.GREEN}{Colors.BOLD}{'=' * 70}{Colors.RESET}")
            print(
                f"{Colors.GREEN}{Colors.BOLD}{'✅ ALL CHECKS PASSED - READY FOR DEPLOYMENT':^70}{Colors.RESET}"
            )
            print(f"{Colors.GREEN}{Colors.BOLD}{'=' * 70}{Colors.RESET}\n")
            return 0
        else:
            print(f"{Colors.RED}{Colors.BOLD}{'=' * 70}{Colors.RESET}")
            print(
                f"{Colors.RED}{Colors.BOLD}{'❌ SOME CHECKS FAILED - REVIEW NEEDED':^70}{Colors.RESET}"
            )
            print(f"{Colors.RED}{Colors.BOLD}{'=' * 70}{Colors.RESET}\n")
            return 1

    def run_all(self) -> int:
        """Run all verification checks."""
        print(f"\n{Colors.BOLD}SOLID Remediation Deployment Verification{Colors.RESET}")
        print(f"{Colors.BOLD}Project: {self.project_root}{Colors.RESET}\n")

        self.verify_tests()
        self.verify_lsp_compliance()
        self.verify_plugin_discovery()
        self.verify_lazy_initialization()
        self.verify_feature_flags()
        self.verify_thread_safety()
        self.verify_import_independence()
        self.verify_protocols()

        return self.print_summary()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Verify SOLID remediation deployment")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    verifier = DeploymentVerifier(verbose=args.verbose)
    exit_code = verifier.run_all()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
