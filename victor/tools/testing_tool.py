import subprocess
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from victor.tools.decorators import tool


@tool
async def run_tests(path: Optional[str] = None, pytest_args: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Runs tests using pytest and returns a structured summary of the results.

    This tool runs pytest on the specified path and captures the output in JSON format,
    providing a clean summary of test outcomes.

    Args:
        path: The specific file or directory to run tests on. If not provided,
              pytest will run on the entire project based on its configuration.
        pytest_args: A list of additional command-line arguments to pass to pytest
                     (e.g., ["-k", "my_test_name", "-v"]).

    Returns:
        A dictionary summarizing the test results, including counts of passed,
        failed, and skipped tests, and detailed error reports for failures.
    """
    report_file = Path(".pytest_report.json")
    command = ["pytest", f"--json-report-file={report_file}"]

    if path:
        command.append(path)
    
    if pytest_args:
        command.extend(pytest_args)

    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300,  # 5-minute timeout for tests
        )

        if not report_file.exists():
            return {
                "error": "pytest execution failed or report was not generated.",
                "stdout": process.stdout,
                "stderr": process.stderr,
            }

        with open(report_file, "r") as f:
            report = json.load(f)

        # Clean up the report file
        report_file.unlink()

        return _summarize_report(report)

    except FileNotFoundError:
        return {"error": "pytest is not installed or not in the system's PATH."}
    except subprocess.TimeoutExpired:
        return {"error": "Test execution timed out after 5 minutes."}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}


def _summarize_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Parses the JSON report from pytest and creates a concise summary."""
    summary = report.get("summary", {})
    total = summary.get("total", 0)
    passed = summary.get("passed", 0)
    failed_count = summary.get("failed", 0)
    
    failures = []
    if failed_count > 0:
        for test in report.get("tests", []):
            if test.get("outcome") == "failed":
                call = test.get("call", {})
                long_repr = call.get("longrepr", "")
                # Ensure long_repr is a string before splitting
                if isinstance(long_repr, str):
                    error_lines = long_repr.split('\n')
                    error_message = error_lines[-1] if error_lines else "No error message captured."
                else:
                    error_message = "Error representation was not a string."

                failures.append({
                    "test_name": test.get("nodeid"),
                    "error_message": error_message,
                    "full_error": long_repr,
                })

    return {
        "summary": {
            "total_tests": total,
            "passed": passed,
            "failed": failed_count,
            "skipped": summary.get("skipped", 0),
            "xfailed": summary.get("xfailed", 0),
            "xpassed": summary.get("xpassed", 0),
        },
        "failures": failures,
    }