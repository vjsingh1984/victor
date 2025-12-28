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

"""Coverage visualization and analysis system.

This module provides comprehensive code coverage collection, parsing,
analysis, and visualization capabilities. Supports multiple coverage
formats and languages.

Example usage:
    from victor.coverage import get_coverage_manager, CoverageThreshold
    from pathlib import Path

    # Get manager with custom thresholds
    manager = get_coverage_manager(project_root=Path("."))
    manager.threshold = CoverageThreshold(
        line_coverage=80.0,
        branch_coverage=70.0,
    )

    # Collect coverage
    report = manager.collect_coverage(
        test_command=["pytest", "--cov=mypackage", "--cov-report=xml"]
    )

    # Or parse existing coverage file
    report = manager.parse_coverage(Path("coverage.xml"))

    # Generate reports
    text_report = manager.generate_text_report()
    print(text_report)

    html_path = manager.generate_html_report(output_dir=Path("htmlcov"))
    print(f"HTML report: {html_path}")

    # Check thresholds
    passed, failures = manager.check_threshold()
    if not passed:
        for failure in failures:
            print(f"FAIL: {failure}")

    # Save to history
    manager.save_report()

    # Compare with previous
    history = manager.get_coverage_history(limit=2)
    if len(history) >= 2:
        _, current = history[0]
        _, previous = history[1]
        diff = manager.compare_reports(previous, current)
        print(f"Coverage change: {diff.coverage_delta:+.1f}%")
"""

from victor.coverage.protocol import (
    BranchCoverage,
    CoverageDiff,
    CoverageReport,
    CoverageStatus,
    CoverageThreshold,
    CoverageType,
    FileCoverage,
    FunctionCoverage,
    LineCoverage,
)
from victor.coverage.parser import (
    BaseCoverageParser,
    CloverParser,
    CoberturaParser,
    CoverageParser,
    GoCoverParser,
    JestCoverageParser,
    LcovParser,
    get_parser_for_file,
    parse_coverage_file,
)
from victor.coverage.visualizer import CoverageVisualizer
from victor.coverage.manager import (
    CoverageManager,
    get_coverage_manager,
    reset_coverage_manager,
)

__all__ = [
    # Protocol types
    "BranchCoverage",
    "CoverageDiff",
    "CoverageReport",
    "CoverageStatus",
    "CoverageThreshold",
    "CoverageType",
    "FileCoverage",
    "FunctionCoverage",
    "LineCoverage",
    # Parsers
    "BaseCoverageParser",
    "CloverParser",
    "CoberturaParser",
    "CoverageParser",
    "GoCoverParser",
    "JestCoverageParser",
    "LcovParser",
    "get_parser_for_file",
    "parse_coverage_file",
    # Visualizer
    "CoverageVisualizer",
    # Manager
    "CoverageManager",
    "get_coverage_manager",
    "reset_coverage_manager",
]
