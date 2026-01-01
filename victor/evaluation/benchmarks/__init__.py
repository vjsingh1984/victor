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

"""Benchmark runners for industry-standard evaluation datasets.

All runners load REAL benchmark data from HuggingFace and execute
REAL tests. Results are not simulated - they represent actual test
execution against actual benchmark problems.

Supported benchmarks:
- SWE-bench: Real-world GitHub issues from Python repositories
- HumanEval: Code generation from docstrings (OpenAI)
- MBPP: Mostly Basic Python Problems (Google Research)

Framework comparison:
- Compare Victor against Aider, Claude Code, Cursor, and others
- Uses published benchmark results for standardized comparison
"""

from victor.evaluation.benchmarks.swe_bench import (
    HumanEvalRunner,
    MBPPRunner,
    SWEBenchRunner,
)
from victor.evaluation.benchmarks.framework_comparison import (
    ComparisonMetrics,
    ComparisonReport,
    Framework,
    FrameworkCapabilities,
    FrameworkResult,
    FRAMEWORK_CAPABILITIES,
    PUBLISHED_RESULTS,
    compute_metrics_from_result,
    create_comparison_report,
    get_published_result,
)

__all__ = [
    # Runners
    "HumanEvalRunner",
    "MBPPRunner",
    "SWEBenchRunner",
    # Framework comparison
    "ComparisonMetrics",
    "ComparisonReport",
    "Framework",
    "FrameworkCapabilities",
    "FrameworkResult",
    "FRAMEWORK_CAPABILITIES",
    "PUBLISHED_RESULTS",
    "compute_metrics_from_result",
    "create_comparison_report",
    "get_published_result",
]
