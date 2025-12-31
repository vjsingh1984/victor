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

"""This module has moved to victor.observability.pipeline.

This module is kept for backward compatibility. Please update imports to use:
    from victor.observability.pipeline import ...
"""

# Re-export from new location for backward compatibility
from victor.observability.pipeline.analyzers import (
    CoberturaAnalyzer,
    GitHubActionsAnalyzer,
    GitLabCIAnalyzer,
    JaCoCoAnalyzer,
    LCOVAnalyzer,
    get_all_coverage_analyzers,
    get_all_pipeline_analyzers,
    get_coverage_analyzer,
    get_pipeline_analyzer,
)
from victor.observability.pipeline.manager import PipelineManager
from victor.observability.pipeline.protocol import (
    CoverageAnalyzerProtocol,
    CoverageMetrics,
    PipelineAnalysisResult,
    PipelineAnalyzerProtocol,
    PipelineConfig,
    PipelineIssue,
    PipelinePlatform,
    PipelineRun,
    PipelineStatus,
    PipelineStep,
    StepType,
)

__all__ = [
    # Manager
    "PipelineManager",
    # Protocols
    "PipelineAnalyzerProtocol",
    "CoverageAnalyzerProtocol",
    # Data classes
    "PipelinePlatform",
    "PipelineStatus",
    "StepType",
    "PipelineStep",
    "PipelineConfig",
    "PipelineRun",
    "PipelineIssue",
    "PipelineAnalysisResult",
    "CoverageMetrics",
    # Analyzers
    "GitHubActionsAnalyzer",
    "GitLabCIAnalyzer",
    "CoberturaAnalyzer",
    "LCOVAnalyzer",
    "JaCoCoAnalyzer",
    # Registry functions
    "get_pipeline_analyzer",
    "get_coverage_analyzer",
    "get_all_pipeline_analyzers",
    "get_all_coverage_analyzers",
]
