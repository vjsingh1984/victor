"""Data Analysis Assistant - Complete vertical for data exploration and insights.

Competitive positioning: ChatGPT Data Analysis, Claude Artifacts, Jupyter AI.
"""

from typing import List, Optional

from victor.verticals.base import StageDefinition, VerticalBase, VerticalConfig
from victor.verticals.protocols import (
    ModeConfigProviderProtocol,
    PromptContributorProtocol,
    SafetyExtensionProtocol,
    ToolDependencyProviderProtocol,
)


class DataAnalysisAssistant(VerticalBase):
    """Data analysis assistant for exploration, visualization, and insights.

    Competitive with: ChatGPT Data Analysis, Claude Artifacts, Jupyter AI.
    """

    name = "data_analysis"
    description = "Data exploration, statistical analysis, visualization, and ML insights"
    version = "1.0.0"

    @classmethod
    def get_tools(cls) -> List[str]:
        """Get the list of tools for data analysis tasks."""
        return [
            # Core filesystem for data files
            "read_file",
            "write_file",
            "edit_files",
            "list_directory",
            # Python execution for analysis
            "bash",  # For running Python scripts
            # Code generation and search
            "code_search",
            "semantic_code_search",
            "codebase_overview",
            # Web for datasets and documentation
            "web_search",
            "web_fetch",
        ]

    @classmethod
    def get_system_prompt(cls) -> str:
        """Get the system prompt for data analysis tasks."""
        return cls._get_system_prompt()

    @classmethod
    def get_config(cls) -> VerticalConfig:
        return VerticalConfig(
            name="data_analysis",
            description="Data analysis assistant for exploration, statistics, and visualization",
            tools=[
                # Core filesystem for data files
                "read_file",
                "write_file",
                "edit_files",
                "list_directory",
                # Python execution for analysis
                "bash",  # For running Python scripts
                # Code generation and search
                "code_search",
                "semantic_code_search",
                "codebase_overview",
                # Web for datasets and documentation
                "web_search",
                "web_fetch",
            ],
            stages=cls._get_stages(),
            system_prompt=cls._get_system_prompt(),
            provider_hints={
                "preferred_providers": ["anthropic", "openai"],
                "min_context_window": 128000,  # Large context for data descriptions
                "features": ["tool_calling", "large_context", "code_execution"],
            },
            evaluation_criteria=[
                "statistical_correctness",
                "visualization_quality",
                "insight_clarity",
                "reproducibility",
                "data_privacy",
                "methodology_transparency",
            ],
        )

    @classmethod
    def _get_stages(cls) -> List[StageDefinition]:
        return [
            StageDefinition(
                name="INITIAL",
                description="Understanding the data and analysis goals",
                allowed_tools=["read_file", "list_directory", "codebase_overview"],
                next_stages=["DATA_LOADING", "EXPLORATION"],
            ),
            StageDefinition(
                name="DATA_LOADING",
                description="Loading and validating data files",
                allowed_tools=["read_file", "bash", "write_file"],
                next_stages=["EXPLORATION", "CLEANING"],
            ),
            StageDefinition(
                name="EXPLORATION",
                description="Exploratory data analysis and profiling",
                allowed_tools=["bash", "read_file", "write_file"],
                next_stages=["CLEANING", "ANALYSIS"],
            ),
            StageDefinition(
                name="CLEANING",
                description="Data cleaning and transformation",
                allowed_tools=["bash", "write_file", "edit_files"],
                next_stages=["ANALYSIS", "EXPLORATION"],
            ),
            StageDefinition(
                name="ANALYSIS",
                description="Statistical analysis and modeling",
                allowed_tools=["bash", "write_file", "read_file"],
                next_stages=["VISUALIZATION", "REPORTING"],
            ),
            StageDefinition(
                name="VISUALIZATION",
                description="Creating charts and visualizations",
                allowed_tools=["bash", "write_file"],
                next_stages=["REPORTING", "ANALYSIS"],
            ),
            StageDefinition(
                name="REPORTING",
                description="Generating insights and reports",
                allowed_tools=["write_file", "edit_files", "read_file"],
                next_stages=["COMPLETION"],
            ),
            StageDefinition(
                name="COMPLETION",
                description="Finalizing analysis deliverables",
                allowed_tools=["write_file", "read_file"],
                next_stages=[],
            ),
        ]

    @classmethod
    def _get_system_prompt(cls) -> str:
        return """You are a data analysis assistant specializing in exploration, statistics, and visualization.

## Core Capabilities

1. **Data Loading**: CSV, Excel, JSON, Parquet, SQL databases
2. **Exploration**: Profiling, summary statistics, distribution analysis
3. **Cleaning**: Missing values, outliers, type conversion, normalization
4. **Analysis**: Correlation, regression, hypothesis testing, clustering
5. **Visualization**: matplotlib, seaborn, plotly for charts and dashboards
6. **ML**: scikit-learn for classification, regression, clustering

## Analysis Workflow

1. **LOAD**: Read data, check structure, identify types
2. **EXPLORE**: Summary stats, distributions, missing values
3. **CLEAN**: Handle nulls, outliers, type issues
4. **ANALYZE**: Apply statistical methods, test hypotheses
5. **VISUALIZE**: Create informative charts
6. **REPORT**: Summarize insights with evidence

## Code Standards

- Always use pandas for data manipulation
- Include comments explaining methodology
- Handle missing data explicitly
- Use descriptive variable names
- Save intermediate results for reproducibility

## Output Format

When presenting analysis:
1. Start with data overview (shape, types, missing)
2. Show key statistics with context
3. Include visualizations with captions
4. State insights with supporting evidence
5. Note limitations and assumptions
6. Provide reproducible code

## Privacy and Ethics

- Never expose personally identifiable information (PII)
- Anonymize sensitive columns before analysis
- Note potential biases in data
- Be transparent about limitations
"""

    @classmethod
    def get_prompt_contributor(cls) -> Optional[PromptContributorProtocol]:
        from victor.verticals.data_analysis.prompts import DataAnalysisPromptContributor
        return DataAnalysisPromptContributor()

    @classmethod
    def get_mode_config_provider(cls) -> Optional[ModeConfigProviderProtocol]:
        from victor.verticals.data_analysis.mode_config import DataAnalysisModeConfigProvider
        return DataAnalysisModeConfigProvider()

    @classmethod
    def get_safety_extension(cls) -> Optional[SafetyExtensionProtocol]:
        from victor.verticals.data_analysis.safety import DataAnalysisSafetyExtension
        return DataAnalysisSafetyExtension()

    @classmethod
    def get_tool_dependency_provider(cls) -> Optional[ToolDependencyProviderProtocol]:
        from victor.verticals.data_analysis.tool_dependencies import DataAnalysisToolDependencyProvider
        return DataAnalysisToolDependencyProvider()
