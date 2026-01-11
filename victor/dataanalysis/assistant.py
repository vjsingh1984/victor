"""Data Analysis Assistant - Complete vertical for data exploration and insights.

Competitive positioning: ChatGPT Data Analysis, Claude Artifacts, Jupyter AI.
"""

from typing import Any, Dict, List, Optional

from victor.core.verticals.base import StageDefinition, VerticalBase
from victor.core.verticals.protocols import (
    ModeConfigProviderProtocol,
    PromptContributorProtocol,
    SafetyExtensionProtocol,
    TieredToolConfig,
    ToolDependencyProviderProtocol,
)

# Phase 3: Import framework capabilities
from victor.framework.capabilities import FileOperationsCapability


class DataAnalysisAssistant(VerticalBase):
    """Data analysis assistant for exploration, visualization, and insights.

    Competitive with: ChatGPT Data Analysis, Claude Artifacts, Jupyter AI.
    """

    name = "data_analysis"
    description = "Data exploration, statistical analysis, visualization, and ML insights"
    version = "1.0.0"

    # Phase 3: Framework file operations capability (read, write, edit, grep)
    _file_ops = FileOperationsCapability()

    @classmethod
    def get_tools(cls) -> List[str]:
        """Get the list of tools for data analysis tasks.

        Phase 3: Uses framework FileOperationsCapability for common file operations
        to reduce code duplication and maintain consistency across verticals.

        Uses canonical tool names from victor.tools.tool_names.
        """
        from victor.tools.tool_names import ToolNames

        # Start with framework file operations (read, write, edit, grep)
        tools = cls._file_ops.get_tool_list()

        # Add data analysis-specific tools
        tools.extend(
            [
                # Directory listing for data file exploration
                ToolNames.LS,  # list_directory → ls
                # Python/Shell execution for analysis
                ToolNames.SHELL,  # bash → shell (for running Python scripts)
                # Code generation and search
                ToolNames.CODE_SEARCH,  # Semantic code search
                ToolNames.OVERVIEW,  # codebase_overview → overview
                ToolNames.GRAPH,  # Code graph analysis (PageRank, dependencies)
                # Web for datasets and documentation
                ToolNames.WEB_SEARCH,  # Web search (internet search)
                ToolNames.WEB_FETCH,  # Fetch URL content
            ]
        )

        return tools

    @classmethod
    def get_system_prompt(cls) -> str:
        """Get the system prompt for data analysis tasks."""
        return cls._get_system_prompt()

    @classmethod
    def get_stages(cls) -> Dict[str, StageDefinition]:
        """Get Data Analysis-specific stage definitions.

        Uses canonical tool names from victor.tools.tool_names.
        """
        from victor.tools.tool_names import ToolNames

        return {
            "INITIAL": StageDefinition(
                name="INITIAL",
                description="Understanding the data and analysis goals",
                tools={ToolNames.READ, ToolNames.LS, ToolNames.OVERVIEW},
                keywords=["what", "data", "analyze", "understand", "explore"],
                next_stages={"DATA_LOADING", "EXPLORATION"},
            ),
            "DATA_LOADING": StageDefinition(
                name="DATA_LOADING",
                description="Loading and validating data files",
                tools={ToolNames.READ, ToolNames.SHELL, ToolNames.WRITE},
                keywords=["load", "import", "read", "open", "fetch"],
                next_stages={"EXPLORATION", "CLEANING"},
            ),
            "EXPLORATION": StageDefinition(
                name="EXPLORATION",
                description="Exploratory data analysis and profiling",
                tools={ToolNames.SHELL, ToolNames.READ, ToolNames.WRITE},
                keywords=["explore", "profile", "describe", "summary", "statistics"],
                next_stages={"CLEANING", "ANALYSIS"},
            ),
            "CLEANING": StageDefinition(
                name="CLEANING",
                description="Data cleaning and transformation",
                tools={ToolNames.SHELL, ToolNames.WRITE, ToolNames.EDIT},
                keywords=["clean", "transform", "fix", "handle", "remove"],
                next_stages={"ANALYSIS", "EXPLORATION"},
            ),
            "ANALYSIS": StageDefinition(
                name="ANALYSIS",
                description="Statistical analysis and modeling",
                tools={ToolNames.SHELL, ToolNames.WRITE, ToolNames.READ},
                keywords=["analyze", "model", "correlate", "test", "predict"],
                next_stages={"VISUALIZATION", "REPORTING"},
            ),
            "VISUALIZATION": StageDefinition(
                name="VISUALIZATION",
                description="Creating charts and visualizations",
                tools={ToolNames.SHELL, ToolNames.WRITE},
                keywords=["plot", "chart", "visualize", "graph", "figure"],
                next_stages={"REPORTING", "ANALYSIS"},
            ),
            "REPORTING": StageDefinition(
                name="REPORTING",
                description="Generating insights and reports",
                tools={ToolNames.WRITE, ToolNames.EDIT, ToolNames.READ},
                keywords=["report", "summarize", "document", "present"],
                next_stages={"COMPLETION"},
            ),
            "COMPLETION": StageDefinition(
                name="COMPLETION",
                description="Finalizing analysis deliverables",
                tools={ToolNames.WRITE, ToolNames.READ},
                keywords=["done", "complete", "finish", "final"],
                next_stages=set(),
            ),
        }

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
        return cls._get_extension_factory("prompt_contributor", "victor.dataanalysis.prompts")

    @classmethod
    def get_mode_config_provider(cls) -> Optional[ModeConfigProviderProtocol]:
        return cls._get_extension_factory("mode_config_provider", "victor.dataanalysis.mode_config")

    @classmethod
    def get_safety_extension(cls) -> Optional[SafetyExtensionProtocol]:
        return cls._get_extension_factory("safety_extension", "victor.dataanalysis.safety")

    @classmethod
    def get_tool_dependency_provider(cls) -> Optional[ToolDependencyProviderProtocol]:
        """Get tool dependency provider using centralized factory.

        Phase 4: Migrated from deprecated DataAnalysisToolDependencyProvider
        wrapper to centralized create_vertical_tool_dependency_provider().
        """

        def _create():
            from victor.core.tool_dependency_loader import create_vertical_tool_dependency_provider

            return create_vertical_tool_dependency_provider("dataanalysis")

        return cls._get_cached_extension("tool_dependency_provider", _create)

    @classmethod
    def get_tiered_tools(cls) -> Optional[TieredToolConfig]:
        """Get tiered tool configuration for Data Analysis."""
        from victor.core.vertical_types import TieredToolTemplate

        return TieredToolTemplate.for_vertical(cls.name)

    # =========================================================================
    # New Framework Integrations (Workflows, RL, Teams)
    # =========================================================================

    @classmethod
    def get_workflow_provider(cls) -> Optional[Any]:
        """Get Data Analysis-specific workflow provider.

        Provides workflows for:
        - eda_workflow: Exploratory Data Analysis
        - data_cleaning: Data cleaning and preparation
        - statistical_analysis: Hypothesis testing and statistical modeling
        - ml_pipeline: End-to-end machine learning pipeline

        Returns:
            DataAnalysisWorkflowProvider instance
        """
        from victor.dataanalysis.workflows import DataAnalysisWorkflowProvider

        return DataAnalysisWorkflowProvider()

    @classmethod
    def get_rl_config_provider(cls) -> Optional[Any]:
        """Get RL configuration provider for Data Analysis vertical.

        Returns:
            DataAnalysisRLConfig instance (implements RLConfigProviderProtocol)
        """
        from victor.dataanalysis.rl import DataAnalysisRLConfig

        return DataAnalysisRLConfig()

    @classmethod
    def get_rl_hooks(cls) -> Optional[Any]:
        """Get RL hooks for Data Analysis vertical.

        Returns:
            DataAnalysisRLHooks instance
        """
        from victor.dataanalysis.rl import DataAnalysisRLHooks

        return DataAnalysisRLHooks()

    @classmethod
    def get_team_spec_provider(cls) -> Optional[Any]:
        """Get team specification provider for Data Analysis tasks.

        Provides pre-configured team specifications for:
        - eda_team: Exploratory data analysis
        - cleaning_team: Data quality and preparation
        - statistics_team: Hypothesis testing
        - ml_team: Machine learning pipeline
        - visualization_team: Charts and dashboards

        Returns:
            DataAnalysisTeamSpecProvider instance (implements TeamSpecProviderProtocol)
        """
        from victor.dataanalysis.teams import DataAnalysisTeamSpecProvider

        return DataAnalysisTeamSpecProvider()

    @classmethod
    def get_capability_provider(cls) -> Optional[Any]:
        """Get capability provider for Data Analysis vertical.

        Provides capabilities for:
        - data_quality: Data quality rules and validation settings
        - visualization_style: Visualization and plotting configuration
        - statistical_analysis: Statistical analysis configuration
        - ml_pipeline: Machine learning pipeline configuration
        - data_privacy: Data privacy and anonymization settings

        Returns:
            DataAnalysisCapabilityProvider instance (implements BaseCapabilityProvider)
        """
        from victor.dataanalysis.capabilities import DataAnalysisCapabilityProvider

        return DataAnalysisCapabilityProvider()

    @classmethod
    def get_handlers(cls) -> Dict[str, Any]:
        """Get compute handlers for DataAnalysis workflows.

        Returns handlers from victor.dataanalysis.handlers for workflow execution.
        This replaces the previous import-side-effect registration pattern.

        Returns:
            Dict mapping handler names to handler instances
        """
        from victor.dataanalysis.handlers import HANDLERS

        return HANDLERS
