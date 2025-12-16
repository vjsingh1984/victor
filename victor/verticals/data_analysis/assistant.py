"""Data Analysis Assistant - Complete vertical for data exploration and insights.

Competitive positioning: ChatGPT Data Analysis, Claude Artifacts, Jupyter AI.
"""

from typing import Any, Dict, List, Optional, Set

from victor.verticals.base import StageDefinition, VerticalBase, VerticalConfig
from victor.verticals.protocols import (
    ModeConfigProviderProtocol,
    PromptContributorProtocol,
    SafetyExtensionProtocol,
    TieredToolConfig,
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
        """Get the list of tools for data analysis tasks.

        Uses canonical tool names from victor.tools.tool_names.
        """
        from victor.tools.tool_names import ToolNames

        return [
            # Core filesystem for data files
            ToolNames.READ,      # read_file → read
            ToolNames.WRITE,     # write_file → write
            ToolNames.EDIT,      # edit_files → edit
            ToolNames.LS,        # list_directory → ls
            # Python/Shell execution for analysis
            ToolNames.SHELL,     # bash → shell (for running Python scripts)
            # Code generation and search
            ToolNames.GREP,        # Keyword search
            ToolNames.CODE_SEARCH, # Semantic code search
            ToolNames.OVERVIEW,    # codebase_overview → overview
            ToolNames.GRAPH,       # Code graph analysis (PageRank, dependencies)
            # Web for datasets and documentation
            ToolNames.WEB_SEARCH,  # Web search (internet search)
            ToolNames.WEB_FETCH,   # Fetch URL content
        ]

    @classmethod
    def get_system_prompt(cls) -> str:
        """Get the system prompt for data analysis tasks."""
        return cls._get_system_prompt()

    @classmethod
    def get_config(cls) -> VerticalConfig:
        """Get the complete configuration for Data Analysis vertical.

        Uses base class implementation with Data Analysis-specific customizations.
        """
        from victor.framework.tools import ToolSet

        return VerticalConfig(
            tools=ToolSet.from_tools(cls.get_tools()),
            system_prompt=cls._get_system_prompt(),
            stages=cls._get_stages(),
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
            metadata={
                "vertical_name": cls.name,
                "vertical_version": cls.version,
                "description": cls.description,
            },
        )

    @classmethod
    def _get_stages(cls) -> Dict[str, StageDefinition]:
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

    @classmethod
    def get_tiered_tools(cls) -> Optional[TieredToolConfig]:
        """Get tiered tool configuration for Data Analysis.

        Simplified configuration using consolidated tool metadata:
        - Mandatory: Core tools always included for any task
        - Vertical Core: Essential tools for data analysis tasks
        - semantic_pool: Derived from ToolMetadataRegistry.get_all_tool_names()
        - stage_tools: Derived from @tool(stages=[...]) decorator metadata

        Returns:
            TieredToolConfig for Data Analysis vertical
        """
        from victor.tools.tool_names import ToolNames

        return TieredToolConfig(
            # Tier 1: Mandatory - always included for any task
            mandatory={
                ToolNames.READ,      # Read files - essential for data
                ToolNames.LS,        # List directory - essential
                ToolNames.GREP,      # Search code/data - essential
            },
            # Tier 2: Vertical Core - essential for data analysis tasks
            vertical_core={
                ToolNames.SHELL,     # Shell commands - core for running Python scripts
                ToolNames.WRITE,     # Write files - core for saving results
                ToolNames.OVERVIEW,  # Codebase overview - core for understanding
            },
            # semantic_pool and stage_tools are now derived from @tool decorator metadata
            # Use get_effective_semantic_pool() and get_tools_for_stage_from_registry()
            # Data analysis often involves exploratory work, allow write tools for analysis
            readonly_only_for_analysis=False,
        )
