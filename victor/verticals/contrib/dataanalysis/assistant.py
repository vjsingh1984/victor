"""Data Analysis Assistant - Complete vertical for data exploration and insights.

Competitive positioning: ChatGPT Data Analysis, Claude Artifacts, Jupyter AI.
"""

from __future__ import annotations

from typing import Any, Dict, List

from victor_sdk import (
    CapabilityIds,
    CapabilityRequirement,
    PromptMetadata,
    StageDefinition,
    ToolNames,
    VerticalBase,
)

from victor.verticals.contrib.dataanalysis.prompt_metadata import (
    DATA_ANALYSIS_GROUNDING_RULES,
    DATA_ANALYSIS_PROMPT_PRIORITY,
    DATA_ANALYSIS_PROMPT_TEMPLATES,
    DATA_ANALYSIS_SYSTEM_PROMPT_SECTION,
    DATA_ANALYSIS_TASK_TYPE_HINTS,
)


class DataAnalysisAssistant(VerticalBase):
    """Data analysis assistant for exploration, visualization, and insights.

    Competitive with: ChatGPT Data Analysis, Claude Artifacts, Jupyter AI.
    """

    name = "dataanalysis"
    description = "Data exploration, statistical analysis, visualization, and ML insights"
    version = "1.0.0"

    @classmethod
    def get_name(cls) -> str:
        """Return the stable identifier for this vertical."""

        return cls.name

    @classmethod
    def get_description(cls) -> str:
        """Return the human-readable vertical description."""

        return cls.description

    @classmethod
    def get_tools(cls) -> List[str]:
        """Get the list of tools for data analysis tasks.

        Uses SDK-owned canonical tool identifiers, including the shared file-operation
        tool group.
        """
        tools = list(ToolNames.file_operations())

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
    def get_prompt_templates(cls) -> Dict[str, str]:
        """Return serializable prompt templates for the data analysis definition."""

        return dict(DATA_ANALYSIS_PROMPT_TEMPLATES)

    @classmethod
    def get_task_type_hints(cls) -> Dict[str, Dict[str, Any]]:
        """Return serializable task-type hints for the data analysis definition."""

        return {
            task_type: dict(config)
            for task_type, config in DATA_ANALYSIS_TASK_TYPE_HINTS.items()
        }

    @classmethod
    def get_prompt_metadata(cls) -> PromptMetadata:
        """Return full prompt metadata, including runtime adapter hints."""

        metadata = super().get_prompt_metadata()
        return PromptMetadata(
            templates=metadata.templates,
            task_type_hints=metadata.task_type_hints,
            metadata={
                "system_prompt_section": DATA_ANALYSIS_SYSTEM_PROMPT_SECTION,
                "grounding_rules": DATA_ANALYSIS_GROUNDING_RULES,
                "priority": DATA_ANALYSIS_PROMPT_PRIORITY,
            },
        )

    @classmethod
    def get_capability_requirements(cls) -> List[CapabilityRequirement]:
        """Declare runtime capabilities required by the data analysis definition."""

        return [
            CapabilityRequirement(
                capability_id=CapabilityIds.FILE_OPS,
                purpose="Inspect datasets, notebooks, and generated analysis artifacts in the workspace.",
            ),
            CapabilityRequirement(
                capability_id=CapabilityIds.SHELL_ACCESS,
                purpose="Run Python, notebook-like, and statistical analysis workflows from the shell.",
            ),
            CapabilityRequirement(
                capability_id=CapabilityIds.VALIDATION,
                purpose="Validate assumptions, calculations, and generated outputs before reporting findings.",
            ),
            CapabilityRequirement(
                capability_id=CapabilityIds.WEB_ACCESS,
                optional=True,
                purpose="Fetch dataset documentation or external references when local context is insufficient.",
            ),
        ]

    @classmethod
    def get_stages(cls) -> Dict[str, StageDefinition]:
        """Get Data Analysis-specific stage definitions.

        Uses SDK-owned canonical tool identifiers.
        """
        return {
            "INITIAL": StageDefinition(
                name="INITIAL",
                description="Understanding the data and analysis goals",
                optional_tools=[ToolNames.READ, ToolNames.LS, ToolNames.OVERVIEW],
                keywords=["what", "data", "analyze", "understand", "explore"],
                next_stages={"DATA_LOADING", "EXPLORATION"},
            ),
            "DATA_LOADING": StageDefinition(
                name="DATA_LOADING",
                description="Loading and validating data files",
                optional_tools=[ToolNames.READ, ToolNames.SHELL, ToolNames.WRITE],
                keywords=["load", "import", "read", "open", "fetch"],
                next_stages={"EXPLORATION", "CLEANING"},
            ),
            "EXPLORATION": StageDefinition(
                name="EXPLORATION",
                description="Exploratory data analysis and profiling",
                optional_tools=[ToolNames.SHELL, ToolNames.READ, ToolNames.WRITE],
                keywords=["explore", "profile", "describe", "summary", "statistics"],
                next_stages={"CLEANING", "ANALYSIS"},
            ),
            "CLEANING": StageDefinition(
                name="CLEANING",
                description="Data cleaning and transformation",
                optional_tools=[ToolNames.SHELL, ToolNames.WRITE, ToolNames.EDIT],
                keywords=["clean", "transform", "fix", "handle", "remove"],
                next_stages={"ANALYSIS", "EXPLORATION"},
            ),
            "ANALYSIS": StageDefinition(
                name="ANALYSIS",
                description="Statistical analysis and modeling",
                optional_tools=[ToolNames.SHELL, ToolNames.WRITE, ToolNames.READ],
                keywords=["analyze", "model", "correlate", "test", "predict"],
                next_stages={"VISUALIZATION", "REPORTING"},
            ),
            "VISUALIZATION": StageDefinition(
                name="VISUALIZATION",
                description="Creating charts and visualizations",
                optional_tools=[ToolNames.SHELL, ToolNames.WRITE],
                keywords=["plot", "chart", "visualize", "graph", "figure"],
                next_stages={"REPORTING", "ANALYSIS"},
            ),
            "REPORTING": StageDefinition(
                name="REPORTING",
                description="Generating insights and reports",
                optional_tools=[ToolNames.WRITE, ToolNames.EDIT, ToolNames.READ],
                keywords=["report", "summarize", "document", "present"],
                next_stages={"COMPLETION"},
            ),
            "COMPLETION": StageDefinition(
                name="COMPLETION",
                description="Finalizing analysis deliverables",
                optional_tools=[ToolNames.WRITE, ToolNames.READ],
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

    # =========================================================================
    # New Framework Integrations (Workflows, RL, Teams)
    # =========================================================================
