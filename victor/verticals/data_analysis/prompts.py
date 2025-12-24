"""Data Analysis Prompt Contributor - Task hints for data science workflows."""

from typing import Dict, Optional

from victor.verticals.protocols import PromptContributorProtocol


# Data analysis-specific task type hints
# Keys align with TaskTypeClassifier task types (data_analysis, visualization)
# Also includes granular hints for specific analysis methods
DATA_ANALYSIS_TASK_TYPE_HINTS: Dict[str, str] = {
    # Classifier task types (matched by TaskTypeClassifier)
    "data_analysis": """[DATA ANALYSIS] Comprehensive data exploration and analysis:
1. Load data and check shape/types with df.info(), df.describe()
2. Calculate summary statistics (mean, median, std, quartiles)
3. Identify missing values and their patterns (df.isnull().sum())
4. Check for duplicates and data quality issues
5. Analyze correlations and distributions before modeling""",

    "visualization": """[VISUALIZATION] Create informative charts and dashboards:
1. Choose appropriate chart type for the data (bar, line, scatter, heatmap)
2. Use clear labels, titles, and legends
3. Add context (units, time periods, annotations)
4. Consider colorblind-friendly palettes (viridis, cividis)
5. Save as high-resolution images (plt.savefig('fig.png', dpi=300))""",

    # Granular hints for specific analysis methods (context_hints)
    "data_profiling": """[PROFILE] Comprehensive data profiling:
1. Load data and check shape/types
2. Calculate summary statistics (mean, median, std, quartiles)
3. Identify missing values and their patterns
4. Check for duplicates and uniqueness
5. Analyze value distributions""",

    "statistical_analysis": """[STATISTICS] Perform statistical analysis:
1. State null and alternative hypotheses
2. Check assumptions (normality, variance)
3. Choose appropriate test (t-test, ANOVA, chi-square)
4. Calculate test statistic and p-value
5. Interpret results with effect size""",

    "correlation_analysis": """[CORRELATION] Analyze variable relationships:
1. Calculate correlation matrix
2. Use appropriate method (Pearson, Spearman)
3. Visualize with heatmap
4. Identify strong correlations
5. Note potential confounders""",

    "regression": """[REGRESSION] Build predictive models:
1. Define target and feature variables
2. Split data into train/test
3. Check for multicollinearity
4. Fit model and assess coefficients
5. Evaluate with R², RMSE, residual plots""",

    "clustering": """[CLUSTERING] Segment data:
1. Scale features appropriately
2. Determine optimal cluster count (elbow, silhouette)
3. Apply clustering algorithm
4. Visualize clusters
5. Profile cluster characteristics""",

    "time_series": """[TIMESERIES] Analyze temporal data:
1. Check datetime format and frequency
2. Plot time series and identify patterns
3. Decompose into trend, seasonal, residual
4. Check stationarity (ADF test)
5. Apply appropriate forecasting method""",

    # Default fallback for 'general' task type
    "general": """[GENERAL DATA] For general data queries:
1. Read available data files (CSV, Excel, databases)
2. Use pandas for data exploration (df.info(), df.describe())
3. Calculate basic statistics and distributions
4. Create simple visualizations for insights
5. Summarize findings clearly""",
}


class DataAnalysisPromptContributor(PromptContributorProtocol):
    """Contributes data analysis-specific prompts and task hints."""

    def get_task_type_hints(self) -> Dict[str, str]:
        """Return data analysis-specific task type hints."""
        return DATA_ANALYSIS_TASK_TYPE_HINTS

    def get_system_prompt_extension(self) -> Optional[str]:
        """Return additional system prompt content for data analysis."""
        return """
## Python Libraries Reference

### Data Manipulation
```python
import pandas as pd
import numpy as np
```

### Visualization
```python
import matplotlib.pyplot as plt
import seaborn as sns
# For interactive: import plotly.express as px
```

### Statistics
```python
from scipy import stats
from statsmodels.api import OLS
```

### Machine Learning
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
```

## Common Data Operations

| Task | Code |
|------|------|
| Read CSV | `pd.read_csv('file.csv')` |
| Summary | `df.describe()` |
| Missing | `df.isnull().sum()` |
| Types | `df.dtypes` |
| Correlation | `df.corr()` |
| Group | `df.groupby('col').agg({'val': 'mean'})` |
"""

    def get_context_hints(self, task_type: Optional[str] = None) -> Optional[str]:
        """Return contextual hints based on detected task type."""
        if task_type and task_type in DATA_ANALYSIS_TASK_TYPE_HINTS:
            return DATA_ANALYSIS_TASK_TYPE_HINTS[task_type]
        return None
