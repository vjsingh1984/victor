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

"""Serializable prompt metadata for the Data Analysis vertical."""

from __future__ import annotations

from typing import Any, Dict

DATA_ANALYSIS_TASK_TYPE_HINTS: Dict[str, Dict[str, Any]] = {
    "data_analysis": {
        "hint": """[DATA ANALYSIS] Comprehensive data exploration and analysis:
1. Load data and check shape/types with df.info(), df.describe()
2. Calculate summary statistics (mean, median, std, quartiles)
3. Identify missing values and their patterns (df.isnull().sum())
4. Check for duplicates and data quality issues
5. Analyze correlations and distributions before modeling""",
        "tool_budget": 15,
        "priority_tools": ["shell", "read", "write", "edit"],
    },
    "visualization": {
        "hint": """[VISUALIZATION] Create informative charts and dashboards:
1. Choose appropriate chart type for the data (bar, line, scatter, heatmap)
2. Use clear labels, titles, and legends
3. Add context (units, time periods, annotations)
4. Consider colorblind-friendly palettes (viridis, cividis)
5. Save as high-resolution images (plt.savefig('fig.png', dpi=300))""",
        "tool_budget": 12,
        "priority_tools": ["shell", "read", "write"],
    },
    "data_profiling": {
        "hint": """[PROFILE] Comprehensive data profiling:
1. Load data and check shape/types
2. Calculate summary statistics (mean, median, std, quartiles)
3. Identify missing values and their patterns
4. Check for duplicates and uniqueness
5. Analyze value distributions""",
        "tool_budget": 10,
        "priority_tools": ["shell", "read"],
    },
    "statistical_analysis": {
        "hint": """[STATISTICS] Perform statistical analysis:
1. State null and alternative hypotheses
2. Check assumptions (normality, variance)
3. Choose appropriate test (t-test, ANOVA, chi-square)
4. Calculate test statistic and p-value
5. Interpret results with effect size""",
        "tool_budget": 12,
        "priority_tools": ["shell", "read", "write"],
    },
    "correlation_analysis": {
        "hint": """[CORRELATION] Analyze variable relationships:
1. Calculate correlation matrix
2. Use appropriate method (Pearson, Spearman)
3. Visualize with heatmap
4. Identify strong correlations
5. Note potential confounders""",
        "tool_budget": 10,
        "priority_tools": ["shell", "read", "write"],
    },
    "regression": {
        "hint": """[REGRESSION] Build predictive models:
1. Define target and feature variables
2. Split data into train/test
3. Check for multicollinearity
4. Fit model and assess coefficients
5. Evaluate with R^2, RMSE, residual plots""",
        "tool_budget": 15,
        "priority_tools": ["shell", "read", "write", "edit"],
    },
    "clustering": {
        "hint": """[CLUSTERING] Segment data:
1. Scale features appropriately
2. Determine optimal cluster count (elbow, silhouette)
3. Apply clustering algorithm
4. Visualize clusters
5. Profile cluster characteristics""",
        "tool_budget": 12,
        "priority_tools": ["shell", "read", "write"],
    },
    "time_series": {
        "hint": """[TIMESERIES] Analyze temporal data:
1. Check datetime format and frequency
2. Plot time series and identify patterns
3. Decompose into trend, seasonal, residual
4. Check stationarity (ADF test)
5. Apply appropriate forecasting method""",
        "tool_budget": 15,
        "priority_tools": ["shell", "read", "write", "edit"],
    },
    "general": {
        "hint": """[GENERAL DATA] For general data queries:
1. Read available data files (CSV, Excel, databases)
2. Use pandas for data exploration (df.info(), df.describe())
3. Calculate basic statistics and distributions
4. Create simple visualizations for insights
5. Summarize findings clearly""",
        "tool_budget": 10,
        "priority_tools": ["read", "ls", "shell"],
    },
}

DATA_ANALYSIS_SYSTEM_PROMPT_SECTION = """
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
""".strip()

DATA_ANALYSIS_GROUNDING_RULES = (
    """GROUNDING: Base ALL responses on tool output only. Never fabricate data or statistics.
Verify calculations with actual data. Always show code that produced results.""".strip()
)

DATA_ANALYSIS_PROMPT_PRIORITY = 5

DATA_ANALYSIS_PROMPT_TEMPLATES: Dict[str, str] = {
    "data_analysis_operations": DATA_ANALYSIS_SYSTEM_PROMPT_SECTION,
}

__all__ = [
    "DATA_ANALYSIS_GROUNDING_RULES",
    "DATA_ANALYSIS_PROMPT_PRIORITY",
    "DATA_ANALYSIS_PROMPT_TEMPLATES",
    "DATA_ANALYSIS_SYSTEM_PROMPT_SECTION",
    "DATA_ANALYSIS_TASK_TYPE_HINTS",
]
