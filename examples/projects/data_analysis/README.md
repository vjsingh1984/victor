# Data Analysis Example Project

Data analysis and visualization assistant using Victor AI.

## Features

- Pandas DataFrame analysis
- Statistical analysis
- Visualization generation (Matplotlib/Seaborn)
- Report generation
- Trend detection and forecasting

## Quick Start

```bash
cd examples/projects/data_analysis
pip install -r requirements.txt
victor init
victor chat "Analyze the data in data/sample.csv and generate insights"
```

## Usage Examples

```bash
# Basic analysis
victor chat "Analyze data/sample.csv and provide summary statistics"

# Visualization
victor chat "Create visualizations for the sales data showing trends over time"

# Statistical analysis
victor chat "Perform correlation analysis on all numerical columns"

# Report generation
victor chat "Generate a comprehensive data analysis report with insights and recommendations"
```

## Sample Code

### src/analyzer.py

```python
"""Data analysis toolkit."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

class DataAnalyzer:
    """Analyze and visualize data."""

    def __init__(self, data_path: str):
        """Initialize analyzer with data."""
        self.data = pd.read_csv(data_path)
        self.report = []

    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive analysis."""
        return {
            "overview": self.get_overview(),
            "statistics": self.get_statistics(),
            "correlations": self.get_correlations(),
            "insights": self.generate_insights()
        }

    def get_overview(self) -> Dict[str, Any]:
        """Get data overview."""
        return {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": self.data.dtypes.to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "memory_usage": self.data.memory_usage(deep=True).sum()
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistical summary."""
        numeric = self.data.select_dtypes(include=['number'])
        return {
            "describe": numeric.describe().to_dict(),
            "skewness": numeric.skew().to_dict(),
            "kurtosis": numeric.kurtosis().to_dict()
        }

    def get_correlations(self) -> Dict[str, float]:
        """Get correlation matrix."""
        numeric = self.data.select_dtypes(include=['number'])
        return numeric.corr().to_dict()

    def visualize(self, output_dir: str):
        """Generate visualizations."""
        # Histograms
        self._plot_histograms(f"{output_dir}/histograms.png")

        # Correlation heatmap
        self._plot_correlation_heatmap(f"{output_dir}/correlations.png")

        # Time series if applicable
        self._plot_time_series(f"{output_dir}/trends.png")

    def _plot_histograms(self, output_path: str):
        """Plot histograms for numerical columns."""
        numeric = self.data.select_dtypes(include=['number'])
        numeric.hist(figsize=(12, 8))
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def _plot_correlation_heatmap(self, output_path: str):
        """Plot correlation heatmap."""
        numeric = self.data.select_dtypes(include=['number'])
        sns.heatmap(numeric.corr(), annot=True, cmap='coolwarm')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def generate_insights(self) -> List[str]:
        """Generate insights from analysis."""
        insights = []

        # Check for missing data
        missing = self.data.isnull().sum()
        if missing.any():
            insights.append(f"Columns with missing data: {missing[missing > 0].to_dict()}")

        # Check for outliers
        numeric = self.data.select_dtypes(include=['number'])
        for col in numeric.columns:
            q1 = numeric[col].quantile(0.25)
            q3 = numeric[col].quantile(0.75)
            iqr = q3 - q1
            outliers = ((numeric[col] < (q1 - 1.5 * iqr)) | (numeric[col] > (q3 + 1.5 * iqr))).sum()
            if outliers > 0:
                insights.append(f"Column '{col}' has {outliers} potential outliers")

        return insights
```

## Victor AI Integration

```bash
# Data exploration
victor chat "Explore the dataset and identify:
1. Key features and their types
2. Missing values and data quality issues
3. Potential correlations
4. Interesting patterns or trends"

# Advanced analysis
victor chat "Perform advanced statistical analysis including:
1. Hypothesis testing
2. Regression analysis
3. Clustering if applicable
4. Feature importance analysis"

# Visualization requests
victor chat "Create the following visualizations:
1. Distribution plots for key variables
2. Correlation heatmap
3. Time series trends if applicable
4. Scatter plots for key relationships"
```

## Learning Objectives

1. Work with Pandas DataFrames
2. Perform statistical analysis
3. Create data visualizations
4. Generate analysis reports
5. Detect patterns and trends

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
scikit-learn>=1.2.0
```

Happy analyzing! 📊
