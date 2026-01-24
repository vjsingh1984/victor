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

"""Integration tests for data analysis workflows.

Tests cover:
- CSV/data file ingestion
- Analysis pipeline execution
- Visualization generation
- Statistical reporting
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Import required modules
pytest.importorskip("pandas")
pytest.importorskip("numpy")
import pandas as pd
import numpy as np

# matplotlib is optional - skip tests that need it if not available
try:
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend for testing
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from victor.dataanalysis.handlers import StatsComputeHandler
from victor.framework.handler_registry import HandlerRegistry


class MockToolResult:
    """Mock tool result for testing."""

    def __init__(self, success: bool = True, output: Any = None, error: str = None):
        self.success = success
        self.output = output
        self.error = error


class MockComputeNode:
    """Mock compute node for testing."""

    def __init__(
        self,
        node_id: str = "test_node",
        input_mapping: Dict[str, Any] = None,
        output_key: str = None,
    ):
        self.id = node_id
        self.input_mapping = input_mapping or {}
        self.output_key = output_key


class MockWorkflowContext:
    """Mock workflow context for testing."""

    def __init__(self, data: Dict[str, Any] = None):
        self._data = data or {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value


class TestDataFileIngestion:
    """Tests for data file ingestion workflows."""

    @pytest.fixture
    def sample_csv_file(self):
        """Create sample CSV file for testing."""
        data = {
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "age": [25, 30, 35, 28, 32],
            "salary": [50000, 60000, 70000, 55000, 65000],
            "department": ["Engineering", "Sales", "Engineering", "Marketing", "Sales"],
        }
        df = pd.DataFrame(data)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
            df.to_csv(tmp_path, index=False)

        yield tmp_path

        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    @pytest.fixture
    def sample_json_file(self):
        """Create sample JSON file for testing."""
        data = [
            {"product": "A", "price": 100, "quantity": 5},
            {"product": "B", "price": 200, "quantity": 3},
            {"product": "C", "price": 150, "quantity": 8},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
            json.dump(data, tmp)

        yield tmp_path

        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    def test_csv_file_reading(self, sample_csv_file):
        """Test reading CSV file into DataFrame."""
        df = pd.read_csv(sample_csv_file)

        assert df.shape == (5, 4)
        assert list(df.columns) == ["name", "age", "salary", "department"]
        assert df["age"].mean() == 30.0

    def test_json_file_reading(self, sample_json_file):
        """Test reading JSON file into DataFrame."""
        df = pd.read_json(sample_json_file)

        assert df.shape == (3, 3)
        assert "product" in df.columns
        assert df["price"].sum() == 450

    def test_data_validation_after_ingestion(self, sample_csv_file):
        """Test data validation after file ingestion."""
        df = pd.read_csv(sample_csv_file)

        # Check for missing values
        assert df.isnull().sum().sum() == 0

        # Check data types
        assert df["age"].dtype in [int, "int64"]
        assert df["salary"].dtype in [int, "int64"]
        assert df["department"].dtype == object

        # Check value ranges
        assert df["age"].min() >= 0
        assert df["age"].max() <= 100
        assert df["salary"].min() > 0


class TestAnalysisPipelineExecution:
    """Tests for analysis pipeline execution."""

    @pytest.fixture
    def mock_registry(self):
        """Mock tool registry."""
        registry = MagicMock()
        registry.execute = AsyncMock(return_value=MockToolResult(success=True, output=""))
        return registry

    @pytest.fixture
    def stats_handler(self):
        """Get stats compute handler."""
        return StatsComputeHandler()

    @pytest.fixture
    def sample_data(self):
        """Create sample data for analysis."""
        return {
            "values": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "categories": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            "weights": [1.0, 2.0, 1.5, 2.5, 1.2, 2.3, 1.8, 2.1, 1.4, 2.6],
        }

    def test_stats_compute_single_operation(self, stats_handler, sample_data):
        """Test computing a single statistic."""
        node = MockComputeNode(
            input_mapping={"data": sample_data["values"], "operations": ["mean"]},
        )
        context = MockWorkflowContext()

        import asyncio

        result, cost = asyncio.run(stats_handler.execute(node, context, MagicMock()))

        assert result["mean"] == 55.0
        assert cost == 0

    def test_stats_compute_multiple_operations(self, stats_handler, sample_data):
        """Test computing multiple statistics."""
        node = MockComputeNode(
            input_mapping={
                "data": sample_data["values"],
                "operations": ["mean", "median", "std", "min", "max"],
            },
        )
        context = MockWorkflowContext()

        import asyncio

        result, cost = asyncio.run(stats_handler.execute(node, context, MagicMock()))

        assert "mean" in result
        assert "median" in result
        assert "std" in result
        assert "min" in result
        assert "max" in result
        assert result["mean"] == 55.0
        assert result["min"] == 10
        assert result["max"] == 100

    def test_stats_compute_with_context_data(self, stats_handler, sample_data):
        """Test computing stats with data from context."""
        node = MockComputeNode(
            input_mapping={"data": sample_data["values"], "operations": ["describe"]},
        )
        context = MockWorkflowContext()

        import asyncio

        result, cost = asyncio.run(stats_handler.execute(node, context, MagicMock()))

        assert result["describe"]["count"] == 10
        assert result["describe"]["mean"] == 55.0

    def test_stats_compute_pipeline_sequence(self, stats_handler, sample_data):
        """Test sequence of statistical operations."""
        import asyncio

        # First operation: mean
        node1 = MockComputeNode(
            input_mapping={"data": sample_data["values"], "operations": ["mean"]},
        )
        context = MockWorkflowContext()
        result1, _ = asyncio.run(stats_handler.execute(node1, context, MagicMock()))
        context.set("mean_result", result1["mean"])

        # Second operation: std
        node2 = MockComputeNode(
            input_mapping={"data": sample_data["values"], "operations": ["std"]},
        )
        result2, _ = asyncio.run(stats_handler.execute(node2, context, MagicMock()))
        context.set("std_result", result2["std"])

        # Third operation: describe
        node3 = MockComputeNode(
            input_mapping={"data": sample_data["values"], "operations": ["describe"]},
        )
        result3, _ = asyncio.run(stats_handler.execute(node3, context, MagicMock()))

        assert result1["mean"] == 55.0
        assert result3["describe"]["count"] == 10

    def test_analysis_with_dataframe(self):
        """Test analysis with pandas DataFrame."""
        df = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": [10, 20, 30, 40, 50],
                "C": [100, 200, 300, 400, 500],
            }
        )

        # Compute correlations
        corr_matrix = df.corr()

        assert corr_matrix.shape == (3, 3)
        assert all(corr_matrix.loc["A", :] >= -1)
        assert all(corr_matrix.loc["A", :] <= 1)

        # Compute summary statistics
        summary = df.describe()

        assert summary.shape[0] >= 5  # At least 5 statistical measures
        assert "mean" in summary.index
        assert "std" in summary.index

    def test_grouped_analysis(self):
        """Test grouped analysis operations."""
        df = pd.DataFrame(
            {
                "category": ["A", "A", "B", "B", "C", "C"],
                "value1": [10, 20, 30, 40, 50, 60],
                "value2": [1, 2, 3, 4, 5, 6],
            }
        )

        # Group by category and compute sum
        grouped = df.groupby("category").sum()

        assert grouped.loc["A", "value1"] == 30
        assert grouped.loc["B", "value1"] == 70
        assert grouped.loc["C", "value1"] == 110

        # Group by category and compute mean
        grouped_mean = df.groupby("category").mean()

        assert grouped_mean.loc["A", "value1"] == 15.0
        assert grouped_mean.loc["A", "value2"] == 1.5


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestVisualizationGeneration:
    """Tests for visualization generation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for visualization."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "x": np.arange(50),
                "y": np.random.randn(50).cumsum(),
                "category": np.random.choice(["A", "B", "C"], 50),
                "value": np.random.randint(1, 100, 50),
            }
        )

    def test_generate_line_plot(self, sample_data):
        """Test generating line plot."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sample_data["x"], sample_data["y"])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Line Plot")

        # Check figure exists
        assert fig is not None
        assert ax is not None

        # Save to file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            plt.savefig(tmp_path, dpi=100, bbox_inches="tight")
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 0
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            plt.close(fig)

    def test_generate_scatter_plot(self, sample_data):
        """Test generating scatter plot."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(sample_data["x"], sample_data["y"], alpha=0.6)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Scatter Plot")

        assert fig is not None

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            plt.savefig(tmp_path)
            assert os.path.exists(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            plt.close(fig)

    def test_generate_bar_plot(self, sample_data):
        """Test generating bar plot."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Aggregate data by category
        category_means = sample_data.groupby("category")["value"].mean()

        fig, ax = plt.subplots(figsize=(10, 6))
        category_means.plot(kind="bar", ax=ax)
        ax.set_xlabel("Category")
        ax.set_ylabel("Mean Value")
        ax.set_title("Bar Plot by Category")

        assert fig is not None

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            plt.savefig(tmp_path)
            assert os.path.exists(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            plt.close(fig)

    def test_generate_histogram(self, sample_data):
        """Test generating histogram."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(sample_data["value"], bins=20, edgecolor="black")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.set_title("Histogram")

        assert fig is not None

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            plt.savefig(tmp_path)
            assert os.path.exists(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            plt.close(fig)

    def test_generate_box_plot(self, sample_data):
        """Test generating box plot."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Prepare data for box plot
        box_data = [
            sample_data[sample_data["category"] == cat]["value"].values for cat in ["A", "B", "C"]
        ]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(box_data, labels=["A", "B", "C"])
        ax.set_xlabel("Category")
        ax.set_ylabel("Value")
        ax.set_title("Box Plot by Category")

        assert fig is not None

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            plt.savefig(tmp_path)
            assert os.path.exists(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            plt.close(fig)

    def test_generate_multi_plot_figure(self, sample_data):
        """Test generating figure with multiple subplots."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Line plot
        axes[0, 0].plot(sample_data["x"], sample_data["y"])
        axes[0, 0].set_title("Line Plot")

        # Scatter plot
        axes[0, 1].scatter(sample_data["x"], sample_data["y"])
        axes[0, 1].set_title("Scatter Plot")

        # Histogram
        axes[1, 0].hist(sample_data["value"], bins=20)
        axes[1, 0].set_title("Histogram")

        # Bar plot
        category_counts = sample_data["category"].value_counts()
        axes[1, 1].bar(category_counts.index, category_counts.values)
        axes[1, 1].set_title("Bar Plot")

        plt.tight_layout()

        assert fig is not None
        assert axes.shape == (2, 2)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            plt.savefig(tmp_path, dpi=150)
            assert os.path.exists(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            plt.close(fig)


class TestStatisticalReporting:
    """Tests for statistical reporting."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for reporting."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "numeric1": np.random.randn(100),
                "numeric2": np.random.randn(100) * 2 + 5,
                "numeric3": np.random.randint(0, 100, 100),
                "category": np.random.choice(["A", "B", "C"], 100),
            }
        )

    def test_generate_descriptive_report(self, sample_dataframe):
        """Test generating descriptive statistics report."""
        report = {
            "shape": sample_dataframe.shape,
            "columns": list(sample_dataframe.columns),
            "dtypes": sample_dataframe.dtypes.to_dict(),
            "missing_values": sample_dataframe.isnull().sum().to_dict(),
            "summary_stats": sample_dataframe.describe().to_dict(),
        }

        assert report["shape"] == (100, 4)
        assert len(report["columns"]) == 4
        assert "numeric1" in report["dtypes"]
        assert report["summary_stats"]["numeric1"]["count"] == 100

    def test_generate_correlation_report(self, sample_dataframe):
        """Test generating correlation analysis report."""
        numeric_cols = ["numeric1", "numeric2", "numeric3"]
        corr_matrix = sample_dataframe[numeric_cols].corr()

        report = {
            "correlation_matrix": corr_matrix.to_dict(),
            "high_correlations": [],
        }

        # Find high correlations
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    report["high_correlations"].append(
                        {
                            "var1": numeric_cols[i],
                            "var2": numeric_cols[j],
                            "correlation": corr_val,
                        }
                    )

        assert "correlation_matrix" in report
        assert isinstance(report["high_correlations"], list)

    def test_generate_category_report(self, sample_dataframe):
        """Test generating category analysis report."""
        category_counts = sample_dataframe["category"].value_counts()

        report = {
            "category_distribution": category_counts.to_dict(),
            "num_categories": len(category_counts),
            "most_common": category_counts.index[0],
            "least_common": category_counts.index[-1],
        }

        assert report["num_categories"] == 3
        assert report["most_common"] in ["A", "B", "C"]
        assert report["least_common"] in ["A", "B", "C"]

    def test_export_report_to_json(self, sample_dataframe):
        """Test exporting report to JSON."""
        report = {
            "shape": sample_dataframe.shape,
            "mean_numeric1": float(sample_dataframe["numeric1"].mean()),
            "std_numeric1": float(sample_dataframe["numeric1"].std()),
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
            json.dump(report, tmp, indent=2)

        try:
            # Read and verify
            with open(tmp_path, "r") as f:
                loaded_report = json.load(f)

            # JSON converts tuples to lists
            assert loaded_report["shape"] == [100, 4]
            assert "mean_numeric1" in loaded_report
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_export_report_to_html(self, sample_dataframe):
        """Test exporting report to HTML."""
        html_content = sample_dataframe.describe().to_html()

        assert isinstance(html_content, str)
        assert "<table" in html_content
        assert "numeric1" in html_content

    def test_generate_markdown_report(self, sample_dataframe):
        """Test generating markdown report."""
        report_lines = []
        report_lines.append("# Data Analysis Report")
        report_lines.append("\n## Overview")
        report_lines.append(f"- Shape: {sample_dataframe.shape}")
        report_lines.append(f"- Columns: {', '.join(sample_dataframe.columns)}")

        report_lines.append("\n## Summary Statistics")
        summary = sample_dataframe.describe()
        for col in summary.columns:
            report_lines.append(f"\n### {col}")
            report_lines.append(f"- Mean: {summary[col]['mean']:.2f}")
            report_lines.append(f"- Std: {summary[col]['std']:.2f}")

        report = "\n".join(report_lines)

        assert isinstance(report, str)
        assert "Data Analysis Report" in report
        assert "Overview" in report
        assert "Summary Statistics" in report


class TestHandlersRegistration:
    """Tests for handler registration."""

    def test_handlers_dict_exists(self):
        """Test that HandlerRegistry contains expected handlers."""
        registry = HandlerRegistry.get_instance()
        handlers = registry.list_handlers("dataanalysis")
        assert isinstance(handlers, list)
        assert len(handlers) > 0

    def test_expected_handlers_present(self):
        """Test that expected handlers are present."""
        registry = HandlerRegistry.get_instance()
        handlers = registry.list_handlers("dataanalysis")

        expected_handlers = [
            "stats_compute",
            "ml_training",
            "pycaret_automl",
            "autosklearn_automl",
            "rl_training",
        ]

        for handler_name in expected_handlers:
            assert handler_name in handlers, f"Handler '{handler_name}' not found"

    def test_stats_handler_instance(self):
        """Test that stats_compute handler is properly registered."""
        from victor.dataanalysis.handlers import StatsComputeHandler

        registry = HandlerRegistry.get_instance()
        handler = registry.get_handler("dataanalysis", "stats_compute")
        assert handler is not None
        assert isinstance(handler, StatsComputeHandler)

    def test_handler_callable_methods(self):
        """Test that handlers have callable execute method."""
        from victor.dataanalysis.handlers import StatsComputeHandler

        registry = HandlerRegistry.get_instance()
        handler = registry.get_handler("dataanalysis", "stats_compute")
        assert hasattr(handler, "execute")
        assert callable(handler.execute)
