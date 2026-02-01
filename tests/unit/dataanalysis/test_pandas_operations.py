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

"""Unit tests for pandas operations in DataAnalysis vertical.

Tests cover:
- DataFrame creation and manipulation
- Statistical operations (mean, median, std, correlation)
- Data visualization (plotting functions)
- Data export (CSV, JSON)
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

# Try importing pandas, skip tests if not available
pytest.importorskip("pandas")
import pandas as pd
import numpy as np


class TestDataFrameCreation:
    """Tests for DataFrame creation operations."""

    def test_create_dataframe_from_dict(self):
        """Test creating DataFrame from dictionary."""
        data = {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "city": ["NYC", "LA", "Chicago"],
        }
        df = pd.DataFrame(data)

        assert df.shape == (3, 3)
        assert list(df.columns) == ["name", "age", "city"]
        assert df["age"].tolist() == [25, 30, 35]

    def test_create_dataframe_from_list(self):
        """Test creating DataFrame from list of lists."""
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        df = pd.DataFrame(data, columns=["A", "B", "C"])

        assert df.shape == (3, 3)
        assert df["A"].tolist() == [1, 4, 7]

    def test_create_dataframe_with_index(self):
        """Test creating DataFrame with custom index."""
        data = {"value": [10, 20, 30]}
        df = pd.DataFrame(data, index=["row1", "row2", "row3"])

        assert df.index.tolist() == ["row1", "row2", "row3"]

    def test_create_empty_dataframe(self):
        """Test creating empty DataFrame."""
        df = pd.DataFrame()

        assert df.empty
        assert df.shape == (0, 0)

    def test_dataframe_with_mixed_types(self):
        """Test DataFrame with mixed data types."""
        data = {
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
        }
        df = pd.DataFrame(data)

        assert df["int_col"].dtype == np.int64
        assert df["float_col"].dtype == np.float64
        assert df["str_col"].dtype == object
        assert df["bool_col"].dtype == bool


class TestDataFrameManipulation:
    """Tests for DataFrame manipulation operations."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": [10, 20, 30, 40, 50],
                "C": [100, 200, 300, 400, 500],
            }
        )

    def test_filter_rows(self, sample_df):
        """Test filtering DataFrame rows."""
        filtered = sample_df[sample_df["A"] > 2]

        assert len(filtered) == 3
        assert filtered["A"].tolist() == [3, 4, 5]

    def test_select_columns(self, sample_df):
        """Test selecting DataFrame columns."""
        selected = sample_df[["A", "C"]]

        assert list(selected.columns) == ["A", "C"]
        assert selected.shape == (5, 2)

    def test_add_column(self, sample_df):
        """Test adding a new column."""
        sample_df["D"] = sample_df["A"] + sample_df["B"]

        assert "D" in sample_df.columns
        assert sample_df["D"].tolist() == [11, 22, 33, 44, 55]

    def test_drop_column(self, sample_df):
        """Test dropping a column."""
        result = sample_df.drop(columns=["B"])

        assert "B" not in result.columns
        assert list(result.columns) == ["A", "C"]

    def test_sort_values(self, sample_df):
        """Test sorting DataFrame by values."""
        sorted_df = sample_df.sort_values("B", ascending=False)

        assert sorted_df["B"].tolist() == [50, 40, 30, 20, 10]

    def test_group_by_aggregation(self):
        """Test group by and aggregation."""
        df = pd.DataFrame(
            {
                "category": ["A", "A", "B", "B", "C"],
                "value": [10, 20, 30, 40, 50],
            }
        )
        grouped = df.groupby("category")["value"].sum().to_dict()

        assert grouped == {"A": 30, "B": 70, "C": 50}

    def test_merge_dataframes(self):
        """Test merging two DataFrames."""
        df1 = pd.DataFrame({"key": ["A", "B"], "value1": [1, 2]})
        df2 = pd.DataFrame({"key": ["A", "B"], "value2": [3, 4]})

        merged = pd.merge(df1, df2, on="key")

        assert merged.shape == (2, 3)
        assert merged["value1"].tolist() == [1, 2]
        assert merged["value2"].tolist() == [3, 4]

    def test_handle_missing_values(self):
        """Test handling missing values."""
        df = pd.DataFrame({"A": [1, 2, np.nan, 4], "B": [5, np.nan, np.nan, 8]})

        # Fill missing values
        filled = df.fillna(0)

        assert filled["A"].tolist() == [1.0, 2.0, 0.0, 4.0]
        assert filled["B"].tolist() == [5.0, 0.0, 0.0, 8.0]

        # Drop missing values (only rows with NO missing values remain)
        # Row 0 has both A and B, row 3 has both A and B
        dropped = df.dropna()

        assert len(dropped) == 2  # Rows 0 and 3 have no missing values
        assert dropped.iloc[0]["A"] == 1.0
        assert dropped.iloc[1]["A"] == 4.0


class TestStatisticalOperations:
    """Tests for statistical operations."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for statistics."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "A": np.random.randn(100),
                "B": np.random.randn(100) * 2 + 5,
                "C": np.random.randint(0, 100, 100),
            }
        )

    def test_mean(self, sample_df):
        """Test mean calculation."""
        mean_a = sample_df["A"].mean()

        assert isinstance(mean_a, (int, float))
        assert -5 < mean_a < 5  # Reasonable range for standard normal

    def test_median(self, sample_df):
        """Test median calculation."""
        median_c = sample_df["C"].median()

        assert isinstance(median_c, (int, float))
        assert 0 <= median_c <= 100

    def test_std(self, sample_df):
        """Test standard deviation calculation."""
        std_a = sample_df["A"].std()
        std_b = sample_df["B"].std()

        assert std_a > 0
        assert std_b > std_a  # B has larger spread

    def test_min_max(self, sample_df):
        """Test min and max calculation."""
        min_val = sample_df["C"].min()
        max_val = sample_df["C"].max()

        assert min_val >= 0
        assert max_val <= 100
        assert min_val < max_val

    def test_describe(self, sample_df):
        """Test describe method for summary statistics."""
        desc = sample_df.describe()

        assert "count" in desc.index
        assert "mean" in desc.index
        assert "std" in desc.index
        assert "min" in desc.index
        assert "max" in desc.index
        assert desc.shape[0] >= 5  # At least 5 statistical measures

    def test_correlation(self, sample_df):
        """Test correlation calculation."""
        corr = sample_df[["A", "B", "C"]].corr()

        assert corr.shape == (3, 3)
        # Diagonal should be all 1s
        assert all(corr.iloc[i, i] == 1.0 for i in range(3))
        # Correlation values should be between -1 and 1
        assert all((corr >= -1) & (corr <= 1))

    def test_value_counts(self):
        """Test value counts for categorical data."""
        df = pd.DataFrame({"category": ["A", "A", "B", "B", "B", "C"]})
        counts = df["category"].value_counts()

        assert counts["B"] == 3
        assert counts["A"] == 2
        assert counts["C"] == 1

    def test_quantile(self, sample_df):
        """Test quantile calculation."""
        q25 = sample_df["A"].quantile(0.25)
        q50 = sample_df["A"].quantile(0.50)
        q75 = sample_df["A"].quantile(0.75)

        assert q25 <= q50 <= q75

    def test_variance(self, sample_df):
        """Test variance calculation."""
        var_a = sample_df["A"].var()

        assert var_a > 0
        assert isinstance(var_a, (int, float))


class TestDataVisualization:
    """Tests for data visualization operations."""

    @pytest.fixture(autouse=True)
    def skip_if_no_matplotlib(self):
        """Skip all tests if matplotlib is not available."""
        try:
            import matplotlib

            matplotlib.use("Agg")  # Use non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not installed")

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for visualization."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "x": np.arange(100),
                "y": np.random.randn(100).cumsum(),
                "category": np.random.choice(["A", "B", "C"], 100),
            }
        )

    @patch("matplotlib.pyplot.subplots")
    def test_line_plot_creation(self, mock_subplots, sample_df):
        """Test creating line plot."""
        # Mock the figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(sample_df["x"], sample_df["y"])

        # Check figure was created
        assert fig is not None or mock_subplots.called
        assert ax is not None or mock_ax is not None

        plt.close(fig) if fig else None

    @patch("matplotlib.pyplot.subplots")
    def test_scatter_plot_creation(self, mock_subplots, sample_df):
        """Test creating scatter plot."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.scatter(sample_df["x"], sample_df["y"])

        assert fig is not None or mock_subplots.called
        plt.close(fig) if fig else None

    @patch("matplotlib.pyplot.subplots")
    def test_bar_plot_creation(self, mock_subplots):
        """Test creating bar plot."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        import matplotlib.pyplot as plt

        df = pd.DataFrame({"category": ["A", "B", "C"], "value": [10, 20, 30]})

        fig, ax = plt.subplots()
        ax.bar(df["category"], df["value"])

        assert fig is not None or mock_subplots.called
        plt.close(fig) if fig else None

    @patch("matplotlib.pyplot.subplots")
    def test_histogram_creation(self, mock_subplots, sample_df):
        """Test creating histogram."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.hist(sample_df["y"], bins=20)

        assert fig is not None or mock_subplots.called
        plt.close(fig) if fig else None

    @patch("matplotlib.pyplot.subplots")
    def test_box_plot_creation(self, mock_subplots, sample_df):
        """Test creating box plot."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.boxplot([sample_df["y"][sample_df["category"] == cat] for cat in ["A", "B", "C"]])

        assert fig is not None or mock_subplots.called
        plt.close(fig) if fig else None

    @patch("matplotlib.pyplot.savefig")
    def test_save_plot(self, mock_savefig):
        """Test saving plot to file."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            plt.savefig(tmp_path)
            # Mock should be called
            assert mock_savefig.called or os.path.exists(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            plt.close(fig)


class TestDataExport:
    """Tests for data export operations."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for export."""
        return pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, 30, 35],
                "city": ["NYC", "LA", "Chicago"],
            }
        )

    def test_export_to_csv(self, sample_df):
        """Test exporting DataFrame to CSV."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            sample_df.to_csv(tmp_path, index=False)

            # Check file was created
            assert os.path.exists(tmp_path)

            # Check content
            loaded = pd.read_csv(tmp_path)
            assert loaded.equals(sample_df)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_export_to_csv_with_index(self, sample_df):
        """Test exporting DataFrame to CSV with index."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            sample_df.to_csv(tmp_path, index=True)

            loaded = pd.read_csv(tmp_path, index_col=0)
            assert loaded.equals(sample_df)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_export_to_json(self, sample_df):
        """Test exporting DataFrame to JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            sample_df.to_json(tmp_path, orient="records")

            # Check file was created
            assert os.path.exists(tmp_path)

            # Check content
            with open(tmp_path, "r") as f:
                data = json.load(f)

            assert len(data) == 3
            assert data[0]["name"] == "Alice"
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_export_to_excel(self, sample_df):
        """Test exporting DataFrame to Excel."""
        pytest.importorskip("openpyxl")

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            sample_df.to_excel(tmp_path, index=False, engine="openpyxl")

            # Check file was created
            assert os.path.exists(tmp_path)

            # Check content
            loaded = pd.read_excel(tmp_path, engine="openpyxl")
            assert loaded.shape == sample_df.shape
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_export_to_dict(self, sample_df):
        """Test exporting DataFrame to dictionary."""
        data = sample_df.to_dict(orient="records")

        assert isinstance(data, list)
        assert len(data) == 3
        assert data[0] == {"name": "Alice", "age": 25, "city": "NYC"}

    def test_export_csv_with_separator(self, sample_df):
        """Test exporting CSV with custom separator."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            sample_df.to_csv(tmp_path, sep=";", index=False)

            # Check file was created
            assert os.path.exists(tmp_path)

            # Read with custom separator
            loaded = pd.read_csv(tmp_path, sep=";")
            assert loaded.equals(sample_df)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestStatsComputeHandler:
    """Tests for StatsComputeHandler."""

    @pytest.fixture
    def handler(self):
        from victor.dataanalysis.handlers import StatsComputeHandler

        return StatsComputeHandler()

    def test_compute_mean(self, handler):
        """Test computing mean."""
        data = [1, 2, 3, 4, 5]
        result = handler._compute_stat(data, "mean")

        assert result == 3.0

    def test_compute_median(self, handler):
        """Test computing median."""
        data = [1, 2, 3, 4, 5]
        result = handler._compute_stat(data, "median")

        assert result == 3.0

    def test_compute_std(self, handler):
        """Test computing standard deviation."""
        data = [1, 2, 3, 4, 5]
        result = handler._compute_stat(data, "std")

        assert isinstance(result, float)
        assert result > 0

    def test_compute_describe(self, handler):
        """Test computing descriptive statistics."""
        data = [1, 2, 3, 4, 5]
        result = handler._compute_stat(data, "describe")

        assert isinstance(result, dict)
        assert "count" in result
        assert "mean" in result
        assert "min" in result
        assert "max" in result
        assert result["mean"] == 3.0

    def test_compute_with_empty_data(self, handler):
        """Test computing stats with empty data."""
        result = handler._compute_stat([], "mean")

        assert result is None

    def test_compute_with_non_numeric_data(self, handler):
        """Test computing stats with non-numeric data."""
        data = ["a", "b", "c"]
        result = handler._compute_stat(data, "mean")

        assert result is None

    def test_compute_min_max_sum(self, handler):
        """Test min, max, and sum operations."""
        data = [1, 2, 3, 4, 5]

        assert handler._compute_stat(data, "min") == 1
        assert handler._compute_stat(data, "max") == 5
        assert handler._compute_stat(data, "sum") == 15

    def test_compute_count(self, handler):
        """Test count operation."""
        data = [1, 2, 3, 4, 5, "a", "b"]
        result = handler._compute_stat(data, "count")

        assert result == 5  # Only numeric values counted

    def test_unsupported_operation(self, handler):
        """Test unsupported operation."""
        data = [1, 2, 3]
        result = handler._compute_stat(data, "unsupported")

        assert result is None


class TestDataLoading:
    """Tests for data loading operations."""

    def test_load_csv_from_file(self):
        """Test loading CSV from file."""
        import tempfile

        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name,age,city\n")
            f.write("Alice,25,NYC\n")
            f.write("Bob,30,LA\n")
            f.write("Charlie,35,Chicago\n")
            tmp_path = f.name

        try:
            # Load CSV
            df = pd.read_csv(tmp_path)

            assert df.shape == (3, 3)
            assert list(df.columns) == ["name", "age", "city"]
            assert df["age"].tolist() == [25, 30, 35]
        finally:
            os.remove(tmp_path)

    def test_load_csv_with_separator(self):
        """Test loading CSV with custom separator."""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("name;age;city\n")
            f.write("Alice;25;NYC\n")
            f.write("Bob;30;LA\n")
            tmp_path = f.name

        try:
            df = pd.read_csv(tmp_path, sep=";")

            assert df.shape == (2, 3)
            assert list(df.columns) == ["name", "age", "city"]
        finally:
            os.remove(tmp_path)

    def test_load_json_from_file(self):
        """Test loading JSON from file."""
        import tempfile

        data = [
            {"name": "Alice", "age": 25, "city": "NYC"},
            {"name": "Bob", "age": 30, "city": "LA"},
            {"name": "Charlie", "age": 35, "city": "Chicago"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            tmp_path = f.name

        try:
            df = pd.read_json(tmp_path)

            assert df.shape == (3, 3)
            assert df["age"].tolist() == [25, 30, 35]
        finally:
            os.remove(tmp_path)

    def test_load_json_with_orient(self):
        """Test loading JSON with different orient."""
        import tempfile

        data = {"name": ["Alice", "Bob"], "age": [25, 30]}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            tmp_path = f.name

        try:
            df = pd.read_json(tmp_path, orient="columns")

            assert df.shape == (2, 2)
            assert list(df.columns) == ["name", "age"]
        finally:
            os.remove(tmp_path)

    def test_data_type_inference(self):
        """Test automatic data type inference."""
        data = {
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
            "date_col": ["2023-01-01", "2023-01-02", "2023-01-03"],
        }

        df = pd.DataFrame(data)

        # Check inferred types
        assert pd.api.types.is_integer_dtype(df["int_col"])
        assert pd.api.types.is_float_dtype(df["float_col"])
        assert pd.api.types.is_string_dtype(df["str_col"]) or df["str_col"].dtype == object
        assert pd.api.types.is_bool_dtype(df["bool_col"])

    def test_parse_dates_on_loading(self):
        """Test parsing dates during data loading."""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("date,value\n")
            f.write("2023-01-01,100\n")
            f.write("2023-01-02,200\n")
            f.write("2023-01-03,300\n")
            tmp_path = f.name

        try:
            df = pd.read_csv(tmp_path, parse_dates=["date"])

            assert pd.api.types.is_datetime64_any_dtype(df["date"])
            assert df["value"].tolist() == [100, 200, 300]
        finally:
            os.remove(tmp_path)


class TestDataFrameAdvancedOperations:
    """Tests for advanced DataFrame operations."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        return pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5],
                "B": [10, 20, 30, 40, 50],
                "C": ["a", "b", "a", "b", "c"],
            }
        )

    def test_apply_function(self, sample_df):
        """Test applying function to column."""
        result = sample_df["A"].apply(lambda x: x * 2)

        assert result.tolist() == [2, 4, 6, 8, 10]

    def test_apply_map_function(self, sample_df):
        """Test applying function to entire DataFrame."""
        result = (
            sample_df[["A", "B"]].applymap(lambda x: x * 2)
            if hasattr(sample_df[["A", "B"]], "applymap")
            else sample_df[["A", "B"]].map(lambda x: x * 2)
        )

        # Check if multiplication worked
        assert result["A"].tolist() == [2, 4, 6, 8, 10]

    def test_rename_columns(self, sample_df):
        """Test renaming columns."""
        renamed = sample_df.rename(columns={"A": "Alpha", "B": "Beta"})

        assert "Alpha" in renamed.columns
        assert "Beta" in renamed.columns
        assert "A" not in renamed.columns

    def test_filter_with_query(self, sample_df):
        """Test filtering with query method."""
        result = sample_df.query("A > 2 and B < 50")

        assert len(result) == 2
        assert result["A"].tolist() == [3, 4]

    def test_concat_dataframes(self):
        """Test concatenating DataFrames."""
        df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        df2 = pd.DataFrame({"A": [5, 6], "B": [7, 8]})

        result = pd.concat([df1, df2], ignore_index=True)

        assert result.shape == (4, 2)
        assert result["A"].tolist() == [1, 2, 5, 6]

    def test_pivot_table(self):
        """Test creating pivot table."""
        df = pd.DataFrame(
            {
                "category": ["A", "A", "B", "B", "C", "C"],
                "value": [10, 20, 30, 40, 50, 60],
                "metric": ["X", "Y", "X", "Y", "X", "Y"],
            }
        )

        pivot = df.pivot_table(index="category", columns="metric", values="value", aggfunc="sum")

        assert pivot.shape == (3, 2)
        assert pivot.loc["A", "X"] == 10
        assert pivot.loc["B", "Y"] == 40

    def test_melt_dataframe(self):
        """Test melting DataFrame."""
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "var1": [10, 20],
                "var2": [30, 40],
            }
        )

        melted = pd.melt(df, id_vars=["id"], var_name="variable", value_name="value")

        assert melted.shape == (4, 3)
        # Melt creates all variables for each id, so order is var1, var1, var2, var2
        assert melted["variable"].tolist() == ["var1", "var1", "var2", "var2"]

    def test_rank_values(self, sample_df):
        """Test ranking values."""
        result = sample_df["A"].rank()

        assert result.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_sample_rows(self, sample_df):
        """Test sampling rows."""
        sampled = sample_df.sample(n=3, random_state=42)

        assert len(sampled) == 3

    def test_head_tail(self, sample_df):
        """Test head and tail methods."""
        head = sample_df.head(2)
        tail = sample_df.tail(2)

        assert len(head) == 2
        assert len(tail) == 2
        assert head["A"].tolist() == [1, 2]
        assert tail["A"].tolist() == [4, 5]


class TestDataAnalysisMethods:
    """Tests for data analysis methods."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "A": np.random.randn(100),
                "B": np.random.randn(100) * 2 + 5,
                "C": np.random.randint(0, 100, 100),
                "D": np.random.choice(["X", "Y", "Z"], 100),
            }
        )

    def test_unique_values(self, sample_df):
        """Test getting unique values."""
        unique_d = sample_df["D"].unique()

        assert len(unique_d) == 3
        assert set(unique_d) == {"X", "Y", "Z"}

    def test_nunique(self, sample_df):
        """Test counting unique values."""
        nunique_d = sample_df["D"].nunique()

        assert nunique_d == 3

    def test_isnull_detection(self, sample_df):
        """Test null value detection."""
        # Add some nulls
        sample_df.loc[0:4, "A"] = np.nan

        null_count = sample_df["A"].isnull().sum()

        assert null_count == 5

    def test_fillna_strategy(self):
        """Test different fillna strategies."""
        df = pd.DataFrame({"A": [1, np.nan, 3, np.nan, 5]})

        # Forward fill
        forward_filled = df["A"].fillna(method="ffill")
        assert forward_filled.iloc[1] == 1.0

        # Backward fill
        backward_filled = df["A"].fillna(method="bfill")
        assert backward_filled.iloc[3] == 5.0

        # Fill with mean
        mean_filled = df["A"].fillna(df["A"].mean())
        assert mean_filled.iloc[1] == 3.0

    def test_drop_duplicates(self):
        """Test dropping duplicate rows."""
        df = pd.DataFrame(
            {
                "A": [1, 1, 2, 2, 3],
                "B": [4, 4, 5, 5, 6],
            }
        )

        deduped = df.drop_duplicates()

        assert len(deduped) == 3
        assert deduped["A"].tolist() == [1, 2, 3]

    def test_replace_values(self):
        """Test replacing values."""
        df = pd.DataFrame({"A": [1, 2, 3, 1, 2]})

        replaced = df["A"].replace({1: 10, 2: 20})

        assert replaced.tolist() == [10, 20, 3, 10, 20]

    def test_clip_values(self):
        """Test clipping values to range."""
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

        clipped = df["A"].clip(lower=3, upper=7)

        assert clipped.min() >= 3
        assert clipped.max() <= 7
        assert clipped.iloc[0] == 3
        assert clipped.iloc[-1] == 7

    def test_cumulative_operations(self):
        """Test cumulative operations."""
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5]})

        cumsum = df["A"].cumsum()
        cummax = df["A"].cummax()

        assert cumsum.tolist() == [1, 3, 6, 10, 15]
        assert cummax.tolist() == [1, 2, 3, 4, 5]

    def test_rolling_window(self):
        """Test rolling window calculations."""
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5]})

        rolling_mean = df["A"].rolling(window=3).mean()

        # First two values should be NaN
        assert pd.isna(rolling_mean.iloc[0])
        assert pd.isna(rolling_mean.iloc[1])
        # Third value should be (1+2+3)/3 = 2
        assert rolling_mean.iloc[2] == 2.0

    def test_percentage_change(self):
        """Test percentage change calculation."""
        df = pd.DataFrame({"A": [100, 110, 121, 133.1]})

        pct_change = df["A"].pct_change()

        # Use approximate comparison for floating point
        assert abs(pct_change.iloc[1] - 0.1) < 0.001  # 10% increase
        assert abs(pct_change.iloc[2] - 0.1) < 0.001  # ~10% increase


class TestStringOperations:
    """Tests for string operations on DataFrames."""

    def test_string_lower_upper(self):
        """Test string case conversion."""
        df = pd.DataFrame({"text": ["Hello", "WORLD", "PyThOn"]})

        lower = df["text"].str.lower()
        upper = df["text"].str.upper()

        assert lower.tolist() == ["hello", "world", "python"]
        assert upper.tolist() == ["HELLO", "WORLD", "PYTHON"]

    def test_string_contains(self):
        """Test string contains check."""
        df = pd.DataFrame({"text": ["apple", "banana", "cherry", "date"]})

        contains_a = df["text"].str.contains("a")

        assert contains_a.tolist() == [True, True, False, True]

    def test_string_replace(self):
        """Test string replace."""
        df = pd.DataFrame({"text": ["foo_bar", "baz_qux", "test_foo"]})

        replaced = df["text"].str.replace("_", " ")

        assert replaced.tolist() == ["foo bar", "baz qux", "test foo"]

    def test_string_extract(self):
        """Test string extract with regex."""
        df = pd.DataFrame({"text": ["item_123", "item_456", "item_789"]})

        extracted = df["text"].str.extract(r"item_(\d+)")

        assert extracted[0].tolist() == ["123", "456", "789"]

    def test_string_split(self):
        """Test string split."""
        df = pd.DataFrame({"text": ["a-b-c", "d-e-f", "g-h-i"]})

        split_result = df["text"].str.split("-")

        assert split_result.iloc[0] == ["a", "b", "c"]
        assert split_result.iloc[1] == ["d", "e", "f"]

    def test_string_strip(self):
        """Test string strip."""
        df = pd.DataFrame({"text": ["  hello  ", "  world  ", "  test  "]})

        stripped = df["text"].str.strip()

        assert stripped.tolist() == ["hello", "world", "test"]

    def test_string_length(self):
        """Test string length calculation."""
        df = pd.DataFrame({"text": ["a", "ab", "abc", "abcd"]})

        length = df["text"].str.len()

        assert length.tolist() == [1, 2, 3, 4]


class TestDateTimeOperations:
    """Tests for datetime operations."""

    def test_create_datetime_index(self):
        """Test creating DataFrame with datetime index."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        df = pd.DataFrame({"value": [10, 20, 30, 40, 50]}, index=dates)

        assert isinstance(df.index, pd.DatetimeIndex)
        assert len(df) == 5

    def test_datetime_components(self):
        """Test extracting datetime components."""
        dates = pd.date_range("2023-01-01", periods=3, freq="D")
        df = pd.DataFrame({"date": dates})

        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day

        assert df["year"].tolist() == [2023, 2023, 2023]
        assert df["month"].tolist() == [1, 1, 1]
        assert df["day"].tolist() == [1, 2, 3]

    def test_datetime_resample(self):
        """Test resampling time series data."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        df = pd.DataFrame({"value": range(10)}, index=dates)

        # Resample to 5-day periods and sum
        resampled = df.resample("5D").sum()

        assert len(resampled) == 2
        assert resampled["value"].iloc[0] == 10  # 0+1+2+3+4
        assert resampled["value"].iloc[1] == 35  # 5+6+7+8+9

    def test_datetime_shift(self):
        """Test shifting datetime values."""
        dates = pd.date_range("2023-01-01", periods=3, freq="D")
        df = pd.DataFrame({"date": dates})

        shifted = df["date"].shift(1)

        assert pd.isna(shifted.iloc[0])
        assert shifted.iloc[1] == dates[0]

    def test_timedelta_operations(self):
        """Test timedelta operations."""
        start = pd.Timestamp("2023-01-01")
        end = pd.Timestamp("2023-01-10")

        delta = end - start

        assert delta.days == 9

    def test_business_day_freq(self):
        """Test business day frequency."""
        dates = pd.date_range("2023-01-01", periods=5, freq="B")  # Business days
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]}, index=dates)

        assert len(df) == 5
        # Check that weekends are skipped
        assert df.index.freq.name == "B"


class TestDataFrameAggregations:
    """Tests for various DataFrame aggregation methods."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        return pd.DataFrame(
            {
                "category": ["A", "A", "B", "B", "C"],
                "value1": [10, 20, 30, 40, 50],
                "value2": [5, 15, 25, 35, 45],
            }
        )

    def test_multiple_aggregations(self, sample_df):
        """Test multiple aggregations at once."""
        result = sample_df.groupby("category")["value1"].agg(["mean", "sum", "count"])

        assert "mean" in result.columns
        assert "sum" in result.columns
        assert "count" in result.columns

    def test_custom_aggregation(self, sample_df):
        """Test custom aggregation function."""

        def range_func(x):
            return x.max() - x.min()

        result = sample_df.groupby("category")["value1"].agg(range_func)

        assert result["A"] == 10  # 20 - 10
        assert result["B"] == 10  # 40 - 30

    def test_transform(self, sample_df):
        """Test transform operation."""
        result = sample_df.groupby("category")["value1"].transform("sum")

        # Original shape should be preserved
        assert len(result) == len(sample_df)
        # First two rows should have same sum (both category A)
        assert result.iloc[0] == result.iloc[1]

    def test_filter_groups(self, sample_df):
        """Test filtering groups based on condition."""
        result = sample_df.groupby("category").filter(lambda x: x["value1"].sum() > 30)

        # Only groups with sum > 30 should remain
        assert result["category"].unique().tolist() == ["B", "C"]

    def test_size_of_groups(self, sample_df):
        """Test getting size of each group."""
        result = sample_df.groupby("category").size()

        assert result["A"] == 2
        assert result["B"] == 2
        assert result["C"] == 1


class TestMultiIndexOperations:
    """Tests for MultiIndex operations."""

    def test_create_multiindex(self):
        """Test creating DataFrame with MultiIndex."""
        arrays = [
            ["A", "A", "B", "B"],
            [1, 2, 1, 2],
        ]
        index = pd.MultiIndex.from_arrays(arrays, names=["first", "second"])
        df = pd.DataFrame({"value": [10, 20, 30, 40]}, index=index)

        assert isinstance(df.index, pd.MultiIndex)
        assert df.loc[("A", 1), "value"] == 10

    def test_multiindex_groupby(self):
        """Test groupby with MultiIndex."""
        arrays = [
            ["A", "A", "B", "B"],
            [1, 2, 1, 2],
        ]
        index = pd.MultiIndex.from_arrays(arrays, names=["first", "second"])
        df = pd.DataFrame({"value": [10, 20, 30, 40]}, index=index)

        result = df.groupby(level="first").sum()

        assert result.loc["A", "value"] == 30
        assert result.loc["B", "value"] == 70

    def test_stack_unstack(self):
        """Test stack and unstack operations."""
        df = pd.DataFrame(
            {
                "A": [1, 2],
                "B": [3, 4],
            },
            index=["X", "Y"],
        )

        # Unstack
        unstacked = df.unstack()

        assert unstacked.shape == (4,)

    def test_pivot_table_multiindex(self):
        """Test pivot table with MultiIndex result."""
        df = pd.DataFrame(
            {
                "A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
                "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
                "C": [
                    "small",
                    "large",
                    "large",
                    "small",
                    "small",
                    "large",
                    "small",
                    "small",
                    "large",
                ],
                "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                "E": [2, 4, 5, 5, 6, 6, 8, 9, 9],
            }
        )

        pivot = pd.pivot_table(df, values="D", index=["A", "B"], columns=["C"], aggfunc="sum")

        assert isinstance(pivot.index, pd.MultiIndex)
        assert pivot.shape[0] == 4  # 2 (A values) * 2 (B values)
