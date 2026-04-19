#!/usr/bin/env python3
"""
Sequential data processing workflow:

1. Load data from data.csv
2. Validate the data (check for missing values, invalid types)
3. If validation fails, clean the data (fill/remove)
4. Transform the data (normalize numeric fields, encode categoricals)
5. Save the processed data to processed.csv

Each step executes only if the previous step succeeded. If any step fails, an error message is printed and the script stops.

Usage:
    python scripts/data_processing.py
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

DATA_PATH = Path("data.csv")
PROCESSED_PATH = Path("processed.csv")


def load_data(path: Path) -> pd.DataFrame:
    """Load CSV into a DataFrame.
    Raises FileNotFoundError if the file does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV: {exc}") from exc
    return df


def validate_data(df: pd.DataFrame) -> bool:
    """Return True if data passes validation.
    Validation checks:
    - No missing values
    - All columns have consistent types (numeric columns are numeric, others are string/categorical)
    """
    # Check missing values
    if df.isnull().values.any():
        return False
    # Check numeric columns
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        if pd.api.types.is_object_dtype(df[col]):
            continue
        # For any other dtype, consider invalid
        return False
    return True


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data by dropping rows with missing values and converting
    numeric-like strings to floats.
    """
    # Drop rows with any missing values
    df = df.dropna()
    # Convert numeric-like strings to numeric types where possible
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass  # keep as object if conversion fails
    return df


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize numeric fields and one-hot encode categoricals.
    Returns transformed DataFrame.
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline([("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="passthrough",
    )

    # Fit and transform
    transformed_array = preprocessor.fit_transform(df)
    # Build column names
    num_features = numeric_cols
    cat_features = (
        preprocessor.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(categorical_cols)
    )
    all_features = list(num_features) + list(cat_features)
    transformed_df = pd.DataFrame(transformed_array, columns=all_features)
    return transformed_df


def save_data(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to CSV."""
    try:
        df.to_csv(path, index=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to write CSV: {exc}") from exc


def main() -> None:
    try:
        df = load_data(DATA_PATH)
        print("Step 1: Data loaded successfully.")
    except Exception as exc:
        print(f"Error in Step 1: {exc}")
        sys.exit(1)

    if not validate_data(df):
        print("Step 2: Validation failed. Cleaning data...")
        try:
            df = clean_data(df)
            print("Step 3: Data cleaned.")
        except Exception as exc:
            print(f"Error in Step 3: {exc}")
            sys.exit(1)
    else:
        print("Step 2: Validation succeeded.")

    try:
        df = transform_data(df)
        print("Step 4: Data transformed.")
    except Exception as exc:
        print(f"Error in Step 4: {exc}")
        sys.exit(1)

    try:
        save_data(df, PROCESSED_PATH)
        print("Step 5: Processed data saved to", PROCESSED_PATH)
    except Exception as exc:
        print(f"Error in Step 5: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
