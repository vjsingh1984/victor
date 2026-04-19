"""Data processing pipeline.

This script implements a sequential workflow:
1. Load data from ``data.csv``.
2. Validate the data – check for missing values and invalid types.
3. If validation fails, clean the data (numeric: mean, categorical: mode).
4. Transform the data – normalize numeric fields and one‑hot encode categoricals.
5. Save the processed data to ``processed.csv``.

The script stops at the first failure and prints a clear error message.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data.csv")
PROCESSED_PATH = Path("processed.csv")


def load_data(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns from {path}")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")


def validate_data(df: pd.DataFrame) -> bool:
    # Check for missing values
    missing = df.isnull().any().any()
    if missing:
        print("Validation failed: missing values detected.")
        return False
    # Check for invalid types – here we simply ensure numeric columns are numeric
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Validation failed: column {col} is not numeric.")
            return False
    print("Validation succeeded.")
    return True


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning data...")
    # Fill numeric with mean, categorical with mode
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=["number"]).columns
    cat_cols = df_clean.select_dtypes(exclude=["number"]).columns
    for col in numeric_cols:
        mean_val = df_clean[col].mean()
        df_clean[col].fillna(mean_val, inplace=True)
    for col in cat_cols:
        mode_val = (
            df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else None
        )
        df_clean[col].fillna(mode_val, inplace=True)
    # After cleaning, ensure no missing values remain
    if df_clean.isnull().any().any():
        raise RuntimeError("Cleaning failed: still missing values.")
    print("Cleaning completed.")
    return df_clean


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Transforming data...")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if cat_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            )
        )

    preprocessor = ColumnTransformer(transformers, remainder="drop")
    transformed_array = preprocessor.fit_transform(df)
    # Build new column names
    num_features = numeric_cols
    cat_features = (
        preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)
        if cat_cols
        else []
    )
    all_features = num_features + cat_features.tolist()
    df_transformed = pd.DataFrame(transformed_array, columns=all_features)
    print(f"Transformed to {df_transformed.shape[1]} features.")
    return df_transformed


def save_data(df: pd.DataFrame, path: Path):
    try:
        df.to_csv(path, index=False)
        print(f"Saved processed data to {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save data: {e}")


def main():
    try:
        df = load_data(DATA_PATH)
        if not validate_data(df):
            df = clean_data(df)
            if not validate_data(df):
                print("Error: Data still invalid after cleaning.")
                sys.exit(1)
        df_transformed = transform_data(df)
        save_data(df_transformed, PROCESSED_PATH)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
