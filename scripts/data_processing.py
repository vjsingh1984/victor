"""Data processing script.

This script demonstrates a simple sequential data processing workflow:
1. Load data from ``data.csv``.
2. Validate the data – check for missing values and invalid types.
3. If validation fails, clean the data (fill missing values).
4. Transform the data – normalize numeric fields and encode categoricals.
5. Save the processed data to ``processed.csv``.

The script is intentionally lightweight and uses only the standard library plus
``pandas`` and ``numpy`` – both of which are already dependencies of the
project.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path("data.csv")
OUTPUT_PATH = Path("processed.csv")


def load_data(path: Path) -> pd.DataFrame:
    """Load CSV data.

    Parameters
    ----------
    path: Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """
    try:
        df = pd.read_csv(path)
        logger.info("Loaded %d rows and %d columns from %s", *df.shape, path)
        return df
    except Exception as exc:
        logger.error("Failed to load data: %s", exc)
        raise


def validate_data(df: pd.DataFrame) -> bool:
    """Validate dataframe.

    Checks for missing values and basic type consistency.

    Returns
    -------
    bool
        ``True`` if validation passes, ``False`` otherwise.
    """
    # Missing values
    missing = df.isnull().any().any()
    if missing:
        logger.warning("Data contains missing values.")
        return False

    # Simple type checks – numeric columns should be numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning("Column %s is expected to be numeric.", col)
            return False
    logger.info("Data validation passed.")
    return True


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data by filling missing values.

    Numeric columns are filled with the mean, categorical with the mode.
    """
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        mean_val = df_clean[col].mean()
        df_clean[col].fillna(mean_val, inplace=True)
        logger.debug("Filled missing numeric %s with mean %s", col, mean_val)

    # For object/string columns, fill with mode
    cat_cols = df_clean.select_dtypes(include=[object, "category"]).columns
    for col in cat_cols:
        mode_val = df_clean[col].mode(dropna=True)
        if not mode_val.empty:
            df_clean[col].fillna(mode_val[0], inplace=True)
            logger.debug("Filled missing categorical %s with mode %s", col, mode_val[0])
        else:
            df_clean[col].fillna("Unknown", inplace=True)
            logger.debug("Filled missing categorical %s with 'Unknown'", col)
    return df_clean


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Transform data.

    Normalizes numeric columns using min‑max scaling and one‑hot encodes
    categorical columns.
    """
    df_trans = df.copy()
    numeric_cols = df_trans.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        min_vals = df_trans[numeric_cols].min()
        max_vals = df_trans[numeric_cols].max()
        # Avoid division by zero
        denom = (max_vals - min_vals).replace(0, 1)
        df_trans[numeric_cols] = (df_trans[numeric_cols] - min_vals) / denom
        logger.debug("Normalized numeric columns: %s", list(numeric_cols))

    cat_cols = df_trans.select_dtypes(include=[object, "category"]).columns
    if len(cat_cols) > 0:
        df_trans = pd.get_dummies(df_trans, columns=cat_cols, drop_first=True)
        logger.debug("One‑hot encoded categorical columns: %s", list(cat_cols))

    return df_trans


def save_data(df: pd.DataFrame, path: Path) -> None:
    """Save dataframe to CSV."""
    try:
        df.to_csv(path, index=False)
        logger.info("Saved processed data to %s", path)
    except Exception as exc:
        logger.error("Failed to save data: %s", exc)
        raise


def main() -> None:
    df = load_data(DATA_PATH)
    if not validate_data(df):
        logger.info("Cleaning data...")
        df = clean_data(df)
        if not validate_data(df):
            logger.error("Data still invalid after cleaning. Aborting.")
            return
    df = transform_data(df)
    save_data(df, OUTPUT_PATH)


if __name__ == "__main__":
    main()

