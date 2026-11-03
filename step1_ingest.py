"""
============================================================
STEP 1 — DATA INGESTION & CLEANING
============================================================
Reads raw AGMARKNET CSVs, cleans them, and returns a
single aggregated daily time-series per crop.
============================================================
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import CROPS, TARGET_COL, DATE_COL, DATE_FORMAT


def load_raw(path: str) -> pd.DataFrame:
    """Load a raw AGMARKNET CSV."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def clean(df: pd.DataFrame, crop: str) -> pd.DataFrame:
    """
    Clean steps:
      1. Parse dates
      2. Drop rows with null prices
      3. Remove duplicates
      4. Clip outliers using IQR
      5. Aggregate to daily median (multiple mandis → one value)
    """
    # 1. Parse dates
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format=DATE_FORMAT, dayfirst=True)

    # 2. Drop nulls in price columns
    price_cols = ["Min_Price", "Max_Price", "Modal_Price", "Arrivals_Tonnes"]
    df = df.dropna(subset=price_cols)

    # 3. Remove duplicates
    df = df.drop_duplicates()

    # 4. Clip outliers via IQR on Modal_Price
    Q1 = df["Modal_Price"].quantile(0.05)
    Q3 = df["Modal_Price"].quantile(0.95)
    IQR = Q3 - Q1
    lower = Q1 - 2.5 * IQR
    upper = Q3 + 2.5 * IQR
    before = len(df)
    df = df[(df["Modal_Price"] >= lower) & (df["Modal_Price"] <= upper)]
    removed = before - len(df)
    if removed:
        print(f"  [{crop}] Outlier rows removed: {removed}")

    # 5. Ensure Min <= Modal <= Max
    df = df[df["Min_Price"] <= df["Modal_Price"]]
    df = df[df["Modal_Price"] <= df["Max_Price"]]

    # 6. Aggregate to daily level (median across all markets)
    daily = df.groupby(DATE_COL).agg(
        Min_Price      = ("Min_Price",      "median"),
        Max_Price      = ("Max_Price",      "median"),
        Modal_Price    = ("Modal_Price",    "median"),
        Arrivals_Tonnes= ("Arrivals_Tonnes","sum"),
        Num_Markets    = (DATE_COL,         "count"),
    ).reset_index()

    daily = daily.sort_values(DATE_COL).reset_index(drop=True)
    daily["Commodity"] = crop
    return daily


def fill_missing_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Fill gaps in the date range with forward-fill."""
    full_range = pd.date_range(df[DATE_COL].min(), df[DATE_COL].max(), freq="D")
    df = df.set_index(DATE_COL).reindex(full_range)
    df.index.name = DATE_COL
    df = df.ffill().reset_index()
    return df


def ingest_all() -> dict:
    """Load, clean and return dict of {crop: DataFrame}."""
    result = {}
    for crop, path in CROPS.items():
        print(f"\n[{crop}] Loading {path} ...")
        raw = load_raw(path)
        print(f"  Raw rows: {len(raw):,}")
        cleaned = clean(raw, crop)
        filled  = fill_missing_dates(cleaned)
        print(f"  Clean daily rows: {len(filled):,} | "
              f"Date range: {filled[DATE_COL].min().date()} → {filled[DATE_COL].max().date()}")
        result[crop] = filled
    return result


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    data = ingest_all()
    for crop, df in data.items():
        print(f"\n{crop} sample:")
        print(df.head(3).to_string())
    print("\n✅ Ingestion complete.")
