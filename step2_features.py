"""
============================================================
STEP 2 — FEATURE ENGINEERING
============================================================
Adds exogenous variables:
  - Rainfall (monthly, Gujarat IMD-based patterns)
  - Temperature (monthly, Gujarat climate patterns)
  - Lag features (1, 3, 7, 14, 30 days)
  - Rolling averages (7, 14, 30 days)
  - Seasonality encodings
  - Price spread & momentum
============================================================
"""

import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import DATE_COL, TARGET_COL, LAG_DAYS, ROLLING_WINDOWS, CLIMATE_MAP


# ── Gujarat Monthly Climate Normals ──────────────────────────────────────────
# Source: IMD Gujarat / NASA POWER historical averages
# Rainfall (mm), Temp_Max (°C), Temp_Min (°C) — indexed by month 1..12
GUJARAT_CLIMATE = {
    #  month: (rainfall_mm, temp_max_c, temp_min_c)
    1:  (2,   28, 12),
    2:  (2,   31, 14),
    3:  (4,   36, 19),
    4:  (5,   40, 24),
    5:  (15,  42, 27),
    6:  (100, 38, 28),
    7:  (280, 33, 26),
    8:  (250, 32, 25),
    9:  (120, 33, 25),
    10: (25,  35, 22),
    11: (8,   32, 17),
    12: (3,   29, 13),
}


def add_weather(df: pd.DataFrame, commodity: str = "Wheat") -> pd.DataFrame:
    """
    Attach monthly climate normals to each row.
    Uses CLIMATE_MAP to pick Gujarat or Maharashtra climate.
    """
    climate = CLIMATE_MAP.get(commodity, CLIMATE_MAP["Wheat"])

    df = df.copy()
    df["Month"] = df[DATE_COL].dt.month

    df["Rainfall_mm"]  = df["Month"].map(lambda m: climate[m][0])
    df["Temp_Max_C"]   = df["Month"].map(lambda m: climate[m][1])
    df["Temp_Min_C"]   = df["Month"].map(lambda m: climate[m][2])
    df["Temp_Avg_C"]   = (df["Temp_Max_C"] + df["Temp_Min_C"]) / 2

    rng = np.random.default_rng(42)
    n   = len(df)
    df["Rainfall_mm"] *= (1 + rng.uniform(-0.1, 0.1, n))
    df["Temp_Max_C"]  *= (1 + rng.uniform(-0.02, 0.02, n))
    df["Temp_Min_C"]  *= (1 + rng.uniform(-0.02, 0.02, n))

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag features for Modal_Price."""
    df = df.copy()
    for lag in LAG_DAYS:
        df[f"Price_Lag_{lag}d"] = df[TARGET_COL].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rolling mean, std, min, max for Modal_Price."""
    df = df.copy()
    for window in ROLLING_WINDOWS:
        df[f"Roll_Mean_{window}d"] = df[TARGET_COL].shift(1).rolling(window).mean()
        df[f"Roll_Std_{window}d"]  = df[TARGET_COL].shift(1).rolling(window).std()
        df[f"Roll_Min_{window}d"]  = df[TARGET_COL].shift(1).rolling(window).min()
        df[f"Roll_Max_{window}d"]  = df[TARGET_COL].shift(1).rolling(window).max()
    return df


def add_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based seasonality features."""
    df = df.copy()
    df["DayOfYear"]   = df[DATE_COL].dt.dayofyear
    df["WeekOfYear"]  = df[DATE_COL].dt.isocalendar().week.astype(int)
    df["Quarter"]     = df[DATE_COL].dt.quarter
    df["Year"]        = df[DATE_COL].dt.year

    # Cyclical encoding for month and day-of-year
    df["Sin_Month"]   = np.sin(2 * np.pi * df["Month"] / 12)
    df["Cos_Month"]   = np.cos(2 * np.pi * df["Month"] / 12)
    df["Sin_DayYear"] = np.sin(2 * np.pi * df["DayOfYear"] / 365)
    df["Cos_DayYear"] = np.cos(2 * np.pi * df["DayOfYear"] / 365)
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add price spread, momentum, and arrival features."""
    df = df.copy()
    df["Price_Spread"]   = df["Max_Price"] - df["Min_Price"]
    df["Price_Momentum"] = df[TARGET_COL].diff(7)  # 7-day change
    df["Log_Price"]      = np.log1p(df[TARGET_COL])
    df["Log_Arrivals"]   = np.log1p(df["Arrivals_Tonnes"])
    return df


def build_features(df: pd.DataFrame, commodity: str = "Wheat") -> pd.DataFrame:
    """Run all feature engineering steps."""
    df = add_weather(df, commodity)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_seasonality(df)
    df = add_price_features(df)

    # Drop rows with NaN introduced by lags/rolling (first 30 rows)
    df = df.dropna().reset_index(drop=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return list of feature column names (everything except target & metadata)."""
    exclude = {DATE_COL, TARGET_COL, "Commodity", "Min_Price", "Max_Price",
               "Log_Price", "Price_Spread"}
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    from src.step1_ingest import ingest_all
    data = ingest_all()
    for crop, df in data.items():
        feat = build_features(df)
        cols = get_feature_columns(feat)
        print(f"\n{crop}: {len(feat)} rows | {len(cols)} features")
        print("Features:", cols[:10], "...")
    print("\n✅ Feature engineering complete.")
