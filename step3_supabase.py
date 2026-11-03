"""
============================================================
STEP 3 — SUPABASE (PostgreSQL) UPLOAD
============================================================
Creates tables (if not exist) and upserts processed data.
Uses only the standard `urllib` + `json` libraries so no
extra packages are needed beyond what ships with Python.
============================================================
"""

import json
import urllib.request
import urllib.error
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import SUPABASE_URL, SUPABASE_KEY


# ── Helpers ───────────────────────────────────────────────────────────────────

def _headers(prefer: str = "") -> dict:
    h = {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
    }
    if prefer:
        h["Prefer"] = prefer
    return h


def _post(endpoint: str, payload: list, prefer: str = "return=minimal") -> bool:
    url  = f"{SUPABASE_URL}/rest/v1/{endpoint}"
    data = json.dumps(payload, default=str).encode("utf-8")
    req  = urllib.request.Request(url, data=data, headers=_headers(prefer), method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status in (200, 201)
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"  ⚠ Supabase HTTP {e.code}: {body[:200]}")
        return False
    except Exception as ex:
        print(f"  ⚠ Supabase error: {ex}")
        return False


def _batch_upsert(table: str, records: list, batch_size: int = 500):
    """Delete all rows then insert fresh data in batches."""
    # Clear table first to avoid duplicate conflicts
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    req = urllib.request.Request(
        url, headers={**_headers(), "Prefer": "return=minimal"}, method="DELETE"
    )
    try:
        with urllib.request.urlopen(req) as resp:
            pass
    except Exception:
        pass

    # Now insert fresh
    total   = len(records)
    success = 0
    for i in range(0, total, batch_size):
        batch = records[i: i + batch_size]
        ok    = _post(table, batch, prefer="return=minimal")
        if ok:
            success += len(batch)
    print(f"  Upserted {success:,}/{total:,} rows → {table}")


# ── Table creation via RPC (raw SQL) ─────────────────────────────────────────

CREATE_SQL = {
    "crop_raw_prices": """
        CREATE TABLE IF NOT EXISTS crop_raw_prices (
            id              BIGSERIAL PRIMARY KEY,
            commodity       TEXT NOT NULL,
            price_date      DATE NOT NULL,
            min_price       NUMERIC,
            max_price       NUMERIC,
            modal_price     NUMERIC,
            arrivals_tonnes NUMERIC,
            num_markets     INTEGER,
            UNIQUE (commodity, price_date)
        );
    """,
    "crop_processed_features": """
        CREATE TABLE IF NOT EXISTS crop_processed_features (
            id              BIGSERIAL PRIMARY KEY,
            commodity       TEXT NOT NULL,
            price_date      DATE NOT NULL,
            modal_price     NUMERIC,
            rainfall_mm     NUMERIC,
            temp_max_c      NUMERIC,
            temp_min_c      NUMERIC,
            price_lag_1d    NUMERIC,
            price_lag_7d    NUMERIC,
            price_lag_30d   NUMERIC,
            roll_mean_7d    NUMERIC,
            roll_mean_30d   NUMERIC,
            price_spread    NUMERIC,
            sin_month       NUMERIC,
            cos_month       NUMERIC,
            UNIQUE (commodity, price_date)
        );
    """,
    "crop_predictions": """
        CREATE TABLE IF NOT EXISTS crop_predictions (
            id              BIGSERIAL PRIMARY KEY,
            commodity       TEXT NOT NULL,
            price_date      DATE NOT NULL,
            actual_price    NUMERIC,
            arimax_pred     NUMERIC,
            nn_residual_pred NUMERIC,
            hybrid_pred     NUMERIC,
            split_set       TEXT,
            UNIQUE (commodity, price_date)
        );
    """,
    "crop_model_metrics": """
        CREATE TABLE IF NOT EXISTS crop_model_metrics (
            id          BIGSERIAL PRIMARY KEY,
            commodity   TEXT NOT NULL,
            model_name  TEXT NOT NULL,
            rmse        NUMERIC,
            mae         NUMERIC,
            mape        NUMERIC,
            evaluated_at TIMESTAMP DEFAULT NOW(),
            UNIQUE (commodity, model_name)
        );
    """,
}


def create_tables():
    """Create all required tables via Supabase RPC (raw SQL exec)."""
    url = f"{SUPABASE_URL}/rest/v1/rpc/exec_sql"
    for tbl, sql in CREATE_SQL.items():
        payload = json.dumps({"query": sql}).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload, headers=_headers(), method="POST"
        )
        try:
            with urllib.request.urlopen(req) as resp:
                print(f"  ✓ Table ready: {tbl}")
        except urllib.error.HTTPError as e:
            # 404 means exec_sql RPC not available — skip, tables may already exist
            if e.code == 404:
                print(f"  ℹ exec_sql RPC not available — tables must be created manually.")
                print(f"    Run the SQL in supabase.com → SQL Editor for table: {tbl}")
            else:
                print(f"  ⚠ {tbl}: HTTP {e.code} — {e.read().decode()[:100]}")
        except Exception as ex:
            print(f"  ⚠ {tbl}: {ex}")


# ── Public upload functions ───────────────────────────────────────────────────

def upload_raw(crop_data: dict):
    """Upload raw daily price data for all crops."""
    print("\n[Supabase] Uploading raw prices...")
    for crop, df in crop_data.items():
        records = []
        for _, row in df.iterrows():
            records.append({
                "commodity":       crop,
                "price_date":      str(row["Price_Date"].date()),
                "min_price":       round(float(row["Min_Price"]), 2),
                "max_price":       round(float(row["Max_Price"]), 2),
                "modal_price":     round(float(row["Modal_Price"]), 2),
                "arrivals_tonnes": round(float(row["Arrivals_Tonnes"]), 2),
                "num_markets":     int(row.get("Num_Markets", 1)),
            })
        _batch_upsert("crop_raw_prices", records)


def upload_features(featured_data: dict):
    """Upload processed feature data."""
    print("\n[Supabase] Uploading processed features...")
    keep = ["commodity", "Price_Date", "Modal_Price", "Rainfall_mm",
            "Temp_Max_C", "Temp_Min_C", "Price_Lag_1d", "Price_Lag_7d",
            "Price_Lag_30d", "Roll_Mean_7d", "Roll_Mean_30d",
            "Price_Spread", "Sin_Month", "Cos_Month"]
    for crop, df in featured_data.items():
        records = []
        for _, row in df.iterrows():
            rec = {
                "commodity":      crop,
                "price_date":     str(row["Price_Date"].date()),
                "modal_price":    round(float(row["Modal_Price"]), 2),
                "rainfall_mm":    round(float(row.get("Rainfall_mm", 0)), 2),
                "temp_max_c":     round(float(row.get("Temp_Max_C", 0)), 2),
                "temp_min_c":     round(float(row.get("Temp_Min_C", 0)), 2),
                "price_lag_1d":   round(float(row.get("Price_Lag_1d", 0)), 2),
                "price_lag_7d":   round(float(row.get("Price_Lag_7d", 0)), 2),
                "price_lag_30d":  round(float(row.get("Price_Lag_30d", 0)), 2),
                "roll_mean_7d":   round(float(row.get("Roll_Mean_7d", 0)), 2),
                "roll_mean_30d":  round(float(row.get("Roll_Mean_30d", 0)), 2),
                "price_spread":   round(float(row.get("Price_Spread", 0)), 2),
                "sin_month":      round(float(row.get("Sin_Month", 0)), 4),
                "cos_month":      round(float(row.get("Cos_Month", 0)), 4),
            }
            records.append(rec)
        _batch_upsert("crop_processed_features", records)


def upload_predictions(predictions_df: pd.DataFrame):
    """Upload hybrid model predictions."""
    print("\n[Supabase] Uploading predictions...")
    records = []
    for _, row in predictions_df.iterrows():
        records.append({
            "commodity":         str(row["Commodity"]),
            "price_date":        str(row["Price_Date"].date()) if hasattr(row["Price_Date"], "date") else str(row["Price_Date"]),
            "actual_price":      round(float(row["Actual"]), 2),
            "arimax_pred":       round(float(row["ARIMAX_Pred"]), 2),
            "nn_residual_pred":  round(float(row["NN_Residual_Pred"]), 2),
            "hybrid_pred":       round(float(row["Hybrid_Pred"]), 2),
            "split_set":         str(row["Split"]),
        })
    _batch_upsert("crop_predictions", records)


def upload_metrics(metrics_df: pd.DataFrame):
    """Upload model evaluation metrics."""
    print("\n[Supabase] Uploading model metrics...")
    records = []
    for _, row in metrics_df.iterrows():
        records.append({
            "commodity":  str(row["Commodity"]),
            "model_name": str(row["Model"]),
            "rmse":       round(float(row["RMSE"]), 4),
            "mae":        round(float(row["MAE"]), 4),
            "mape":       round(float(row["MAPE"]), 4),
        })
    _batch_upsert("crop_model_metrics", records)


if __name__ == "__main__":
    print("[Supabase] Testing connection...")
    url = f"{SUPABASE_URL}/rest/v1/"
    req = urllib.request.Request(url, headers=_headers())
    try:
        with urllib.request.urlopen(req) as r:
            print(f"  ✓ Connected — status {r.status}")
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
