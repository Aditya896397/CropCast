"""
============================================================
CROP PRICE FORECASTING SYSTEM — MAIN PIPELINE
============================================================
Data Source:    AGMARKNET (agmarknet.gov.in) — Gujarat
Crops:          Wheat, Tomato, Onion
Date Range:     01-Jan-2022 to 27-Feb-2025
Models:         ARIMAX + Neural Network (Hybrid)
Storage:        Supabase (PostgreSQL)

Run:
    cd crop_forecasting
    python run_pipeline.py

Model Assumptions:
    1. Daily modal price is the best single-point price indicator
    2. Seasonal patterns are consistent year-over-year (with variation)
    3. Exogenous weather variables follow historical Gujarat IMD patterns
    4. ARIMAX captures linear temporal dependencies and trend
    5. Neural Network captures non-linear residual patterns
    6. Train/test split is chronological (no data leakage)

Limitations:
    1. Weather data uses historical IMD climate normals, not real-time
    2. AGMARKNET data may have market-reporting gaps → forward-filled
    3. ARIMAX walk-forward test uses simplified AR approximation
    4. Neural Network importances are approximate (weight-based method)
    5. Model does not account for sudden policy changes (e.g. MSP revision)
    6. Short-term shocks (floods, export bans) not modelled explicitly
============================================================
"""

import os
import sys
import time

# Change to project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from config import OUTPUT_DIR, MODEL_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

from src.step1_ingest       import ingest_all
from src.step2_features     import build_features
from src.step3_supabase     import (upload_raw, upload_features,
                                     upload_predictions, upload_metrics,
                                     create_tables)
from src.step4_arimax       import train_arimax
from src.step5_nn_residuals import train_nn_residuals, get_feature_importance
from src.step6_hybrid_eval  import build_hybrid
from src.step7_export       import (export_predictions, export_metrics,
                                     export_feature_importance,
                                     plot_predictions, plot_metrics,
                                     plot_feature_importance, plot_residuals)


def run():
    t0 = time.time()
    print("\n" + "="*60)
    print("  CROP PRICE FORECASTING SYSTEM — STARTING")
    print("="*60)

    # ── Step 1: Ingest & Clean ────────────────────────────────
    print("\n📥 STEP 1: Ingesting & Cleaning Data...")
    raw_data = ingest_all()

    # ── Step 2: Feature Engineering ──────────────────────────
    print("\n🔧 STEP 2: Engineering Features...")
    featured_data = {}
    for crop, df in raw_data.items():
        featured_data[crop] = build_features(df, commodity=crop)
        print(f"  [{crop}] {len(featured_data[crop])} rows | "
              f"{featured_data[crop].shape[1]} columns")

    # ── Step 3: Upload to Supabase ────────────────────────────
    print("\n☁️  STEP 3: Uploading to Supabase...")
    try:
        create_tables()
        upload_raw(raw_data)
        upload_features(featured_data)
        print("  ✓ Supabase upload complete")
    except Exception as e:
        print(f"  ⚠ Supabase upload failed (continuing): {e}")

    # ── Steps 4–6: Train Models & Build Hybrid ────────────────
    print("\n🤖 STEPS 4-6: Training Models...")
    all_preds       = []
    all_metrics     = []
    importance_dict = {}

    for crop, df in featured_data.items():
        print(f"\n{'─'*50}")
        print(f"  Processing: {crop}")
        print(f"{'─'*50}")

        # ARIMAX
        arimax_result = train_arimax(crop, df)

        # Neural Network on residuals
        nn_result = train_nn_residuals(crop, df, arimax_result["residuals"])

        # Feature importance
        importance_dict[crop] = get_feature_importance(
            nn_result["pipeline"], nn_result["feat_cols"]
        )

        # Hybrid evaluation
        pred_df, metrics_df = build_hybrid(crop, df, arimax_result, nn_result)
        all_preds.append(pred_df)
        all_metrics.append(metrics_df)

    # ── Step 7: Export & Visualize ────────────────────────────
    print("\n📊 STEP 7: Exporting Results & Generating Charts...")
    pred_path    = export_predictions(all_preds)
    metrics_path = export_metrics(all_metrics)
    imp_path     = export_feature_importance(importance_dict)

    plot_predictions(all_preds)
    plot_metrics(all_metrics)
    plot_feature_importance(importance_dict)
    plot_residuals(all_preds)

    # ── Upload predictions & metrics to Supabase ─────────────
    print("\n☁️  Uploading predictions & metrics to Supabase...")
    try:
        import pandas as pd
        combined_preds   = pd.concat(all_preds,   ignore_index=True)
        combined_metrics = pd.concat(all_metrics, ignore_index=True)
        upload_predictions(combined_preds)
        upload_metrics(combined_metrics)
        print("  ✓ Done")
    except Exception as e:
        print(f"  ⚠ Final Supabase upload failed: {e}")

    # ── Summary ───────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "="*60)
    print("  ✅ PIPELINE COMPLETE")
    print(f"  Time elapsed: {elapsed:.1f}s")
    print("="*60)
    print("\n📁 Output Files:")
    for f in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, f)
        size  = os.path.getsize(fpath)
        print(f"  {f:<40} {size/1024:>8.1f} KB")

    import pandas as pd
    print("\n📈 Final Metrics Summary (Test Set):")
    metrics_df = pd.concat(all_metrics)
    test_df    = metrics_df[metrics_df["Split"] == "Test"]
    print(test_df[["Commodity","Model","RMSE","MAE","MAPE"]].to_string(index=False))
    print()


if __name__ == "__main__":
    run()
