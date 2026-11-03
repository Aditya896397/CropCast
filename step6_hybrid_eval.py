"""
============================================================
STEP 6 — HYBRID MODEL + EVALUATION
============================================================
Combines ARIMAX and Neural Network:

  final_prediction = ARIMAX_pred + NN(residual_prediction)

Evaluates using RMSE, MAE, MAPE for:
  - ARIMAX alone
  - NN alone (ARIMAX + NN residual)
  - Hybrid (same as NN here, explicitly labeled)

Produces:
  - predictions DataFrame
  - metrics DataFrame
  - feature importance DataFrame
============================================================
"""

import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import TARGET_COL, DATE_COL, TRAIN_RATIO


# ── Metrics ───────────────────────────────────────────────────────────────────

def rmse(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2)))

def mae(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))

def mape(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred)) & (y_true != 0)
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ── Hybrid prediction builder ─────────────────────────────────────────────────

def build_hybrid(crop: str,
                  df: pd.DataFrame,
                  arimax_result: dict,
                  nn_result: dict) -> tuple:
    """
    Assemble final predictions and compute metrics.

    Returns:
      (predictions_df, metrics_df)
    """
    n_train = arimax_result["n_train"]
    y       = arimax_result["y"].astype(float)
    dates   = arimax_result["dates"]

    arimax_preds  = arimax_result["preds"].astype(float)
    nn_res_preds  = nn_result["nn_preds"].astype(float)

    # Hybrid = ARIMAX prediction + NN residual correction
    hybrid_preds  = arimax_preds + nn_res_preds

    # Clamp to realistic price floors (no negative prices)
    hybrid_preds  = np.clip(hybrid_preds, 100, None)
    arimax_preds  = np.clip(arimax_preds, 100, None)

    # Build split labels
    split_labels = np.array(["train"] * n_train + ["test"] * (len(y) - n_train))

    # Predictions DataFrame
    pred_df = pd.DataFrame({
        "Commodity":      crop,
        "Price_Date":     dates,
        "Actual":         y,
        "ARIMAX_Pred":    arimax_preds,
        "NN_Residual_Pred": nn_res_preds,
        "Hybrid_Pred":    hybrid_preds,
        "Split":          split_labels,
    })

    # Metrics — compute on TEST set only
    test_mask = split_labels == "test"
    y_test    = y[test_mask]
    ar_test   = arimax_preds[test_mask]
    hy_test   = hybrid_preds[test_mask]

    # Also compute train metrics
    y_tr  = y[~test_mask]
    ar_tr = arimax_preds[~test_mask]
    hy_tr = hybrid_preds[~test_mask]

    metrics_rows = [
        {
            "Commodity": crop, "Model": "ARIMAX",
            "Split": "Test",
            "RMSE": rmse(y_test, ar_test),
            "MAE":  mae(y_test,  ar_test),
            "MAPE": mape(y_test, ar_test),
        },
        {
            "Commodity": crop, "Model": "Hybrid (ARIMAX + NN)",
            "Split": "Test",
            "RMSE": rmse(y_test, hy_test),
            "MAE":  mae(y_test,  hy_test),
            "MAPE": mape(y_test, hy_test),
        },
        {
            "Commodity": crop, "Model": "ARIMAX",
            "Split": "Train",
            "RMSE": rmse(y_tr, ar_tr),
            "MAE":  mae(y_tr,  ar_tr),
            "MAPE": mape(y_tr, ar_tr),
        },
        {
            "Commodity": crop, "Model": "Hybrid (ARIMAX + NN)",
            "Split": "Train",
            "RMSE": rmse(y_tr, hy_tr),
            "MAE":  mae(y_tr,  hy_tr),
            "MAPE": mape(y_tr, hy_tr),
        },
    ]
    metrics_df = pd.DataFrame(metrics_rows)

    # Print summary
    print(f"\n{'='*55}")
    print(f"  {crop} — MODEL EVALUATION (TEST SET)")
    print(f"{'='*55}")
    print(f"  {'Model':<25} {'RMSE':>8} {'MAE':>8} {'MAPE':>8}")
    print(f"  {'-'*50}")
    for _, row in metrics_df[metrics_df["Split"]=="Test"].iterrows():
        print(f"  {row['Model']:<25} {row['RMSE']:>8.2f} {row['MAE']:>8.2f} {row['MAPE']:>7.2f}%")

    return pred_df, metrics_df


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    from src.step1_ingest       import ingest_all
    from src.step2_features     import build_features
    from src.step4_arimax       import train_arimax
    from src.step5_nn_residuals import train_nn_residuals

    data = ingest_all()
    all_preds   = []
    all_metrics = []

    for crop, df in data.items():
        feat    = build_features(df)
        arimax  = train_arimax(crop, feat)
        nn      = train_nn_residuals(crop, feat, arimax["residuals"])
        p_df, m_df = build_hybrid(crop, feat, arimax, nn)
        all_preds.append(p_df)
        all_metrics.append(m_df)

    print("\n✅ Hybrid model evaluation complete.")
