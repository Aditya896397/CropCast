"""
============================================================
STEP 5 — NEURAL NETWORK ON ARIMAX RESIDUALS
============================================================
Trains an MLPRegressor to predict ARIMAX residuals.
The NN captures non-linear patterns the linear ARIMAX misses.

Architecture: 128 → 64 → 32 neurons, ReLU activation
Input: All engineered features
Target: ARIMAX residuals
============================================================
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings, sys, os
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (TARGET_COL, DATE_COL, TRAIN_RATIO,
                    NN_HIDDEN_LAYERS, NN_MAX_ITER, NN_RANDOM_STATE)


NN_FEATURE_COLS = [
    "Rainfall_mm", "Temp_Max_C", "Temp_Min_C", "Temp_Avg_C",
    "Price_Lag_1d", "Price_Lag_3d", "Price_Lag_7d", "Price_Lag_14d", "Price_Lag_30d",
    "Roll_Mean_7d", "Roll_Mean_14d", "Roll_Mean_30d",
    "Roll_Std_7d",  "Roll_Std_14d",  "Roll_Std_30d",
    "Roll_Min_7d",  "Roll_Max_7d",
    "Price_Spread", "Price_Momentum", "Log_Arrivals",
    "Sin_Month", "Cos_Month", "Sin_DayYear", "Cos_DayYear",
    "Quarter", "WeekOfYear",
]


def train_nn_residuals(crop: str,
                        df: pd.DataFrame,
                        residuals: np.ndarray) -> dict:
    """
    Train Neural Network to predict ARIMAX residuals.

    Args:
        crop      : crop name (for logging)
        df        : featured DataFrame
        residuals : ARIMAX residuals array (len == len(df))

    Returns dict with pipeline, predictions, and split info.
    """
    n_train = int(len(df) * TRAIN_RATIO)

    # Select available feature columns
    feat_cols = [c for c in NN_FEATURE_COLS if c in df.columns]
    X = df[feat_cols].values.astype(float)
    y = residuals.astype(float)

    # Remove NaN rows
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]

    # Align n_train with clean data
    n_valid_train = int(len(X_clean) * TRAIN_RATIO)

    X_train = X_clean[:n_valid_train]
    y_train = y_clean[:n_valid_train]
    X_test  = X_clean[n_valid_train:]
    y_test  = y_clean[n_valid_train:]

    print(f"\n[{crop}] Training Neural Network on residuals ...")
    print(f"  Features: {len(feat_cols)} | Train: {len(X_train)} | Test: {len(X_test)}")

    # Build pipeline: scaler + MLP
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp",    MLPRegressor(
            hidden_layer_sizes = NN_HIDDEN_LAYERS,
            activation         = "relu",
            solver             = "adam",
            learning_rate_init = 0.001,
            max_iter           = NN_MAX_ITER,
            early_stopping     = True,
            validation_fraction= 0.1,
            n_iter_no_change   = 20,
            random_state       = NN_RANDOM_STATE,
            verbose            = False,
        )),
    ])

    pipeline.fit(X_train, y_train)

    # Predict on all clean data
    all_preds = pipeline.predict(X_clean)

    # Map back to original indices
    full_nn_preds = np.zeros(len(df))
    full_nn_preds[valid_mask] = all_preds
    full_nn_preds[~valid_mask] = 0.0  # zero residual correction where no data

    train_residual_preds = full_nn_preds[:n_train]
    test_residual_preds  = full_nn_preds[n_train:]

    train_rmse = np.sqrt(np.mean((y_train - pipeline.predict(X_train))**2))
    test_rmse  = np.sqrt(np.mean((y_test  - pipeline.predict(X_test))**2)) if len(X_test) else 0
    print(f"  Residual Train RMSE: {train_rmse:.2f} | Test RMSE: {test_rmse:.2f}")

    return {
        "pipeline":           pipeline,
        "feat_cols":          feat_cols,
        "nn_preds":           full_nn_preds,
        "train_residual_preds": train_residual_preds,
        "test_residual_preds":  test_residual_preds,
        "n_train":            n_train,
    }


def get_feature_importance(pipeline: Pipeline, feat_cols: list) -> pd.DataFrame:
    """
    Extract feature importance from the MLP using input-weight method.
    |W1| × |W2| × ... aggregated per input feature.
    """
    mlp    = pipeline.named_steps["mlp"]
    coefs  = mlp.coefs_  # list of weight matrices

    # Input layer weights → absolute sum per input feature
    # coefs[0] shape: (n_features, n_hidden_1)
    input_weights = np.abs(coefs[0]).sum(axis=1)  # shape: (n_features,)

    # Normalize directly — no propagation to avoid shape mismatch
    importance = input_weights[:len(feat_cols)]
    importance /= importance.sum() + 1e-8

    df_imp = pd.DataFrame({
        "Feature":    feat_cols,
        "Importance": importance,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    df_imp["Rank"] = df_imp.index + 1
    return df_imp


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    from src.step1_ingest   import ingest_all
    from src.step2_features import build_features
    from src.step4_arimax   import train_arimax
    data = ingest_all()
    for crop, df in data.items():
        feat      = build_features(df)
        arimax_r  = train_arimax(crop, feat)
        nn_r      = train_nn_residuals(crop, feat, arimax_r["residuals"])
        imp       = get_feature_importance(nn_r["pipeline"], nn_r["feat_cols"])
        print(f"\n  Top 5 features for {crop}:")
        print(imp.head(5).to_string(index=False))
    print("\n✅ Neural Network training complete.")
