"""
============================================================
STEP 7 — VISUALIZATIONS & EXPORTS
============================================================
Generates:
  - Actual vs Predicted plots per crop
  - Residual distribution plots
  - Feature importance bar charts
  - Model comparison bar chart

Exports:
  - predictions.csv
  - model_metrics.csv
  - feature_importance.csv
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import OUTPUT_DIR

CROP_COLORS = {
    "Wheat":      "#E8A838",
    "Tomato":     "#E84040",
    "Onion":      "#9B59B6",
    "Wheat_MH":   "#C8881E",
    "Tomato_MH":  "#C02020",
    "Onion_MH":   "#6B3990",
}


# ── CSV Exports ───────────────────────────────────────────────────────────────

def export_predictions(all_preds: list) -> str:
    df   = pd.concat(all_preds, ignore_index=True)
    path = os.path.join(OUTPUT_DIR, "predictions.csv")
    df.to_csv(path, index=False)
    print(f"  ✓ Saved: {path} ({len(df):,} rows)")
    return path


def export_metrics(all_metrics: list) -> str:
    df   = pd.concat(all_metrics, ignore_index=True)
    df   = df.round(4)
    path = os.path.join(OUTPUT_DIR, "model_metrics.csv")
    df.to_csv(path, index=False)
    print(f"  ✓ Saved: {path}")
    return path


def export_feature_importance(importance_dict: dict) -> str:
    frames = []
    for crop, imp_df in importance_dict.items():
        imp_df = imp_df.copy()
        imp_df["Commodity"] = crop
        frames.append(imp_df)
    df   = pd.concat(frames, ignore_index=True)
    path = os.path.join(OUTPUT_DIR, "feature_importance.csv")
    df.to_csv(path, index=False)
    print(f"  ✓ Saved: {path}")
    return path


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_predictions(all_preds: list):
    """3-panel actual vs predicted plot for all crops."""
    crops  = [df["Commodity"].iloc[0] for df in all_preds]
    fig, axes = plt.subplots(len(crops), 1, figsize=(16, 5 * len(crops)))
    if len(crops) == 1:
        axes = [axes]

    for ax, pred_df, crop in zip(axes, all_preds, crops):
        color = CROP_COLORS.get(crop, "#3498DB")
        test_df  = pred_df[pred_df["Split"] == "test"]
        train_df = pred_df[pred_df["Split"] == "train"]

        ax.plot(train_df["Price_Date"], train_df["Actual"],
                color="gray", alpha=0.5, lw=1, label="Actual (train)")
        ax.plot(test_df["Price_Date"],  test_df["Actual"],
                color="black", lw=1.5, label="Actual (test)")
        ax.plot(train_df["Price_Date"], train_df["Hybrid_Pred"],
                color=color, alpha=0.6, lw=1, linestyle="--", label="Hybrid (train)")
        ax.plot(test_df["Price_Date"],  test_df["Hybrid_Pred"],
                color=color, lw=2, label="Hybrid Prediction (test)")

        # Shade train/test split
        split_date = pred_df[pred_df["Split"] == "test"]["Price_Date"].iloc[0]
        ax.axvline(split_date, color="red", linestyle=":", alpha=0.7, label="Train/Test Split")
        ax.set_title(f"{crop} — Actual vs Hybrid Prediction", fontsize=13, fontweight="bold")
        ax.set_ylabel("Modal Price (₹/Quintal)")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "actual_vs_predicted.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def plot_metrics(all_metrics: list):
    """Model comparison bar chart."""
    df    = pd.concat(all_metrics)
    test  = df[df["Split"] == "Test"].copy()
    crops = test["Commodity"].unique()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics   = ["RMSE", "MAE", "MAPE"]
    titles    = ["RMSE (lower = better)", "MAE (lower = better)", "MAPE % (lower = better)"]

    for ax, metric, title in zip(axes, metrics, titles):
        x     = np.arange(len(crops))
        width = 0.35
        models = test["Model"].unique()
        for i, model in enumerate(models):
            vals = [test[(test["Commodity"]==c) & (test["Model"]==model)][metric].values[0]
                    if len(test[(test["Commodity"]==c) & (test["Model"]==model)]) > 0
                    else 0 for c in crops]
            ax.bar(x + i * width, vals, width,
                   label=model,
                   color=["#3498DB", "#E74C3C"][i % 2],
                   alpha=0.8)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(crops)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Model Performance Comparison — Test Set", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def plot_feature_importance(importance_dict: dict):
    """Feature importance bar charts per crop."""
    crops = list(importance_dict.keys())
    fig, axes = plt.subplots(1, len(crops), figsize=(7 * len(crops), 6))
    if len(crops) == 1:
        axes = [axes]

    for ax, crop in zip(axes, crops):
        imp   = importance_dict[crop].head(12)
        color = CROP_COLORS.get(crop, "#3498DB")
        ax.barh(imp["Feature"][::-1], imp["Importance"][::-1],
                color=color, alpha=0.8)
        ax.set_title(f"{crop}\nTop Feature Importances", fontweight="bold")
        ax.set_xlabel("Relative Importance")
        ax.grid(True, alpha=0.3, axis="x")

    plt.suptitle("Neural Network Feature Importance (Input Weight Method)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def plot_residuals(all_preds: list):
    """Residual distribution histograms."""
    crops = [df["Commodity"].iloc[0] for df in all_preds]
    fig, axes = plt.subplots(1, len(crops), figsize=(6 * len(crops), 4))
    if len(crops) == 1:
        axes = [axes]

    for ax, pred_df, crop in zip(axes, all_preds, crops):
        color = CROP_COLORS.get(crop, "#3498DB")
        test  = pred_df[pred_df["Split"] == "test"]
        res   = test["Actual"] - test["Hybrid_Pred"]
        ax.hist(res, bins=40, color=color, alpha=0.8, edgecolor="white")
        ax.axvline(0, color="black", linestyle="--", lw=1.5)
        ax.axvline(res.mean(), color="red", linestyle="-", lw=1.5,
                   label=f"Mean: {res.mean():.1f}")
        ax.set_title(f"{crop} — Residual Distribution", fontweight="bold")
        ax.set_xlabel("Residual (₹/Quintal)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Hybrid Model Residuals — Test Set", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "residual_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")
