"""
Random Forest Predictive Modeling for ADM3 Flood Risk & Combined Exposure
=======================================================================
Trains a Random Forest Classifier to identify high-risk ADM3 payams based on 
multi-factor exposure, historical hazard intensity, and asset indicators.

Outputs:
- outputs/adm3_random_forest_predictions.csv
- outputs/random_forest_feature_importance.png
- outputs/random_forest_roc_curve.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# ==============================================================================
# 1. Setup & Data Loading
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

INPUT_CSV = PROJECT_ROOT / "outputs" / "adm3_two_factor_exposure_analysis.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Could not find {INPUT_CSV}. Run your two-factor analysis script first.")
    print(f"[INFO] Loading dataset from {INPUT_CSV}...")
    return pd.read_csv(INPUT_CSV)


# ==============================================================================
# 2. Feature Engineering & Target Definition
# ==============================================================================
def prepare_modeling_data(df: pd.DataFrame):
    print("[INFO] Preparing features and target variable...")
    
    # Calculate overall combined risk score if not already present
    if "overall_risk" not in df.columns:
        df["overall_risk"] = df["factor1_pop_livestock_risk"] + df["factor2_farm_infra_risk"]
    
    # Define binary target: Top 25% highest combined risk payams = High Risk Priority (1), else (0)
    risk_threshold = df["overall_risk"].quantile(0.75)
    df["high_risk_target"] = (df["overall_risk"] >= risk_threshold).astype(int)
    print(f"[INFO] High-risk threshold (75th percentile overall risk): {risk_threshold:.4f}")
    print(f"       Class distribution: {df['high_risk_target'].value_counts().to_dict()}")

    # Select predictive features
    feature_cols = [
        "population_2024_norm",
        "cattle_sum_norm",
        "cropland_score_norm",
        "health_facilities_count_norm",
        "flood_hazard_2022_count_norm"
    ]
    
    model_df = df.dropna(subset=feature_cols + ["high_risk_target"]).copy()
    
    X = model_df[feature_cols]
    y = model_df["high_risk_target"]
    
    return model_df, X, y, feature_cols


# ==============================================================================
# 3. Random Forest Training & Evaluation
# ==============================================================================
def train_random_forest(X, y):
    print("[INFO] Training Random Forest Classifier...")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Initialize Random Forest with robust hyperparameters
    rf_model = RandomForestClassifier(
        n_estimators=250,
        max_depth=6,
        min_samples_split=5,
        random_state=42,
        class_weight="balanced"
    )
    
    rf_model.fit(X_train, y_train)
    
    # Predictions & Evaluation
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    
    print("\n--- Model Evaluation Report ---")
    print(classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {auc:.4f}")
        
    return rf_model, X_test, y_test, y_prob


# ==============================================================================
# 4. Visualizations (Feature Importance & ROC Curve)
# ==============================================================================
def plot_feature_importance(model, feature_names):
    print("[VISUALIZATION] Generating Feature Importance Chart...")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)
    
    clean_labels = [
        name.replace("_norm", "").replace("_", " ").title() 
        for name in feature_names
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(
        range(len(indices)), 
        importances[indices], 
        color="#2563EB", 
        alpha=0.85,
        edgecolor="#1E3A8A",
        linewidth=0.6
    )
    
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([clean_labels[i] for i in indices], fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean Decrease in Impurity (Feature Importance)", fontsize=11, fontweight="bold")
    ax.set_title("Random Forest Feature Importance:\nDrivers of High Combined Exposure & Flood Risk in ADM3 Payams", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6, axis="x")
    
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height()/2, f"{w:.3f}", va="center", fontsize=9, fontweight="bold", color="#1E3A8A")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "random_forest_feature_importance.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] {out_path.name}")


def plot_roc_curve(y_test, y_prob):
    print("[VISUALIZATION] Generating ROC Curve Plot...")
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#2563EB", lw=2.5, label=f"Random Forest Model (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Random Guess Baseline (AUC = 0.5000)")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve\nModel Classification Performance", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11, framealpha=0.95)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "random_forest_roc_curve.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] {out_path.name}")


# ==============================================================================
# 5. Full Dataset Scoring & Export
# ==============================================================================
def score_and_export_all(df, model, feature_cols):
    print("[INFO] Scoring all ADM3 payams with trained Random Forest model...")
    
    X_all = df[feature_cols]
    df["rf_high_risk_probability"] = model.predict_proba(X_all)[:, 1]
    df["rf_predicted_class"] = model.predict(X_all)
    
    df_sorted = df.sort_values("rf_high_risk_probability", ascending=False)
    
    out_csv = OUTPUT_DIR / "adm3_random_forest_predictions.csv"
    df_sorted.to_csv(out_csv, index=False)
    print(f"[SUCCESS] Saved Random Forest predictions to:\n  -> {out_csv}")
    
    print("\n--- Top 10 Payams Predicted as Highest Risk by Random Forest ---")
    top10 = df_sorted.head(10)[["adm3_name", "adm2_name", "adm1_name", "rf_high_risk_probability", "overall_risk"]]
    print(top10.to_string(index=False))


# ==============================================================================
# 6. Main Execution
# ==============================================================================
def main():
    print("=" * 70)
    print("RANDOM FOREST FLOOD RISK & EXPOSURE PREDICTION MODEL")
    print("=" * 70)
    
    df = load_data()
    model_df, X, y, feature_cols = prepare_modeling_data(df)
    
    model, X_test, y_test, y_prob = train_random_forest(X, y)
    
    plot_feature_importance(model, feature_cols)
    plot_roc_curve(y_test, y_prob)
    
    score_and_export_all(model_df, model, feature_cols)
    
    print("\n[COMPLETE] Random Forest modeling finished successfully.\n")


if __name__ == "__main__":
    main()