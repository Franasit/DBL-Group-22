"""
Random Forest Predictive Modeling for ADM3 Flood Risk
=======================================================================
Trains a Random Forest Classifier to PREDICT high flood-hazard payams
based on socio-economic, geographic, and asset features.

The model is trained on real data:
  - Features : raw observable indicators (population, livestock, cropland,
               health facilities, area, geographic coordinates)
  - Target   : whether the payam experienced HIGH flood intensity in 2022
               (flood_hazard_2022_count >= 75th-percentile threshold),
               i.e. an actual historical measurement — NOT a formula.

Training uses Stratified K-Fold cross-validation + GridSearchCV to tune
the Random Forest so it generalises beyond the training set.

Outputs:
  - outputs/adm3_random_forest_predictions.csv   – full scored dataset
  - outputs/random_forest_feature_importance.png – feature importance bar chart
  - outputs/random_forest_roc_curve.png          – ROC curve (hold-out test set)
  - outputs/random_forest_cv_results.csv         – cross-validation scores
  - outputs/random_forest_model.joblib           – saved trained model
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 0. Paths
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

INPUT_CSV = PROJECT_ROOT / "outputs" / "adm3_two_factor_exposure_analysis.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ==============================================================================
# 1. Data Loading
# ==============================================================================
def load_data() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_CSV}.\n"
            "Run the two-factor exposure analysis script first."
        )
    print(f"[INFO] Loading dataset from {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV)
    print(f"       {len(df)} rows, {df.shape[1]} columns loaded.")
    return df


# ==============================================================================
# 2. Feature Engineering & Target Definition
# ==============================================================================
def prepare_modeling_data(df: pd.DataFrame):
    """
    Build features and target from raw data columns.

    Target
    ------
    Binary label derived from the ACTUAL historical flood observation
    `flood_hazard_2022_count`:
      1  →  High flood intensity  (>= 75th-percentile of the raw count)
      0  →  Low / no flood

    Features
    --------
    Raw socio-economic / geographic indicators that an analyst could observe
    BEFORE a flood season — these are NOT derived from the flood count:
      • population_2024          – total population
      • cattle_sum               – livestock headcount
      • cropland_score           – area under crops
      • health_facilities_count  – number of health facilities
      • area_sqkm                – physical area of the payam
      • center_lat / center_lon  – geographic location

    We intentionally EXCLUDE columns that were computed FROM the flood count
    (e.g. flood_hazard_2022_count_norm, combined_exposure_score, factor*_risk)
    to avoid data leakage.
    """
    print("[INFO] Preparing features and target variable ...")

    # ── Target: high flood intensity from real historical data ──────────────
    flood_col = "flood_hazard_2022_count"
    threshold = df[flood_col].quantile(0.75)
    df = df.copy()
    df["high_flood_target"] = (df[flood_col] >= threshold).astype(int)

    print(f"       Flood count 75th-percentile threshold : {threshold:.1f}")
    print(f"       Class distribution : {df['high_flood_target'].value_counts().to_dict()}")

    # ── Features: raw, pre-flood observable indicators ──────────────────────
    feature_cols = [
        "population_2024",
        "cattle_sum",
        "cropland_score",
        "health_facilities_count",
        "area_sqkm",
        "center_lat",
        "center_lon",
    ]

    model_df = df.dropna(subset=feature_cols + ["high_flood_target"]).copy()
    dropped = len(df) - len(model_df)
    if dropped:
        print(f"       Dropped {dropped} rows with missing values.")

    X = model_df[feature_cols]
    y = model_df["high_flood_target"]

    return model_df, X, y, feature_cols


# ==============================================================================
# 3. Random Forest Training with GridSearchCV
# ==============================================================================
def train_random_forest(X: pd.DataFrame, y: pd.Series):
    """
    Split data into train/test, tune a Random Forest with GridSearchCV
    (5-fold stratified CV on the training set), then evaluate on the hold-out
    test set.

    Returns the best fitted model and test-set data for evaluation plots.
    """
    print("[INFO] Splitting data (75 % train / 25 % test, stratified) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    # ── Hyperparameter grid ─────────────────────────────────────────────────
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [4, 6, 8, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    base_rf = RandomForestClassifier(
        class_weight="balanced",   # handles class imbalance
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("[INFO] Running GridSearchCV (5-fold stratified CV) ...")
    print(f"       Grid size: {np.prod([len(v) for v in param_grid.values()])} combinations")

    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        refit=True,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    print(f"\n       Best CV ROC-AUC   : {best_cv_score:.4f}")
    print(f"       Best parameters  : {best_params}")

    # ── Save CV results ─────────────────────────────────────────────────────
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    cv_out = OUTPUT_DIR / "random_forest_cv_results.csv"
    cv_results_df.to_csv(cv_out, index=False)
    print(f"[SAVED] CV results → {cv_out.name}")

    # ── Evaluate best model on hold-out test set ────────────────────────────
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    print("\n--- Hold-out Test Set Evaluation Report ---")
    print(classification_report(y_test, y_pred, target_names=["Low/No Flood", "High Flood"]))
    test_auc = roc_auc_score(y_test, y_prob)
    print(f"Hold-out Test ROC-AUC : {test_auc:.4f}")

    # ── Additional cross-validation for final reporting ─────────────────────
    print("\n[INFO] Running final cross-validation on full training set ...")
    cv_scores = cross_validate(
        best_model, X_train, y_train,
        cv=cv,
        scoring=["roc_auc", "f1", "precision", "recall"],
        n_jobs=-1,
    )
    for metric, scores in cv_scores.items():
        if metric.startswith("test_"):
            name = metric.replace("test_", "").upper()
            print(f"       CV {name:<12}: {scores.mean():.4f} ± {scores.std():.4f}")

    return best_model, X_test, y_test, y_prob


# ==============================================================================
# 4. Save Trained Model
# ==============================================================================
def save_model(model: RandomForestClassifier, feature_cols: list):
    """Persist the trained model and metadata using joblib."""
    model_path = OUTPUT_DIR / "random_forest_model.joblib"
    joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)
    print(f"[SAVED] Trained model → {model_path.name}")
    print(f"        Load with: data = joblib.load('{model_path.name}')")
    print(f"                   model = data['model']")


# ==============================================================================
# 5. Visualizations
# ==============================================================================
def plot_feature_importance(model: RandomForestClassifier, feature_names: list):
    print("[VISUALIZATION] Generating Feature Importance Chart ...")

    importances = model.feature_importances_
    std = np.std(
        [tree.feature_importances_ for tree in model.estimators_], axis=0
    )
    indices = np.argsort(importances)

    clean_labels = [
        name.replace("_2024", "").replace("_sum", "").replace("_score", "")
           .replace("_count", "").replace("_sqkm", " (km²)").replace("_", " ").title()
        for name in feature_names
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(
        range(len(indices)),
        importances[indices],
        xerr=std[indices],
        color="#2563EB",
        alpha=0.85,
        edgecolor="#1E3A8A",
        linewidth=0.6,
        capsize=4,
        error_kw={"elinewidth": 1.2, "ecolor": "#93C5FD"},
    )

    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(
        [clean_labels[i] for i in indices], fontsize=11, fontweight="bold"
    )
    ax.set_xlabel(
        "Mean Decrease in Impurity (Feature Importance)", fontsize=11, fontweight="bold"
    )
    ax.set_title(
        "Random Forest — Feature Importance\n"
        "Drivers of High Flood Intensity in ADM3 Payams (2022)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.6, axis="x")

    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.3f}",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#1E3A8A",
        )

    plt.tight_layout()
    out_path = OUTPUT_DIR / "random_forest_feature_importance.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] {out_path.name}")


def plot_roc_curve(y_test, y_prob):
    print("[VISUALIZATION] Generating ROC Curve Plot ...")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        fpr, tpr,
        color="#2563EB", lw=2.5,
        label=f"Random Forest — Tuned Model (AUC = {auc_score:.4f})",
    )
    ax.plot(
        [0, 1], [0, 1],
        color="gray", linestyle="--", lw=1.5,
        label="Random Guess Baseline (AUC = 0.5000)",
    )

    ax.fill_between(fpr, tpr, alpha=0.08, color="#2563EB")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Receiver Operating Characteristic (ROC) Curve\n"
        "Hold-out Test Set Performance — Flood Risk Classifier",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=11, framealpha=0.95)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "random_forest_roc_curve.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] {out_path.name}")


def plot_confusion_matrix(model, X_test, y_test):
    print("[VISUALIZATION] Generating Confusion Matrix ...")

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=["Low/No Flood", "High Flood"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title(
        "Confusion Matrix — Hold-out Test Set\nRandom Forest Flood Risk Classifier",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    out_path = OUTPUT_DIR / "random_forest_confusion_matrix.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[SAVED] {out_path.name}")


# ==============================================================================
# 6. Score Full Dataset & Export
# ==============================================================================
def score_and_export_all(df: pd.DataFrame, model: RandomForestClassifier, feature_cols: list):
    print("[INFO] Scoring all ADM3 payams with the trained model ...")

    score_df = df.dropna(subset=feature_cols).copy()
    X_all = score_df[feature_cols]

    score_df["rf_flood_probability"] = model.predict_proba(X_all)[:, 1]
    score_df["rf_predicted_high_flood"] = model.predict(X_all)

    # Risk tier for readability
    score_df["rf_risk_tier"] = pd.cut(
        score_df["rf_flood_probability"],
        bins=[0, 0.25, 0.50, 0.75, 1.01],
        labels=["Low", "Moderate", "High", "Very High"],
        right=False,
    )

    df_sorted = score_df.sort_values("rf_flood_probability", ascending=False)

    out_csv = OUTPUT_DIR / "adm3_random_forest_predictions.csv"
    df_sorted.to_csv(out_csv, index=False)
    print(f"[SUCCESS] Saved predictions → {out_csv.name}")

    print("\n--- Top 10 Payams Predicted as Highest Flood Risk ---")
    display_cols = [
        "adm3_name", "adm2_name", "adm1_name",
        "rf_flood_probability", "rf_risk_tier",
        "flood_hazard_2022_count",
    ]
    available = [c for c in display_cols if c in df_sorted.columns]
    print(df_sorted.head(10)[available].to_string(index=False))


# ==============================================================================
# 7. Main
# ==============================================================================
def main():
    print("=" * 70)
    print("RANDOM FOREST FLOOD RISK PREDICTION MODEL")
    print("(trained on real historical data — no static formulas)")
    print("=" * 70)

    df = load_data()
    model_df, X, y, feature_cols = prepare_modeling_data(df)

    model, X_test, y_test, y_prob = train_random_forest(X, y)

    save_model(model, feature_cols)

    plot_feature_importance(model, feature_cols)
    plot_roc_curve(y_test, y_prob)
    plot_confusion_matrix(model, X_test, y_test)

    score_and_export_all(model_df, model, feature_cols)

    print("\n[COMPLETE] Random Forest flood risk modeling finished successfully.\n")


if __name__ == "__main__":
    main()