#!/usr/bin/env python3
"""14a | Shared utilities for leakage-aware ADM3 flood prediction tests.

Models:
- Persistence baseline
- Logistic Regression
- Random Forest

Evaluation:
- TP / FP / FN / TN
- Recall, Precision, F1
- False Negative Rate (FNR)
- False Positive Rate (FPR)
- PR-AUC, ROC-AUC, Brier score
- Calibration table
- Per-year and per-ADM3 error summaries
- Threshold sweep on validation only
- Final threshold locked before testing 2023-2025

This module does not read exposure factors. Health facilities, cropland,
rangeland and population belong to downstream exposure/prioritization analysis,
not to the core hazard model.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from sklearn.calibration import calibration_curve
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise SystemExit(
        "Missing scikit-learn. Install with:\n"
        "python -m pip install scikit-learn"
    ) from exc


@dataclass(frozen=True)
class SplitConfig:
    train_end: int = 2018
    val_start: int = 2019
    val_end: int = 2022
    test_start: int = 2023
    test_end: int = 2025


def safe_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    x = series.astype("string").str.strip().str.casefold()
    mapping = {
        "true": True, "1": True, "1.0": True,
        "false": False, "0": False, "0.0": False,
    }
    return x.map(mapping).astype("boolean")


def ensure_unique(df: pd.DataFrame, keys: list[str], label: str) -> None:
    dup = df.duplicated(keys, keep=False)
    if dup.any():
        raise ValueError(
            f"{label}: duplicate rows for {keys}. Examples:\n"
            + df.loc[dup, keys].head(12).to_string(index=False)
        )


def training_quantile_threshold(
    df: pd.DataFrame,
    target_value_col: str,
    year_col: str,
    train_end: int,
    quantile: float,
    positive_only: bool = True,
) -> float:
    train = pd.to_numeric(
        df.loc[df[year_col].le(train_end), target_value_col],
        errors="coerce",
    ).dropna()
    if positive_only:
        positive = train[train.gt(0)]
        if not positive.empty:
            train = positive
    if train.empty:
        raise ValueError("No usable training target values for threshold estimation.")
    threshold = float(train.quantile(quantile))
    if not np.isfinite(threshold):
        raise ValueError("Training target threshold is non-finite.")
    return threshold


def add_binary_target(
    df: pd.DataFrame,
    target_value_col: str,
    threshold: float,
    output_col: str = "target_high_flood",
) -> pd.DataFrame:
    out = df.copy()
    x = pd.to_numeric(out[target_value_col], errors="coerce")
    out[output_col] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    known = x.notna()
    out.loc[known, output_col] = x.loc[known].ge(threshold).to_numpy()
    return out


def validate_split_years(df: pd.DataFrame, year_col: str, split: SplitConfig) -> None:
    years = sorted(pd.to_numeric(df[year_col], errors="coerce").dropna().astype(int).unique())
    if not years:
        raise ValueError("No years available.")
    required_ranges = {
        "training": (min(years), split.train_end),
        "validation": (split.val_start, split.val_end),
        "test": (split.test_start, split.test_end),
    }
    counts = {
        name: sum(lo <= y <= hi for y in years)
        for name, (lo, hi) in required_ranges.items()
    }
    missing = [k for k, v in counts.items() if v == 0]
    if missing:
        raise ValueError(
            "The available panel cannot support the requested time split. "
            f"Available years={years[0]}-{years[-1]}; missing={missing}. "
            "Use explicit split arguments only if they preserve a true future holdout."
        )


def split_frames(df: pd.DataFrame, year_col: str, split: SplitConfig):
    train = df.loc[df[year_col].le(split.train_end)].copy()
    val = df.loc[df[year_col].between(split.val_start, split.val_end)].copy()
    test = df.loc[df[year_col].between(split.test_start, split.test_end)].copy()
    return train, val, test


def make_models(random_state: int = 42):
    logistic = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        )),
    ])
    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )),
    ])
    return {
        "logistic_regression": logistic,
        "random_forest": rf,
    }


def metric_dict(y_true, prob, threshold: float) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    recall = recall_score(y_true, pred, zero_division=0)
    precision = precision_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)
    fnr = fn / (fn + tp) if (fn + tp) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) else np.nan

    out = {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "negatives": int((1 - y_true).sum()),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "false_negative_rate": float(fnr) if np.isfinite(fnr) else None,
        "false_positive_rate": float(fpr) if np.isfinite(fpr) else None,
        "brier_score": float(brier_score_loss(y_true, prob)),
    }
    if len(np.unique(y_true)) == 2:
        out["pr_auc"] = float(average_precision_score(y_true, prob))
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
    else:
        out["pr_auc"] = None
        out["roc_auc"] = None
    return out


def threshold_sweep(
    y_true,
    prob,
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.05)
    return pd.DataFrame([metric_dict(y_true, prob, float(t)) for t in thresholds])


def choose_threshold(
    sweep: pd.DataFrame,
    max_fnr: float = 0.20,
) -> float:
    """Choose on validation only.

    Primary objective:
    - satisfy FNR <= max_fnr when possible;
    - among those, maximize precision then F1;
    - otherwise maximize F1, then recall.

    This encodes the anticipatory-action preference to avoid missed high-flood
    cases while still limiting false alarms.
    """
    s = sweep.copy()
    valid = s["false_negative_rate"].notna()
    acceptable = s.loc[valid & s["false_negative_rate"].le(max_fnr)].copy()
    if not acceptable.empty:
        row = acceptable.sort_values(
            ["precision", "f1", "threshold"],
            ascending=[False, False, False],
            kind="stable",
        ).iloc[0]
    else:
        row = s.sort_values(
            ["f1", "recall", "threshold"],
            ascending=[False, False, False],
            kind="stable",
        ).iloc[0]
    return float(row["threshold"])


def calibration_table(y_true, prob, n_bins: int = 10) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    bins = pd.cut(
        p,
        bins=np.linspace(0, 1, n_bins + 1),
        include_lowest=True,
        duplicates="drop",
    )
    return (
        pd.DataFrame({"y": y, "prob": p, "bin": bins})
        .groupby("bin", observed=True)
        .agg(
            n=("y", "size"),
            mean_predicted_probability=("prob", "mean"),
            observed_high_flood_rate=("y", "mean"),
        )
        .reset_index()
    )


def prediction_rows(
    frame: pd.DataFrame,
    prob,
    threshold: float,
    id_col: str,
    year_col: str,
    time_cols: list[str],
    target_col: str,
    model_name: str,
) -> pd.DataFrame:
    out = frame[[id_col, year_col, *time_cols, target_col]].copy()
    out["model"] = model_name
    out["predicted_probability"] = np.asarray(prob, dtype=float)
    out["predicted_high_flood"] = out["predicted_probability"].ge(threshold)
    y = out[target_col].astype(int)
    p = out["predicted_high_flood"].astype(int)
    out["outcome"] = np.select(
        [
            y.eq(1) & p.eq(1),
            y.eq(0) & p.eq(1),
            y.eq(1) & p.eq(0),
            y.eq(0) & p.eq(0),
        ],
        ["TP", "FP", "FN", "TN"],
        default="UNKNOWN",
    )
    return out


def summarize_by_year(pred: pd.DataFrame, year_col: str) -> pd.DataFrame:
    rows = []
    for (model, year), g in pred.groupby(["model", year_col], sort=True):
        y = g["target_high_flood"].astype(int)
        p = g["predicted_high_flood"].astype(int)
        prob = g["predicted_probability"]
        row = {"model": model, year_col: int(year)}
        row.update(metric_dict(y, prob, threshold=0.5))
        # Recompute confusion at actual stored prediction threshold/class.
        tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
        row.update({"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)})
        row["recall"] = tp / (tp + fn) if tp + fn else np.nan
        row["false_negative_rate"] = fn / (tp + fn) if tp + fn else np.nan
        row["false_positive_rate"] = fp / (fp + tn) if fp + tn else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_region(pred: pd.DataFrame, id_col: str) -> pd.DataFrame:
    rows = []
    for (model, rid), g in pred.groupby(["model", id_col], sort=True):
        counts = g["outcome"].value_counts()
        tp = int(counts.get("TP", 0))
        fp = int(counts.get("FP", 0))
        fn = int(counts.get("FN", 0))
        tn = int(counts.get("TN", 0))
        rows.append({
            "model": model,
            id_col: rid,
            "n_test_periods": int(len(g)),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "recall": tp / (tp + fn) if tp + fn else np.nan,
            "false_negative_rate": fn / (tp + fn) if tp + fn else np.nan,
            "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
            "mean_predicted_probability": float(g["predicted_probability"].mean()),
            "actual_high_flood_periods": int(g["target_high_flood"].astype(int).sum()),
        })
    return pd.DataFrame(rows)


def fit_evaluate_models(
    df: pd.DataFrame,
    feature_cols: list[str],
    id_col: str,
    year_col: str,
    time_cols: list[str],
    target_col: str,
    split: SplitConfig,
    outdir: Path,
    max_validation_fnr: float = 0.20,
    random_state: int = 42,
) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)

    model_df = df.dropna(subset=[target_col]).copy()
    model_df[target_col] = model_df[target_col].astype(int)

    validate_split_years(model_df, year_col, split)
    train, val, test = split_frames(model_df, year_col, split)

    if train[target_col].nunique() < 2:
        raise ValueError("Training target has only one class.")
    if val[target_col].nunique() < 2:
        raise ValueError("Validation target has only one class; threshold tuning is not meaningful.")
    if test[target_col].nunique() < 2:
        raise ValueError("Test target has only one class; FP/FN comparison is not meaningful.")

    X_train = train[feature_cols]
    y_train = train[target_col].astype(int)
    X_val = val[feature_cols]
    y_val = val[target_col].astype(int)
    X_test = test[feature_cols]
    y_test = test[target_col].astype(int)

    outputs = {}
    all_test_predictions = []

    # Persistence baseline uses the first lagged flood feature.
    persistence_candidates = [c for c in feature_cols if c.startswith("lag1_")]
    if not persistence_candidates:
        raise ValueError("At least one lag1_ feature is required for the persistence baseline.")
    persistence_col = persistence_candidates[0]
    # Normalize persistence values into a probability using training empirical CDF.
    train_vals = pd.to_numeric(train[persistence_col], errors="coerce").dropna().sort_values().to_numpy()
    if len(train_vals) == 0:
        raise ValueError("Persistence baseline has no training values.")

    def empirical_prob(values):
        x = pd.to_numeric(values, errors="coerce").fillna(np.nanmedian(train_vals)).to_numpy()
        return np.searchsorted(train_vals, x, side="right") / len(train_vals)

    val_prob = empirical_prob(val[persistence_col])
    val_sweep = threshold_sweep(y_val, val_prob)
    threshold = choose_threshold(val_sweep, max_validation_fnr)
    test_prob = empirical_prob(test[persistence_col])

    val_sweep.to_csv(outdir / "persistence_validation_threshold_sweep.csv", index=False)
    pd.DataFrame([metric_dict(y_test, test_prob, threshold)]).to_csv(
        outdir / "persistence_test_metrics.csv", index=False
    )
    pred = prediction_rows(
        test, test_prob, threshold, id_col, year_col, time_cols, target_col, "persistence"
    )
    all_test_predictions.append(pred)
    calibration_table(y_test, test_prob).to_csv(
        outdir / "persistence_test_calibration.csv", index=False
    )
    outputs["persistence"] = {
        "threshold": threshold,
        "test_metrics": metric_dict(y_test, test_prob, threshold),
    }

    for model_name, model in make_models(random_state).items():
        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        sweep = threshold_sweep(y_val, val_prob)
        threshold = choose_threshold(sweep, max_validation_fnr)
        test_prob = model.predict_proba(X_test)[:, 1]

        sweep.to_csv(outdir / f"{model_name}_validation_threshold_sweep.csv", index=False)
        metrics = metric_dict(y_test, test_prob, threshold)
        pd.DataFrame([metrics]).to_csv(
            outdir / f"{model_name}_test_metrics.csv", index=False
        )
        calibration_table(y_test, test_prob).to_csv(
            outdir / f"{model_name}_test_calibration.csv", index=False
        )

        pred = prediction_rows(
            test, test_prob, threshold, id_col, year_col, time_cols, target_col, model_name
        )
        all_test_predictions.append(pred)
        outputs[model_name] = {
            "threshold": threshold,
            "test_metrics": metrics,
        }

    combined = pd.concat(all_test_predictions, ignore_index=True)
    combined.to_csv(outdir / "all_test_predictions.csv", index=False)
    summarize_by_year(combined, year_col).to_csv(
        outdir / "test_metrics_by_year.csv", index=False
    )
    summarize_by_region(combined, id_col).to_csv(
        outdir / "test_errors_by_adm3.csv", index=False
    )

    # Dedicated FP/FN extracts for mapping/review.
    combined.loc[combined.outcome.eq("FN")].to_csv(
        outdir / "false_negatives.csv", index=False
    )
    combined.loc[combined.outcome.eq("FP")].to_csv(
        outdir / "false_positives.csv", index=False
    )

    with (outdir / "model_test_summary.json").open("w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    return outputs
