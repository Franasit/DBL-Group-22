#!/usr/bin/env python3
"""14d | Standalone same-month next-year ADM3 flood prediction test.

Standalone version:
- does NOT require 14a_prediction_common.py;
- includes all shared model/evaluation/report utilities internally;
- generates full CSV tables, figures, FP/FN summaries, calibration,
  confusion matrices, PR/ROC curves, yearly metrics and optional ADM3 maps.
"""
from __future__ import annotations

import argparse
import sys
from importlib import util as importlib_util

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
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
    if not 0 < quantile < 1:
        raise ValueError("target quantile must be between 0 and 1.")
    years = pd.to_numeric(df[year_col], errors="coerce")
    values = pd.to_numeric(
        df.loc[years.le(train_end), target_value_col],
        errors="coerce",
    ).dropna()
    if positive_only:
        positive = values[values.gt(0)]
        if not positive.empty:
            values = positive
    if values.empty:
        raise ValueError("No usable training target values for threshold estimation.")
    threshold = float(values.quantile(quantile))
    if not np.isfinite(threshold):
        raise ValueError("Training-only target threshold is non-finite.")
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
    years = sorted(
        pd.to_numeric(df[year_col], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    if not years:
        raise ValueError("No split years available.")

    ranges = {
        "training": (years[0], split.train_end),
        "validation": (split.val_start, split.val_end),
        "test": (split.test_start, split.test_end),
    }
    missing = []
    for name, (lo, hi) in ranges.items():
        if not any(lo <= y <= hi for y in years):
            missing.append(name)
    if missing:
        raise ValueError(
            "The available panel cannot support the requested time split. "
            f"Available years={years[0]}-{years[-1]}; missing={missing}."
        )
    if not (
        split.train_end < split.val_start <= split.val_end
        < split.test_start <= split.test_end
    ):
        raise ValueError("Time split must be strictly ordered train -> validation -> test.")


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
            max_iter=2500,
            class_weight="balanced",
            random_state=random_state,
        )),
    ])
    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=500,
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


def probability_metrics(y_true, prob) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    result = {
        "brier_score": float(brier_score_loss(y, p)),
        "pr_auc": None,
        "roc_auc": None,
    }
    if len(np.unique(y)) == 2:
        result["pr_auc"] = float(average_precision_score(y, p))
        result["roc_auc"] = float(roc_auc_score(y, p))
    return result


def classification_metrics_from_class(y_true, pred) -> dict:
    y = np.asarray(y_true, dtype=int)
    c = np.asarray(pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, c, labels=[0, 1]).ravel()
    return {
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "recall": float(recall_score(y, c, zero_division=0)),
        "precision": float(precision_score(y, c, zero_division=0)),
        "f1": float(f1_score(y, c, zero_division=0)),
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else None,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
    }


def metric_dict(y_true, prob, threshold: float) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    pred = (p >= threshold).astype(int)
    result = {
        "threshold": float(threshold),
        "n": int(len(y)),
        "positives": int(y.sum()),
        "negatives": int((1 - y).sum()),
    }
    result.update(classification_metrics_from_class(y, pred))
    result.update(probability_metrics(y, p))
    return result


def threshold_sweep(
    y_true,
    prob,
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.05)
    return pd.DataFrame(
        [metric_dict(y_true, prob, float(t)) for t in thresholds]
    )


def choose_threshold(
    sweep: pd.DataFrame,
    max_fnr: float = 0.20,
) -> float:
    if not 0 <= max_fnr <= 1:
        raise ValueError("max_validation_fnr must be between 0 and 1.")

    acceptable = sweep.loc[
        sweep["false_negative_rate"].notna()
        & sweep["false_negative_rate"].le(max_fnr)
    ].copy()

    if not acceptable.empty:
        row = acceptable.sort_values(
            ["precision", "f1", "threshold"],
            ascending=[False, False, False],
            kind="stable",
        ).iloc[0]
    else:
        row = sweep.sort_values(
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
        pd.DataFrame({"actual": y, "probability": p, "bin": bins})
        .groupby("bin", observed=True)
        .agg(
            n=("actual", "size"),
            mean_predicted_probability=("probability", "mean"),
            observed_high_flood_rate=("actual", "mean"),
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
    keep = [id_col, year_col, *time_cols, target_col]
    out = frame[keep].copy()
    out["model"] = model_name
    out["model_threshold"] = float(threshold)
    out["predicted_probability"] = np.asarray(prob, dtype=float)
    out["predicted_high_flood"] = out["predicted_probability"].ge(threshold)

    y = out[target_col].astype(int)
    c = out["predicted_high_flood"].astype(int)
    out["outcome"] = np.select(
        [
            y.eq(1) & c.eq(1),
            y.eq(0) & c.eq(1),
            y.eq(1) & c.eq(0),
            y.eq(0) & c.eq(0),
        ],
        ["TP", "FP", "FN", "TN"],
        default="UNKNOWN",
    )
    return out


def summarize_predictions(
    pred: pd.DataFrame,
    group_cols: list[str],
    target_col: str = "target_high_flood",
) -> pd.DataFrame:
    rows = []
    for keys, g in pred.groupby(["model", *group_cols], sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        model = keys[0]
        group_values = keys[1:]
        y = g[target_col].astype(int)
        c = g["predicted_high_flood"].astype(int)
        prob = g["predicted_probability"].astype(float)

        row = {
            "model": model,
            "n": int(len(g)),
            "positives": int(y.sum()),
            "negatives": int((1 - y).sum()),
            "mean_predicted_probability": float(prob.mean()),
        }
        for col, value in zip(group_cols, group_values):
            row[col] = value
        row.update(classification_metrics_from_class(y, c))
        row.update(probability_metrics(y, prob))
        rows.append(row)
    return pd.DataFrame(rows)


def write_model_metric_comparison(
    results: dict,
    outdir: Path,
) -> pd.DataFrame:
    rows = []
    for model, details in results.items():
        row = {"model": model}
        row.update(details["test_metrics"])
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(outdir / "01_model_metrics_comparison.csv", index=False)

    metrics = [
        "recall", "precision", "f1", "false_negative_rate",
        "false_positive_rate", "pr_auc", "roc_auc", "brier_score",
    ]
    long = table.melt(
        id_vars=["model"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )
    long.to_csv(outdir / "02_model_metrics_long.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = long.pivot(index="metric", columns="model", values="value")
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Final test metrics by model")
    ax.set_ylabel("Metric value")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Model")
    fig.tight_layout()
    fig.savefig(outdir / "01_model_metrics_comparison.png", dpi=220)
    plt.close(fig)

    return table


def plot_confusion_matrix(pred: pd.DataFrame, outdir: Path) -> None:
    for model, g in pred.groupby("model"):
        y = g["target_high_flood"].astype(int)
        c = g["predicted_high_flood"].astype(int)
        cm = confusion_matrix(y, c, labels=[0, 1])

        fig, ax = plt.subplots(figsize=(5.4, 4.8))
        im = ax.imshow(cm)
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, f"{value:,}", ha="center", va="center", fontsize=12)
        ax.set_xticks([0, 1], ["Predicted low", "Predicted high"])
        ax.set_yticks([0, 1], ["Actual low", "Actual high"])
        ax.set_title(f"Confusion matrix — {model}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(outdir / f"02_confusion_matrix_{model}.png", dpi=220)
        plt.close(fig)


def plot_threshold_sweeps(outdir: Path, sweeps: dict[str, pd.DataFrame]) -> None:
    for model, s in sweeps.items():
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for col in [
            "recall", "precision", "false_negative_rate",
            "false_positive_rate", "f1",
        ]:
            ax.plot(s["threshold"], s[col], marker="o", label=col)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Probability threshold")
        ax.set_ylabel("Rate / score")
        ax.set_title(f"Validation threshold trade-off — {model}")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(outdir / f"03_threshold_tradeoff_{model}.png", dpi=220)
        plt.close(fig)


def plot_pr_roc(pred: pd.DataFrame, outdir: Path) -> None:
    fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))

    for model, g in pred.groupby("model"):
        y = g["target_high_flood"].astype(int).to_numpy()
        p = g["predicted_probability"].astype(float).to_numpy()
        if len(np.unique(y)) < 2:
            continue

        precision, recall, _ = precision_recall_curve(y, p)
        ax_pr.plot(recall, precision, label=f"{model} (AP={average_precision_score(y, p):.3f})")

        fpr, tpr, _ = roc_curve(y, p)
        ax_roc.plot(fpr, tpr, label=f"{model} (AUC={roc_auc_score(y, p):.3f})")

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision–Recall curves — final test")
    ax_pr.legend()
    ax_pr.grid(alpha=0.2)
    fig_pr.tight_layout()
    fig_pr.savefig(outdir / "04_precision_recall_curves.png", dpi=220)
    plt.close(fig_pr)

    ax_roc.plot([0, 1], [0, 1], linestyle="--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC curves — final test")
    ax_roc.legend()
    ax_roc.grid(alpha=0.2)
    fig_roc.tight_layout()
    fig_roc.savefig(outdir / "05_roc_curves.png", dpi=220)
    plt.close(fig_roc)


def plot_calibration(pred: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", label="perfect calibration")

    for model, g in pred.groupby("model"):
        tab = calibration_table(
            g["target_high_flood"].astype(int),
            g["predicted_probability"].astype(float),
        )
        tab.to_csv(outdir / f"calibration_{model}.csv", index=False)
        ax.plot(
            tab["mean_predicted_probability"],
            tab["observed_high_flood_rate"],
            marker="o",
            label=model,
        )

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed high-flood rate")
    ax.set_title("Calibration — final test")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / "06_calibration_curves.png", dpi=220)
    plt.close(fig)


def plot_year_metrics(
    by_year: pd.DataFrame,
    year_col: str,
    outdir: Path,
) -> None:
    for metric in ["recall", "false_negative_rate", "false_positive_rate", "precision", "f1"]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for model, g in by_year.groupby("model"):
            g = g.sort_values(year_col)
            ax.plot(g[year_col], g[metric], marker="o", label=model)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Test year")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by test year")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(outdir / f"07_year_{metric}.png", dpi=220)
        plt.close(fig)


def plot_top_region_errors(
    by_region: pd.DataFrame,
    id_col: str,
    outdir: Path,
    top_n: int = 20,
) -> None:
    for error_col in ["FN", "FP"]:
        for model, g in by_region.groupby("model"):
            top = g.sort_values(
                [error_col, id_col],
                ascending=[False, True],
            ).head(top_n)
            if top.empty:
                continue
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.barh(top[id_col].astype(str), top[error_col])
            ax.invert_yaxis()
            ax.set_xlabel(f"{error_col} count")
            ax.set_ylabel("ADM3")
            ax.set_title(f"Top ADM3 by {error_col} count — {model}")
            fig.tight_layout()
            fig.savefig(
                outdir / f"08_top_adm3_{error_col.lower()}_{model}.png",
                dpi=220,
            )
            plt.close(fig)


def plot_fp_fn_by_year(
    pred: pd.DataFrame,
    year_col: str,
    outdir: Path,
) -> None:
    counts = (
        pred.loc[pred["outcome"].isin(["FP", "FN"])]
        .groupby(["model", year_col, "outcome"])
        .size()
        .rename("count")
        .reset_index()
    )
    counts.to_csv(outdir / "03_fp_fn_counts_by_year.csv", index=False)

    for model, g in counts.groupby("model"):
        pivot = g.pivot(index=year_col, columns="outcome", values="count").fillna(0)
        fig, ax = plt.subplots(figsize=(8, 5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(f"False positives vs false negatives by year — {model}")
        ax.set_ylabel("ADM3-period cases")
        ax.set_xlabel("Test year")
        ax.tick_params(axis="x", rotation=0)
        fig.tight_layout()
        fig.savefig(outdir / f"09_fp_fn_by_year_{model}.png", dpi=220)
        plt.close(fig)


def try_plot_region_maps(
    by_region: pd.DataFrame,
    id_col: str,
    boundary_path: Path | None,
    outdir: Path,
) -> None:
    if boundary_path is None or not boundary_path.is_file():
        return
    try:
        import geopandas as gpd
    except ImportError:
        return

    geo = gpd.read_file(boundary_path)
    if geo.crs is None:
        return

    candidates = ["adm3_pcode", "ADM3_PCODE", "adm3_id", "ADM3_ID", "adm_id"]
    code_col = next((c for c in candidates if c in geo.columns), None)
    if code_col is None:
        return

    geo = geo.copy()
    geo[id_col] = geo[code_col].astype("string").str.strip().str.upper()

    for model, g in by_region.groupby("model"):
        merged = geo.merge(g, on=id_col, how="left", validate="one_to_one")
        for metric in ["FN", "FP", "false_negative_rate", "false_positive_rate"]:
            fig, ax = plt.subplots(figsize=(9, 10))
            merged.plot(
                column=metric,
                ax=ax,
                legend=True,
                missing_kwds={"color": "lightgrey"},
                edgecolor="white",
                linewidth=0.2,
            )
            geo.boundary.plot(ax=ax, color="black", linewidth=0.5)
            ax.set_axis_off()
            ax.set_title(f"{metric} by ADM3 — {model}")
            fig.tight_layout()
            fig.savefig(
                outdir / f"10_map_{metric}_{model}.png",
                dpi=220,
            )
            plt.close(fig)


def generate_report_artifacts(
    results: dict,
    combined_predictions: pd.DataFrame,
    validation_sweeps: dict[str, pd.DataFrame],
    id_col: str,
    year_col: str,
    outdir: Path,
    boundary_path: Path | None = None,
) -> None:
    tables = outdir / "tables"
    figures = outdir / "figures"
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)

    metric_table = write_model_metric_comparison(results, tables)

    by_year = summarize_predictions(
        combined_predictions,
        [year_col],
    )
    by_year.to_csv(tables / "04_test_metrics_by_year.csv", index=False)

    by_region = summarize_predictions(
        combined_predictions,
        [id_col],
    )
    by_region.to_csv(tables / "05_test_metrics_by_adm3.csv", index=False)

    fp = combined_predictions.loc[combined_predictions["outcome"].eq("FP")].copy()
    fn = combined_predictions.loc[combined_predictions["outcome"].eq("FN")].copy()
    fp.to_csv(tables / "06_false_positives.csv", index=False)
    fn.to_csv(tables / "07_false_negatives.csv", index=False)

    # Save sweeps into tables folder too.
    for model, sweep in validation_sweeps.items():
        sweep.to_csv(
            tables / f"validation_threshold_sweep_{model}.csv",
            index=False,
        )

    combined_predictions.to_csv(
        tables / "08_all_test_predictions.csv",
        index=False,
    )

    plot_confusion_matrix(combined_predictions, figures)
    plot_threshold_sweeps(figures, validation_sweeps)
    plot_pr_roc(combined_predictions, figures)
    plot_calibration(combined_predictions, figures)
    plot_year_metrics(by_year, year_col, figures)
    plot_top_region_errors(by_region, id_col, figures)
    plot_fp_fn_by_year(combined_predictions, year_col, figures)
    try_plot_region_maps(by_region, id_col, boundary_path, figures)


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
    boundary_path: Path | None = None,
) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)

    model_df = df.dropna(subset=[target_col]).copy()
    model_df[target_col] = model_df[target_col].astype(int)
    validate_split_years(model_df, year_col, split)

    train, val, test = split_frames(model_df, year_col, split)
    if train[target_col].nunique() < 2:
        raise ValueError("Training target has only one class.")
    if val[target_col].nunique() < 2:
        raise ValueError("Validation target has only one class.")
    if test[target_col].nunique() < 2:
        raise ValueError("Test target has only one class.")

    X_train = train[feature_cols]
    y_train = train[target_col].astype(int)
    X_val = val[feature_cols]
    y_val = val[target_col].astype(int)
    X_test = test[feature_cols]
    y_test = test[target_col].astype(int)

    results: dict = {}
    validation_sweeps: dict[str, pd.DataFrame] = {}
    all_test_predictions = []

    # Persistence baseline: use current-period flood signal if available;
    # otherwise first lag1 feature.
    persistence_candidates = [
        c for c in feature_cols
        if c in {"current_flood_days", "current_flood_event_days"}
    ]
    if not persistence_candidates:
        persistence_candidates = [c for c in feature_cols if c.startswith("lag1_")]
    if not persistence_candidates:
        raise ValueError("No persistence feature available.")
    persistence_col = persistence_candidates[0]

    train_vals = (
        pd.to_numeric(train[persistence_col], errors="coerce")
        .dropna()
        .sort_values()
        .to_numpy()
    )
    if len(train_vals) == 0:
        raise ValueError("Persistence baseline has no training values.")

    def empirical_prob(values):
        x = pd.to_numeric(values, errors="coerce").to_numpy()
        fill = float(np.nanmedian(train_vals))
        x = np.where(np.isfinite(x), x, fill)
        return np.searchsorted(train_vals, x, side="right") / len(train_vals)

    val_prob = empirical_prob(val[persistence_col])
    sweep = threshold_sweep(y_val, val_prob)
    validation_sweeps["persistence"] = sweep
    threshold = choose_threshold(sweep, max_validation_fnr)
    test_prob = empirical_prob(test[persistence_col])
    metrics = metric_dict(y_test, test_prob, threshold)
    results["persistence"] = {
        "threshold": threshold,
        "test_metrics": metrics,
        "persistence_feature": persistence_col,
    }
    all_test_predictions.append(
        prediction_rows(
            test, test_prob, threshold,
            id_col, year_col, time_cols, target_col, "persistence",
        )
    )

    for model_name, model in make_models(random_state).items():
        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        sweep = threshold_sweep(y_val, val_prob)
        validation_sweeps[model_name] = sweep
        threshold = choose_threshold(sweep, max_validation_fnr)
        test_prob = model.predict_proba(X_test)[:, 1]
        metrics = metric_dict(y_test, test_prob, threshold)
        results[model_name] = {
            "threshold": threshold,
            "test_metrics": metrics,
        }
        all_test_predictions.append(
            prediction_rows(
                test, test_prob, threshold,
                id_col, year_col, time_cols, target_col, model_name,
            )
        )

    combined = pd.concat(all_test_predictions, ignore_index=True)

    with (outdir / "model_test_summary.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    generate_report_artifacts(
        results,
        combined,
        validation_sweeps,
        id_col=id_col,
        year_col=year_col,
        outdir=outdir,
        boundary_path=boundary_path,
    )
    return results

# ===== SAME-MONTH NEXT-YEAR PIPELINE =====

def project_root() -> Path:
    for start in (Path(__file__).resolve().parent, Path.cwd()):
        for p in (start, *start.parents):
            if (p / "processed_data").is_dir() and (p / "scripts").is_dir():
                return p
    return Path.cwd()


def build_same_month_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    out = out.sort_values(["adm_id", "month", "year"]).reset_index(drop=True)

    group = out.groupby(["adm_id", "month"], sort=False)

    # Information known in year t for the same calendar month.
    out["current_same_month_flood_days"] = out["flood_days"]
    out["lag1_same_month_flood_days"] = group["flood_days"].shift(1)
    out["lag2_same_month_flood_days"] = group["flood_days"].shift(2)
    out["lag3_same_month_flood_days"] = group["flood_days"].shift(3)

    out["previous_3yr_same_month_mean"] = (
        group["flood_days"]
        .rolling(3, min_periods=2)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    # Target = same month in the next calendar year.
    out["target_year"] = group["year"].shift(-1)
    out["target_month"] = group["month"].shift(-1)
    out["target_flood_days"] = group["flood_days"].shift(-1)

    consecutive_year = out["target_year"].eq(out["year"] + 1)
    same_month = out["target_month"].eq(out["month"])
    valid = consecutive_year & same_month

    out.loc[
        ~valid,
        ["target_year", "target_month", "target_flood_days"],
    ] = np.nan

    # Month encoding is retained because the model pools all months nationally.
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)

    features = [
        "current_same_month_flood_days",
        "lag1_same_month_flood_days",
        "lag2_same_month_flood_days",
        "lag3_same_month_flood_days",
        "previous_3yr_same_month_mean",
        "month_sin",
        "month_cos",
    ]
    return out, features


def merge_optional_same_month_drivers(
    panel: pd.DataFrame,
    driver_path: Path | None,
    features: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    if driver_path is None:
        return panel, features

    d = pd.read_csv(driver_path, dtype={"adm_id": "string"})
    required = {"adm_id", "year", "month"}
    if not required.issubset(d.columns):
        raise ValueError(f"Monthly driver CSV requires {sorted(required)}.")

    d["adm_id"] = d["adm_id"].astype("string").str.strip().str.upper()
    for c in ["year", "month"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    ensure_unique(d, ["adm_id", "year", "month"], "monthly drivers")

    numeric = [
        c for c in d.columns
        if c not in {"adm_id", "year", "month"}
        and pd.api.types.is_numeric_dtype(d[c])
    ]
    if not numeric:
        raise ValueError("No numeric monthly driver columns found.")

    # Year-t same-month drivers are known before predicting year t+1 same month.
    rename = {c: f"driver_same_month__{c}" for c in numeric}
    d = d.rename(columns=rename)
    keep = ["adm_id", "year", "month", *rename.values()]

    merged = panel.merge(
        d[keep],
        on=["adm_id", "year", "month"],
        how="left",
        validate="one_to_one",
    )
    return merged, features + list(rename.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path)
    ap.add_argument("--panel", type=Path)
    ap.add_argument("--driver-csv", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--target-quantile", type=float, default=0.75)
    ap.add_argument("--max-validation-fnr", type=float, default=0.20)
    ap.add_argument("--train-end", type=int, default=2018)
    ap.add_argument("--val-start", type=int, default=2019)
    ap.add_argument("--val-end", type=int, default=2022)
    ap.add_argument("--test-start", type=int, default=2023)
    ap.add_argument("--test-end", type=int, default=2025)
    args = ap.parse_args()

    root = args.root.resolve() if args.root else project_root()
    panel_path = args.panel or (
        root
        / "processed_data/multivariate_flood_modeling/"
        "05a_monthly_panel/adm3_monthly_panel.csv"
    )
    out = args.out or (
        root
        / "processed_data/multivariate_flood_modeling/"
        "14d_same_month_next_year_prediction_test"
    )
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(panel_path, dtype={"adm_id": "string"})
    required = {"adm_id", "year", "month", "flood_days"}
    if not required.issubset(df.columns):
        raise ValueError(f"Monthly panel requires {sorted(required)}.")

    ensure_unique(df, ["adm_id", "year", "month"], "monthly flood panel")

    df["adm_id"] = df["adm_id"].astype("string").str.strip().str.upper()
    for c in ["year", "month", "flood_days"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.loc[df["month"].between(1, 12)].copy()

    if "target_status" in df.columns:
        (
            df.groupby("target_status", dropna=False)
            .size()
            .rename("rows")
            .reset_index()
            .to_csv(out / "target_status_inventory.csv", index=False)
        )

    panel, features = build_same_month_features(df)

    driver = None
    if args.driver_csv:
        driver = (
            args.driver_csv
            if args.driver_csv.is_absolute()
            else root / args.driver_csv
        )
    panel, features = merge_optional_same_month_drivers(
        panel, driver, features
    )

    split = SplitConfig(
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )

    threshold_source = panel.rename(columns={"target_year": "split_year"})
    threshold = training_quantile_threshold(
        threshold_source,
        "target_flood_days",
        "split_year",
        split.train_end,
        args.target_quantile,
        positive_only=True,
    )

    panel = add_binary_target(
        panel,
        "target_flood_days",
        threshold,
    )
    panel["split_year"] = panel["target_year"]

    model_cols = [
        "adm_id",
        "split_year",
        "target_month",
        "target_high_flood",
        *features,
    ]
    model = panel[model_cols].dropna(
        subset=["split_year", "target_high_flood"]
    ).copy()
    model["split_year"] = model["split_year"].astype(int)
    model["target_month"] = model["target_month"].astype(int)

    panel.to_csv(
        out / "same_month_next_year_prediction_panel.csv",
        index=False,
    )

    boundary = (
        root
        / "raw_data/administrative_boundaries/ssd_admin3.geojson"
    )

    pd.DataFrame([{
        "forecast_unit": "ADM3-month",
        "forecast_horizon": "same month, one year ahead",
        "example": "September 2023 -> September 2024",
        "target_definition": (
            "same-month next-year flood_days >= "
            "training-only positive-value quantile"
        ),
        "target_quantile": args.target_quantile,
        "training_only_target_threshold_flood_days": threshold,
        "train_end": split.train_end,
        "validation_target_years": f"{split.val_start}-{split.val_end}",
        "test_target_years": f"{split.test_start}-{split.test_end}",
        "features": ";".join(features),
        "exposure_factors_used": False,
    }]).to_csv(
        out / "00_target_and_feature_definition.csv",
        index=False,
    )

    results = fit_evaluate_models(
        model,
        feature_cols=features,
        id_col="adm_id",
        year_col="split_year",
        time_cols=["target_month"],
        target_col="target_high_flood",
        split=split,
        outdir=out,
        max_validation_fnr=args.max_validation_fnr,
        boundary_path=boundary if boundary.is_file() else None,
    )

    print(json.dumps(results, indent=2))
    print(f"[OK] Same-month next-year tables and figures: {out}")


if __name__ == "__main__":
    main()
