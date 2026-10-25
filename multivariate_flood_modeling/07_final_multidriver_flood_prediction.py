#!/usr/bin/env python3
"""JBG060 Script 07: out-of-time multi-driver monthly event-proxy forecasting.

Run directly in VS Code, no parameters required:
    python scripts/multivariate_flood_modeling/07_final_multidriver_flood_prediction.py

Inputs (already produced by 05a):
 processed_data/multivariate_flood_modeling/05a_monthly_panel/
   adm2_monthly_panel.csv
   adm3_monthly_panel.csv

Forecast horizon: next month. All features from t-1, t-2 and t-12.
Training 2001-2019 | Validation 2020-2022 | Test 2023-2025.
Uses exactly the same complete-case region-month rows for ALL variants
within a spatial level, for a paired/fair driver comparison.

Targets are EVENT-DATE PROXIES if 05a was run with --event-proxy, NOT
validated daily inundation observations or operational warnings. Never
reinterpret zero recorded events as scientifically verified zero flooding.

NO automatically inferred Lake/DFO relationships. Use separate reviewed
spatial integration if/when those data have documented connections.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, confusion_matrix,
)

YEARS = {"train": (2001, 2019), "validation": (2020, 2022), "test": (2023, 2025)}
LAGS = (1, 2, 12)
DRIVERS = ("rainfall_mm", "runoff_mm", "et0_mm")
MODELS = {
    "M0_history": (),
    "M1_rainfall": ("rainfall_mm",),
    "M2_runoff": ("runoff_mm",),
    "M3_et0": ("et0_mm",),
    "M4_all_drivers": DRIVERS,
}
LABEL = "EXPLORATORY_EVENT_DATE_PROXY"


def project_root() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name == "multivariate_flood_modeling" and p.parent.parent.name == "scripts":
        return p.parents[2]
    return Path.cwd()


def load_panel(path: Path, level: str) -> tuple[pd.DataFrame, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run 05a_build_monthly_panel.py first.")
    df = pd.read_csv(path, dtype={"adm_id": "string"}, low_memory=False)
    required = {"adm_id", "year", "month", "flood_days", *DRIVERS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{level} missing columns: {sorted(missing)}")
    df["adm_id"] = df["adm_id"].str.strip()
    for c in ["year", "month", "flood_days", *DRIVERS]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[["adm_id", "year", "month"]].isna().any().any():
        raise ValueError(f"{level}: missing/invalid region-year-month keys")
    if ((df.year % 1 != 0) | (df.month % 1 != 0) |
        (~df.month.between(1, 12))).any():
        raise ValueError(f"{level}: invalid year/month")
    df[["year", "month"]] = df[["year", "month"]].astype(int)
    if df.duplicated(["adm_id", "year", "month"]).any():
        raise ValueError(f"{level}: duplicate administrative region/month keys")
    days = pd.to_datetime(dict(year=df.year, month=df.month, day=1)).dt.days_in_month
    if (df.flood_days.dropna() < 0).any() or (df.flood_days > days).any():
        raise ValueError(f"{level}: flood_days outside valid calendar bounds")
    statuses = df["target_status"].dropna().astype(str).unique().tolist() if "target_status" in df else []
    return df.sort_values(["adm_id", "year", "month"]).reset_index(drop=True), {
        "level": level, "input_rows": len(df), "unique_regions": int(df.adm_id.nunique()),
        "target_status_values": statuses,
    }


def make_lags(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["period"] = x.year * 12 + x.month
    x = x.sort_values(["adm_id", "period"]).reset_index(drop=True)
    for lag in LAGS:
        previous = x.groupby("adm_id", sort=False)["period"].shift(lag)
        calendar_aligned = previous.eq(x.period - lag)
        for c in ("flood_days", *DRIVERS):
            x[f"{c}_lag{lag}"] = (
                x.groupby("adm_id", sort=False)[c].shift(lag)
                .where(calendar_aligned)
            )
    return x


def features(name: str) -> list[str]:
    return [f"flood_days_lag{k}" for k in LAGS] + [
        f"{driver}_lag{k}" for driver in MODELS[name] for k in LAGS
    ] + ["month"]


def split_data(cohort: pd.DataFrame, split: str) -> pd.DataFrame:
    lo, hi = YEARS[split]
    return cohort.loc[cohort.year.between(lo, hi)]


def regression_metrics(y, pred) -> dict:
    if not len(y):
        return {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}
    return {"MAE": float(mean_absolute_error(y, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
            "R2": float(r2_score(y, pred)) if len(y) > 1 else float("nan")}


def classify_metrics(y, pred, event_threshold: float, alert_threshold: float) -> dict:
    true = (np.asarray(y) >= event_threshold).astype(int)
    alert = (np.asarray(pred) >= alert_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(true, alert, labels=[0, 1]).ravel()
    return {
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "Precision": float(precision_score(true, alert, zero_division=0)),
        "Recall": float(recall_score(true, alert, zero_division=0)),
        "F1": float(f1_score(true, alert, zero_division=0)),
        "FNR": float(fn / (fn + tp)) if fn + tp else float("nan"),
        "FPR": float(fp / (fp + tn)) if fp + tn else float("nan"),
        "positives": int(true.sum()), "alerts": int(alert.sum()),
    }


def pick_alert_threshold(y, pred, event_threshold: float, min_precision: float) -> dict:
    """Tune on VALIDATION only. Higher F2 weights recall more than precision.

    Threshold selection is performed independently for each already-trained
    algorithm. It is NOT tuned on test outcomes.
    """
    if len(y) == 0 or not (np.asarray(y) >= event_threshold).any():
        return {"alert_threshold": event_threshold, "validation_F2": None,
                "selection_note": "No validation positives: threshold fixed to event threshold"}
    # Distinct candidate values, including 0 to test catch-all behavior;
    # candidate grid is based on validation predictions, not labels.
    grid = np.unique(np.r_[0.0, float(event_threshold), np.quantile(pred, np.linspace(0, 1, 81))])
    choices = []
    for t in grid:
        m = classify_metrics(y, pred, event_threshold, float(t))
        p, r = m["Precision"], m["Recall"]
        f2 = 5 * p * r / (4 * p + r) if 4 * p + r else 0.0
        choices.append((float(t), p, r, f2, m["FP"]))
    admissible = [x for x in choices if x[1] >= min_precision]
    if not admissible:
        return {"alert_threshold": event_threshold, "validation_F2": None,
                "selection_note": "No threshold met minimum precision; fallback to event threshold"}
    best = sorted(admissible, key=lambda x: (-x[3], -x[1], x[4], -x[0]))[0]
    return {"alert_threshold": best[0], "validation_F2": best[3],
            "selection_note": "Max validation F2, subject to minimum precision"}


def plot_metrics(metric: pd.DataFrame, output: Path, level: str) -> None:
    d = metric[(metric.level == level) & (metric.split == "test")]
    if d.empty:
        return
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    names = list(MODELS)
    vals = [float(d.loc[d.model == m, "MAE"].iloc[0]) for m in names]
    rec = [float(d.loc[d.model == m, "Recall"].iloc[0]) for m in names]
    axs[0].bar(np.arange(len(names)), vals)
    axs[0].set_ylabel("MAE (recorded-event days / month)")
    axs[0].set_title("Monthly prediction error (lower is better)")
    axs[1].bar(np.arange(len(names)), rec)
    axs[1].set_ylim(0, 1)
    axs[1].set_ylabel("Recall")
    axs[1].set_title("High-activity alert recall (higher is better)")
    for ax in axs:
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels([s.split("_", 1)[1] for s in names], rotation=25, ha="right")
    fig.suptitle(f"{level} | Strategy C | Test 2023–2025 | Event-date proxy")
    fig.savefig(output / f"{level.lower()}_model_comparison.png", dpi=175)
    plt.close(fig)


def plot_confusion(cm: dict, level: str, output: Path) -> None:
    data = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(data, cmap="Blues")
    ax.set_xticks([0, 1], ["No alert", "Alert"])
    ax.set_yticks([0, 1], ["Below threshold", "At/above threshold"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed event-date proxy")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(data[i, j]), ha="center", va="center", fontsize=14)
    ax.set_title(f"{level} M4 | Test 2023–2025")
    fig.tight_layout()
    fig.savefig(output / f"{level.lower()}_M4_confusion_matrix.png", dpi=175)
    plt.close(fig)


def run_level(df: pd.DataFrame, level: str, trees: int, seed: int,
              min_precision: float, event_quantile: float):
    x = make_lags(df)
    # Critically: use SAME cohort across M0–M4, including all meteorological lags.
    common = ["flood_days"] + features("M4_all_drivers")
    cohort = x.dropna(subset=common).copy()
    train, val, test = [split_data(cohort, p) for p in YEARS]
    if len(train) < 40 or len(val) < 10 or len(test) < 10:
        raise ValueError(f"{level} insufficient common cohort: "
                         f"train={len(train)}, val={len(val)}, test={len(test)}")
    # Event threshold set solely from TRAINING target distribution.
    # If proxy is sparse, positive-only quantile avoids selecting zero as alert event.
    positive = train.loc[train.flood_days > 0, "flood_days"]
    if positive.empty:
        raise ValueError(f"{level}: training set has no positive recorded-event months")
    event_threshold = float(max(1.0, positive.quantile(event_quantile)))
    event_threshold = float(math.ceil(event_threshold))

    coverage = {"level": level, "common_regions": int(cohort.adm_id.nunique()),
                "train_n": len(train), "validation_n": len(val), "test_n": len(test),
                "alert_event_threshold": event_threshold,
                "training_positive_fraction": float((train.flood_days >= event_threshold).mean())}
    scores, test_pred, importance, alerts = [], [], [], []
    for model_name in MODELS:
        cols = features(model_name)
        reg = RandomForestRegressor(n_estimators=trees, min_samples_leaf=3,
                                    random_state=seed, n_jobs=-1)
        reg.fit(train[cols], train.flood_days.to_numpy(float))
        for feat, val_importance in zip(cols, reg.feature_importances_):
            importance.append({"level": level, "model": model_name,
                               "feature": feat, "importance": float(val_importance)})
        v_pred = np.clip(reg.predict(val[cols]), 0, pd.to_datetime(dict(year=val.year, month=val.month, day=1)).dt.days_in_month)
        chosen = pick_alert_threshold(val.flood_days.to_numpy(float), v_pred,
                                      event_threshold, min_precision)
        alerts.append({"level": level, "model": model_name,
                       "observed_event_threshold": event_threshold, **chosen})
        for label, subset in [("train", train), ("validation", val), ("test", test)]:
            limit = pd.to_datetime(dict(year=subset.year, month=subset.month, day=1)).dt.days_in_month.to_numpy()
            y = subset.flood_days.to_numpy(float)
            pred = np.clip(reg.predict(subset[cols]), 0, limit)
            metrics = regression_metrics(y, pred)
            warning = classify_metrics(y, pred, event_threshold, chosen["alert_threshold"])
            scores.append({"level": level, "model": model_name, "split": label,
                           "n": len(subset), "observed_event_threshold": event_threshold,
                           "predicted_alert_threshold": chosen["alert_threshold"],
                           **metrics, **warning})
            if label == "test":
                for adm, yr, mo, obs, guess in zip(subset.adm_id, subset.year, subset.month, y, pred):
                    is_pos = int(obs >= event_threshold)
                    warned = int(guess >= chosen["alert_threshold"])
                    test_pred.append({"level": level, "model": model_name, "adm_id": str(adm),
                                      "year": int(yr), "month": int(mo),
                                      "actual_event_dates": float(obs), "predicted_event_dates": float(guess),
                                      "observed_high_activity": is_pos, "predicted_warning": warned,
                                      "TP": int(is_pos and warned), "FN": int(is_pos and not warned),
                                      "FP": int(not is_pos and warned), "TN": int(not is_pos and not warned)})
        print(f"{level} {model_name}: test MAE {scores[-1]['MAE']:.3f}, "
              f"Recall {scores[-1]['Recall']:.3f}, F1 {scores[-1]['F1']:.3f}, "
              f"FN {scores[-1]['FN']}")
    return coverage, scores, test_pred, importance, alerts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    root = project_root()
    ap.add_argument("--adm2-panel", type=Path,
                    default=root / "processed_data/multivariate_flood_modeling/05a_monthly_panel/adm2_monthly_panel.csv")
    ap.add_argument("--adm3-panel", type=Path,
                    default=root / "processed_data/multivariate_flood_modeling/05a_monthly_panel/adm3_monthly_panel.csv")
    ap.add_argument("--output", type=Path,
                    default=root / "processed_data/multivariate_flood_modeling/07_final_multidriver_prediction")
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--event-quantile", type=float, default=0.75,
                    help="Training positive-event-day quantile for high-activity event threshold")
    ap.add_argument("--min-precision", type=float, default=0.30,
                    help="Minimum validation precision when tuning alert threshold")
    args = ap.parse_args()
    if not 0 < args.event_quantile < 1 or not 0 <= args.min_precision <= 1:
        ap.error("Require 0 < event-quantile < 1 and 0 <= min-precision <= 1")
    args.output.mkdir(parents=True, exist_ok=True)
    coverage_rows, metric_rows, pred_rows, importance_rows, alert_rows, source_info = [], [], [], [], [], []
    for level, path in [("ADM2", args.adm2_panel), ("ADM3", args.adm3_panel)]:
        raw, info = load_panel(path, level)
        source_info.append(info)
        coverage, metrics, predictions, importances, thresholds = run_level(
            raw, level, args.trees, args.seed, args.min_precision, args.event_quantile
        )
        coverage_rows.append(coverage)
        metric_rows.extend(metrics)
        pred_rows.extend(predictions)
        importance_rows.extend(importances)
        alert_rows.extend(thresholds)
    metric_df = pd.DataFrame(metric_rows)
    pred_df = pd.DataFrame(pred_rows)
    metric_df.to_csv(args.output / "model_metrics_regression_and_warning.csv", index=False)
    pred_df.to_csv(args.output / "test_predictions_and_warning_flags.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(args.output / "common_sample_coverage.csv", index=False)
    pd.DataFrame(importance_rows).to_csv(args.output / "feature_importances.csv", index=False)
    pd.DataFrame(alert_rows).to_csv(args.output / "validation_warning_thresholds.csv", index=False)
    improvements = []
    for level in ["ADM2", "ADM3"]:
        d = metric_df[(metric_df.level == level) & (metric_df.split == "test")]
        base = float(d.loc[d.model == "M0_history", "MAE"].iloc[0])
        for r in d.to_dict("records"):
            improvements.append({"level": level, "model": r["model"], "test_n": r["n"],
                                 "baseline_MAE": base, "model_MAE": r["MAE"],
                                 "MAE_reduction": base - r["MAE"],
                                 "MAE_reduction_pct": 100 * (base - r["MAE"]) / base if base > 0 else np.nan})
        plot_metrics(metric_df, args.output, level)
        best_cm = d.loc[d.model == "M4_all_drivers"].iloc[0].to_dict()
        plot_confusion(best_cm, level, args.output)
    pd.DataFrame(improvements).to_csv(args.output / "driver_contributions_same_cohort.csv", index=False)
    metadata = {
        "target": "Monthly recorded flood-event-date proxy; not validated flood duration or disaster alerts",
        "strategy": "C: next-month prediction using previous 1, 2 and 12 calendar months",
        "train_validation_test_years": YEARS,
        "model_feature_sets": {m: features(m) for m in MODELS},
        "samples": "Common complete-case cohort within each ADM level, identical for M0 through M4",
        "warning_event_threshold": "ceil(75th percentile of positive training recorded-event-date counts) by ADM level",
        "predicted_alert_threshold": "Maximum validation F2 among options meeting minimum validation precision; never selected on test",
        "rainfall_runoff_et0": "Historical inputs only; no contemporaneous target-month meteorology",
        "lake_DFO": "NOT included: geographic candidate links are not approved hydrological predictive features",
        "warning_limitations": [
            "Proxy labels cannot validate operational flood warning recall or real inundation",
            "Cross-region unequal spatial coverage; do not compare ADM2 and ADM3 MAE as if same task",
            "Correlation and predictive improvements are not causal evidence",
            "MAE/F1/Recall do not evaluate real emergency costs; action trigger requires domain validation",
            "Model hyperparameters fixed, not validation-tuned; validation only selects warning cutoffs",
        ],
        "input_sources": source_info,
    }
    (args.output / "methodology.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nOUTPUT:", args.output)
    print(metric_df[metric_df.split == "test"]["level model n MAE RMSE R2 Precision Recall F1 FN FP".split()].to_string(index=False))
    print("\nIMPORTANT: These are event-date-proxy predictions; not validated real-world flood warnings.")


if __name__ == "__main__":
    main()
