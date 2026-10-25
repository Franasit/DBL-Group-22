#!/usr/bin/env python3
"""JBG060 Script 10a — Fixed vs rolling-origin monthly flood-proxy forecasting.

Run in VS Code (no arguments required):
  python scripts/multivariate_flood_modeling/10a_compare_fixed_vs_rolling_origin.py

Requires alongside this file:
  07_final_multidriver_flood_prediction.py

Requires existing Script 05a panel CSVs:
  processed_data/multivariate_flood_modeling/05a_monthly_panel/
    adm2_monthly_panel.csv
    adm3_monthly_panel.csv

Protocol (mirrors Script 07):
  Training:   2001–2019
  Validation: 2020–2022 (freeze warning cutoff per model)
  Final test: 2023–2025

Fixed: fit only 2001–2019, predict all 2023–2025.
Rolling: refit at the beginning of each test year using all complete historical
         months through the preceding December (2023: through 2022; 2024:
         through 2023; 2025: through 2024).

Common data rows, M0–M4 model features, regressor hyperparameters, observed-event
threshold and warning cutoffs are the same across strategies. No test-year labels
are used to select hyperparameters or thresholds. Predictions for each test year
use only preceding months' predictor values (lags 1, 2 and 12).

IMPORTANT: the target is a recorded FLOOD EVENT-DATE PROXY, not validated true
flood duration, inundated area or a real-world warning system. This script does
not assess spatial transfer to unseen administrative regions.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

BLUE = "#7EC9F2"
PURPLE = "#7546B7"


def repo_root() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name == "multivariate_flood_modeling" and p.parent.parent.name == "scripts":
        return p.parents[2]
    return Path.cwd()


def load_script07(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Required existing Script 07 not found: {path}\n"
            "Place 10a next to 07_final_multidriver_flood_prediction.py."
        )
    spec = importlib.util.spec_from_file_location("jbg060_script07", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clipped_predict(reg, data: pd.DataFrame, cols: list[str]) -> np.ndarray:
    limit = pd.to_datetime(dict(year=data.year, month=data.month, day=1)).dt.days_in_month.to_numpy()
    return np.clip(reg.predict(data[cols]), 0, limit)


def train_model(s07, data: pd.DataFrame, model_name: str, trees: int, seed: int):
    reg = RandomForestRegressor(
        n_estimators=trees, min_samples_leaf=3, random_state=seed, n_jobs=-1
    )
    reg.fit(data[s07.features(model_name)], data.flood_days.to_numpy(float))
    return reg


def compute_metrics(s07, y, pred, event_threshold, alert_threshold):
    return {
        **s07.regression_metrics(y, pred),
        **s07.classify_metrics(y, pred, event_threshold, alert_threshold),
    }


def add_predictions(rows, level, model_name, strategy, year, df, preds, event_thr, alert_thr):
    obs = df.flood_days.to_numpy(float)
    for adm, mo, actual, predicted in zip(df.adm_id, df.month, obs, preds):
        actual_high = int(actual >= event_thr)
        warning = int(predicted >= alert_thr)
        rows.append({
            "level": level, "model": model_name, "strategy": strategy,
            "adm_id": str(adm), "year": int(year), "month": int(mo),
            "actual_event_dates": float(actual),
            "predicted_event_dates": float(predicted),
            "observed_high_activity": actual_high, "predicted_warning": warning,
            "TP": int(actual_high and warning), "FP": int(not actual_high and warning),
            "FN": int(actual_high and not warning), "TN": int(not actual_high and not warning),
            "observed_event_threshold": float(event_thr),
            "predicted_alert_threshold": float(alert_thr),
        })


def run_level(s07, path: Path, level: str, trees: int, seed: int,
              event_quantile: float, min_precision: float):
    df, info = s07.load_panel(path, level)
    lagged = s07.make_lags(df)
    # Use the identical complete-case region-month cohort for all models.
    common_cols = ["flood_days"] + s07.features("M4_all_drivers")
    cohort = lagged.dropna(subset=common_cols).copy()
    dev_train = cohort[cohort.year.between(2001, 2019)].copy()
    dev_val = cohort[cohort.year.between(2020, 2022)].copy()
    final_test = cohort[cohort.year.between(2023, 2025)].copy()
    if min(map(len, (dev_train, dev_val, final_test))) < 10:
        raise ValueError(f"{level}: insufficient complete rows train={len(dev_train)} "
                         f"validation={len(dev_val)} test={len(final_test)}")

    positive = dev_train.loc[dev_train.flood_days > 0, "flood_days"]
    if positive.empty:
        raise ValueError(f"{level}: no positive labels in training data")
    event_threshold = float(math.ceil(max(1.0, positive.quantile(event_quantile))))
    preds, score_rows, config_rows = [], [], []
    for model_name in s07.MODELS:
        cols = s07.features(model_name)
        # Threshold chosen exactly once, on 2020-2022 validation, using a
        # 2001-2019 fitted model. Then frozen across BOTH test strategies.
        fixed = train_model(s07, dev_train, model_name, trees, seed)
        val_preds = clipped_predict(fixed, dev_val, cols)
        chosen = s07.pick_alert_threshold(
            dev_val.flood_days.to_numpy(float), val_preds,
            event_threshold, min_precision,
        )
        alert_threshold = float(chosen["alert_threshold"])
        config_rows.append({
            "level": level, "model": model_name,
            "event_threshold": event_threshold,
            "warning_threshold": alert_threshold,
            "validation_F2": chosen.get("validation_F2"),
            "selection_note": chosen.get("selection_note"),
            "train_years_for_threshold": "2001-2019",
            "validation_years": "2020-2022",
        })
        for year in (2023, 2024, 2025):
            test_year = final_test[final_test.year == year]
            if test_year.empty:
                raise ValueError(f"{level}: no test observations for {year}")
            # Fixed regressor: NEVER refit on validation/test data.
            # Rolling: refit using past years only; 2023 includes 2020-2022.
            rolling_train = cohort[cohort.year.between(2001, year - 1)]
            rolling = train_model(s07, rolling_train, model_name, trees, seed)
            for strategy, reg, train_end in (
                ("Fixed", fixed, 2019),
                ("Rolling", rolling, year - 1),
            ):
                yp = clipped_predict(reg, test_year, cols)
                y = test_year.flood_days.to_numpy(float)
                add_predictions(preds, level, model_name, strategy, year,
                                test_year, yp, event_threshold, alert_threshold)
                score_rows.append({
                    "level": level, "model": model_name, "strategy": strategy,
                    "year": year, "training_start": 2001,
                    "training_end": train_end,
                    "training_n": len(dev_train) if strategy == "Fixed" else len(rolling_train),
                    "test_n": len(test_year),
                    "event_threshold": event_threshold,
                    "warning_threshold": alert_threshold,
                    **compute_metrics(s07, y, yp, event_threshold, alert_threshold),
                })
        print(f"{level} {model_name}: finished Fixed + Rolling / 2023–2025")

    return pd.DataFrame(preds), pd.DataFrame(score_rows), pd.DataFrame(config_rows), {
        **info,
        "complete_case_rows": len(cohort),
        "train_n": len(dev_train), "validation_n": len(dev_val),
        "test_n": len(final_test), "event_threshold": event_threshold,
    }


def pooled_metrics(s07, preds: pd.DataFrame):
    out = []
    for (level, model, strategy), group in preds.groupby(
        ["level", "model", "strategy"], sort=False
    ):
        event_thr = float(group.observed_event_threshold.iloc[0])
        alert_thr = float(group.predicted_alert_threshold.iloc[0])
        m = compute_metrics(
            s07,
            group.actual_event_dates.to_numpy(float),
            group.predicted_event_dates.to_numpy(float),
            event_thr, alert_thr,
        )
        out.append({"level": level, "model": model, "strategy": strategy,
                    "years": "2023-2025", "test_n": len(group),
                    "event_threshold": event_thr,
                    "warning_threshold": alert_thr, **m})
    return pd.DataFrame(out)


def plot_comparison(pooled: pd.DataFrame, level: str, out: Path):
    data = pooled[pooled.level == level]
    order = ["M0_history", "M1_rainfall", "M2_runoff", "M3_et0", "M4_all_drivers"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0), constrained_layout=True)
    x = np.arange(len(order))
    for ax, metric, title in zip(axes, ["MAE", "Recall", "F1"],
                                 ["MAE (lower is better)", "Recall (higher is better)", "F1 (higher is better)"]):
        for pos, (strategy, colour) in enumerate((("Fixed", BLUE), ("Rolling", PURPLE))):
            values = [float(data.loc[(data.model == name) & (data.strategy == strategy), metric].iloc[0])
                      for name in order]
            ax.bar(x + (pos - 0.5) * 0.36, values, 0.34,
                   color=colour, label=strategy)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([a.split("_", 1)[0] for a in order])
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        if metric in ("Recall", "F1"):
            ax.set_ylim(0, 1.02)
    axes[0].legend(frameon=False)
    fig.suptitle(f"{level} | Fixed vs Rolling-Origin | 2023–2025 | Event-date proxy",
                 fontsize=14, weight="bold")
    fig.savefig(out / f"{level.lower()}_10a_fixed_vs_rolling.png", dpi=250)
    fig.savefig(out / f"{level.lower()}_10a_fixed_vs_rolling.pdf")
    plt.close(fig)


def plot_by_year(score: pd.DataFrame, level: str, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for ax, metric in zip(axes, ("Recall", "MAE", "FP")):
        for strategy, colour in (("Fixed", BLUE), ("Rolling", PURPLE)):
            d = score[(score.level == level) & (score.model == "M4_all_drivers") &
                      (score.strategy == strategy)].sort_values("year")
            ax.plot(d.year, d[metric], marker="o", linewidth=2, color=colour,
                    label=strategy)
        ax.set_xticks([2023, 2024, 2025])
        ax.set_title("M4 — " + metric)
        ax.grid(alpha=0.2)
        if metric == "Recall":
            ax.set_ylim(0, 1.03)
    axes[0].legend(frameon=False)
    fig.suptitle(f"{level} | Year-by-year test comparison | 2023–2025")
    fig.savefig(out / f"{level.lower()}_10a_M4_yearly.png", dpi=250)
    fig.savefig(out / f"{level.lower()}_10a_M4_yearly.pdf")
    plt.close(fig)


def main():
    root = repo_root()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--script07", type=Path, default=Path(__file__).resolve().with_name(
        "07_final_multidriver_flood_prediction.py"))
    ap.add_argument("--adm2-panel", type=Path, default=root / "processed_data/multivariate_flood_modeling/05a_monthly_panel/adm2_monthly_panel.csv")
    ap.add_argument("--adm3-panel", type=Path, default=root / "processed_data/multivariate_flood_modeling/05a_monthly_panel/adm3_monthly_panel.csv")
    ap.add_argument("--output", type=Path, default=root / "processed_data/multivariate_flood_modeling/10a_fixed_vs_rolling_origin")
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--event-quantile", type=float, default=0.75)
    ap.add_argument("--min-precision", type=float, default=0.30)
    ap.add_argument("--level", choices=["ADM2", "ADM3", "both"], default="both")
    args = ap.parse_args()
    if not (0 < args.event_quantile < 1 and 0 <= args.min_precision <= 1 and args.trees >= 1):
        ap.error("Invalid event-quantile, min-precision or trees setting")

    s07 = load_script07(args.script07)
    if s07.YEARS != {"train": (2001, 2019), "validation": (2020, 2022), "test": (2023, 2025)}:
        raise RuntimeError("Script 07 split differs from 10a protocol; review before comparison")
    args.output.mkdir(parents=True, exist_ok=True)
    all_preds, all_yearly, all_configs, audit = [], [], [], []
    levels = [("ADM2", args.adm2_panel), ("ADM3", args.adm3_panel)]
    if args.level != "both":
        levels = [(l, p) for l, p in levels if l == args.level]
    for level, path in levels:
        pred, yearly, configs, audit_row = run_level(
            s07, path, level, args.trees, args.seed, args.event_quantile, args.min_precision
        )
        all_preds.append(pred)
        all_yearly.append(yearly)
        all_configs.append(configs)
        audit.append(audit_row)
    preds = pd.concat(all_preds, ignore_index=True)
    yearly = pd.concat(all_yearly, ignore_index=True)
    configs = pd.concat(all_configs, ignore_index=True)
    pooled = pooled_metrics(s07, preds)
    preds.to_csv(args.output / "01_test_predictions_fixed_vs_rolling.csv", index=False)
    yearly.to_csv(args.output / "02_metrics_by_test_year.csv", index=False)
    pooled.to_csv(args.output / "03_pooled_test_metrics.csv", index=False)
    configs.to_csv(args.output / "04_validation_thresholds_frozen.csv", index=False)
    pd.DataFrame(audit).to_csv(args.output / "05_sample_audit.csv", index=False)
    for level, _ in levels:
        plot_comparison(pooled, level, args.output)
        plot_by_year(yearly, level, args.output)
    meta = {
        "purpose": "Compare fixed deployment with annual rolling-origin retraining, not independent split-selection competitions",
        "original_07_splits": s07.YEARS,
        "fixed": "Train 2001-2019; threshold tuned on validation 2020-2022; test 2023-2025 without refitting",
        "rolling": "2023 train through 2022; 2024 through 2023; 2025 through 2024; refit at each annual origin",
        "warning_cutoffs": "Identical fixed validation-chosen warning threshold for each model across both strategies and all test years",
        "event_cutoffs": "Positive-target 75th percentile computed on 2001-2019 training data only, fixed for all years",
        "target": "Monthly flood event-date proxy, NOT verified real-world flood duration or disaster severity",
        "feature_timing": "Lagged historical months 1, 2 and 12; no actual contemporaneous target-month weather used",
        "same_test_samples": True,
        "spatial_transfer_test": False,
        "holdout_integrity_note": "If test outcomes were inspected during model development, 2023-2025 cannot be claimed untouched; disclose retrospectively.",
        "parameters": {"trees": args.trees, "seed": args.seed,
                       "event_quantile": args.event_quantile, "min_precision": args.min_precision},
    }
    (args.output / "06_methodology.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\nOUTPUT:", args.output)
    print(pooled[["level", "model", "strategy", "test_n", "MAE", "Recall", "Precision", "F1", "FN", "FP"]].to_string(index=False))
    print("\nNOTE: Results evaluate recorded event-date proxies, not confirmed flood warnings.")


if __name__ == "__main__":
    main()
