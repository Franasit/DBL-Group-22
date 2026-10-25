#!/usr/bin/env python3
"""JBG060 Script 07b: evidence-gated, monthly hydrological forecast extensions.

English, standalone and directly runnable from VS Code:
  python scripts/multivariate_flood_modeling/07b_extended_hydrological_prediction.py

Compares, on an identical ADM-month cohort for EACH comparison:
  M4: historical flood activity + rainfall + runoff + reference ET0
  M5: M4 + approved upstream lake water level information
  M6: M4 + approved DFO river discharge information
  M7: M4 + both approved lake and DFO information

IMPORTANT:
 - Without explicit, documented and approved source-to-ADM hydrological links,
   the corresponding extension is NOT TRAINED. This is intentional.
 - Never equate a candidate basin route with verified flood propagation.
 - Hydro observations are historic (at least 1 month before target), not same-month.
 - The outcome is a monthly flood EVENT-DATE PROXY, not validated actual flood
   duration or operational disaster warning. Zero recorded event dates is not
   proof of a fully observed flood-free month.
 - M4 is refitted on each matched comparison cohort; do not compare M4 scores
   across different cohorts to rank M5 vs M6 vs M7.
 - Alert-event threshold derives from training labels; predicted alert cutoff
   derives from validation data, never the test set.
 - No approved links? Outputs a clear SKIPPED status, not fabricated metrics.

Inputs:
  processed_data/multivariate_flood_modeling/05a_monthly_panel/
    adm2_monthly_panel.csv, adm3_monthly_panel.csv
  processed_data/multivariate_flood_modeling/06a_lake_dfo_monthly/
    station_lake_monthly.csv
  processed_data/multivariate_flood_modeling/06g_reviewed_hydrology/
    03_approved_connections.csv  (produced by 06g after human review)

Outputs:
  processed_data/multivariate_flood_modeling/07b_extended_hydrological_prediction/
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

SPLITS = {"train": (2001, 2019), "validation": (2020, 2022), "test": (2023, 2025)}
CORE = ("rainfall_mm", "runoff_mm", "et0_mm")
LAGS = (1, 2, 12)
MODELS = {"M5_lake": ("lake",), "M6_dfo": ("dfo",), "M7_lake_dfo": ("lake", "dfo")}
REQUIRED_APPROVAL = ("source_type", "source_id", "level", "adm_id", "decision",
                     "evidence_reference", "reviewer", "review_date", "connection_notes")


def root_dir() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name == "multivariate_flood_modeling" and p.parent.parent.name == "scripts":
        return p.parents[2]
    return Path.cwd()


def read_panel(path: Path, level: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing. Run Script 05a first.")
    df = pd.read_csv(path, dtype={"adm_id": "string"}, low_memory=False)
    cols = ["adm_id", "year", "month", "flood_days", *CORE]
    absent = set(cols).difference(df.columns)
    if absent:
        raise ValueError(f"{level} panel is missing: {sorted(absent)}")
    df["adm_id"] = df["adm_id"].str.strip()
    for col in cols[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["adm_id", "year", "month"]].isna().any().any():
        raise ValueError(f"{level}: null ADM/year/month keys")
    if (df["year"] % 1 != 0).any() or (df["month"] % 1 != 0).any() or not df["month"].between(1, 12).all():
        raise ValueError(f"{level}: invalid year or month")
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    if df.duplicated(["adm_id", "year", "month"]).any():
        raise ValueError(f"{level}: duplicate ADM-month keys")
    df["period"] = df["year"] * 12 + df["month"]
    df = df.sort_values(["adm_id", "period"]).reset_index(drop=True)
    if "target_status" in df and not df["target_status"].dropna().astype(str).str.contains("EVENT_DATE_PROXY").all():
        print(f"WARNING {level}: target_status is not exclusively the event-date proxy; check target semantics")
    return df


def make_core_lags(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for lag in LAGS:
        per = x.groupby("adm_id")["period"].shift(lag)
        aligned = per.eq(x["period"] - lag)
        for name in ("flood_days", *CORE):
            x[f"{name}_lag{lag}"] = x.groupby("adm_id")[name].shift(lag).where(aligned)
    return x


def approved_links(path: Path) -> tuple[pd.DataFrame, str]:
    columns = list(REQUIRED_APPROVAL)
    if not path.exists():
        return pd.DataFrame(columns=columns), f"Approved mapping not found: {path}"
    df = pd.read_csv(path, dtype={"source_id": "string", "adm_id": "string"}, keep_default_na=False)
    if df.empty:
        return pd.DataFrame(columns=columns), "The approved mapping is empty (no verified connections)."
    absent = set(columns) - set(df.columns)
    if absent:
        raise ValueError(f"Approved mapping lacks original review evidence fields: {sorted(absent)}")
    for col in ("source_type", "source_id", "level", "adm_id", "decision",
                "evidence_reference", "reviewer", "review_date", "connection_notes"):
        df[col] = df[col].astype(str).str.strip()
    df["source_type"] = df["source_type"].str.lower()
    df["level"] = df["level"].str.upper()
    df["decision"] = df["decision"].str.upper()
    # Defense-in-depth: independently validate the evidence, not simply trust a file name.
    ok_date = pd.to_datetime(df["review_date"], format="%Y-%m-%d", errors="coerce").notna()
    ok = (df["decision"].eq("APPROVED") & ok_date &
          df["evidence_reference"].str.len().ge(12) &
          df["reviewer"].str.len().gt(0) &
          df["connection_notes"].str.len().ge(12) &
          df["source_type"].isin(["lake", "dfo"]) &
          df["level"].isin(["ADM2", "ADM3"]))
    rejected = len(df) - int(ok.sum())
    if rejected:
        print(f"WARNING: {rejected} mapping rows lack complete explicit approval/evidence and will be ignored")
    df = df.loc[ok].copy()
    if df.empty:
        return df, "No fully documented APPROVED hydrological connections."
    df = df.drop_duplicates(["source_type", "source_id", "level", "adm_id"])
    return df, "OK"


def monthly_sources(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"source_id": "string"})
    req = {"source_type", "source_id", "year", "month", "hydro_mean"}
    if missing := req - set(df.columns):
        raise ValueError(f"06a monthly file missing {sorted(missing)}")
    df["source_type"] = df["source_type"].astype(str).str.lower().str.strip()
    df["source_id"] = df["source_id"].astype(str).str.strip()
    for col in ("year", "month", "hydro_mean"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["year", "month", "hydro_mean"]).copy()
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df = df[df["month"].between(1, 12)].copy()
    df["period"] = df["year"] * 12 + df["month"]
    # Repeated raw rows on same source-month are reduced to one source measurement.
    return df.groupby(["source_type", "source_id", "period"], as_index=False)["hydro_mean"].mean()


def build_hydrology_lags(base: pd.DataFrame, links: pd.DataFrame, src: pd.DataFrame, level: str) -> pd.DataFrame:
    """Produce one feature per hydrology TYPE and historical lag.

    Source-specific water elevations have incompatible physical datums. Therefore
    compute each station's training-period z-score, then average mapped sources'
    standardized anomalies at each ADM-month (NOT raw lake elevations).
    Standardisation statistics use source dates no later than 2019-12.
    Missing measurements remain missing, never silently converted to zero.
    """
    part = links.loc[links["level"] == level, ["source_type", "source_id", "adm_id"]].copy()
    if part.empty:
        return base
    train_sources = src[src["period"] <= 2019 * 12 + 12]
    standards = train_sources.groupby(["source_type", "source_id"])["hydro_mean"].agg(["mean", "std", "count"]).reset_index()
    standards = standards[(standards["count"] >= 12) & standards["std"].gt(0)]
    if standards.empty:
        print(f"{level}: no hydrology source has >=12 training measurements and nonzero variance")
        return base
    normalized = src.merge(standards[["source_type", "source_id", "mean", "std"]],
                           on=["source_type", "source_id"], how="inner", validate="many_to_one")
    normalized["anomaly"] = (normalized["hydro_mean"] - normalized["mean"]) / normalized["std"]
    mapped = part.merge(normalized[["source_type", "source_id", "period", "anomaly"]],
                        on=["source_type", "source_id"], how="inner")
    out = base.copy()
    # Exact prior monthly observations at t-1, t-2, and t-12; no forward fill.
    for typ in ("lake", "dfo"):
        t = mapped[mapped.source_type.eq(typ)]
        if t.empty:
            continue
        agg = t.groupby(["adm_id", "period"], as_index=False)["anomaly"].mean()
        for lag in LAGS:
            col = f"{typ}_anomaly_lag{lag}"
            temp = agg.rename(columns={"anomaly": col}).copy()
            temp["period"] = temp["period"] + lag
            out = out.merge(temp, on=["adm_id", "period"], how="left", validate="one_to_one")
    return out


def time_rows(df: pd.DataFrame, split: str) -> pd.DataFrame:
    first, last = SPLITS[split]
    return df[df["year"].between(first, last)]


def core_columns() -> list[str]:
    return [f"flood_days_lag{k}" for k in LAGS] + [f"{name}_lag{k}" for name in CORE for k in LAGS] + ["month"]


def type_columns(typ: str) -> list[str]:
    return [f"{typ}_anomaly_lag{k}" for k in LAGS]


def reg_scores(obs, pred) -> dict:
    return {"MAE": float(mean_absolute_error(obs, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(obs, pred))),
            "R2": float(r2_score(obs, pred)) if len(obs) > 1 else None}


def warning_scores(obs, pred, observed_cutoff, alert_cutoff) -> dict:
    true = np.asarray(obs) >= observed_cutoff
    alerts = np.asarray(pred) >= alert_cutoff
    tn, fp, fn, tp = confusion_matrix(true, alerts, labels=[False, True]).ravel()
    return {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "Precision": float(precision_score(true, alerts, zero_division=0)),
            "Recall": float(recall_score(true, alerts, zero_division=0)),
            "F1": float(f1_score(true, alerts, zero_division=0)),
            "FNR": float(fn / (tp + fn)) if tp + fn else None,
            "FPR": float(fp / (tn + fp)) if tn + fp else None,
            "observed_positive_count": int(true.sum())}


def choose_cutoff(obs, prediction, event_cutoff: float, min_precision: float) -> tuple[float, str]:
    if not np.any(np.asarray(obs) >= event_cutoff):
        return event_cutoff, "No validation positives: use event cutoff"
    grid = np.unique(np.r_[0, event_cutoff, np.quantile(prediction, np.linspace(0, 1, 81))])
    options = []
    for cutoff in grid:
        s = warning_scores(obs, prediction, event_cutoff, cutoff)
        p, r = s["Precision"], s["Recall"]
        f2 = 5*p*r/(4*p+r) if 4*p+r else 0
        if p >= min_precision:
            options.append((f2, p, -s["FP"], cutoff))
    if not options:
        return event_cutoff, "No validation cutoff met minimum precision"
    best = max(options)
    return float(best[-1]), "Max validation F2 subject to minimum precision"


def report_plot(table: pd.DataFrame, out: Path, level: str, experiment: str) -> None:
    t = table[(table.level == level) & (table.experiment == experiment) & (table.split == "test")]
    if len(t) != 2:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for axis, metric, title in ((ax[0], "MAE", "MAE (lower better)"),
                                (ax[1], "Recall", "Recall (higher better)")):
        axis.bar(t["model"], t[metric])
        axis.set_title(title)
        if metric == "Recall":
            axis.set_ylim(0, 1)
        axis.tick_params(axis="x", rotation=10)
    fig.suptitle(f"{level} | {experiment} | matched test cohort")
    fig.savefig(out / f"{level.lower()}_{experiment}_comparison.png", dpi=180)
    plt.close(fig)


def run_experiment(panel, level, kind, trees, seed, quantile, min_precision):
    model_extended = f"M{ {'M5_lake': 5, 'M6_dfo': 6, 'M7_lake_dfo': 7}[kind] }"
    needed_types = MODELS[kind]
    hydroc = [c for t in needed_types for c in type_columns(t)]
    missing = set(hydroc).difference(panel.columns)
    if missing:
        return {"level": level, "experiment": kind, "status": "SKIPPED_NO_APPROVED_FEATURES",
                "reason": f"Missing historical feature columns: {sorted(missing)}"}, [], [], []
    all_columns = core_columns() + hydroc
    cohort = panel.dropna(subset=["flood_days", *all_columns]).copy()
    datasets = {n: time_rows(cohort, n) for n in SPLITS}
    counts = {k: len(v) for k, v in datasets.items()}
    region_counts = {k: int(v.adm_id.nunique()) for k, v in datasets.items()}
    if counts["train"] < 60 or counts["validation"] < 20 or counts["test"] < 20:
        return {"level": level, "experiment": kind, "status": "SKIPPED_INSUFFICIENT_COHORT",
                "reason": f"Insufficient matched complete cases: {counts}", **counts}, [], [], []
    train = datasets["train"]
    positive = train.loc[train.flood_days > 0, "flood_days"]
    if positive.empty:
        return {"level": level, "experiment": kind, "status": "SKIPPED_NO_TRAIN_POSITIVES",
                "reason": "Training proxy target has no positive counts"}, [], [], []
    event_cutoff = math.ceil(max(1.0, positive.quantile(quantile)))
    metrics, predictions, thresholds = [], [], []
    for model, cols in [("M4_matched_baseline", core_columns()),
                        (model_extended + "_extended", all_columns)]:
        rf = RandomForestRegressor(n_estimators=trees, min_samples_leaf=3,
                                   random_state=seed, n_jobs=-1)
        rf.fit(train[cols], train.flood_days.to_numpy(float))
        validation = datasets["validation"]
        vpred = np.clip(rf.predict(validation[cols]), 0, validation.apply(
            lambda r: pd.Period(year=int(r.year), month=int(r.month), freq="M").days_in_month, axis=1).to_numpy())
        cutoff, reason = choose_cutoff(validation.flood_days.to_numpy(), vpred,
                                       event_cutoff, min_precision)
        thresholds.append({"level": level, "experiment": kind, "model": model,
                           "event_threshold": event_cutoff, "predicted_alert_threshold": cutoff,
                           "selection_note": reason})
        for split, frame in datasets.items():
            maxdays = pd.to_datetime(dict(year=frame.year, month=frame.month, day=1)).dt.days_in_month.to_numpy()
            y = frame.flood_days.to_numpy(float)
            pred = np.clip(rf.predict(frame[cols]), 0, maxdays)
            metrics.append({"level": level, "experiment": kind, "model": model,
                            "split": split, "n": len(frame), "regions": region_counts[split],
                            "event_threshold": event_cutoff, "alert_threshold": cutoff,
                            **reg_scores(y, pred), **warning_scores(y, pred, event_cutoff, cutoff)})
            if split == "test":
                true = y >= event_cutoff
                alert = pred >= cutoff
                predictions.extend({"level": level, "experiment": kind, "model": model,
                                    "adm_id": str(adm), "year": int(year), "month": int(month),
                                    "actual_event_dates": float(a), "predicted_event_dates": float(p),
                                    "actual_high_activity": int(t), "predicted_warning": int(w),
                                    "TP": int(t and w), "FP": int(not t and w),
                                    "FN": int(t and not w), "TN": int(not t and not w)}
                                   for adm, year, month, a, p, t, w in
                                   zip(frame.adm_id, frame.year, frame.month, y, pred, true, alert))
    return {"level": level, "experiment": kind, "status": "TRAINED",
            "train_n": counts["train"], "validation_n": counts["validation"],
            "test_n": counts["test"], "test_regions": region_counts["test"],
            "event_threshold": event_cutoff}, metrics, predictions, thresholds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    root = root_dir()
    base = root / "processed_data/multivariate_flood_modeling"
    p.add_argument("--adm2-panel", type=Path, default=base / "05a_monthly_panel/adm2_monthly_panel.csv")
    p.add_argument("--adm3-panel", type=Path, default=base / "05a_monthly_panel/adm3_monthly_panel.csv")
    p.add_argument("--monthly-sources", type=Path, default=base / "06a_lake_dfo_monthly/station_lake_monthly.csv")
    p.add_argument("--approved-mapping", type=Path, default=base / "06g_reviewed_hydrology/03_approved_connections.csv")
    p.add_argument("--output", type=Path, default=base / "07b_extended_hydrological_prediction")
    p.add_argument("--trees", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--event-quantile", type=float, default=0.75)
    p.add_argument("--min-precision", type=float, default=0.30)
    args = p.parse_args()
    if not 0 < args.event_quantile < 1 or not 0 <= args.min_precision <= 1:
        p.error("Invalid event quantile or minimum precision")
    args.output.mkdir(parents=True, exist_ok=True)
    links, link_status = approved_links(args.approved_mapping)
    if not links.empty and not args.monthly_sources.is_file():
        raise FileNotFoundError(f"Approved links exist, but monthly sources missing: {args.monthly_sources}")
    source = monthly_sources(args.monthly_sources) if not links.empty else pd.DataFrame()
    outcomes, scores, predictions, thresholds = [], [], [], []
    for level, path in (("ADM2", args.adm2_panel), ("ADM3", args.adm3_panel)):
        panel = make_core_lags(read_panel(path, level))
        if not links.empty:
            panel = build_hydrology_lags(panel, links, source, level)
        for experiment in MODELS:
            outcome, m, pr, th = run_experiment(panel, level, experiment,
                                               args.trees, args.seed, args.event_quantile,
                                               args.min_precision)
            outcomes.append(outcome)
            scores.extend(m)
            predictions.extend(pr)
            thresholds.extend(th)
            print(f"{level} {experiment}: {outcome['status']} {outcome.get('reason', '')}")
    status = pd.DataFrame(outcomes)
    status.to_csv(args.output / "experiment_status.csv", index=False)
    metric_df = pd.DataFrame(scores)
    metric_df.to_csv(args.output / "model_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(args.output / "test_predictions.csv", index=False)
    pd.DataFrame(thresholds).to_csv(args.output / "warning_thresholds.csv", index=False)
    comparison = []
    for (level, experiment), d in (metric_df[metric_df["split"] == "test"].groupby(["level", "experiment"])
                                   if "split" in metric_df.columns else []):
        if len(d) != 2:
            continue
        base_row = d[d.model == "M4_matched_baseline"].iloc[0]
        new_row = d[d.model != "M4_matched_baseline"].iloc[0]
        comparison.append({"level": level, "experiment": experiment,
                           "same_test_n": int(base_row.n),
                           "M4_MAE": float(base_row.MAE), "extended_MAE": float(new_row.MAE),
                           "MAE_reduction": float(base_row.MAE - new_row.MAE),
                           "M4_Recall": float(base_row.Recall), "extended_Recall": float(new_row.Recall),
                           "Recall_change": float(new_row.Recall - base_row.Recall),
                           "M4_F1": float(base_row.F1), "extended_F1": float(new_row.F1),
                           "M4_FN": int(base_row.FN), "extended_FN": int(new_row.FN),
                           "M4_FP": int(base_row.FP), "extended_FP": int(new_row.FP)})
        report_plot(metric_df, args.output, level, experiment)
    pd.DataFrame(comparison).to_csv(args.output / "paired_test_comparisons.csv", index=False)
    notes = {"input_mapping": str(args.approved_mapping), "mapping_status": link_status,
             "approved_source_to_ADM_links": int(len(links)),
             "forecast_horizon": "next month; hydrological lags t-1, t-2, t-12",
             "training_years": "2001-2019", "validation_years": "2020-2022",
             "test_years": "2023-2025", "lake_dfo_data": "06a monthly hydro_mean",
             "hydro_standardization": "source-specific means/standard deviations fitted using dates through 2019 only",
             "spatial_mappings": "only explicit reviewed APPROVED links, no automatic 06e candidate adoption",
             "same_cohort": "M4 retrained on exact same ADM-month rows as each corresponding extended model",
             "warning_metric": "Recall, F1 etc are proxy-label high activity alerts, not validated disaster warning skill",
             "important": ["No extrapolation of station readings to unreviewed ADM regions",
                           "No same-month predictor values are used",
                           "Sparse lake records produce missing monthly features and may prevent M5/M7 fitting",
                           "Models with different eligible cohorts must not be ranked directly",
                           "HydroBASINS topology alone is not adequate approval evidence"]}
    (args.output / "methodology.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    print("\nResults:", args.output)
    if not comparison:
        print("NO EXTENDED MODEL WAS FIT. This is expected if 06g still has 0 approved links.")
        print("Keep Script 07 as the current completed core-model result. Do not use geographic candidates as verified features.")
    else:
        print(pd.DataFrame(comparison).to_string(index=False))

if __name__ == "__main__":
    main()
