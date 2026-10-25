"""JBG060 — Monthly flood prediction diagnostics (no re-training).

Reads outputs of 05b and writes per-month/per-year metrics, B-vs-C paired
comparisons, heatmaps and a short poster-ready interpretation.

Target limitation: when built with 05a --event-proxy, the outcome is the
number of recorded flood-event dates, NOT validated flood duration or rainfall.

Run without arguments from VS Code or the project directory:
    python scripts/multivariate_flood_modeling/05e_analyze_monthly_prediction_results.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLS = ["level", "strategy", "model", "algorithm", "split", "adm_id", "year", "month", "actual_flood_days", "predicted_flood_days"]


def root_dir() -> Path:
    for p in [Path.cwd(), *Path(__file__).resolve().parents]:
        if (p / "processed_data").is_dir():
            return p
    return Path.cwd()


def metrics(d: pd.DataFrame) -> dict:
    a = d["actual_flood_days"].to_numpy(dtype=float)
    p = d["predicted_flood_days"].to_numpy(dtype=float)
    ok = np.isfinite(a) & np.isfinite(p)
    a, p = a[ok], p[ok]
    if len(a) == 0:
        return {"n": 0, "MAE": np.nan, "RMSE": np.nan, "Bias": np.nan,
                "overpredicted_pct": np.nan, "R2": np.nan}
    error = p - a
    denom = np.square(a - a.mean()).sum()
    return {"n": len(a), "MAE": float(np.abs(error).mean()),
            "RMSE": float(np.sqrt(np.square(error).mean())),
            "Bias": float(error.mean()),
            "overpredicted_pct": float(100 * (error > 0).mean()),
            "R2": float(1 - np.square(error).sum() / denom) if denom > 0 else np.nan}


def grouped_metrics(data: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    records = []
    for key, frame in data.groupby(keys, observed=True, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        records.append({**dict(zip(keys, key)), **metrics(frame)})
    return pd.DataFrame(records)


def heatmap(monthly: pd.DataFrame, level: str, out: Path):
    sub = monthly[(monthly.level == level) & (monthly.strategy == "C") &
                  (monthly.model == "M2") & (monthly.algorithm == "RandomForest")]
    if sub.empty:
        return
    pivot = sub.pivot(index="year", columns="month", values="MAE").reindex(columns=range(1, 13))
    fig, ax = plt.subplots(figsize=(12, 3.5))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="Blues", interpolation="nearest")
    ax.set_xticks(np.arange(12), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index.tolist())
    ax.set_title(f"{level} | C + M2 Random Forest | MAE by test month")
    for r in range(len(pivot)):
        for c in range(12):
            v = pivot.iloc[r, c]
            if pd.notna(v):
                ax.text(c, r, f"{v:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if v > np.nanmax(pivot.to_numpy()) * .68 else "black")
    fig.colorbar(image, ax=ax, label="MAE (recorded event dates)")
    fig.tight_layout()
    fig.savefig(out / f"{level.lower()}_monthly_mae_heatmap.png", dpi=220)
    plt.close(fig)


def b_c_pairs(data: pd.DataFrame):
    base = data[(data.model == "M2") & (data.algorithm == "RandomForest") &
                (data.strategy.isin(["B", "C"]))].copy()
    key = ["level", "adm_id", "year", "month"]
    # Check uniqueness before pairing: avoids accidental duplication and fake improvement.
    if base.duplicated(key + ["strategy"]).any():
        raise ValueError("Duplicate B/C predictions for the same region, year and month")
    b = base[base.strategy == "B"][key + ["actual_flood_days", "predicted_flood_days"]].rename(
        columns={"actual_flood_days": "actual_B", "predicted_flood_days": "prediction_B"})
    c = base[base.strategy == "C"][key + ["actual_flood_days", "predicted_flood_days"]].rename(
        columns={"actual_flood_days": "actual_C", "predicted_flood_days": "prediction_C"})
    pairs = b.merge(c, on=key, how="inner", validate="1:1")
    if pairs.empty:
        raise ValueError("No paired B/C samples were found")
    if not np.allclose(pairs.actual_B, pairs.actual_C, rtol=0, atol=1e-8):
        raise ValueError("Observed outcomes differ across B and C for the same sample")
    pairs["actual"] = pairs.actual_B
    pairs["abs_error_B"] = (pairs.prediction_B - pairs.actual).abs()
    pairs["abs_error_C"] = (pairs.prediction_C - pairs.actual).abs()
    pairs["C_minus_B_abs_error"] = pairs.abs_error_C - pairs.abs_error_B
    results = []
    for level, d in pairs.groupby("level"):
        results.append({"level": level, "matched_samples": len(d),
                        "B_MAE": d.abs_error_B.mean(), "C_MAE": d.abs_error_C.mean(),
                        "C_minus_B_MAE": d.C_minus_B_abs_error.mean(),
                        "C_better_pct": 100 * (d.C_minus_B_abs_error < 0).mean(),
                        "tied_pct": 100 * (d.C_minus_B_abs_error == 0).mean(),
                        "coverage_note": "Paired valid predictions only; not necessarily all admin regions"})
    return pairs, pd.DataFrame(results)


def strategy_bar(whole: pd.DataFrame, out: Path):
    d = whole[(whole.model == "M2") & (whole.algorithm == "RandomForest")]
    if d.empty:
        return
    levels = sorted(d.level.unique())
    fig, axes = plt.subplots(1, len(levels), figsize=(6 * len(levels), 4), squeeze=False)
    for ax, level in zip(axes.flat, levels):
        x = d[d.level == level].set_index("strategy")["MAE"]
        s = ["A", "B", "C"]
        v = [x.get(k, np.nan) for k in s]
        ax.bar(s, v)
        ax.set_title(f"{level}: strategy comparison (M2 RF)")
        ax.set_ylabel("MAE (recorded event dates)")
        for idx, val in enumerate(v):
            if pd.notna(val):
                ax.text(idx, val, f"{val:.2f}", ha="center", va="bottom", fontsize=10)
        ax.text(.5, -.23, "A: ~12 months ahead; B/C: ~1 month ahead", transform=ax.transAxes,
                ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "strategy_comparison_M2_RF.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def poster_selection(monthly: pd.DataFrame) -> pd.DataFrame:
    """Select months by test MAE, without claiming scientifically best months.

    Representative best/worst examples are chosen solely from C/M2/RF test errors,
    for each year+level. Selection uses the TEST set, so these months are suitable
    for illustrating test behaviour, not model selection/hyperparameter tuning.
    """
    d = monthly[(monthly.strategy == "C") & (monthly.model == "M2") &
                (monthly.algorithm == "RandomForest")].copy()
    picks = []
    for (level, year), sub in d.groupby(["level", "year"]):
        if sub.empty:
            continue
        for name, chosen in [("lowest_test_MAE", sub.loc[sub.MAE.idxmin()]),
                             ("highest_test_MAE", sub.loc[sub.MAE.idxmax()])]:
            picks.append({"level": level, "year": int(year), "month": int(chosen.month),
                          "reason": name, "MAE": round(float(chosen.MAE), 4),
                          "Bias": round(float(chosen.Bias), 4), "n": int(chosen.n)})
    return pd.DataFrame(picks)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None, type=Path)
    ap.add_argument("--input", default=None, type=Path, help="05b all_predictions.csv")
    ap.add_argument("--output", default=None, type=Path)
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    args = ap.parse_args()
    root = args.root.resolve() if args.root else root_dir()
    source = args.input if args.input else root / "processed_data/multivariate_flood_modeling/05b_monthly_random_forest/all_predictions.csv"
    out = args.output if args.output else root / "processed_data/multivariate_flood_modeling/05e_monthly_prediction_analysis"
    source, out = source.resolve(), out.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Missing prediction file: {source}. Run 05b first.")
    out.mkdir(parents=True, exist_ok=True)

    # Read relevant split only: prevents a huge 2M+ row memory footprint.
    chunks = []
    for chunk in pd.read_csv(source, chunksize=200_000, usecols=COLS):
        chosen = chunk[chunk.split == args.split]
        if not chosen.empty:
            chunks.append(chosen)
    if not chunks:
        raise ValueError(f"No '{args.split}' predictions in {source}")
    df = pd.concat(chunks, ignore_index=True)
    df = df.dropna(subset=["actual_flood_days", "predicted_flood_days"])
    for col in ["actual_flood_days", "predicted_flood_days"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[np.isfinite(df.actual_flood_days) & np.isfinite(df.predicted_flood_days)].copy()
    if df.empty:
        raise ValueError("No valid numerical prediction rows")

    by_month = grouped_metrics(df, ["level", "strategy", "model", "algorithm", "year", "month"])
    by_year = grouped_metrics(df, ["level", "strategy", "model", "algorithm", "year"])
    overall = grouped_metrics(df, ["level", "strategy", "model", "algorithm"])
    pairs, paired_summary = b_c_pairs(df)
    picks = poster_selection(by_month)

    by_month.to_csv(out / "01_metrics_by_month.csv", index=False)
    by_year.to_csv(out / "02_metrics_by_year.csv", index=False)
    overall.to_csv(out / "03_metrics_overall.csv", index=False)
    paired_summary.to_csv(out / "04_paired_B_vs_C.csv", index=False)
    picks.to_csv(out / "05_poster_month_candidates.csv", index=False)
    for level in sorted(df.level.unique()):
        heatmap(by_month, level, out)
    strategy_bar(overall, out)

    lines = [
        "# JBG060 — Monthly prediction interpretation",
        "",
        "**Target limitation:** The current 05a `--event-proxy` target is the number of recorded flood-event dates, not independently validated flood duration, rainfall, or flood extent.",
        f"**Data split:** {args.split}; figures are from historical predictions, not forecasts for future 2026 conditions.",
        "",
        "## Fair comparison: B versus C",
        "B and C are both one-month-ahead approaches and are compared on identical district-month targets.",
        "A has an approximately 12-month lead and should be interpreted separately.",
        "",
    ]
    for _, s in paired_summary.iterrows():
        lines.append(f"- {s.level}: {int(s.matched_samples):,} matched rows. "
                     f"B MAE = {s.B_MAE:.3f}; C MAE = {s.C_MAE:.3f}; "
                     f"C minus B = {s.C_minus_B_MAE:+.3f} recorded event dates.")
    lines += ["", "## How to use maps",
              "Use selected months to display observed proxy, predicted proxy and signed error, "
              "and state the selection rule. A month with low MAE is not proof of overall predictive skill.",
              "",
              "## Known limitations",
              "- Zero recorded event days are not independently verified zero-flood days.",
              "- Regions with missing predictors may be excluded; compare metrics on matched regions.",
              "- Compare the same target, horizon, sample, and time split; do not use the test set to tune parameters.",
              "- Errors expressed as 'days' describe the event-date proxy, not verified flood duration.",
              ""]
    (out / "README_results.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "methodology.json").write_text(json.dumps({
        "input_file": str(source), "split": args.split,
        "target": "exploratory_flood_event_record_date_proxy",
        "prediction_leads": {"A": "about 12 months", "B": "one month", "C": "one month"},
        "metrics": ["MAE", "RMSE", "Bias", "overpredicted_pct", "R2"],
        "poster_selection": "For each year and spatial level, select lowest and highest test-month MAE (descriptive only)",
        "no_retraining": True}, indent=2), encoding="utf-8")
    print(f"Analysis complete: {out}")
    print("\nMatched-sample B vs C:")
    print(paired_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nPoster map candidates:")
    print(picks.to_string(index=False))


if __name__ == "__main__":
    main()
