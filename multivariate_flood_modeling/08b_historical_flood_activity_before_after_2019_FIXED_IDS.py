#!/usr/bin/env python3
"""
08b_historical_flood_activity_before_after_2019.py

Purpose
-------
Create ADM2- and ADM3-level "before and after 2019" historical flood-activity
comparison outputs for South Sudan using already-built annual panel datasets.

Outputs
-------
processed_data/multivariate_flood_modeling/08b_before_after_2019/
    01_adm2_period_summary.csv
    02_adm3_period_summary.csv
    03_adm2_top_regions.csv
    04_adm3_top_regions.csv
    05_adm2_before_after_2019.png
    06_adm3_before_after_2019.png
    07_methodology.json

Notes
-----
- This script summarizes *historical flood activity days* already present in the
  annual panel datasets. It does NOT reprocess flood masks.
- The metric should be interpreted as "days with recorded flood activity in the
  administrative unit", not automatically as full-area inundation or disaster severity.
- Default comparison periods:
    2000-2010
    2011-2018
    2019-2025
- Ranking defaults to the 2019-2025 mean annual flood-activity days.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
PERIODS = [
    ("2000-2010", 2000, 2010),
    ("2011-2018", 2011, 2018),
    ("2019-2025", 2019, 2025),
]

RANK_PERIOD = "2019-2025"
TOP_N = 12

# Consistent colors (similar palette previously preferred)
COLORS = {
    "2000-2010": "#7EC9F2",  # bright light blue, matching 08c
    "2011-2018": "#C18AE0",  # bright lavender, matching 08c
    "2019-2025": "#7546B7",  # rich purple, matching 08c
}


def root_default() -> Path:
    p = Path(__file__).resolve()
    if p.parent.name == "multivariate_flood_modeling":
        return p.parents[2]
    return Path.cwd()


# ------------------------------------------------------------
# Generic helpers
# ------------------------------------------------------------
def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def find_one_csv(folder: Path) -> Optional[Path]:
    if not folder.exists():
        return None
    csvs = sorted(folder.rglob("*.csv"))
    # Prefer files with panel in name
    preferred = [p for p in csvs if "panel" in p.name.lower()]
    if preferred:
        return preferred[0]
    return csvs[0] if csvs else None


def guess_annual_panel(root: Path, level: str) -> Path:
    level = level.lower()
    folder_candidates = []
    if level == "adm2":
        folder_candidates = [
            root / "processed_data/multivariate_flood_modeling/01_adm2_panel_dataset",
            root / "processed_data/01_adm2_panel_dataset",
            root / "processed_data/multivariate_flood_modeling",
        ]
        file_candidates = [
            root / "processed_data/multivariate_flood_modeling/01_adm2_panel_dataset/adm2_panel_dataset.csv",
            root / "processed_data/multivariate_flood_modeling/01_adm2_panel_dataset/adm2_annual_panel.csv",
            root / "processed_data/adm2_panel_dataset.csv",
        ]
    else:
        folder_candidates = [
            root / "processed_data/multivariate_flood_modeling/02_adm3_panel_dataset",
            root / "processed_data/02_adm3_panel_dataset",
            root / "processed_data/multivariate_flood_modeling",
        ]
        file_candidates = [
            root / "processed_data/multivariate_flood_modeling/02_adm3_panel_dataset/adm3_panel_dataset.csv",
            root / "processed_data/multivariate_flood_modeling/02_adm3_panel_dataset/adm3_annual_panel.csv",
            root / "processed_data/adm3_panel_dataset.csv",
        ]
    # Full panel is the authoritative product with a stable `adm_id`.
    full = root / f"processed_data/multivariate_flood_modeling/{'01_adm2_panel_dataset' if level == 'adm2' else '02_adm3_panel_dataset'}/full_analysis_panel.csv"
    if full.exists():
        return full
    hit = first_existing(file_candidates)
    if hit:
        return hit
    for folder in folder_candidates:
        hit = find_one_csv(folder)
        if hit:
            return hit
    raise FileNotFoundError(f"Could not locate annual {level.upper()} panel CSV.")


def pick_col(columns: Sequence[str], candidates: Sequence[str], label: str) -> str:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    # fallback: substring contains all tokens
    for cand in candidates:
        tokens = [t for t in cand.lower().replace("__", "_").split("_") if t]
        for c in columns:
            cl = c.lower()
            if all(t in cl for t in tokens):
                return c
    raise ValueError(f"Could not identify column for {label}. Available columns: {list(columns)}")


# ------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------
def load_and_standardize(path: Path, level: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = list(df.columns)

    year_col = pick_col(cols, ["year", "flood__year"], "year")
    flood_col = pick_col(cols, ["flood__flood_days", "flood_days", "annual_flood_days"], "flood days")

    if level.lower() == "adm2":
        name_col = pick_col(cols, ["flood__ADM2_NAME", "ADM2_NAME", "adm2_name", "admin2name"], "ADM2 name")
        id_candidates = [c for c in cols if c.lower() in {"adm2_pcode", "adm2_code", "adm2_id", "admin2pcode"}]
        parent_col = None
    else:
        name_col = pick_col(cols, ["flood__ADM3_NAME", "ADM3_NAME", "adm3_name", "admin3name"], "ADM3 name")
        parent_col = pick_col(cols, ["flood__ADM2_NAME", "ADM2_NAME", "adm2_name", "admin2name"], "parent ADM2 name")
        id_candidates = [c for c in cols if c.lower() in {"adm3_pcode", "adm3_code", "adm3_id", "admin3pcode"}]

    # The integrated annual panels store official stable administrative IDs in `adm_id`.
    # Never silently fall back to the region name: names are not guaranteed unique.
    id_col = next((c for c in cols if c.lower() == "adm_id"), None)
    if id_col is None:
        id_col = id_candidates[0] if id_candidates else None
    if id_col is None:
        raise ValueError(f"{level.upper()} panel has no administrative code (adm_id/pcode): {path}")

    out = pd.DataFrame()
    out["year"] = pd.to_numeric(df[year_col], errors="coerce")
    out["flood_days"] = pd.to_numeric(df[flood_col], errors="coerce")
    out["region_name"] = df[name_col].astype(str)
    out["region_id"] = df[id_col].astype("string").str.strip().str.upper()
    out["parent_adm2_name"] = df[parent_col].astype(str) if parent_col else pd.NA

    out["region_name"] = df[name_col].astype("string").str.strip()
    if parent_col:
        out["parent_adm2_name"] = df[parent_col].astype("string").str.strip()
    out = out.dropna(subset=["year", "flood_days", "region_name", "region_id"]).copy()
    out = out[out["region_id"].ne("") & out["region_name"].ne("")].copy()
    out["year"] = out["year"].astype(int)
    out = out[out.year.between(2000, 2025)].copy()
    duplicates = out.duplicated(["region_id", "year"], keep=False)
    if duplicates.any():
        raise ValueError(f"Duplicate {level.upper()} region-year rows: " +
                         str(out.loc[duplicates, ["region_id", "year"]].head(8).to_dict("records")))
    inconsistent_names = out.groupby("region_id")["region_name"].nunique(dropna=True)
    if (inconsistent_names > 1).any():
        raise ValueError(f"Multiple region names for same {level.upper()} ID: " +
                         str(inconsistent_names[inconsistent_names > 1].head().to_dict()))
    if level.lower() == "adm3":
        inconsistent_parent = out.groupby("region_id")["parent_adm2_name"].nunique(dropna=True)
        if (inconsistent_parent > 1).any():
            raise ValueError("Conflicting ADM2 parent names for ADM3 IDs: " +
                             str(inconsistent_parent[inconsistent_parent > 1].head().to_dict()))
    return out


def summarize_periods(df: pd.DataFrame, level: str) -> pd.DataFrame:
    rows = []
    # Group by the stable key ONLY, not by mutable names.
    for region_id, sub in df.groupby("region_id", dropna=False):
        region_name = sub["region_name"].iloc[0]
        parent = sub["parent_adm2_name"].iloc[0] if level.lower() == "adm3" else pd.NA

        row = {
            "region_id": region_id,
            "region_name": region_name,
            "parent_adm2_name": parent,
        }

        # overall historical mean
        row["mean_annual_flood_days_2000_2025"] = sub["flood_days"].mean()
        row["years_with_recorded_flood_activity"] = int((sub["flood_days"] > 0).sum())
        row["total_years_observed"] = int(sub["year"].nunique())

        pre_mask = (sub["year"] >= 2000) & (sub["year"] <= 2018)
        row["mean_annual_flood_days_2000_2018"] = sub.loc[pre_mask, "flood_days"].mean()

        for label, start, end in PERIODS:
            mask = (sub["year"] >= start) & (sub["year"] <= end)
            vals = sub.loc[mask, "flood_days"]
            row[f"mean_{label}"] = vals.mean() if len(vals) else np.nan
            row[f"years_{label}"] = int(mask.sum())

        row["change_after_2019"] = row["mean_2019-2025"] - row["mean_annual_flood_days_2000_2018"]
        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values([f"mean_{RANK_PERIOD}", "mean_annual_flood_days_2000_2025"], ascending=[False, False])
    out.insert(0, "rank_recent", range(1, len(out) + 1))
    # Keep explicit administrative IDs for safe downstream joins.
    out.insert(1, "adm_id", out["region_id"])
    return out


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
def make_grouped_bar_chart(summary: pd.DataFrame, level_label: str, output_png: Path, top_n: int = TOP_N) -> pd.DataFrame:
    top = summary.head(top_n).copy()
    display_names = []
    for _, row in top.iterrows():
        if level_label == "ADM3":
            nm = f"{row['region_name']}\n({row['parent_adm2_name']})"
        else:
            nm = row["region_name"]
        display_names.append(nm)
    top["display_name"] = display_names

    x = np.arange(len(top))
    width = 0.24

    fig, ax = plt.subplots(figsize=(max(13, top_n * 0.95), 8))

    for i, (label, _, _) in enumerate(PERIODS):
        values = top[f"mean_{label}"].values
        ax.bar(x + (i - 1) * width, values, width=width, label=label, color=COLORS[label], edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(top["display_name"], rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Mean annual days with recorded flood activity", fontsize=12)
    ax.set_title(f"Historical Flood Activity: Before and After 2019 ({level_label})", fontsize=16, weight="bold")
    ax.legend(title="Period")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    note = (
        "Interpretation note: values summarize days with recorded flood activity in the administrative unit.\n"
        "They do not automatically equal full-area inundation duration or humanitarian impact severity."
    )
    fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=9)

    plt.tight_layout(rect=(0, 0.04, 1, 1))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return top


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def audit_against_old(new_df: pd.DataFrame, old_path: Path, level: str) -> pd.DataFrame:
    """Compare numerical values without relying on the old `region_id` field.

    For old outputs grouped by name, match by name + parent (ADM3) only if unique.
    Ambiguous names remain UNMATCHED; they are NEVER force-joined to IDs.
    """
    audit = new_df[["adm_id", "region_name", "parent_adm2_name"]].copy()
    audit["status"] = "NO_PRIOR_OUTPUT"
    audit["max_absolute_metric_difference"] = np.nan
    if not old_path.exists():
        return audit
    old = pd.read_csv(old_path)
    names = ["region_name"] + (["parent_adm2_name"] if level == "ADM3" else [])
    for col in names:
        old[col] = old[col].astype("string").str.strip().str.casefold()
        audit[col] = audit[col].astype("string").str.strip().str.casefold()
    numeric = [c for c in new_df.columns if (c.startswith("mean_") or c.startswith("years_")
                    or c in ("change_after_2019", "total_years_observed", "years_with_recorded_flood_activity"))
               and c in old.columns]
    if old.duplicated(names).any() or audit.duplicated(names).any():
        audit["status"] = "AMBIGUOUS_NAME_NO_COMPARISON"
        return audit
    old_keep = old[names + numeric].rename(columns={c: c+"_old" for c in numeric})
    new_keep = new_df[["adm_id"] + numeric]
    compared = audit.merge(old_keep, on=names, how="left", validate="one_to_one")
    compared = compared.merge(new_keep, on="adm_id", how="left", validate="one_to_one")
    if numeric:
        diffs = [(pd.to_numeric(compared[c], errors="coerce") -
                  pd.to_numeric(compared[c+"_old"], errors="coerce")).abs()
                 for c in numeric]
        compared["max_absolute_metric_difference"] = pd.concat(diffs, axis=1).max(axis=1)
    else:
        compared["max_absolute_metric_difference"] = np.nan
    compared["status"] = np.select([
        compared["max_absolute_metric_difference"].isna(),
        compared["max_absolute_metric_difference"].le(1e-8)],
        ["UNMATCHED_OR_NO_NUMERIC_METRICS", "UNCHANGED"], default="CHANGED")
    return compared[["adm_id"] + names + ["status", "max_absolute_metric_difference"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(root_default()))
    ap.add_argument("--adm2-panel", default=None)
    ap.add_argument("--adm3-panel", default=None)
    ap.add_argument("--top-n", type=int, default=TOP_N)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    outdir = Path(args.output).expanduser().resolve() if args.output else root / "processed_data/multivariate_flood_modeling/08b_before_after_2019"
    outdir.mkdir(parents=True, exist_ok=True)

    adm2_path = Path(args.adm2_panel).resolve() if args.adm2_panel else guess_annual_panel(root, "adm2")
    adm3_path = Path(args.adm3_panel).resolve() if args.adm3_panel else guess_annual_panel(root, "adm3")

    # Read prior summary BEFORE overwriting it.
    previous_adm2 = outdir / "01_adm2_period_summary.csv"
    previous_adm3 = outdir / "02_adm3_period_summary.csv"
    adm2_df = load_and_standardize(adm2_path, "adm2")
    adm3_df = load_and_standardize(adm3_path, "adm3")

    # Restrict to analysis range
    adm2_df = adm2_df[(adm2_df["year"] >= 2000) & (adm2_df["year"] <= 2025)].copy()
    adm3_df = adm3_df[(adm3_df["year"] >= 2000) & (adm3_df["year"] <= 2025)].copy()

    adm2_summary = summarize_periods(adm2_df, "adm2")
    adm3_summary = summarize_periods(adm3_df, "adm3")

    old_adm2_audit = audit_against_old(adm2_summary, previous_adm2, "ADM2")
    old_adm3_audit = audit_against_old(adm3_summary, previous_adm3, "ADM3")
    old_adm2_audit.to_csv(outdir / "08_adm2_prior_result_comparison.csv", index=False)
    old_adm3_audit.to_csv(outdir / "09_adm3_prior_result_comparison.csv", index=False)

    adm2_top = make_grouped_bar_chart(adm2_summary, "ADM2", outdir / "05_adm2_before_after_2019.png", top_n=args.top_n)
    adm3_top = make_grouped_bar_chart(adm3_summary, "ADM3", outdir / "06_adm3_before_after_2019.png", top_n=args.top_n)

    # Export summaries and tops
    adm2_summary.to_csv(outdir / "01_adm2_period_summary.csv", index=False)
    adm3_summary.to_csv(outdir / "02_adm3_period_summary.csv", index=False)
    adm2_top.to_csv(outdir / "03_adm2_top_regions.csv", index=False)
    adm3_top.to_csv(outdir / "04_adm3_top_regions.csv", index=False)

    methodology = {
        "adm2_panel": str(adm2_path),
        "adm3_panel": str(adm3_path),
        "periods": [{"label": p[0], "start": p[1], "end": p[2]} for p in PERIODS],
        "ranking_period": RANK_PERIOD,
        "top_n": int(args.top_n),
        "metric": "Mean annual days with recorded flood activity in each administrative unit.",
        "id_source": "adm_id from full_analysis_panel.csv (or an explicit administrative pcode)",
        "id_policy": "Never fall back to region names; group strictly by stable administrative ID",
        "prior_comparison": "08_adm2_prior_result_comparison.csv and 09_adm3_prior_result_comparison.csv",
        "interpretation_warning": (
            "This metric should not automatically be interpreted as full-area inundation duration or humanitarian severity. "
            "It is best treated as historical flood-activity intensity within the available flood-observation framework."
        ),
        "adm2_regions": int(adm2_summary["region_id"].nunique()),
        "adm3_regions": int(adm3_summary["region_id"].nunique()),
        "analysis_year_range": [2000, 2025],
        "outputs": {
            "adm2_summary": str(outdir / "01_adm2_period_summary.csv"),
            "adm3_summary": str(outdir / "02_adm3_period_summary.csv"),
            "adm2_top": str(outdir / "03_adm2_top_regions.csv"),
            "adm3_top": str(outdir / "04_adm3_top_regions.csv"),
            "adm2_plot": str(outdir / "05_adm2_before_after_2019.png"),
            "adm3_plot": str(outdir / "06_adm3_before_after_2019.png"),
        "adm2_audit_statuses": old_adm2_audit.status.value_counts().to_dict(),
        "adm3_audit_statuses": old_adm3_audit.status.value_counts().to_dict(),
        },
    }

    (outdir / "07_methodology.json").write_text(json.dumps(methodology, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "status": "OK",
        "output_dir": str(outdir),
        "adm2_rows": int(len(adm2_summary)),
        "adm3_rows": int(len(adm3_summary)),
        "adm2_plot": str(outdir / "05_adm2_before_after_2019.png"),
        "adm3_plot": str(outdir / "06_adm3_before_after_2019.png"),
        "adm2_audit_statuses": old_adm2_audit.status.value_counts().to_dict(),
        "adm3_audit_statuses": old_adm3_audit.status.value_counts().to_dict(),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
