"""JBG060: batch monthly flood event-date proxy prediction maps.

Default: 2023-2025 x 12 months x ADM2/ADM3, plus annual summaries.
Input: 05b_monthly_random_forest/all_predictions.csv.
Run directly from project root or by VS Code's Run button (script locates repo root).

Figures: Observed / Predicted (identical 'Blues' scale) / Error (blue = under,
red = over). Grey = unavailable / unmatched, never zero by implication.

IMPORTANT: Targets from 05a --event-proxy are exploratory RECORDED flood-event
DATES, not validated true flood duration or rainfall forecasts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable


def project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [Path.cwd(), *here.parents]:
        if (p / "raw_data").is_dir() and (p / "processed_data").is_dir():
            return p
    return Path.cwd()


def boundary_candidates(folder: Path, level: str):
    items = [p for p in folder.rglob("*") if p.suffix.lower() in {".geojson", ".gpkg", ".shp"}]
    def score(p):
        name = p.stem.lower()
        match = level.lower() in name or f"admin{level[-1]}" in name
        fallback = level == "ADM2" and ("adm3" in name or "admin3" in name)
        return (0 if match else 1 if fallback else 2, 0 if p.suffix.lower() == ".geojson" else 1, len(str(p)))
    return sorted(items, key=score)


def load_boundary(folder: Path, level: str, explicit: str | None = None) -> tuple[gpd.GeoDataFrame, str]:
    choices = [Path(explicit)] if explicit else boundary_candidates(folder, level)
    keys = (["adm2_pcode", "ADM2_PCODE", "admin2Pcod"] if level == "ADM2" else
            ["adm3_pcode", "ADM3_PCODE", "admin3Pcod"])
    for f in choices:
        if not f.is_file():
            continue
        g = gpd.read_file(f)
        key = next((c for c in keys if c in g.columns), None)
        if key is None:
            continue
        g = g[[key, "geometry"]].dropna(subset=["geometry"]).copy()
        g["adm_id"] = g[key].astype(str).str.strip()
        g = g[["adm_id", "geometry"]]
        if g["adm_id"].duplicated().any():
            g = g.dissolve(by="adm_id", as_index=False)
        return g, str(f)
    raise FileNotFoundError(f"No suitable {level} boundaries in {folder}; files checked: {[str(x) for x in choices[:8]]}")


def create_figure(merged: gpd.GeoDataFrame, level: str, label: str, path: Path, vmax: float, errmax: float):
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
    columns = [("actual", "Observed", "Blues", 0, vmax),
               ("predicted", "Predicted", "Blues", 0, vmax),
               ("error", "Prediction error\nPredicted − Observed", "RdBu_r", -errmax, errmax)]
    for ax, (col, title, cmap, lo, hi) in zip(axes, columns):
        merged.plot(ax=ax, color="#e0e0e0", edgecolor="#ffffff", linewidth=0.2)
        valid = merged.loc[merged[col].notna()]
        if len(valid):
            valid.plot(ax=ax, column=col, cmap=cmap, vmin=lo, vmax=hi,
                       edgecolor="#ffffff", linewidth=0.2)
        norm = TwoSlopeNorm(vcenter=0, vmin=lo, vmax=hi) if col == "error" else Normalize(vmin=lo, vmax=hi)
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.66, fraction=0.04,
                     label="Error (event-record days)" if col == "error" else "Recorded event dates")
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()
    fig.suptitle(f"South Sudan | {level} | {label} | C / M2 / Random Forest", fontsize=13)
    fig.text(0.5, 0.005, "Exploratory event-date proxy; grey = no matched prediction. NOT validated flood duration.",
             ha="center", fontsize=8)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    p.add_argument("--levels", nargs="+", choices=["ADM2", "ADM3"], default=["ADM2", "ADM3"])
    p.add_argument("--strategy", choices=["A", "B", "C"], default="C")
    p.add_argument("--model", choices=["M0", "M1", "M2"], default="M2")
    p.add_argument("--split", choices=["train", "validation", "test"], default="test")
    p.add_argument("--adm2-boundaries", type=str, default=None)
    p.add_argument("--adm3-boundaries", type=str, default=None)
    p.add_argument("--predictions-csv", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--no-monthly", action="store_true")
    p.add_argument("--no-annual", action="store_true")
    args = p.parse_args()
    root = (args.root or project_root()).resolve()
    predfile = Path(args.predictions_csv) if args.predictions_csv else root / "processed_data/multivariate_flood_modeling/05b_monthly_random_forest/all_predictions.csv"
    out = Path(args.output) if args.output else root / "processed_data/multivariate_flood_modeling/05d_monthly_prediction_maps"
    out.mkdir(parents=True, exist_ok=True)
    if not predfile.is_file():
        raise FileNotFoundError(f"Missing 05b predictions: {predfile}. Run 05b first.")
    df = pd.read_csv(predfile)
    required = {"level", "strategy", "model", "algorithm", "split", "adm_id", "year", "month", "actual_flood_days", "predicted_flood_days"}
    if required - set(df.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(df.columns))}")
    df = df[(df.strategy == args.strategy) & (df.model == args.model) &
            (df.algorithm == "RandomForest") & (df.split == args.split) &
            df.year.isin(args.years)].copy()
    df["adm_id"] = df["adm_id"].astype(str).str.strip()
    df["actual_flood_days"] = pd.to_numeric(df["actual_flood_days"], errors="coerce")
    df["predicted_flood_days"] = pd.to_numeric(df["predicted_flood_days"], errors="coerce")
    if df.empty:
        raise ValueError("No selected prediction rows. Check --years / --split / --strategy / --model.")
    report = []
    for level in args.levels:
        sub = df[df.level == level].copy()
        if sub.empty:
            print(f"SKIP {level}: no rows")
            continue
        folder = root / "raw_data/administrative_boundaries"
        b, source = load_boundary(folder, level, getattr(args, f"{level.lower()}_boundaries"))
        leveldir = out / level.lower()
        leveldir.mkdir(exist_ok=True)
        sub["actual"] = sub["actual_flood_days"]
        sub["predicted"] = sub["predicted_flood_days"]
        sub["error"] = sub.predicted - sub.actual
        vmax_month = float(max(1, sub.actual.max(), sub.predicted.max()))
        errmax_month = float(max(1, sub.error.abs().max()))
        if not args.no_monthly:
            for year in args.years:
                for month in range(1, 13):
                    data = sub[(sub.year == year) & (sub.month == month)]
                    if data.empty:
                        continue
                    if data.adm_id.duplicated().any():
                        raise ValueError(f"Duplicate prediction IDs: {level} {year}-{month:02d}")
                    merged = b.merge(data[["adm_id", "actual", "predicted", "error"]], on="adm_id", how="left", validate="1:1")
                    n = int(merged.predicted.notna().sum())
                    if n == 0:
                        raise ValueError(f"No prediction IDs matched {level} map! boundary={source}")
                    dest = leveldir / f"{level.lower()}_{args.strategy}_{args.model}_{year}_{month:02d}.png"
                    create_figure(merged, level, f"{year}-{month:02d}", dest, vmax_month, errmax_month)
                    report.append(dict(level=level, period=f"{year}-{month:02d}", kind="monthly", matched=n, total=len(b), png=str(dest)))
                    print(f"{level} {year}-{month:02d}: {n}/{len(b)} matching polygons")
        if not args.no_annual:
            annual = sub.groupby(["adm_id", "year"], as_index=False).agg(
                actual=("actual", "sum"), predicted=("predicted", "sum"),
                observed_months=("actual", "count"), predicted_months=("predicted", "count"),
                distinct_months=("month", "nunique"))
            # Complete cases only: 12 months with both labels and predictions
            annual.loc[(annual.observed_months != 12) | (annual.predicted_months != 12) |
                       (annual.distinct_months != 12), ["actual", "predicted"]] = np.nan
            annual["error"] = annual.predicted - annual.actual
            vmax_annual = float(max(1, annual.actual.max(skipna=True), annual.predicted.max(skipna=True)))
            errmax_annual = float(max(1, annual.error.abs().max(skipna=True)))
            for year in args.years:
                data = annual[annual.year == year]
                if data.empty:
                    continue
                merged = b.merge(data[["adm_id", "actual", "predicted", "error"]], on="adm_id", how="left", validate="1:1")
                n = int(merged.predicted.notna().sum())
                if n == 0:
                    print(f"SKIP {level} {year}: no region has all 12 valid monthly predictions")
                    continue
                dest = leveldir / f"{level.lower()}_{args.strategy}_{args.model}_{year}_annual_SUM.png"
                create_figure(merged, level, f"{year} annual sum of 12 monthly event-date counts", dest, vmax_annual, errmax_annual)
                report.append(dict(level=level, period=str(year), kind="annual_sum", matched=n, total=len(b), png=str(dest)))
                print(f"{level} {year} annual summary: {n}/{len(b)} complete 12-month regions")
    if not report:
        raise RuntimeError("No maps generated.")
    pd.DataFrame(report).to_csv(out / "map_inventory.csv", index=False)
    (out / "README.txt").write_text(
        "Maps show observed and predicted monthly event-record-date proxies, NOT validated actual flood days or rainfall forecasts.\n"
        "Observed and predicted use the SAME color limits over all selected months for each administrative level.\n"
        "Annual summaries SUM the 12 monthly event-date counts only where all 12 months have observations and predictions.\n"
        "Grey means no matched prediction; red positive prediction error; blue negative error.\n"
        "Boundary match counts and figure paths are in map_inventory.csv.\n", encoding="utf-8")
    print(f"DONE: {len(report)} maps; output={out}")


if __name__ == "__main__":
    main()
