"""
Upstream Analysis - Step 2: Hydrological Lag & Cross-Correlation Analysis
========================================================================
This script quantifies the travel time and statistical coupling between upstream
hydrological signals (Lake Victoria, Lake Albert, Lake Kyoga, and river discharge)
and downstream flood extent observed across South Sudan (NASA MODIS/VIIRS flood masks).

Key Steps:
1. Aggregate daily/weekly downstream flood extents (flooded area in km² and pixel counts)
   across South Sudan from NASA VIIRS/MODIS flood masks (2012-2024).
2. Align downstream flood dynamics with upstream lake elevations and discharge.
3. Compute cross-correlation functions across lags from -4 to +26 weeks (up to 180 days).
4. Identify optimal travel time lags (peak correlation) and assess how upstream pulses
   propagate down into South Sudan's Sudd floodplains.

Outputs:
- Weekly aligned time-series dataset in `outputs/upstream_analysis/data/`
- Lag-correlation summary statistics table in `outputs/upstream_analysis/data/`
- Publication-quality figures in `outputs/upstream_analysis/figures/`
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ==============================================================================
# 1. Directory & Path Setup
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def find_data_root() -> Path:
    candidates = [
        PROJECT_ROOT / "data-JBG060-2026",
        PROJECT_ROOT / "raw_data",
        PROJECT_ROOT,
    ]
    for c in candidates:
        if (c / "flood_masks").exists():
            return c
    raise FileNotFoundError("Could not find data directory containing flood_masks.")


DATA_ROOT = find_data_root()
OUTPUT_DATA_DIR = PROJECT_ROOT / "outputs" / "upstream_analysis" / "data"
OUTPUT_FIG_DIR = PROJECT_ROOT / "outputs" / "upstream_analysis" / "figures"

OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

# Pixel resolution for 250m product: ~0.0625 km² per pixel
KM2_PER_PIXEL = (250.0 * 250.0) / 1e6


# ==============================================================================
# 2. Downstream Flood Mask Aggregator
# ==============================================================================
def load_and_aggregate_flood_masks(start_year=2012, end_year=2024) -> pd.DataFrame:
    """
    Reads NASA compact flood mask parquets across tiles h20v08 and h21v08
    and aggregates daily counts of recurring, unusual, and total flooded pixels.
    """
    print(f"[DATA] Aggregating downstream flood masks ({start_year}-{end_year})...")
    p_rec = DATA_ROOT / "flood_masks" / "compact_recurring"
    p_unu = DATA_ROOT / "flood_masks" / "compact_unusual"

    records = []
    years = range(start_year, end_year + 1)
    tiles = ["h20v08", "h21v08"]

    for y in years:
        daily_y = {}
        for tile in tiles:
            f_rec = p_rec / f"flood_events_{tile}_{y}.parquet"
            if f_rec.exists():
                df_r = pd.read_parquet(f_rec, columns=["date"])
                counts = df_r.groupby("date").size()
                for d, val in counts.items():
                    daily_y[d] = daily_y.get(d, 0) + val

        for d, count in daily_y.items():
            records.append({"date": pd.to_datetime(d), "recurring_pixels": count})

    df_rec = pd.DataFrame(records)
    if not df_rec.empty:
        df_rec = df_rec.groupby("date").sum().sort_index()

    # Repeat for unusual
    records_unu = []
    for y in years:
        daily_y = {}
        for tile in tiles:
            f_unu = p_unu / f"flood_events_{tile}_{y}.parquet"
            if f_unu.exists():
                df_u = pd.read_parquet(f_unu, columns=["date"])
                counts = df_u.groupby("date").size()
                for d, val in counts.items():
                    daily_y[d] = daily_y.get(d, 0) + val

        for d, count in daily_y.items():
            records_unu.append({"date": pd.to_datetime(d), "unusual_pixels": count})

    df_unu = pd.DataFrame(records_unu)
    if not df_unu.empty:
        df_unu = df_unu.groupby("date").sum().sort_index()

    # Combine into unified daily series
    full_idx = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    df_flood = pd.DataFrame(index=full_idx)
    df_flood["recurring_pixels"] = df_rec["recurring_pixels"].reindex(full_idx).fillna(0) if not df_rec.empty else 0
    df_flood["unusual_pixels"] = df_unu["unusual_pixels"].reindex(full_idx).fillna(0) if not df_unu.empty else 0
    df_flood["total_flooded_pixels"] = df_flood["recurring_pixels"] + df_flood["unusual_pixels"]
    df_flood["total_flood_area_km2"] = df_flood["total_flooded_pixels"] * KM2_PER_PIXEL

    print(f"  -> Processed {len(df_flood)} days of flood observations.")
    return df_flood


# ==============================================================================
# 3. Load Upstream Data
# ==============================================================================
def load_upstream_features() -> pd.DataFrame:
    """Loads upstream data from Step 1 output or generates it."""
    step1_csv = OUTPUT_DATA_DIR / "upstream_water_levels_and_discharge.csv"
    if step1_csv.exists():
        print(f"[DATA] Loading existing upstream time-series from:\n  -> {step1_csv}")
        df = pd.read_csv(step1_csv, index_col="date", parse_dates=True)
        return df
    else:
        raise FileNotFoundError(f"Missing Step 1 dataset: {step1_csv}. Please run Step 1 first.")


# ==============================================================================
# 4. Cross-Correlation Engine
# ==============================================================================
def compute_cross_correlation(x: pd.Series, y: pd.Series, max_lag_weeks=26, min_lag_weeks=-4):
    """
    Computes Pearson cross-correlation r(lag) where positive lag means x LEADS y
    (i.e., x at t - lag correlates with y at t).
    """
    lags = range(min_lag_weeks, max_lag_weeks + 1)
    corrs = []

    for lag in lags:
        # Shift x by lag weeks: x_shifted[t] = x[t - lag]
        x_shifted = x.shift(lag)
        valid = pd.concat([x_shifted, y], axis=1).dropna()
        if len(valid) > 20:
            r = np.corrcoef(valid.iloc[:, 0], valid.iloc[:, 1])[0, 1]
        else:
            r = np.nan
        corrs.append({"lag_weeks": lag, "lag_days": lag * 7, "pearson_r": r})

    df_corr = pd.DataFrame(corrs)
    best_row = df_corr.loc[df_corr["pearson_r"].idxmax()]
    return df_corr, best_row


# ==============================================================================
# 5. Visualizations
# ==============================================================================
def plot_cross_correlograms(corr_results: dict, save_path: Path):
    """Plots cross-correlation curves with optimal lag markers."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    for ax, (name, (df_c, best)), color in zip(axes, corr_results.items(), colors):
        ax.plot(df_c["lag_weeks"], df_c["pearson_r"], color=color, linewidth=2.2, label="Pearson r(lag)")
        ax.axhline(0, color="gray", linestyle="-", linewidth=0.8)
        ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7, label="Zero Lag")

        # Highlight optimal peak lag
        opt_lag = best["lag_weeks"]
        max_r = best["pearson_r"]
        ax.plot(opt_lag, max_r, "ro", markersize=8,
                label=f"Peak: Lag = +{opt_lag:.0f} wks ({best['lag_days']} d)\nr = {max_r:.3f}")
        ax.axvline(opt_lag, color="red", linestyle=":", linewidth=1.5, alpha=0.8)

        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_ylabel("Cross-Correlation (r)", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    axes[2].set_xlabel("Upstream Lead Time Lag [Weeks]", fontsize=11)
    axes[3].set_xlabel("Upstream Lead Time Lag [Weeks]", fontsize=11)

    plt.suptitle("Hydrological Travel Time & Cross-Correlation: Upstream Drivers vs. Downstream Flood Extent",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  [SAVED] {save_path.name}")


def plot_lag_aligned_timeseries(df_weekly: pd.DataFrame, best_lags: dict, save_path: Path):
    """Dual-axis time series plot comparing shifted upstream drivers with downstream flood extent."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top Panel: Lake Albert shifted by optimal lag vs Flooded Area
    alb_lag = best_lags["Lake Albert"]["lag_weeks"]
    shifted_albert = df_weekly["water_level_albert_m"].shift(int(alb_lag))

    color_fl = "crimson"
    color_alb = "navy"

    ax1.set_ylabel("Flooded Area in South Sudan [km²]", color=color_fl, fontsize=11, fontweight="bold")
    line1 = ax1.plot(df_weekly.index, df_weekly["total_flood_area_km2"], color=color_fl, linewidth=1.8,
                     label="Downstream Flooded Area (km²)")
    ax1.tick_params(axis="y", labelcolor=color_fl)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax1_twin = ax1.twinx()
    ax1_twin.set_ylabel(f"Lake Albert Level [m] (Shifted +{int(alb_lag)} wks)", color=color_alb, fontsize=11, fontweight="bold")
    line2 = ax1_twin.plot(df_weekly.index, shifted_albert, color=color_alb, linewidth=1.8, linestyle="--",
                          label=f"Lake Albert Elevation (t - {int(alb_lag)} wks)")
    ax1_twin.tick_params(axis="y", labelcolor=color_alb)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", framealpha=0.9)
    ax1.set_title("Downstream Inundation vs. Lag-Shifted Lake Albert Water Level", fontsize=12, fontweight="bold")

    # Bottom Panel: Malakal Discharge shifted vs Flooded Area
    mal_lag = best_lags["Malakal Discharge"]["lag_weeks"]
    shifted_mal = df_weekly["discharge_malakal_m3s"].shift(int(mal_lag))
    color_mal = "darkgreen"

    ax2.set_ylabel("Flooded Area in South Sudan [km²]", color=color_fl, fontsize=11, fontweight="bold")
    line3 = ax2.plot(df_weekly.index, df_weekly["total_flood_area_km2"], color=color_fl, linewidth=1.8,
                     label="Downstream Flooded Area (km²)")
    ax2.tick_params(axis="y", labelcolor=color_fl)
    ax2.grid(True, linestyle=":", alpha=0.6)

    ax2_twin = ax2.twinx()
    ax2_twin.set_ylabel(f"Malakal Discharge [m³/s] (Shifted +{int(mal_lag)} wks)", color=color_mal, fontsize=11, fontweight="bold")
    line4 = ax2_twin.plot(df_weekly.index, shifted_mal, color=color_mal, linewidth=1.8, linestyle="--",
                          label=f"Malakal Discharge (t - {int(mal_lag)} wks)")
    ax2_twin.tick_params(axis="y", labelcolor=color_mal)

    lines_b = line3 + line4
    labels_b = [l.get_label() for l in lines_b]
    ax2.legend(lines_b, labels_b, loc="upper left", framealpha=0.9)
    ax2.set_title("Downstream Inundation vs. Lag-Shifted Malakal River Discharge", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Year", fontsize=12)

    plt.suptitle("Lag-Synchronized Dynamic Coupling (2015-2024)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  [SAVED] {save_path.name}")


def plot_lagged_scatter(df_weekly: pd.DataFrame, best_lags: dict, save_path: Path):
    """Scatter plot and linear regression of flood extent vs. optimally lagged lake levels."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Lake Albert
    lag_a = int(best_lags["Lake Albert"]["lag_weeks"])
    x1 = df_weekly["water_level_albert_m"].shift(lag_a)
    y1 = df_weekly["total_flood_area_km2"]
    valid1 = pd.concat([x1, y1], axis=1).dropna()

    p1 = np.polyfit(valid1.iloc[:, 0], valid1.iloc[:, 1], 1)
    r1 = np.corrcoef(valid1.iloc[:, 0], valid1.iloc[:, 1])[0, 1]

    ax1.scatter(valid1.iloc[:, 0], valid1.iloc[:, 1], color="royalblue", alpha=0.6, edgecolors="none")
    x_range1 = np.linspace(valid1.iloc[:, 0].min(), valid1.iloc[:, 0].max(), 100)
    ax1.plot(x_range1, np.polyval(p1, x_range1), color="darkred", linewidth=2.5,
             label=f"Trendline (r = {r1:.3f}, R² = {r1**2:.2f})")
    ax1.set_xlabel(f"Lake Albert Elevation [m] (Lag = +{lag_a} wks)", fontsize=11)
    ax1.set_ylabel("South Sudan Flooded Area [km²]", fontsize=11)
    ax1.set_title(f"Lake Albert Level vs. Inundation Extent (Lag: {lag_a} wks)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", fontsize=10)

    # Lake Victoria
    lag_v = int(best_lags["Lake Victoria"]["lag_weeks"])
    x2 = df_weekly["vic_egm2008_m"].shift(lag_v)
    y2 = df_weekly["total_flood_area_km2"]
    valid2 = pd.concat([x2, y2], axis=1).dropna()

    p2 = np.polyfit(valid2.iloc[:, 0], valid2.iloc[:, 1], 1)
    r2 = np.corrcoef(valid2.iloc[:, 0], valid2.iloc[:, 1])[0, 1]

    ax2.scatter(valid2.iloc[:, 0], valid2.iloc[:, 1], color="forestgreen", alpha=0.6, edgecolors="none")
    x_range2 = np.linspace(valid2.iloc[:, 0].min(), valid2.iloc[:, 0].max(), 100)
    ax2.plot(x_range2, np.polyval(p2, x_range2), color="darkred", linewidth=2.5,
             label=f"Trendline (r = {r2:.3f}, R² = {r2**2:.2f})")
    ax2.set_xlabel(f"Lake Victoria Elevation [m] (Lag = +{lag_v} wks)", fontsize=11)
    ax2.set_ylabel("South Sudan Flooded Area [km²]", fontsize=11)
    ax2.set_title(f"Lake Victoria Level vs. Inundation Extent (Lag: {lag_v} wks)", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", fontsize=10)

    plt.suptitle("Statistical Coupling: Flood Area as a Function of Upstream Lake Storage",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  [SAVED] {save_path.name}")


# ==============================================================================
# 6. Main Execution Pipeline
# ==============================================================================
def main():
    print("=" * 70)
    print("STEP 2: HYDROLOGICAL LAG & CROSS-CORRELATION ANALYSIS")
    print("=" * 70)

    # 1. Load upstream and downstream datasets
    df_upstream = load_upstream_features()
    df_flood = load_and_aggregate_flood_masks(start_year=2015, end_year=2024)

    # Combine into shared daily timeline
    combined_daily = df_upstream.join(df_flood, how="inner")
    combined_daily.index.name = "date"

    # 2. Resample to Weekly Frequency
    # (Weekly smoothing accounts for satellite return periods and cloud gaps)
    print("\n[ANALYSIS] Resampling to weekly frequency for lag correlation...")
    df_weekly = combined_daily.resample("W").agg({
        "vic_egm2008_m": "mean",
        "kyo_egm2008_m": "mean",
        "water_level_albert_m": "mean",
        "discharge_malakal_m3s": "mean",
        "total_flood_area_km2": "mean",
        "total_flooded_pixels": "mean",
        "recurring_pixels": "mean",
        "unusual_pixels": "mean",
    }).dropna(subset=["total_flood_area_km2"])

    # Interpolate any tiny weekly missing lake observations
    df_weekly["vic_egm2008_m"] = df_weekly["vic_egm2008_m"].interpolate(method="linear")
    df_weekly["kyo_egm2008_m"] = df_weekly["kyo_egm2008_m"].interpolate(method="linear")
    df_weekly["water_level_albert_m"] = df_weekly["water_level_albert_m"].interpolate(method="linear")
    df_weekly["discharge_malakal_m3s"] = df_weekly["discharge_malakal_m3s"].interpolate(method="linear")

    # 3. Compute cross-correlations
    target_var = df_weekly["total_flood_area_km2"]
    drivers = [
        ("Lake Albert", df_weekly["water_level_albert_m"]),
        ("Lake Victoria", df_weekly["vic_egm2008_m"]),
        ("Lake Kyoga", df_weekly["kyo_egm2008_m"]),
        ("Malakal Discharge", df_weekly["discharge_malakal_m3s"]),
    ]

    corr_results = {}
    best_summary = {}
    summary_rows = []

    print("\n[RESULTS] Optimal Lag & Peak Cross-Correlation:")
    for name, series in drivers:
        df_c, best = compute_cross_correlation(series, target_var, max_lag_weeks=26, min_lag_weeks=-4)
        corr_results[name] = (df_c, best)
        best_summary[name] = best

        summary_rows.append({
            "driver_name": name,
            "optimal_lag_weeks": int(best["lag_weeks"]),
            "optimal_lag_days": int(best["lag_days"]),
            "peak_pearson_r": round(best["pearson_r"], 3),
            "r_squared": round(best["pearson_r"] ** 2, 3),
            "interpretation": f"Upstream {name} leads downstream flood extent by {int(best['lag_weeks'])} weeks."
        })
        print(f"  * {name:18s}: Peak r = {best['pearson_r']:+.3f} (R² = {best['pearson_r']**2:.2f}) at Lag = +{int(best['lag_weeks'])} weeks ({best['lag_days']} days)")

    # 4. Save Outputs
    df_summary = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DATA_DIR / "lag_correlation_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"\n[DATA] Saved lag-correlation summary to:\n  -> {summary_path}")

    aligned_path = OUTPUT_DATA_DIR / "weekly_upstream_downstream_aligned.csv"
    df_weekly.to_csv(aligned_path)
    print(f"[DATA] Saved aligned weekly dataset to:\n  -> {aligned_path}")

    # 5. Visualizations
    print("\n[VISUALIZATION] Generating Publication Figures:")
    plot_cross_correlograms(corr_results, OUTPUT_FIG_DIR / "02_cross_correlation_correlograms.png")
    plot_lag_aligned_timeseries(df_weekly, best_summary, OUTPUT_FIG_DIR / "02_lag_aligned_timeseries_comparison.png")
    plot_lagged_scatter(df_weekly, best_summary, OUTPUT_FIG_DIR / "02_flood_extent_vs_lake_scatter.png")

    print("\n[COMPLETE] Step 2 Hydrological Lag & Cross-Correlation successfully finished.\n")


if __name__ == "__main__":
    main()
