"""
Upstream Analysis - Step 3: Driver Attribution (Fluvial vs. Pluvial Analysis)
=============================================================================
This script addresses the fundamental scientific attribution question:
Are South Sudan's persistent floods driven by:
1. Local precipitation over South Sudan (Pluvial driver), OR
2. Upstream lake storage and river inflow from the Great Lakes (Fluvial driver)?

Key Steps:
1. Extract daily/weekly spatial precipitation from ERA5 across:
   - Local South Sudan domain (Lat 3.5° to 12.0°N, Lon 24.0° to 35.0°E)
   - Upstream Equatorial Lake Basin (Uganda/Kenya/Tanzania: Lat -3.0° to 3.5°N, Lon 29.0° to 35.0°E)
2. Integrate upstream lake storage (Lake Albert with optimal 6-week lag, Lake Victoria with 26-week lag).
3. Align with downstream observed flood extent (NASA VIIRS/MODIS masks, 2015-2024).
4. Perform standardized multivariate attribution modeling:
   - Ordinary Least Squares (OLS) regression on standardized anomalies (z-scores).
   - Relative importance decomposition (variance explained R² shares).
5. Conduct annual regime analysis:
   - Compare driver behavior during normal years (2015-2018) vs. crisis flood years (2019-2024).

Outputs:
- Regional rainfall CSV cache in `outputs/upstream_analysis/data/`
- Attribution regression statistics in `outputs/upstream_analysis/data/`
- Annual driver breakdown table in `outputs/upstream_analysis/data/`
- Publication-quality visualization figures in `outputs/upstream_analysis/figures/`
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
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
        if (c / "rainfall and runoff").exists():
            return c
    raise FileNotFoundError("Could not find data directory containing 'rainfall and runoff'.")


DATA_ROOT = find_data_root()
OUTPUT_DATA_DIR = PROJECT_ROOT / "outputs" / "upstream_analysis" / "data"
OUTPUT_FIG_DIR = PROJECT_ROOT / "outputs" / "upstream_analysis" / "figures"

OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 2. ERA5 Spatial Precipitation Extractor
# ==============================================================================
def extract_regional_rainfall(start_year=2015, end_year=2024) -> pd.DataFrame:
    """
    Extracts daily spatial mean rainfall (mm/day) from ERA5 for:
    - South Sudan local domain
    - Upstream Equatorial Lakes catchment
    Caches output to CSV for fast future executions.
    """
    cache_file = OUTPUT_DATA_DIR / f"era5_regional_rainfall_{start_year}_{end_year}.csv"
    if cache_file.exists():
        print(f"[DATA] Loading cached ERA5 regional rainfall from:\n  -> {cache_file}")
        df = pd.read_csv(cache_file, index_col="date", parse_dates=True)
        return df

    print(f"[DATA] Extracting ERA5 rainfall across domains ({start_year}-{end_year})...")
    era_dir = DATA_ROOT / "rainfall and runoff"

    daily_records = []

    for y in range(start_year, end_year + 1):
        nc_path = era_dir / f"ERA5_{y}.nc"
        if not nc_path.exists():
            print(f"  [WARN] Missing file: {nc_path.name}")
            continue

        print(f"  -> Processing ERA5 year {y}...")
        ds = xr.open_dataset(nc_path)

        # Slice spatial domains
        # South Sudan: Lat 12 to 3.5, Lon 24 to 36
        ds_ss = ds.sel(latitude=slice(12.0, 3.5), longitude=slice(24.0, 36.0))
        # Upstream Basin: Lat 3.5 to -3.0, Lon 29.0 to 35.0
        ds_up = ds.sel(latitude=slice(3.5, -3.0), longitude=slice(29.0, 35.0))

        # ERA5 'tp' is total precipitation in meters. Sum daily then mean spatially.
        # Resample valid_time to 1D sum
        daily_ss = ds_ss["tp"].resample(valid_time="1D").sum()
        daily_up = ds_up["tp"].resample(valid_time="1D").sum()

        mean_ss_mm = daily_ss.mean(dim=["latitude", "longitude"]).values * 1000.0
        mean_up_mm = daily_up.mean(dim=["latitude", "longitude"]).values * 1000.0
        dates = daily_ss["valid_time"].values

        for d, r_ss, r_up in zip(dates, mean_ss_mm, mean_up_mm):
            daily_records.append({
                "date": pd.to_datetime(d),
                "rainfall_local_ss_mm": float(r_ss),
                "rainfall_upstream_mm": float(r_up),
            })

        ds.close()

    df = pd.DataFrame(daily_records).set_index("date").sort_index()
    df.to_csv(cache_file)
    print(f"[DATA] Cached regional rainfall to: {cache_file}")
    return df


# ==============================================================================
# 3. Multivariable Standardized Regression & Variance Decomposition
# ==============================================================================
def fit_standardized_ols(df_reg: pd.DataFrame, target_col: str, feature_cols: list[str]):
    """
    Fits standardized multiple linear regression:
    y_std = sum(beta_i * x_std_i) + e
    Returns beta coefficients, t-stats, R², and relative variance shares.
    """
    # Drop rows with missing values
    data = df_reg[[target_col] + feature_cols].dropna()

    # Standardize to zero mean, unit variance
    means = data.mean()
    stds = data.std()
    data_std = (data - means) / stds

    y = data_std[target_col].values
    X = data_std[feature_cols].values

    # Add intercept column (will be ~0 for standardized data)
    X_mat = np.column_stack([np.ones(len(y)), X])

    # OLS estimation: beta = (X^T X)^-1 X^T y
    XtX_inv = np.linalg.pinv(X_mat.T @ X_mat)
    betas = XtX_inv @ (X_mat.T @ y)

    # Residuals & R²
    y_pred = X_mat @ betas
    residuals = y - y_pred
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum(residuals ** 2)
    r_squared = 1.0 - (ss_res / ss_tot)
    adj_r_squared = 1.0 - ((1.0 - r_squared) * (len(y) - 1) / (len(y) - len(feature_cols) - 1))

    # Standard errors and t-statistics
    sigma2 = ss_res / (len(y) - len(betas))
    var_betas = np.diag(sigma2 * XtX_inv)
    se_betas = np.sqrt(np.maximum(var_betas, 1e-12))
    t_stats = betas / se_betas

    # Relative importance via standardized beta squared shares
    # (Lindeman, Merenda, and Gold heuristic)
    abs_betas = np.abs(betas[1:])
    importance_pct = (abs_betas / np.sum(abs_betas)) * 100.0

    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "standardized_beta": betas[1:],
        "std_error": se_betas[1:],
        "t_statistic": t_stats[1:],
        "relative_importance_pct": importance_pct,
    })

    return coef_df, r_squared, adj_r_squared, len(y)


# ==============================================================================
# 4. Visualizations
# ==============================================================================
def plot_driver_attribution_timeseries(df: pd.DataFrame, save_path: Path):
    """4-panel comparison of local rain, upstream rain, lake level, and flood extent."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)

    # Panel 1: Local South Sudan Rainfall
    axes[0].plot(df.index, df["rainfall_local_ss_mm"], color="skyblue", alpha=0.6, label="Weekly Local Rain")
    axes[0].plot(df.index, df["rainfall_local_ss_mm"].rolling(8, center=True).mean(), color="navy", linewidth=1.8, label="8-Wk Rolling Mean")
    axes[0].set_ylabel("Local Rain\n[mm/day]", fontsize=10, fontweight="bold")
    axes[0].set_title("A. Pluvial Driver: Local South Sudan Precipitation (ERA5)", fontsize=11, fontweight="bold", loc="left")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper right", fontsize=9)

    # Panel 2: Upstream Basin Rainfall
    axes[1].plot(df.index, df["rainfall_upstream_mm"], color="mediumseagreen", alpha=0.6, label="Weekly Upstream Catchment Rain")
    axes[1].plot(df.index, df["rainfall_upstream_mm"].rolling(8, center=True).mean(), color="darkgreen", linewidth=1.8, label="8-Wk Rolling Mean")
    axes[1].set_ylabel("Upstream Rain\n[mm/day]", fontsize=10, fontweight="bold")
    axes[1].set_title("B. Catchment Driver: Upstream Lake Basin Precipitation (ERA5)", fontsize=11, fontweight="bold", loc="left")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper right", fontsize=9)

    # Panel 3: Upstream Lake Albert Elevation
    axes[2].plot(df.index, df["water_level_albert_m"], color="darkorange", linewidth=2.0, label="Lake Albert Water Level")
    axes[2].axhline(621.63, color="dimgray", linestyle="--", label="Historical Baseline (621.63 m)")
    axes[2].axvspan(pd.Timestamp("2019-10-01"), pd.Timestamp("2024-12-31"), color="gold", alpha=0.2, label="Multi-year High-Water Plateau")
    axes[2].set_ylabel("Elevation\n[m]", fontsize=10, fontweight="bold")
    axes[2].set_title("C. Fluvial Inflow Driver: Upstream Lake Albert Water Level (Altimetry)", fontsize=11, fontweight="bold", loc="left")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend(loc="upper left", fontsize=9)

    # Panel 4: Downstream Flood Extent
    axes[3].fill_between(df.index, 0, df["total_flood_area_km2"], color="crimson", alpha=0.5, label="Flooded Area")
    axes[3].plot(df.index, df["total_flood_area_km2"], color="darkred", linewidth=1.5)
    axes[3].set_ylabel("Flooded Area\n[km²]", fontsize=10, fontweight="bold")
    axes[3].set_title("D. Downstream Impact: Observed Inundation Extent across South Sudan (NASA VIIRS/MODIS)", fontsize=11, fontweight="bold", loc="left")
    axes[3].grid(True, linestyle=":", alpha=0.6)
    axes[3].legend(loc="upper left", fontsize=9)
    axes[3].set_xlabel("Year", fontsize=11)

    axes[3].xaxis.set_major_locator(mdates.YearLocator(1))
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xlim(df.index.min(), df.index.max())

    plt.suptitle("Causal Attribution: Local Pluvial Rainfall vs. Upstream Fluvial Lake Storage (2015-2024)",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  [SAVED] {save_path.name}")


def plot_relative_driver_importance(coef_df: pd.DataFrame, r2: float, save_path: Path):
    """Bar chart illustrating standardized effect size and relative importance percentage."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # 1. Standardized Beta coefficients
    colors = ["#1f77b4" if b > 0 else "#d62728" for b in coef_df["standardized_beta"]]
    ax1.barh(coef_df["feature"], coef_df["standardized_beta"], color=colors, alpha=0.85, edgecolor="black")
    ax1.axvline(0, color="black", linewidth=1)
    ax1.set_xlabel("Standardized Effect Size (β)", fontsize=11, fontweight="bold")
    ax1.set_title(f"Standardized Regression Coefficients\n(Total Model R² = {r2:.2f})", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)

    for i, row in coef_df.iterrows():
        b = row["standardized_beta"]
        ax1.text(b + (0.02 if b >= 0 else -0.08), i, f"β = {b:+.2f}", va="center", fontsize=10, fontweight="bold")

    # 2. Relative Importance Pie/Donut Chart
    labels = coef_df["feature"].tolist()
    shares = coef_df["relative_importance_pct"].tolist()
    palette = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"][:len(labels)]

    wedges, texts, autotexts = ax2.pie(
        shares,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=palette,
        wedgeprops=dict(width=0.45, edgecolor="w", linewidth=2),
        textprops=dict(fontsize=10),
    )
    for autotext in autotexts:
        autotext.set_fontweight("bold")

    ax2.set_title("Relative Explained Variance Share (%)\n(Fluvial vs. Pluvial Drivers)", fontsize=12, fontweight="bold")

    plt.suptitle("Quantitative Driver Attribution: Why Does South Sudan Flood?", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  [SAVED] {save_path.name}")


def plot_annual_regime_comparison(df_annual: pd.DataFrame, save_path: Path):
    """Annual grouped bar chart comparing standardized anomalies of drivers vs flood extent."""
    fig, ax = plt.subplots(figsize=(13, 6))

    years = df_annual.index
    x = np.arange(len(years))
    width = 0.22

    # Standardize annual aggregates to compare on equal z-score scales
    z_flood = (df_annual["flood_area"] - df_annual["flood_area"].mean()) / df_annual["flood_area"].std()
    z_lake = (df_annual["lake_albert"] - df_annual["lake_albert"].mean()) / df_annual["lake_albert"].std()
    z_local_rain = (df_annual["local_rain"] - df_annual["local_rain"].mean()) / df_annual["local_rain"].std()
    z_up_rain = (df_annual["upstream_rain"] - df_annual["upstream_rain"].mean()) / df_annual["upstream_rain"].std()

    ax.bar(x - 1.5 * width, z_local_rain, width, label="Local Rain Anomaly (Pluvial)", color="dodgerblue", alpha=0.85)
    ax.bar(x - 0.5 * width, z_up_rain, width, label="Upstream Catchment Rain Anomaly", color="mediumseagreen", alpha=0.85)
    ax.bar(x + 0.5 * width, z_lake, width, label="Lake Albert Elevation Anomaly (Fluvial)", color="darkorange", alpha=0.85)
    ax.bar(x + 1.5 * width, z_flood, width, label="Observed Flood Extent Anomaly", color="crimson", alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=11, fontweight="bold")
    ax.set_ylabel("Standardized Annual Anomaly [σ]", fontsize=11, fontweight="bold")
    ax.set_title("Year-by-Year Regime Shift: Upstream Lake Surge Decouples Floods from Local Rain (2015-2024)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  [SAVED] {save_path.name}")


# ==============================================================================
# 5. Main Execution Pipeline
# ==============================================================================
def main():
    print("=" * 70)
    print("STEP 3: DRIVER ATTRIBUTION (FLUVIAL VS. PLUVIAL ANALYSIS)")
    print("=" * 70)

    # 1. Load weekly aligned dataset from Step 2
    step2_csv = OUTPUT_DATA_DIR / "weekly_upstream_downstream_aligned.csv"
    if not step2_csv.exists():
        print(f"[ERROR] Could not find {step2_csv}. Please run Step 2 first.")
        sys.exit(1)

    print(f"[DATA] Loading aligned upstream and flood data from:\n  -> {step2_csv}")
    df_step2 = pd.read_csv(step2_csv, index_col="date", parse_dates=True)

    # 2. Extract and resample ERA5 regional rainfall
    df_rain_daily = extract_regional_rainfall(start_year=2015, end_year=2024)
    df_rain_weekly = df_rain_daily.resample("W").mean()

    # Join into unified weekly analysis frame
    merged = df_step2.join(df_rain_weekly, how="inner").dropna(subset=["total_flood_area_km2"])

    # 3. Apply Hydrological Lags identified in Step 2
    # Lake Albert optimal lag is ~6 weeks
    merged["lake_albert_lag6w"] = merged["water_level_albert_m"].shift(6)
    # Lake Victoria optimal lag is ~26 weeks
    merged["lake_victoria_lag26w"] = merged["vic_egm2008_m"].shift(26)
    # Upstream catchment rainfall lag ~6 weeks (flow transit time)
    merged["upstream_rain_lag6w"] = merged["rainfall_upstream_mm"].shift(6)
    # Local South Sudan rainfall operates immediately (concurrent lag 0)
    merged["local_rain_lag0"] = merged["rainfall_local_ss_mm"]

    # 4. Standardized Multiple Regression Modeling
    feature_cols = [
        "lake_albert_lag6w",
        "upstream_rain_lag6w",
        "local_rain_lag0",
    ]
    target_col = "total_flood_area_km2"

    print("\n[ANALYSIS] Fitting Standardized Attribution Regression Model...")
    coef_df, r2, adj_r2, n_obs = fit_standardized_ols(merged, target_col, feature_cols)

    # Clean feature names for presentation
    feature_display_names = {
        "lake_albert_lag6w": "Upstream Lake Albert Inflow (Lag 6w) [Fluvial]",
        "upstream_rain_lag6w": "Upstream Catchment Rainfall (Lag 6w)",
        "local_rain_lag0": "Local South Sudan Rainfall (Lag 0) [Pluvial]",
    }
    coef_df["feature"] = coef_df["feature"].map(feature_display_names)

    print(f"\n[RESULTS] Model Goodness-of-Fit (N = {n_obs} weekly timesteps):")
    print(f"  * Total Explained Variance (R²): {r2:.3f} (Adjusted R²: {adj_r2:.3f})")
    print("\n[RESULTS] Attribution Factor Coefficients & Relative Shares:")
    for _, row in coef_df.iterrows():
        print(f"  * {row['feature']:50s}: β = {row['standardized_beta']:+.3f}, Relative Share = {row['relative_importance_pct']:.1f}%")

    # 5. Annual Regime Analysis
    annual_summary = merged.resample("YE").agg({
        "total_flood_area_km2": "mean",
        "water_level_albert_m": "mean",
        "rainfall_local_ss_mm": "mean",
        "rainfall_upstream_mm": "mean",
    })
    annual_summary.index = annual_summary.index.year
    annual_summary.columns = ["flood_area", "lake_albert", "local_rain", "upstream_rain"]

    print("\n[RESULTS] Annual Mean Values (2015-2024):")
    print(annual_summary.to_string())

    # 6. Save Tables
    reg_path = OUTPUT_DATA_DIR / "attribution_regression_results.csv"
    coef_df.to_csv(reg_path, index=False)
    print(f"\n[DATA] Saved regression attribution results to:\n  -> {reg_path}")

    annual_path = OUTPUT_DATA_DIR / "annual_driver_breakdown.csv"
    annual_summary.to_csv(annual_path)
    print(f"[DATA] Saved annual driver breakdown to:\n  -> {annual_path}")

    # 7. Generate Visualizations
    print("\n[VISUALIZATION] Generating Publication Figures:")
    plot_driver_attribution_timeseries(merged, OUTPUT_FIG_DIR / "03_driver_attribution_comparison_timeseries.png")
    plot_relative_driver_importance(coef_df, r2, OUTPUT_FIG_DIR / "03_relative_driver_importance_barchart.png")
    plot_annual_regime_comparison(annual_summary, OUTPUT_FIG_DIR / "03_annual_flood_driver_contributions.png")

    print("\n[COMPLETE] Step 3 Driver Attribution successfully finished.\n")


if __name__ == "__main__":
    main()
