"""
Upstream Analysis - Step 1: Upstream Trends & Anomaly Detection
===============================================================
This script performs long-term trend analysis, baseline climatology, and anomaly
detection for upstream hydrological drivers of the White Nile basin:
1. Upstream Lake Water Levels:
   - Lake Victoria (Uganda/Tanzania/Kenya)
   - Lake Kyoga (Uganda)
   - Lake Albert (Uganda/DRC)
2. River Discharge:
   - Dartmouth Flood Observatory station 100205 (Malakal, South Sudan - primary Nile outflow/inflow)
   - Station 1541 (Sudan border) and Station 1505 (Ethiopia)

Outputs:
- Cleaned and aligned time-series CSV in `outputs/upstream_analysis/data/`
- Detected high-water surge events CSV in `outputs/upstream_analysis/data/`
- Publication-quality visualization figures in `outputs/upstream_analysis/figures/`
"""

import re
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
    """Find the root data directory (supporting data-JBG060-2026 or raw_data)."""
    candidates = [
        PROJECT_ROOT / "data-JBG060-2026",
        PROJECT_ROOT / "raw_data",
        PROJECT_ROOT,
    ]
    for c in candidates:
        if (c / "Water levels lakes").exists():
            return c
    raise FileNotFoundError("Could not find 'Water levels lakes' directory in candidate paths.")


DATA_ROOT = find_data_root()
OUTPUT_DATA_DIR = PROJECT_ROOT / "outputs" / "upstream_analysis" / "data"
OUTPUT_FIG_DIR = PROJECT_ROOT / "outputs" / "upstream_analysis" / "figures"

OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

print(f"[INFO] Using Data Root: {DATA_ROOT}")
print(f"[INFO] Outputs will be saved to: {OUTPUT_DATA_DIR} and {OUTPUT_FIG_DIR}")


# ==============================================================================
# 2. Upstream Lake Data Loaders
# ==============================================================================
def parse_altimetry_txt(file_path: Path) -> pd.DataFrame:
    """
    Parse satellite altimetry TXT files (Lake Victoria, Lake Kyoga) covering
    all satellite missions (TOPEX, JASN1, JASN2, JASN3, SEN6A, etc.) through 2026.
    """
    pattern = re.compile(
        r"^(TOPEX|POSDN|JASN1|JASN2|JASN3|JASN4|SEN6A|SENT6A|SENT3A|SENT3B|ENVIS|ERS1|ERS2|GFO|SARAL|CRYOS|HY2A|HY2B|ICESAT)"
    )

    records = []
    with open(file_path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line or not pattern.match(line):
                continue
            parts = line.split()
            if len(parts) >= 16:
                mission = parts[0]
                date_str = parts[2]
                try:
                    rel_level = float(parts[5])
                    err = float(parts[6])
                    egm2008 = float(parts[14])
                    # Filter out default fill / missing values (999.99, etc.)
                    if rel_level < 800.0 and err < 10.0:
                        records.append({
                            "date": pd.to_datetime(date_str, format="%Y%m%d"),
                            "water_level_rel": rel_level,
                            "height_egm2008": egm2008,
                            "mission": mission,
                        })
                except (ValueError, IndexError):
                    continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values("date").drop_duplicates(subset=["date"])
    df = df.set_index("date")
    return df


def load_lake_albert(file_path: Path) -> pd.DataFrame:
    """Load NetCDF altimetry for Lake Albert."""
    ds = xr.open_dataset(file_path)
    df = ds.to_dataframe().reset_index()
    # Normalize timestamp to date (midnight)
    df["date"] = pd.to_datetime(df["datetime"]).dt.floor("D")
    # Clean outlier values and aggregate duplicate observations on same day
    df = df[df["water_level"] > 600.0]
    df = df.groupby("date")[["water_level", "error"]].mean()
    df = df.rename(columns={"water_level": "water_level_albert_m", "error": "error_albert_m"})
    ds.close()
    return df


def load_all_upstream_lakes() -> pd.DataFrame:
    """Loads and resamples all 3 upstream lakes into a unified daily DataFrame."""
    lakes_dir = DATA_ROOT / "Water levels lakes"

    print("  -> Loading Lake Victoria...")
    df_vic = parse_altimetry_txt(lakes_dir / "water_level_victoria.txt")
    df_vic = df_vic.rename(columns={
        "water_level_rel": "vic_rel_m",
        "height_egm2008": "vic_egm2008_m"
    })[["vic_rel_m", "vic_egm2008_m"]]

    print("  -> Loading Lake Kyoga...")
    df_kyo = parse_altimetry_txt(lakes_dir / "water_level_Kyoga.txt")
    df_kyo = df_kyo.rename(columns={
        "water_level_rel": "kyo_rel_m",
        "height_egm2008": "kyo_egm2008_m"
    })[["kyo_rel_m", "kyo_egm2008_m"]]

    print("  -> Loading Lake Albert...")
    df_alb = load_lake_albert(lakes_dir / "water_level_altimetry_Albert.nc")

    # Resample each to daily (linear interpolation for gaps <= 30 days)
    daily_idx = pd.date_range(start="1998-01-01", end="2025-12-31", freq="D")
    combined = pd.DataFrame(index=daily_idx)

    for name, df_lake, col in [
        ("Lake Victoria", df_vic, "vic_egm2008_m"),
        ("Lake Kyoga", df_kyo, "kyo_egm2008_m"),
        ("Lake Albert", df_alb, "water_level_albert_m"),
    ]:
        reindexed = df_lake[[col]].reindex(daily_idx)
        # Interpolate small measurement gaps between satellite overpasses
        interpolated = reindexed[col].interpolate(method="time", limit=35)
        combined[col] = interpolated

    return combined


# ==============================================================================
# 3. River Discharge Loaders
# ==============================================================================
def load_dartmouth_stations() -> pd.DataFrame:
    """Load Dartmouth Flood Observatory discharge series for key stations."""
    dfo_dir = DATA_ROOT / "Darthmouth Flood Observatory"

    stations = {
        "100205": "discharge_malakal_m3s",  # South Sudan (White Nile / Malakal)
        "1541": "discharge_station_1541_m3s",  # Sudan border (White Nile)
        "1505": "discharge_station_1505_m3s",  # Ethiopia (Omo / Lake Turkana basin)
    }

    daily_idx = pd.date_range(start="1998-01-01", end="2025-12-31", freq="D")
    df_discharge = pd.DataFrame(index=daily_idx)

    for st_id, col_name in stations.items():
        fpath = dfo_dir / f"{st_id}_discharge.csv"
        if fpath.exists():
            df = pd.read_csv(fpath)
            df["date"] = pd.to_datetime(df["Date"])
            df = df.set_index("date").sort_index()
            # Clean possible negative or invalid discharge values
            val_col = [c for c in df.columns if "discharge" in c.lower()][0]
            series = df[val_col].copy()
            series[series < 0] = np.nan
            series = series.reindex(daily_idx)
            df_discharge[col_name] = series

    return df_discharge


# ==============================================================================
# 4. Anomaly and Surge Detection Calculations
# ==============================================================================
def compute_anomalies_and_surges(df: pd.DataFrame, col: str, baseline_years=(2000, 2018)):
    """
    Computes climatological baseline, rolling means, z-score anomalies, and surge flags.
    Surge defined as z-score >= +1.5 standard deviations sustained over a rolling window.
    """
    series = df[col].dropna()
    baseline = series[(series.index.year >= baseline_years[0]) & (series.index.year <= baseline_years[1])]

    if len(baseline) == 0:
        base_mean = series.mean()
        base_std = series.std()
    else:
        base_mean = baseline.mean()
        base_std = baseline.std()

    # 30-day rolling mean to remove short-term high frequency noise
    rolling_30d = df[col].rolling(30, min_periods=7, center=True).mean()
    z_score = (rolling_30d - base_mean) / base_std
    surge_flag = (z_score >= 1.5).astype(int)

    return base_mean, base_std, rolling_30d, z_score, surge_flag


# ==============================================================================
# 5. Visualization Functions
# ==============================================================================
def plot_lake_trends(df: pd.DataFrame, save_path: Path):
    """Plot multi-lake elevation levels and long-term trends."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    lake_configs = [
        ("vic_egm2008_m", "Lake Victoria (Upstream Headwaters, Uganda/TZ/KE)", "navy", 1135.0),
        ("kyo_egm2008_m", "Lake Kyoga (Middle Lake, Uganda)", "teal", 1033.0),
        ("water_level_albert_m", "Lake Albert (Direct Inflow to Sudd, Uganda/DRC)", "darkred", 621.5),
    ]

    for ax, (col, title, color, ref_line) in zip(axes, lake_configs):
        ax.plot(df.index, df[col], label=f"Observed Elevation (m)", color=color, alpha=0.6, linewidth=1.2)
        ax.plot(df.index, df[f"{col}_rolling30"], label="30-day Rolling Mean", color="black", linewidth=1.5)

        # Highlight post-2019 unprecedented surge
        ax.axvspan(pd.Timestamp("2019-10-01"), pd.Timestamp("2024-12-31"), color="gold", alpha=0.18,
                   label="2019-2024 Unprecedented High-Water Period")

        ax.axhline(ref_line, color="gray", linestyle="--", alpha=0.6, label="Reference Baseline")
        ax.set_ylabel("Water Level [m]", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

    axes[-1].set_xlabel("Year", fontsize=12)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xlim(pd.Timestamp("2000-01-01"), pd.Timestamp("2025-12-31"))
    plt.suptitle("Upstream Great Lakes Water Level Dynamics (2000-2025)", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  [SAVED] {save_path.name}")


def plot_discharge_and_anomalies(df: pd.DataFrame, save_path: Path):
    """Plot river discharge at Malakal (South Sudan) with z-score anomaly bars."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    col = "discharge_malakal_m3s"
    ax1.plot(df.index, df[col], color="royalblue", alpha=0.5, label="Daily Discharge (m³/s)")
    ax1.plot(df.index, df[f"{col}_rolling30"], color="midnightblue", linewidth=1.8, label="30-Day Rolling Mean")
    ax1.axhline(df[col].mean(), color="crimson", linestyle="--", label=f"Long-term Mean ({df[col].mean():.0f} m³/s)")
    ax1.set_ylabel("Discharge [m³/s]", fontsize=11)
    ax1.set_title("River Discharge at Malakal, South Sudan (Station 100205)", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", fontsize=10)

    # Anomaly bar plot
    z_col = f"{col}_zscore"
    pos_mask = df[z_col] >= 0
    neg_mask = df[z_col] < 0
    ax2.bar(df.index[pos_mask], df[z_col][pos_mask], width=2, color="crimson", alpha=0.7, label="Positive Anomaly (+Z)")
    ax2.bar(df.index[neg_mask], df[z_col][neg_mask], width=2, color="dodgerblue", alpha=0.7, label="Negative Anomaly (-Z)")
    ax2.axhline(1.5, color="darkred", linestyle=":", linewidth=1.5, label="High Surge Threshold (+1.5σ)")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Standardized Anomaly (z)", fontsize=11)
    ax2.set_xlabel("Year", fontsize=12)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", fontsize=9)

    plt.xlim(pd.Timestamp("2000-01-01"), pd.Timestamp("2025-12-31"))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  [SAVED] {save_path.name}")


def plot_combined_surge_timeline(df: pd.DataFrame, save_path: Path):
    """Plot synchronized normalized timeline comparing Lake Albert, Lake Victoria, and Malakal discharge."""
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df.index, df["vic_egm2008_m_zscore"], label="Lake Victoria Level (z-score)", color="blue", linewidth=1.5)
    ax.plot(df.index, df["water_level_albert_m_zscore"], label="Lake Albert Level (z-score)", color="green", linewidth=1.5)
    ax.plot(df.index, df["discharge_malakal_m3s_zscore"], label="Malakal River Discharge (z-score)", color="red", linewidth=1.5)

    ax.axhline(1.5, color="black", linestyle="--", linewidth=1, label="Surge Threshold (+1.5σ)")
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.8)

    ax.fill_between(df.index, 1.5, np.maximum(df["water_level_albert_m_zscore"], 1.5),
                    where=(df["water_level_albert_m_zscore"] >= 1.5), color="lightcoral", alpha=0.4, label="Upstream Surge Zone")

    ax.set_ylabel("Normalized Anomaly [Standard Deviations (σ)]", fontsize=11)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_title("Synchronized Upstream Driver Anomalies & Flood Surge Genesis (2000-2025)", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    plt.xlim(pd.Timestamp("2005-01-01"), pd.Timestamp("2025-12-31"))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  [SAVED] {save_path.name}")


# ==============================================================================
# 6. Main Execution Pipeline
# ==============================================================================
def main():
    print("=" * 70)
    print("STEP 1: UPSTREAM TRENDS & ANOMALY DETECTION")
    print("=" * 70)

    # 1. Load data
    df_lakes = load_all_upstream_lakes()
    df_discharge = load_dartmouth_stations()

    combined = df_lakes.join(df_discharge, how="outer")
    combined.index.name = "date"

    # 2. Calculate rolling averages, anomalies, and surge detection
    target_cols = [
        ("vic_egm2008_m", "Lake Victoria"),
        ("kyo_egm2008_m", "Lake Kyoga"),
        ("water_level_albert_m", "Lake Albert"),
        ("discharge_malakal_m3s", "Malakal Discharge"),
        ("discharge_station_1541_m3s", "Station 1541 Discharge"),
    ]

    surge_summary = []

    print("\n[ANALYSIS] Calculating Climatological Baselines & Surge Anomalies:")
    for col, label in target_cols:
        if col in combined.columns:
            mean, std, roll30, z, surge = compute_anomalies_and_surges(combined, col)
            combined[f"{col}_rolling30"] = roll30
            combined[f"{col}_zscore"] = z
            combined[f"{col}_surge_flag"] = surge

            # Identify surge intervals
            surge_days = surge.sum()
            recent_surge_days = surge[surge.index.year >= 2019].sum()
            max_val = combined[col].max()
            max_date = combined[col].idxmax()

            surge_summary.append({
                "variable": label,
                "column": col,
                "baseline_mean": round(mean, 2),
                "baseline_std": round(std, 2),
                "all_time_max": round(max_val, 2),
                "max_date": str(max_date.date()) if pd.notna(max_date) else "N/A",
                "total_surge_days": int(surge_days),
                "surge_days_post_2019": int(recent_surge_days),
            })
            print(f"  * {label}: Baseline Mean = {mean:.2f}, Std = {std:.2f}, Max = {max_val:.2f} ({max_date.date()})")
            print(f"    Total Surge Days (> +1.5σ): {surge_days} days ({recent_surge_days} days occurred since 2019!)")

    # 3. Save datasets
    csv_timeseries_path = OUTPUT_DATA_DIR / "upstream_water_levels_and_discharge.csv"
    combined.to_csv(csv_timeseries_path)
    print(f"\n[DATA] Saved unified upstream time-series to:\n  -> {csv_timeseries_path}")

    df_surge = pd.DataFrame(surge_summary)
    csv_surge_path = OUTPUT_DATA_DIR / "upstream_surge_events.csv"
    df_surge.to_csv(csv_surge_path, index=False)
    print(f"[DATA] Saved surge summary table to:\n  -> {csv_surge_path}")

    # 4. Generate Visualizations
    print("\n[VISUALIZATION] Generating Publication Figures:")
    plot_lake_trends(combined, OUTPUT_FIG_DIR / "01_upstream_water_levels_multilake_trend.png")
    plot_discharge_and_anomalies(combined, OUTPUT_FIG_DIR / "01_upstream_discharge_trends_and_anomalies.png")
    plot_combined_surge_timeline(combined, OUTPUT_FIG_DIR / "01_upstream_lake_discharge_surge_timeline.png")

    print("\n[COMPLETE] Step 1 Upstream Trends & Anomaly Detection successfully finished.\n")


if __name__ == "__main__":
    main()
