from pathlib import Path
import re

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_ROOT = PROJECT_ROOT / "raw_data"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "water_level_qc"
FIGURE_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Research period
# ============================================================

RESEARCH_START = pd.Timestamp("2013-01-01")
RESEARCH_END = pd.Timestamp("2025-12-31")


# ============================================================
# 3. File finder
# ============================================================

def find_file(search_root, patterns):
    """
    Search recursively below search_root and return the first matching file.

    patterns should be a list such as:
        ["*Albert*.nc", "*albert*.nc"]
    """

    matches = []

    for pattern in patterns:
        matches.extend(search_root.rglob(pattern))

    # Remove duplicate paths
    matches = list(dict.fromkeys(matches))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"\nNo file found below:\n{search_root}\n"
            f"matching any of these patterns:\n{patterns}\n"
        )

    if len(matches) > 1:
        print("\nMultiple matching files found:")

        for match in matches:
            print(f"  - {match}")

        print(f"\nUsing first match:\n{matches[0]}\n")

    return matches[0]


# ============================================================
# 4. Automatically locate files
# ============================================================

ALBERT_FILE = find_file(
    RAW_ROOT,
    [
        "*Albert*.nc",
        "*albert*.nc",
    ],
)

KYOGA_FILE = find_file(
    RAW_ROOT,
    [
        "*Kyoga*.txt",
        "*kyoga*.txt",
    ],
)

VICTORIA_FILE = find_file(
    RAW_ROOT,
    [
        "*Victoria*.txt",
        "*victoria*.txt",
    ],
)


# ============================================================
# 5. General helper functions
# ============================================================

def safe_float(value):
    """
    Convert a value to float.
    Return NaN if conversion fails.
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def calculate_gap_statistics(dates):
    """
    Calculate spacing between consecutive valid observations.
    """

    dates = pd.Series(
        pd.to_datetime(dates, errors="coerce")
    )

    dates = (
        dates
        .dropna()
        .sort_values()
        .drop_duplicates()
    )

    if len(dates) < 2:
        return {
            "min_gap_days": np.nan,
            "median_gap_days": np.nan,
            "mean_gap_days": np.nan,
            "max_gap_days": np.nan,
        }

    gaps = (
        dates.diff()
        .dropna()
        .dt.total_seconds()
        / 86400
    )

    return {
        "min_gap_days": float(gaps.min()),
        "median_gap_days": float(gaps.median()),
        "mean_gap_days": float(gaps.mean()),
        "max_gap_days": float(gaps.max()),
    }


# ============================================================
# 6. Read Lake Albert NetCDF
# ============================================================

def read_albert(file_path):
    """
    Read Lake Albert NetCDF file.

    Expected variables are likely:
        datetime
        water_level
        error
        latitude / lat
        longitude / lon
    """

    print("\n" + "=" * 80)
    print("READING LAKE ALBERT")
    print("=" * 80)

    print(f"File: {file_path}")

    ds = xr.open_dataset(file_path)

    print("\nDataset structure:")
    print(ds)

    print("\nAvailable variables:")
    print(list(ds.variables))

    # --------------------------------------------------------
    # Detect time variable
    # --------------------------------------------------------

    possible_time_vars = [
        "datetime",
        "time",
        "date",
        "dates",
    ]

    time_var = None

    for candidate in possible_time_vars:
        if candidate in ds.variables:
            time_var = candidate
            break

    if time_var is None:
        ds.close()

        raise ValueError(
            "Could not automatically identify a time variable.\n"
            f"Available variables:\n{list(ds.variables)}"
        )

    # --------------------------------------------------------
    # Detect water-level variable
    # --------------------------------------------------------

    possible_level_vars = [
        "water_level",
        "waterlevel",
        "water_level_altimetry",
        "level",
        "height",
        "elevation",
    ]

    level_var = None

    for candidate in possible_level_vars:
        if candidate in ds.variables:
            level_var = candidate
            break

    if level_var is None:
        ds.close()

        raise ValueError(
            "Could not automatically identify the water-level variable.\n"
            f"Available variables:\n{list(ds.variables)}"
        )

    # --------------------------------------------------------
    # Detect error variable
    # --------------------------------------------------------

    possible_error_vars = [
        "error",
        "water_level_error",
        "uncertainty",
        "sigma",
        "std",
    ]

    error_var = None

    for candidate in possible_error_vars:
        if candidate in ds.variables:
            error_var = candidate
            break

    # --------------------------------------------------------
    # Convert to dataframe
    # --------------------------------------------------------

    date_values = pd.to_datetime(
        np.asarray(ds[time_var].values).reshape(-1),
        errors="coerce",
    )

    level_values = pd.to_numeric(
        np.asarray(ds[level_var].values).reshape(-1),
        errors="coerce",
    )

    data = {
        "date": date_values,
        "water_level": level_values,
    }

    if error_var is not None:

        error_values = pd.to_numeric(
            np.asarray(ds[error_var].values).reshape(-1),
            errors="coerce",
        )

        data["error"] = error_values

    else:
        data["error"] = np.nan

    df = pd.DataFrame(data)

    # --------------------------------------------------------
    # Get coordinates
    # --------------------------------------------------------

    latitude = np.nan
    longitude = np.nan

    for candidate in ["latitude", "lat"]:

        if candidate in ds.variables:

            try:
                latitude = float(
                    np.asarray(ds[candidate].values)
                    .reshape(-1)[0]
                )
            except Exception:
                pass

            break

    for candidate in ["longitude", "lon"]:

        if candidate in ds.variables:

            try:
                longitude = float(
                    np.asarray(ds[candidate].values)
                    .reshape(-1)[0]
                )
            except Exception:
                pass

            break

    ds.close()

    print(f"\nDetected time variable: {time_var}")
    print(f"Detected water-level variable: {level_var}")
    print(f"Detected error variable: {error_var}")

    print(f"Latitude: {latitude}")
    print(f"Longitude: {longitude}")

    return df, latitude, longitude


# ============================================================
# 7. Read Kyoga / Victoria TPJOJS TXT files
# ============================================================

def read_altimetry_txt(file_path):
    """
    Read TPJOJS lake-altimetry text files.

    According to the supplied file documentation:

    Column 1  = satellite mission
    Column 2  = repeat cycle
    Column 3  = YYYYMMDD date
    Column 4  = hour
    Column 5  = minute
    Column 6  = water-level variation relative to reference
    Column 7  = estimated error
    Column 8  = backscatter coefficient
    Column 9  = wet tropospheric correction
    Column 10 = ionosphere correction
    Column 11 = dry tropospheric correction
    Column 12 = instrument mode 1
    Column 13 = instrument mode 2
    Column 14 = frozen-surface flag
    Column 15 = EGM2008 water level
    Column 16 = data source flag

    Main water-level variable used for QC:
        Column 15

    Sentinel/default invalid values include:
        date = 99999999
        relative level = 999.99
        error = 99.999
        backscatter = 999.99
        EGM2008 level = 9999.99
    """

    print("\n" + "=" * 80)
    print(f"READING {file_path.name}")
    print("=" * 80)

    print(f"File: {file_path}")

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as file:
        lines = file.readlines()

    latitude = np.nan
    longitude = np.nan

    # --------------------------------------------------------
    # Extract target midpoint coordinates from header
    # --------------------------------------------------------

    for line in lines[:50]:

        if "Latitude and longitude" in line:

            coordinate_text = line.split(":")[0].strip()

            coordinate_parts = coordinate_text.split()

            if len(coordinate_parts) >= 2:

                latitude = safe_float(
                    coordinate_parts[0]
                )

                longitude = safe_float(
                    coordinate_parts[1]
                )

            break

    print(f"Latitude: {latitude}")
    print(f"Longitude: {longitude}")

    # --------------------------------------------------------
    # Recognized mission names
    # --------------------------------------------------------

    data_line_pattern = re.compile(
        r"^(TOPEX|POSDN|JASN1|JASN2|JASN3|JASN4|"
        r"SENT6A|SENT3A|SENT3B|ENVIS|ERS1|ERS2|"
        r"GFO|SARAL|CRYOS|HY2A|HY2B|ICESAT)"
    )

    rows = []

    for line in lines:

        stripped = line.strip()

        if not stripped:
            continue

        if not data_line_pattern.match(stripped):
            continue

        parts = stripped.split()

        if len(parts) < 16:
            continue

        mission = parts[0]

        cycle = safe_float(parts[1])

        date_raw = parts[2]

        hour = safe_float(parts[3])
        minute = safe_float(parts[4])

        relative_level = safe_float(parts[5])
        error = safe_float(parts[6])
        backscatter = safe_float(parts[7])

        wet_tropo = parts[8]
        ionosphere = parts[9]
        dry_tropo = parts[10]

        mode_1 = safe_float(parts[11])
        mode_2 = safe_float(parts[12])

        frozen_flag = safe_float(parts[13])

        water_level_egm2008 = safe_float(parts[14])

        source_flag = safe_float(parts[15])

        # ----------------------------------------------------
        # Date cleaning
        # ----------------------------------------------------

        if date_raw == "99999999":

            date = pd.NaT

        else:

            date = pd.to_datetime(
                date_raw,
                format="%Y%m%d",
                errors="coerce",
            )

        # ----------------------------------------------------
        # Sentinel values -> NaN
        # ----------------------------------------------------

        if relative_level == 999.99:
            relative_level = np.nan

        if error == 99.999:
            error = np.nan

        if backscatter == 999.99:
            backscatter = np.nan

        if water_level_egm2008 == 9999.99:
            water_level_egm2008 = np.nan

        rows.append(
            {
                "mission": mission,
                "cycle": cycle,
                "date": date,
                "hour": hour,
                "minute": minute,
                "relative_water_level": relative_level,
                "error": error,
                "backscatter": backscatter,
                "wet_tropo_correction": wet_tropo,
                "ionosphere_correction": ionosphere,
                "dry_tropo_correction": dry_tropo,
                "mode_1": mode_1,
                "mode_2": mode_2,
                "frozen_flag": frozen_flag,
                "water_level": water_level_egm2008,
                "source_flag": source_flag,
            }
        )

    df = pd.DataFrame(rows)

    print(f"Parsed rows: {len(df)}")

    return df, latitude, longitude


# ============================================================
# 8. Common QC
# ============================================================

def calculate_common_qc(
    df,
    lake_name,
    source_file,
    latitude=np.nan,
    longitude=np.nan,
):
    """
    Calculate main QC statistics.
    """

    total_rows = len(df)

    valid_dates = int(
        df["date"]
        .notna()
        .sum()
    )

    valid_water_levels = int(
        df["water_level"]
        .notna()
        .sum()
    )

    missing_dates = int(
        df["date"]
        .isna()
        .sum()
    )

    missing_water_levels = int(
        df["water_level"]
        .isna()
        .sum()
    )

    duplicate_dates = int(
        df.loc[
            df["date"].notna(),
            "date"
        ]
        .duplicated()
        .sum()
    )

    valid_df = df[
        df["date"].notna()
        & df["water_level"].notna()
    ].copy()

    valid_df = valid_df.sort_values("date")

    # --------------------------------------------------------
    # Time coverage
    # --------------------------------------------------------

    if not valid_df.empty:

        first_date = valid_df["date"].min()
        last_date = valid_df["date"].max()

    else:

        first_date = pd.NaT
        last_date = pd.NaT

    # --------------------------------------------------------
    # Water-level statistics
    # --------------------------------------------------------

    if not valid_df.empty:

        water_level_min = float(
            valid_df["water_level"].min()
        )

        water_level_max = float(
            valid_df["water_level"].max()
        )

        water_level_mean = float(
            valid_df["water_level"].mean()
        )

        water_level_median = float(
            valid_df["water_level"].median()
        )

        water_level_std = float(
            valid_df["water_level"].std()
        )

    else:

        water_level_min = np.nan
        water_level_max = np.nan
        water_level_mean = np.nan
        water_level_median = np.nan
        water_level_std = np.nan

    # --------------------------------------------------------
    # Error statistics
    # --------------------------------------------------------

    if "error" in df.columns:

        valid_errors = df["error"].dropna()

    else:

        valid_errors = pd.Series(dtype=float)

    if len(valid_errors) > 0:

        error_min = float(
            valid_errors.min()
        )

        error_max = float(
            valid_errors.max()
        )

        error_mean = float(
            valid_errors.mean()
        )

        error_median = float(
            valid_errors.median()
        )

    else:

        error_min = np.nan
        error_max = np.nan
        error_mean = np.nan
        error_median = np.nan

    # --------------------------------------------------------
    # Observation gaps
    # --------------------------------------------------------

    gap_stats = calculate_gap_statistics(
        valid_df["date"]
    )

    # --------------------------------------------------------
    # Research-period overlap
    # --------------------------------------------------------

    research_df = valid_df[
        (valid_df["date"] >= RESEARCH_START)
        &
        (valid_df["date"] <= RESEARCH_END)
    ]

    research_observations = len(
        research_df
    )

    if research_observations > 0:

        research_first_date = (
            research_df["date"].min()
        )

        research_last_date = (
            research_df["date"].max()
        )

    else:

        research_first_date = pd.NaT
        research_last_date = pd.NaT

    # --------------------------------------------------------
    # Percentage calculations
    # --------------------------------------------------------

    if total_rows > 0:

        valid_pct = (
            valid_water_levels
            / total_rows
            * 100
        )

        missing_pct = (
            missing_water_levels
            / total_rows
            * 100
        )

    else:

        valid_pct = np.nan
        missing_pct = np.nan

    summary = {
        "lake": lake_name,
        "source_file": source_file.name,
        "source_path": str(source_file),

        "latitude": latitude,
        "longitude": longitude,

        "total_rows": total_rows,

        "valid_dates": valid_dates,
        "missing_dates": missing_dates,

        "valid_water_levels": valid_water_levels,
        "missing_water_levels": missing_water_levels,

        "valid_water_level_pct": valid_pct,
        "missing_water_level_pct": missing_pct,

        "duplicate_dates": duplicate_dates,

        "first_valid_date": first_date,
        "last_valid_date": last_date,

        "water_level_min_m": water_level_min,
        "water_level_max_m": water_level_max,
        "water_level_mean_m": water_level_mean,
        "water_level_median_m": water_level_median,
        "water_level_std_m": water_level_std,

        "error_min_m": error_min,
        "error_max_m": error_max,
        "error_mean_m": error_mean,
        "error_median_m": error_median,

        "research_period_start": RESEARCH_START,
        "research_period_end": RESEARCH_END,

        "research_period_valid_observations": research_observations,
        "research_period_first_date": research_first_date,
        "research_period_last_date": research_last_date,

        **gap_stats,
    }

    return summary


# ============================================================
# 9. Missing-value summary
# ============================================================

def create_missing_summary(
    df,
    lake_name,
):
    """
    Missing-value summary per variable.
    """

    preferred_columns = [
        "date",
        "water_level",
        "relative_water_level",
        "error",
        "backscatter",
    ]

    columns_to_check = [
        column
        for column in preferred_columns
        if column in df.columns
    ]

    rows = []

    for column in columns_to_check:

        missing_count = int(
            df[column]
            .isna()
            .sum()
        )

        if len(df) > 0:

            missing_pct = (
                missing_count
                / len(df)
                * 100
            )

        else:

            missing_pct = np.nan

        rows.append(
            {
                "lake": lake_name,
                "variable": column,
                "total_rows": len(df),
                "missing_count": missing_count,
                "missing_pct": missing_pct,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 10. Annual observation coverage
# ============================================================

def create_annual_coverage(
    df,
    lake_name,
):
    """
    Count valid water-level observations per year.
    """

    valid_df = df[
        df["date"].notna()
        &
        df["water_level"].notna()
    ].copy()

    if valid_df.empty:

        return pd.DataFrame(
            columns=[
                "lake",
                "year",
                "valid_observations",
            ]
        )

    valid_df["year"] = (
        valid_df["date"]
        .dt.year
    )

    annual_df = (
        valid_df
        .groupby("year")
        .size()
        .reset_index(
            name="valid_observations"
        )
    )

    annual_df.insert(
        0,
        "lake",
        lake_name,
    )

    return annual_df


# ============================================================
# 11. Plot raw water-level time series
# ============================================================

def plot_timeseries(
    df,
    lake_name,
):
    """
    Create and save raw water-level time-series plot.
    """

    plot_df = df[
        df["date"].notna()
        &
        df["water_level"].notna()
    ].copy()

    if plot_df.empty:

        print(
            f"No valid water-level data "
            f"available for {lake_name}."
        )

        return

    plot_df = plot_df.sort_values(
        "date"
    )

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        plot_df["date"],
        plot_df["water_level"],
        linewidth=1,
    )

    plt.title(
        f"{lake_name} Raw Water-Level Time Series"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Water level (m)"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    file_name = (
        lake_name
        .lower()
        .replace(" ", "_")
        + "_raw_timeseries.png"
    )

    output_path = (
        FIGURE_DIR
        / file_name
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved figure:\n{output_path}"
    )


# ============================================================
# 12. Print lake-level summary
# ============================================================

def print_lake_summary(
    lake_name,
    df,
):
    """
    Display useful quick checks in Terminal.
    """

    print("\n" + "-" * 80)
    print(f"{lake_name.upper()} QUICK CHECK")
    print("-" * 80)

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Valid dates: "
        f"{df['date'].notna().sum()}"
    )

    print(
        f"Valid water levels: "
        f"{df['water_level'].notna().sum()}"
    )

    if df["date"].notna().any():

        print(
            f"First date: "
            f"{df['date'].min()}"
        )

        print(
            f"Last date: "
            f"{df['date'].max()}"
        )

    if df["water_level"].notna().any():

        print(
            f"Minimum water level: "
            f"{df['water_level'].min():.3f}"
        )

        print(
            f"Maximum water level: "
            f"{df['water_level'].max():.3f}"
        )


# ============================================================
# 13. Main
# ============================================================

def main():

    print("\n")
    print("=" * 80)
    print("WATER LEVEL RAW DATA QUALITY CHECK")
    print("=" * 80)

    print("\nDetected input files:")

    print(
        f"Albert:\n{ALBERT_FILE}"
    )

    print(
        f"\nKyoga:\n{KYOGA_FILE}"
    )

    print(
        f"\nVictoria:\n{VICTORIA_FILE}"
    )

    summaries = []
    annual_coverages = []
    missing_summaries = []

    # ========================================================
    # Lake Albert
    # ========================================================

    albert_df, albert_lat, albert_lon = (
        read_albert(
            ALBERT_FILE
        )
    )

    print_lake_summary(
        "Albert",
        albert_df,
    )

    summaries.append(
        calculate_common_qc(
            df=albert_df,
            lake_name="Albert",
            source_file=ALBERT_FILE,
            latitude=albert_lat,
            longitude=albert_lon,
        )
    )

    annual_coverages.append(
        create_annual_coverage(
            albert_df,
            "Albert",
        )
    )

    missing_summaries.append(
        create_missing_summary(
            albert_df,
            "Albert",
        )
    )

    plot_timeseries(
        albert_df,
        "Albert",
    )

    # ========================================================
    # Lake Kyoga
    # ========================================================

    kyoga_df, kyoga_lat, kyoga_lon = (
        read_altimetry_txt(
            KYOGA_FILE
        )
    )

    print_lake_summary(
        "Kyoga",
        kyoga_df,
    )

    summaries.append(
        calculate_common_qc(
            df=kyoga_df,
            lake_name="Kyoga",
            source_file=KYOGA_FILE,
            latitude=kyoga_lat,
            longitude=kyoga_lon,
        )
    )

    annual_coverages.append(
        create_annual_coverage(
            kyoga_df,
            "Kyoga",
        )
    )

    missing_summaries.append(
        create_missing_summary(
            kyoga_df,
            "Kyoga",
        )
    )

    plot_timeseries(
        kyoga_df,
        "Kyoga",
    )

    # ========================================================
    # Lake Victoria
    # ========================================================

    victoria_df, victoria_lat, victoria_lon = (
        read_altimetry_txt(
            VICTORIA_FILE
        )
    )

    print_lake_summary(
        "Victoria",
        victoria_df,
    )

    summaries.append(
        calculate_common_qc(
            df=victoria_df,
            lake_name="Victoria",
            source_file=VICTORIA_FILE,
            latitude=victoria_lat,
            longitude=victoria_lon,
        )
    )

    annual_coverages.append(
        create_annual_coverage(
            victoria_df,
            "Victoria",
        )
    )

    missing_summaries.append(
        create_missing_summary(
            victoria_df,
            "Victoria",
        )
    )

    plot_timeseries(
        victoria_df,
        "Victoria",
    )

    # ========================================================
    # Combine QC outputs
    # ========================================================

    summary_df = pd.DataFrame(
        summaries
    )

    annual_coverage_df = pd.concat(
        annual_coverages,
        ignore_index=True,
    )

    missing_summary_df = pd.concat(
        missing_summaries,
        ignore_index=True,
    )

    # ========================================================
    # Save output files
    # ========================================================

    summary_path = (
        OUTPUT_DIR
        / "water_level_qc_summary.csv"
    )

    annual_path = (
        OUTPUT_DIR
        / "water_level_annual_observation_coverage.csv"
    )

    missing_path = (
        OUTPUT_DIR
        / "water_level_missing_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    annual_coverage_df.to_csv(
        annual_path,
        index=False,
    )

    missing_summary_df.to_csv(
        missing_path,
        index=False,
    )

    # ========================================================
    # Terminal summary
    # ========================================================

    print("\n")
    print("=" * 80)
    print("QC SUMMARY")
    print("=" * 80)

    display_columns = [
        "lake",
        "total_rows",
        "valid_water_levels",
        "missing_water_levels",
        "valid_water_level_pct",
        "first_valid_date",
        "last_valid_date",
        "median_gap_days",
        "max_gap_days",
        "research_period_valid_observations",
    ]

    print(
        summary_df[
            display_columns
        ]
        .round(3)
        .to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)

    print(
        f"\nQC summary:\n{summary_path}"
    )

    print(
        f"\nAnnual coverage:\n{annual_path}"
    )

    print(
        f"\nMissing-value summary:\n{missing_path}"
    )

    print(
        f"\nFigures:\n{FIGURE_DIR}"
    )

    print("\n")
    print("=" * 80)
    print("WATER-LEVEL QC COMPLETED")
    print("=" * 80)


# ============================================================
# 14. Run script
# ============================================================

if __name__ == "__main__":
    main()