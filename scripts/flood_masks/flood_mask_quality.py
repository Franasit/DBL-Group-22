import pandas as pd
from pathlib import Path
import re

# --------------------------------------------------
# 1. Define folders
# --------------------------------------------------
BASE_DIR = Path("raw_data") / "flood_masks"

DATASET_DIRS = {
    "compact_unusual": BASE_DIR / "compact_unusual",
    "compact_recurring": BASE_DIR / "compact_recurring",
}

OUTPUT_DIR = Path("processed_data") / "flood_masks_processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Helper function: extract tile and year
# --------------------------------------------------
def extract_tile_year(file_name):
    """
    Example:
    flood_events_h20v08_2025.parquet
    -> tile = h20v08
    -> year = 2025
    """

    match = re.search(r"(h\d+v\d+)_(\d{4})", file_name)

    if match:
        tile = match.group(1)
        year = int(match.group(2))
        return tile, year

    return None, None


# --------------------------------------------------
# 3. Store QC results
# --------------------------------------------------
results = []

# --------------------------------------------------
# 4. Loop through both dataset types
# --------------------------------------------------
for dataset_type, folder in DATASET_DIRS.items():

    print("\n" + "=" * 80)
    print(f"Checking dataset: {dataset_type}")
    print(f"Folder: {folder}")
    print("=" * 80)

    if not folder.exists():
        print(f"Folder not found: {folder}")
        continue

    files = sorted(folder.glob("*.parquet"))

    print(f"Found {len(files)} parquet files.")

    for file_path in files:

        print(f"\nChecking: {file_path.name}")

        tile_from_name, year_from_name = extract_tile_year(file_path.name)

        try:
            df = pd.read_parquet(file_path)

            # ------------------------------------------
            # Basic structure
            # ------------------------------------------
            rows = len(df)
            columns = df.columns.tolist()

            expected_columns = {
                "date",
                "lat",
                "lon",
                "tile",
                "cloud_frac",
            }

            columns_ok = expected_columns.issubset(set(columns))

            # ------------------------------------------
            # Date processing
            # ------------------------------------------
            date_start = None
            date_end = None
            missing_date = None
            wrong_year_rows = None

            if "date" in df.columns:

                dates = pd.to_datetime(
                    df["date"].astype(str),
                    errors="coerce"
                )

                missing_date = int(dates.isna().sum())

                if dates.notna().any():
                    date_start = dates.min()
                    date_end = dates.max()

                    if year_from_name is not None:
                        wrong_year_rows = int(
                            (dates.dt.year != year_from_name).sum()
                        )

            # ------------------------------------------
            # Latitude / longitude
            # ------------------------------------------
            lat_min = None
            lat_max = None
            lon_min = None
            lon_max = None

            missing_lat = None
            missing_lon = None

            if "lat" in df.columns:
                missing_lat = int(df["lat"].isna().sum())

                if df["lat"].notna().any():
                    lat_min = float(df["lat"].min())
                    lat_max = float(df["lat"].max())

            if "lon" in df.columns:
                missing_lon = int(df["lon"].isna().sum())

                if df["lon"].notna().any():
                    lon_min = float(df["lon"].min())
                    lon_max = float(df["lon"].max())

            # ------------------------------------------
            # Tile checks
            # ------------------------------------------
            tile_values = None
            tile_matches_file = None
            missing_tile = None

            if "tile" in df.columns:

                missing_tile = int(df["tile"].isna().sum())

                tile_values = sorted(
                    df["tile"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                if tile_from_name is not None:
                    tile_matches_file = (
                        len(tile_values) == 1
                        and tile_values[0] == tile_from_name
                    )

            # ------------------------------------------
            # Cloud fraction
            # ------------------------------------------
            cloud_min = None
            cloud_max = None
            cloud_mean = None
            missing_cloud = None

            if "cloud_frac" in df.columns:

                missing_cloud = int(
                    df["cloud_frac"].isna().sum()
                )

                if df["cloud_frac"].notna().any():

                    cloud_min = float(
                        df["cloud_frac"].min()
                    )

                    cloud_max = float(
                        df["cloud_frac"].max()
                    )

                    cloud_mean = float(
                        df["cloud_frac"].mean()
                    )

            # ------------------------------------------
            # Duplicate spatial-date records
            # ------------------------------------------
            duplicate_date_lat_lon = None

            if {"date", "lat", "lon"}.issubset(df.columns):

                duplicate_date_lat_lon = int(
                    df.duplicated(
                        subset=["date", "lat", "lon"]
                    ).sum()
                )

            # ------------------------------------------
            # Overall status
            # ------------------------------------------
            status = "OK"

            if rows == 0:
                status = "CHECK"

            if not columns_ok:
                status = "CHECK"

            if missing_date not in (None, 0):
                status = "CHECK"

            if missing_lat not in (None, 0):
                status = "CHECK"

            if missing_lon not in (None, 0):
                status = "CHECK"

            if missing_tile not in (None, 0):
                status = "CHECK"

            if tile_matches_file is False:
                status = "CHECK"

            if wrong_year_rows not in (None, 0):
                status = "CHECK"

            if duplicate_date_lat_lon not in (None, 0):
                status = "CHECK"

            # ------------------------------------------
            # Save one row
            # ------------------------------------------
            results.append({
                "dataset_type": dataset_type,
                "file": file_path.name,
                "tile": tile_from_name,
                "year": year_from_name,

                "rows": rows,

                "columns_ok": columns_ok,
                "columns": ", ".join(columns),

                "date_start": date_start,
                "date_end": date_end,
                "wrong_year_rows": wrong_year_rows,

                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,

                "tile_values": ", ".join(tile_values)
                if tile_values
                else None,

                "tile_matches_file": tile_matches_file,

                "cloud_min": cloud_min,
                "cloud_max": cloud_max,
                "cloud_mean": cloud_mean,

                "missing_date": missing_date,
                "missing_lat": missing_lat,
                "missing_lon": missing_lon,
                "missing_tile": missing_tile,
                "missing_cloud": missing_cloud,

                "duplicate_date_lat_lon":
                    duplicate_date_lat_lon,

                "status": status,
            })

        except Exception as e:

            print(f"ERROR: {e}")

            results.append({
                "dataset_type": dataset_type,
                "file": file_path.name,
                "tile": tile_from_name,
                "year": year_from_name,
                "status": "ERROR",
                "error": str(e),
            })


# --------------------------------------------------
# 5. Convert results into DataFrame
# --------------------------------------------------
qc_df = pd.DataFrame(results)

if not qc_df.empty:

    qc_df = qc_df.sort_values(
        by=[
            "dataset_type",
            "tile",
            "year",
            "file",
        ],
        na_position="last"
    )

# --------------------------------------------------
# 6. Save summary
# --------------------------------------------------
output_file = (
    OUTPUT_DIR
    / "flood_mask_qc_summary.csv"
)

qc_df.to_csv(
    output_file,
    index=False
)

# --------------------------------------------------
# 7. Print summary
# --------------------------------------------------
print("\n" + "=" * 80)
print("QC COMPLETE")
print("=" * 80)

print(f"\nTotal files checked: {len(qc_df)}")

if not qc_df.empty and "status" in qc_df.columns:

    print("\nStatus counts:")
    print(qc_df["status"].value_counts(dropna=False))

print(f"\nSaved QC summary to:")
print(output_file)