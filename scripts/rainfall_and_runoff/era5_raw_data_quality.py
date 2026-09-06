import xarray as xr
import pandas as pd
from pathlib import Path
import calendar

# -----------------------------
# 1. Define folders
# -----------------------------
raw_dir = Path("raw_data") / "rainfall_and_runoff"
output_dir = Path("processed_data") / "rainfall_and_runoff_processed"

# Make sure output folder exists
output_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 2. Find all ERA5 NetCDF files
# -----------------------------
files = sorted(raw_dir.glob("ERA5_*.nc"))

print(f"Found {len(files)} ERA5 files.")

# Store QC results here
results = []

# -----------------------------
# 3. Loop through every file
# -----------------------------
for file_path in files:

    print(f"\nChecking: {file_path.name}")

    # Extract year from file name
    year = int(file_path.stem.split("_")[1].replace("(1)", ""))

    try:
        ds = xr.open_dataset(file_path)

        # -----------------------------
        # Basic dimensions
        # -----------------------------
        n_time = ds.sizes.get("valid_time", 0)
        n_lat = ds.sizes.get("latitude", 0)
        n_lon = ds.sizes.get("longitude", 0)

        # Expected hours
        if calendar.isleap(year):
            expected_hours = 8784
        else:
            expected_hours = 8760

        # -----------------------------
        # Variable presence
        # -----------------------------
        has_tp = "tp" in ds.data_vars
        has_ro = "ro" in ds.data_vars

        # -----------------------------
        # Time checks
        # -----------------------------
        duplicate_times = 0
        missing_hours = None

        if "valid_time" in ds.coords:
            times = pd.DatetimeIndex(ds["valid_time"].values)

            duplicate_times = times.duplicated().sum()

            expected_times = pd.date_range(
                start=f"{year}-01-01 00:00:00",
                end=f"{year}-12-31 23:00:00",
                freq="h"
            )

            missing_hours = len(expected_times.difference(times))

        # -----------------------------
        # Grid resolution
        # -----------------------------
        lat_step = None
        lon_step = None

        if n_lat > 1:
            lat_values = ds["latitude"].values
            lat_step = abs(float(lat_values[1] - lat_values[0]))

        if n_lon > 1:
            lon_values = ds["longitude"].values
            lon_step = abs(float(lon_values[1] - lon_values[0]))

        # -----------------------------
        # tp checks
        # -----------------------------
        tp_nan = None
        tp_min = None
        tp_max = None
        tp_negative = None
        tp_unit = None

        if has_tp:
            tp = ds["tp"]

            tp_nan = int(tp.isnull().sum().item())
            tp_min = float(tp.min(skipna=True).item())
            tp_max = float(tp.max(skipna=True).item())
            tp_negative = int((tp < 0).sum().item())
            tp_unit = tp.attrs.get("units", "unknown")

        # -----------------------------
        # ro checks
        # -----------------------------
        ro_nan = None
        ro_min = None
        ro_max = None
        ro_negative = None
        ro_unit = None

        if has_ro:
            ro = ds["ro"]

            ro_nan = int(ro.isnull().sum().item())
            ro_min = float(ro.min(skipna=True).item())
            ro_max = float(ro.max(skipna=True).item())
            ro_negative = int((ro < 0).sum().item())
            ro_unit = ro.attrs.get("units", "unknown")

        # -----------------------------
        # Overall status
        # -----------------------------
        status = "OK"

        if n_time != expected_hours:
            status = "CHECK"

        if missing_hours != 0:
            status = "CHECK"

        if duplicate_times != 0:
            status = "CHECK"

        if not has_tp or not has_ro:
            status = "CHECK"

        # -----------------------------
        # Save one row
        # -----------------------------
        results.append({
            "year": year,
            "file": file_path.name,
            "hours": n_time,
            "expected_hours": expected_hours,
            "missing_hours": missing_hours,
            "duplicate_times": duplicate_times,
            "latitude_points": n_lat,
            "longitude_points": n_lon,
            "latitude_step": lat_step,
            "longitude_step": lon_step,
            "has_tp": has_tp,
            "has_ro": has_ro,
            "tp_unit": tp_unit,
            "ro_unit": ro_unit,
            "tp_nan": tp_nan,
            "ro_nan": ro_nan,
            "tp_min": tp_min,
            "tp_max": tp_max,
            "ro_min": ro_min,
            "ro_max": ro_max,
            "tp_negative": tp_negative,
            "ro_negative": ro_negative,
            "status": status
        })

        ds.close()

    except Exception as e:

        results.append({
            "year": year,
            "file": file_path.name,
            "status": "ERROR",
            "error": str(e)
        })

        print(f"Error: {e}")


# -----------------------------
# 4. Convert results to a table
# -----------------------------
qc_df = pd.DataFrame(results)

# Sort by year
qc_df = qc_df.sort_values("year")

# -----------------------------
# 5. Save QC report
# -----------------------------
output_file = output_dir / "era5_qc_summary.csv"

qc_df.to_csv(output_file, index=False)

print("\nQC complete.")
print(qc_df)

print(f"\nSaved to: {output_file}")