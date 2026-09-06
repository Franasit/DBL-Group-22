import xarray as xr
import pandas as pd
from pathlib import Path

file_path = Path("raw_data") / "rainfall_and_runoff" / "ERA5_2025.nc"

ds = xr.open_dataset(file_path)

# Choose one location
point = ds.sel(
    latitude=0,
    longitude=32,
    method="nearest"
)

# Show first 24 hours
table = pd.DataFrame({
    "time": point["valid_time"].values[:24],
    "tp_m": point["tp"].values[:24],
    "tp_mm": point["tp"].values[:24] * 1000,
    "ro_m": point["ro"].values[:24],
    "ro_mm": point["ro"].values[:24] * 1000,
})

print(table)