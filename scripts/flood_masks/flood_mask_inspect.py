import pandas as pd
from pathlib import Path

base_dir = Path("raw_data") / "flood_masks"

files = {
    "compact_unusual_2025": (
        base_dir
        / "compact_unusual"
        / "flood_events_h20v08_2025.parquet"
    ),
    "compact_recurring_2025": (
        base_dir
        / "compact_recurring"
        / "flood_events_h20v08_2025.parquet"
    ),
}

for name, file_path in files.items():

    print("\n" + "=" * 80)
    print(f"DATASET: {name}")
    print(f"FILE: {file_path}")
    print("=" * 80)

    if not file_path.exists():
        print("File not found.")
        continue

    df = pd.read_parquet(file_path)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 10 rows:")
    print(df.head(10))

    print("\nDate range:")
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"].astype(str))
        print(dates.min(), "to", dates.max())

    print("\nLatitude range:")
    if "lat" in df.columns:
        print(df["lat"].min(), "to", df["lat"].max())

    print("\nLongitude range:")
    if "lon" in df.columns:
        print(df["lon"].min(), "to", df["lon"].max())

    print("\nTile values:")
    if "tile" in df.columns:
        print(df["tile"].unique())

    print("\nCloud fraction summary:")
    if "cloud_frac" in df.columns:
        print(df["cloud_frac"].describe())