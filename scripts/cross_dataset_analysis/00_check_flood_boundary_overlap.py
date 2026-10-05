from pathlib import Path
import pandas as pd
import geopandas as gpd


PROJECT_ROOT = Path("/Users/shibai/Documents/GitHub/DBL-Group-22")

# ============================================================
# CHANGE THESE ONLY IF NECESSARY
# ============================================================

# Use ONE flood file only for checking
FLOOD_FILE = (
    PROJECT_ROOT
    / "raw_data"
    / "flood_masks"
    / "flood_events_h21v08_2025.parquet"
)

# If the above path does not exist, the script will search automatically.


def find_one_flood_file():
    if FLOOD_FILE.exists():
        return FLOOD_FILE

    files = list(PROJECT_ROOT.rglob("flood_events_h21v08_2025*.parquet"))

    if not files:
        files = list(PROJECT_ROOT.rglob("flood_events*.parquet"))

    if not files:
        raise FileNotFoundError("No flood parquet file found.")

    return files[0]


def find_all_boundaries():
    files = []

    for ext in ["*.shp", "*.gpkg", "*.geojson"]:
        files.extend(PROJECT_ROOT.rglob(ext))

    return sorted(set(files))


def main():

    print("=" * 80)
    print("CHECK FLOOD / BOUNDARY SPATIAL OVERLAP")
    print("=" * 80)

    # ========================================================
    # 1. FLOOD RANGE
    # ========================================================

    flood_file = find_one_flood_file()

    print("\nFlood file:")
    print(flood_file)

    df = pd.read_parquet(flood_file)

    print("\nFlood columns:")
    print(df.columns.tolist())

    # detect coordinates
    lat_col = None
    lon_col = None

    for c in df.columns:
        c_low = str(c).lower()

        if c_low in ["lat", "latitude"]:
            lat_col = c

        if c_low in ["lon", "lng", "longitude", "long"]:
            lon_col = c

    if lat_col is None or lon_col is None:
        raise ValueError("Could not identify lat/lon columns.")

    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")

    print("\nFLOOD COORDINATE RANGE")
    print("-" * 50)

    print(
        f"Latitude : "
        f"{df[lat_col].min():.6f} "
        f"to "
        f"{df[lat_col].max():.6f}"
    )

    print(
        f"Longitude: "
        f"{df[lon_col].min():.6f} "
        f"to "
        f"{df[lon_col].max():.6f}"
    )

    # ========================================================
    # 2. CHECK EVERY BOUNDARY
    # ========================================================

    print("\n" + "=" * 80)
    print("BOUNDARY FILES")
    print("=" * 80)

    boundary_files = find_all_boundaries()

    print(f"\nFound {len(boundary_files)} spatial files.\n")

    flood_min_lon = df[lon_col].min()
    flood_max_lon = df[lon_col].max()
    flood_min_lat = df[lat_col].min()
    flood_max_lat = df[lat_col].max()

    possible_matches = []

    for i, path in enumerate(boundary_files, start=1):

        try:
            gdf = gpd.read_file(path)

            if gdf.empty:
                continue

            if gdf.crs is None:
                crs_text = "NONE"
                gdf = gdf.set_crs("EPSG:4326")
            else:
                crs_text = str(gdf.crs)

            gdf_wgs = gdf.to_crs("EPSG:4326")

            minx, miny, maxx, maxy = gdf_wgs.total_bounds

            overlap = (
                maxx >= flood_min_lon
                and minx <= flood_max_lon
                and maxy >= flood_min_lat
                and miny <= flood_max_lat
            )

            print("-" * 80)
            print(f"[{i}] {path}")
            print(f"CRS: {crs_text}")

            print(
                "Bounds:"
                f" lon {minx:.4f} to {maxx:.4f},"
                f" lat {miny:.4f} to {maxy:.4f}"
            )

            print("Columns:")
            print(gdf.columns.tolist())

            print(
                "Bounding-box overlap with flood:",
                "YES" if overlap else "NO"
            )

            if overlap:
                possible_matches.append(path)

        except Exception as exc:
            print("-" * 80)
            print(f"[{i}] {path}")
            print(f"Could not read: {exc}")

    # ========================================================
    # 3. SUMMARY
    # ========================================================

    print("\n" + "=" * 80)
    print("POSSIBLE MATCHING BOUNDARIES")
    print("=" * 80)

    if possible_matches:

        for path in possible_matches:
            print(path)

    else:
        print(
            "\nNo boundary has a bounding-box overlap "
            "with this flood file."
        )

    print(
        "\nExpected South Sudan approximate extent:"
        "\nLongitude: roughly 24E-36E"
        "\nLatitude : roughly 3N-13N"
    )


if __name__ == "__main__":
    main()