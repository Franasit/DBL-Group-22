import os
import glob
import sys
import os
# Ensure script runs from the project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np

directories = [
    "/Users/maksadamowicz/Desktop/DBL-Group-22/Water_levels_lakes",
    "/Users/maksadamowicz/Desktop/DBL-Group-22/Darthmouth_Flood_Observatory"
]

def check_csv(file_path):
    folder_name = os.path.basename(os.path.dirname(file_path))
    file_name = os.path.basename(file_path)
    
    stats = {
        "Folder Name": folder_name,
        "File Name": file_name,
        "Rows": 0,
        "Columns": 0,
        "Data Types": None,
        "Missing Values Count": 0,
        "Temporal Coverage": None,
        "Spatial Coverage": None,
        "Resolution": None,
        "CRS": "Not natively supported in CSV",
        "Completely Empty Columns": None,
        "Status": "Success"
    }

    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        stats["Status"] = f"Error reading file: {e}"
        return stats
    
    if df.empty:
        stats["Status"] = "File is empty"
        return stats
        
    stats["Rows"] = len(df)
    stats["Columns"] = len(df.columns)
    
    # Store data types as a string summary
    stats["Data Types"] = ", ".join([f"{col} ({dtype})" for col, dtype in df.dtypes.items()])
    
    # Missing Values
    missing = df.isnull().sum()
    stats["Missing Values Count"] = missing.sum()
    
    # Temporal Coverage
    time_cols = [c for c in df.columns if any(x in c.lower() for x in ['date', 'time', 'year', 'month'])]
    temporal_coverage = []
    for c in time_cols:
        try:
            dt = pd.to_datetime(df[c], errors='coerce')
            if dt.notna().any():
                temporal_coverage.append(f"{c}: {dt.min()} to {dt.max()}")
            elif pd.api.types.is_numeric_dtype(df[c]):
                temporal_coverage.append(f"{c}: {df[c].min()} to {df[c].max()}")
        except:
            pass
    if temporal_coverage:
        stats["Temporal Coverage"] = " | ".join(temporal_coverage)

    # Spatial Coverage
    lat_cols = [c for c in df.columns if 'lat' in c.lower() or 'y' == c.lower()]
    lon_cols = [c for c in df.columns if 'lon' in c.lower() or 'lng' in c.lower() or 'x' == c.lower()]
    spatial_coverage = []
    for c in lat_cols + lon_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            spatial_coverage.append(f"{c}: min {df[c].min()}, max {df[c].max()}")
    if spatial_coverage:
        stats["Spatial Coverage"] = " | ".join(spatial_coverage)
        
    # Resolution
    resolutions = []
    for c in time_cols + lat_cols + lon_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            try:
                unique_sorted = np.sort(df[c].dropna().unique())
                if len(unique_sorted) > 1:
                    diffs = np.diff(unique_sorted)
                    median_diff = np.median(diffs[diffs > 0]) # avoid zero diffs
                    if not np.isnan(median_diff):
                         resolutions.append(f"{c}: ~{median_diff}")
            except:
                pass
    if resolutions:
        stats["Resolution"] = " | ".join(resolutions)
        
    # File Completeness
    empty_cols = [col for col in df.columns if df[col].isnull().all()]
    if empty_cols:
        stats["Completely Empty Columns"] = ", ".join(empty_cols)
    else:
        stats["Completely Empty Columns"] = "None"
        
    return stats

if __name__ == "__main__":
    all_stats = []
    
    for d in directories:
        if not os.path.exists(d):
            print(f"Directory not found: {d}")
            continue
        
        csv_files = glob.glob(os.path.join(d, "**", "*.csv"), recursive=True)
        print(f"Processing {len(csv_files)} CSV files in {d}...")
        
        for f in csv_files:
            all_stats.append(check_csv(f))
            
    if all_stats:
        results_df = pd.DataFrame(all_stats)
        
        # Displaying the DataFrame
        print("\n--- Summary DataFrame ---")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(results_df.head(20)) # Print first 20 rows to avoid console clutter
        print(f"\nTotal files processed: {len(results_df)}")
        
        # Saving to CSV for convenience
        output_csv = "/Users/maksadamowicz/Desktop/DBL-Group-22/csv_summary_stats.csv"
        results_df.to_csv(output_csv, index=False)
        print(f"\nSaved full results to: {output_csv}")

